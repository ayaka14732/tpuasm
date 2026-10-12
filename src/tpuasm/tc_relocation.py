"""在 TC 程序 overlay 内插入、删除 bundle，迁移机器分支、overlay 装载参数与编译器元数据。"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

from ._protobuf import message, replace_fields
from .assembly_model import Form
from .assembly_syntax import AssemblyProgram, branch_target, integer, parse_assembly
from .printer import _verify_image
from .targets import HardwareTarget, TPU_V4_TC, TPU_V6E_TC
from .tc_solver import BundleSolver
from .tc_source_mapping import _ENCODED_WORD_BYTES, Overlay, _Program, _one, _overlays, _text, _values

# sbr.rel 与 scall.rel 位移字段的位数。v4 的字段只有 16 位，放不下的位移由编译器按 65536 回绕；v6e 的字段足够放下任何 overlay 内的位移。
_DISPLACEMENT_BITS = {TPU_V4_TC.identifier: 16, TPU_V6E_TC.identifier: 20}

@dataclass(frozen=True)
class BundleInsertion:
    """在原映像的 image_pc 之前插入汇编片段，并可以同时删除从 image_pc 起的若干个原 bundle。

    source 含 .target 声明和一个或多个 bundle，无需块对齐。片段中的直接分支只能用相对形式（sbr.rel、scall.rel），引用片段内标签或片段局部编号（允许指向末尾以继续执行原程序）；程序 overlay 装入指令存储的位置在运行时确定，片段中不能使用 sbr.abs 和 scall.abs。branch_target='inserted' 使原程序中指向 image_pc 的直接分支先执行片段；'original' 则跳过片段。fallthrough 总会执行片段。多个插入点的编号均相对于输入映像，可以落在不同的程序 overlay 中。

    delete 是从 image_pc 起删除的原 bundle 个数，默认不删；delete 大于零时 source 可以只有 .target 声明（纯删除）。删除的 bundle 必须在同一个 overlay 的正文内，其中不能有分支、call、shalt，不能落在原分支的延迟窗口里，也不能是除 image_pc 之外的分支目标；指向 image_pc 的分支按 branch_target 落到片段开头或删除区间之后的第一个 bundle。
    """
    image_pc: int
    source: str
    branch_target: Literal['inserted', 'original'] = 'inserted'
    delete: int = 0

def _set(data: bytes, number: int, value: int | bytes) -> bytes:
    return replace_fields(data, {number: [value]})

def _main_limit(metadata: bytes) -> int:
    ranges = [r for table in _values(metadata, 8) for r in _values(table, 2) if _one(r, 4) == 1]
    if not ranges:
        raise ValueError('bundle insertion requires TensorCore symbol ranges')
    return max(_one(r, 2) for r in ranges)

def _relocate_metadata(metadata: bytes, overlays: Sequence[Overlay], growth: Mapping[int, tuple[int, int]], emitted: Mapping[int, int], counts: Mapping[int, int], deletes: Mapping[int, int], block_bundles: int, block_words: int, old_size: int, new_size: int, trap_halts: set[int]) -> bytes:
    # overlays 是全部程序 overlay；growth[index] 是该 overlay 的 (净增 bundle 数, 新增对齐填充)；emitted 把编辑点的映像位置换成发射坐标。
    # 前一个 overlay 正文末尾与后一个 overlay 正文开头的发射坐标相同，两处的插入在发射坐标中合并。
    edits: dict[int, int] = {}
    for pc, count in counts.items():
        edits[emitted[pc]] = edits.get(emitted[pc], 0) + count
    removed = {emitted[pc]: count for pc, count in deletes.items() if count}
    by_index = {overlay.index: overlay for overlay in overlays}
    emitted_limit = overlays[-1].emitted_limit
    main_limit = _main_limit(metadata)

    def gone(pc: int) -> bool:
        return any(at <= pc < at + count for at, count in removed.items())

    def shifted(pc: int) -> int:
        # 调用者保证 pc 本身没有被删除。
        return pc + sum(count for at, count in edits.items() if at <= pc) - sum(min(max(pc - at, 0), count) for at, count in removed.items())

    def trap_location(location: bytes) -> bytes:
        # v4 TagAndPc identifies the bundle following shalt, relative to the image start of the overlay named by the tag.
        # Validate the actual halt instead of assuming every metadata PC uses this coordinate system.
        pc = _one(location, 2)
        overlay = by_index.get(_one(location, 1))
        halt = overlay.image_start + pc - 1 if overlay is not None else -1
        if overlay is None or _one(location, 3) != 1 or halt not in trap_halts:
            raise ValueError('unsupported trap tag, sequencer or halt PC coordinate')
        return _set(location, 2, pc + sum(count for at, count in counts.items() if overlay.image_start <= at <= halt) - sum(min(max(halt - at, 0), count) for at, count in deletes.items() if at >= overlay.image_start))

    def traps(table: bytes) -> bytes:
        if _values(table, 3):
            raise ValueError('compressed trap metadata relocation is not supported')
        entries = []
        for entry in _values(table, 2):
            locations = _values(entry, 1)
            if len(locations) != 1:
                raise ValueError('trap entry needs one tag and PC')
            entries.append(_set(entry, 1, trap_location(locations[0])))
        return replace_fields(table, {1: [trap_location(t) for t in _values(table, 1)], 2: entries})

    def ranges(data: bytes) -> list[bytes]:
        if _one(data, 4) != 1:
            return [data]
        start, limit = _one(data, 1), _one(data, 2)
        if not 0 <= start < limit <= main_limit:
            raise ValueError('unsupported TensorCore symbol range')
        # 区间在每个编辑点断开：插入的 bundle 不属于任何符号，删除的 bundle 从区间里去掉。
        pieces = []
        current = start
        for at in sorted(edits):
            if at >= limit:
                break
            if current < at:
                pieces.append((current, at))
            end = at + removed.get(at, 0)
            if end > start:
                current = max(current, end)
        if current < limit:
            pieces.append((current, limit))
        return [replace_fields(data, {1: [shifted(a)], 2: [shifted(b - 1) + 1]}) for a, b in pieces]

    def symbol_table(table: bytes) -> bytes:
        symbols = []
        for entry in _values(table, 1):
            symbol = _one(entry, 2, b'')
            children = [part for r in _values(symbol, 23) for part in ranges(r)]
            symbols.append(_set(entry, 2, replace_fields(symbol, {23: children})))
        return replace_fields(table, {1: symbols, 2: [part for r in _values(table, 2) for part in ranges(r)]})

    def annotations(table: bytes) -> bytes:
        entries = []
        for entry in _values(table, 1):
            key, annotation = _one(entry, 1), _one(entry, 2, b'')
            pc = _one(annotation, 1)
            if gone(key):
                continue
            if key < main_limit and pc == key:
                annotation = _set(annotation, 1, shifted(pc))
            elif not (main_limit <= key < emitted_limit and pc == key - main_limit):
                raise ValueError('unsupported annotation key/PC coordinate relation')
            entries.append(replace_fields(entry, {1: [shifted(key)], 2: [annotation]}))
        # New instructions have no original source or symbol owner.
        for at, count in sorted(edits.items()):
            first = at + sum(other for position, other in edits.items() if position < at) - sum(min(max(at - position, 0), other) for position, other in removed.items())
            for pc in range(first, first + count):
                entries.append(message([(1, pc), (2, message([(1, pc), (3, b'tpuasm: inserted bundle')]))]))
        return replace_fields(table, {1: entries})

    tables = _values(metadata, 10)
    raw_overlays = _values(tables[0], 1)
    # 发射坐标和 encoded_word_offset 都连续编号：一个 overlay 变长，其后的 overlay 整体后移。
    emitted_shift = word_shift = 0
    for overlay in overlays:
        added, padding = growth[overlay.index]
        moved = {2: (overlay.emitted_start, emitted_shift), 3: (overlay.emitted_limit, emitted_shift + added), 4: (overlay.suffix_size, padding), 5: (overlay.encoded_word_offset, word_shift)}
        raw_overlays[overlay.index] = replace_fields(raw_overlays[overlay.index], {number: [value + delta] for number, (value, delta) in moved.items() if delta})
        emitted_shift += added
        word_shift += (added + padding) // block_bundles * block_words
    tables[0] = replace_fields(tables[0], {1: raw_overlays})
    memory_tables = _values(metadata, 12)
    matched = 0
    for i, table in enumerate(memory_tables):
        summaries = _values(table, 1)
        for j, summary in enumerate(summaries):
            if _one(summary, 1) != 0:  # HBM
                continue
            blocks = _values(summary, 3)
            resized = False
            for k, block in enumerate(blocks):
                if _one(block, 1) != 5:  # OVERLAYS
                    continue
                allocations = _values(block, 4)
                if len(allocations) != 1 or _one(allocations[0], 1) != b'overlays' or _one(allocations[0], 4) != old_size or _one(block, 2) != old_size:
                    raise ValueError('unsupported HBM overlay allocation')
                matched += 1
                resized = True
                allocations[0] = _set(allocations[0], 4, new_size)
                blocks[k] = replace_fields(block, {2: [new_size], 4: allocations})
            if resized:
                summaries[j] = replace_fields(summary, {2: [_one(summary, 2) + new_size - old_size], 3: blocks})
        memory_tables[i] = replace_fields(table, {1: summaries})
    if matched != 1 or _one(metadata, 18) != old_size:
        raise ValueError('metadata code size does not match the program image')
    if any(_values(metadata, n) for n in (32, 35, 39, 43)):
        raise ValueError('relocation of fusion or breakpoint metadata is not supported')
    annotation_tables = _values(metadata, 4) or [b'']
    return replace_fields(metadata, {4: [annotations(t) for t in annotation_tables], 8: [symbol_table(t) for t in _values(metadata, 8)], 10: tables, 11: [traps(t) for t in _values(metadata, 11)], 12: memory_tables, 18: [new_size]})

def _routine(routines: Sequence[tuple[int, str]], name: str) -> int:
    found = [pc for pc, text in routines if text == name]
    if len(found) != 1:
        raise ValueError(f'insertion requires the {name} annotation')
    return found[0]

def _immediate(operand: str) -> int | None:
    try:
        return integer(operand)
    except ValueError:
        return None

def _overlay_capacity(original: AssemblyProgram, routines: Sequence[tuple[int, str]]) -> int:
    """一个程序 overlay 最多占用的块数，读自前缀中的装载例程。

    routine:branch 把指令存储中第一个可用的块号放进 s7，分配例程用 ``ssub.s32 s6, 总块数, s4`` 计算从当前位置到指令存储末尾还剩多少块。装不下时例程清空分配表，再从第一个可用块重新装载，所以一个 overlay 的上限是两者之差。
    """
    branch = _routine(routines, 'routine:branch')
    allocate = _routine(routines, 'routine:allocation-table-get-or-add')
    limit = min(pc for pc, _ in routines if pc > allocate)
    first = [integer(i.operands[1]) for i in original.bundles[branch].instructions if i.mnemonic == 'simm.s32' and i.operands[0] == 's7']
    total = [value for bundle in original.bundles[allocate:limit] for i in bundle.instructions if i.mnemonic == 'ssub.s32' and (value := _immediate(i.operands[1])) is not None]
    if len(first) != 1 or len(total) != 1:
        raise ValueError('cannot identify the instruction memory limits in the overlay loader')
    return total[0] - first[0]

def insert_program(program: _Program, insertions: list[BundleInsertion], hardware: HardwareTarget) -> tuple[bytes, bytes]:
    if hardware == TPU_V4_TC:
        from .tpu_v4_tc_assembler import ISA, _decoded_source as decode_v4
        from .tpu_v4_tc_codec import encode_program
    else:
        from .tpu_v6e_tc_assembler import ISA, _decoded_source as decode_v6e
        from .tpu_v6e_tc_codec import encode_program
    if program.diagnostic or len(_values(program.metadata, 10)) != 1:
        raise ValueError('insertion requires one code image with a prelude and at least one program overlay')
    original, encoded_raw = decode_v4(program.image) if hardware == TPU_V4_TC else decode_v6e(program.image)
    encoded: list[tuple[int, tuple[Form, ...]]] = [(word, tuple(forms)) for word, forms in encoded_raw]
    # 只读 overlay 表和前缀例程的名称；完整的来源映射要解码每个槽的来源记录，大程序上比插入本身慢得多。
    table = _overlays(program.metadata, hardware, len(original.bundles))
    if len(table) < 2:
        raise ValueError('insertion requires one code image with a prelude and at least one program overlay')
    prelude, *overlays = table
    if prelude.image_start != 0 or prelude.emitted_start != prelude.emitted_limit:
        raise ValueError('unsupported prelude overlay layout')
    # 程序 overlay 在映像和发射坐标中都首尾相接，各自按块对齐。
    image_limit, emitted_limit = prelude.body_limit + prelude.suffix_size, 0
    for overlay in overlays:
        if overlay.image_start != image_limit or overlay.emitted_start != emitted_limit or overlay.hlo_function_overlay or overlay.image_start % hardware.bundles_per_block:
            raise ValueError('unsupported program overlay layout')
        image_limit, emitted_limit = overlay.body_limit + overlay.suffix_size, overlay.emitted_limit
    if image_limit != len(original.bundles):
        raise ValueError('unsupported program overlay layout')
    routines = [(_one(entry, 1), _text(_one(entry, 2, b''), 3)) for table in _values(program.metadata, 30) for entry in _values(table, 1)]
    main_limit = _main_limit(program.metadata)
    if not 0 < main_limit <= emitted_limit:
        raise ValueError('TensorCore symbol ranges exceed the program overlays')
    if any(pc >= overlays[0].image_start for pc, _ in routines):
        raise ValueError('overlayer annotations outside the prelude are not supported')
    delay = hardware.branch_delay_bundles
    if delay is None:
        raise ValueError('branch delay is not established for this target')
    delayed = {
        pc + after: (pc, instruction.mnemonic)
        for pc, bundle in enumerate(original.bundles)
        for instruction in bundle.instructions if instruction.mnemonic.startswith(('sbr.', 'scall.'))
        for after in range(1, delay + 1)
    }
    fragments = {}
    policies = {}
    deletes: dict[int, int] = {}
    owners: dict[int, Overlay] = {}
    emitted: dict[int, int] = {}
    for insertion in insertions:
        pc = insertion.image_pc
        if pc in fragments:
            raise ValueError(f'duplicate insertion position {pc}; combine its bundles in one fragment')
        if insertion.branch_target not in ('inserted', 'original'):
            raise ValueError('branch_target must be inserted or original')
        # 正文之后的第一个 bundle 之前也可以插入：片段接在该 overlay 正文末尾，位于它的出口 trampoline 之前。
        owner = next((overlay for overlay in overlays if overlay.body_start <= pc < overlay.body_limit + min(overlay.suffix_size, 1)), None)
        if owner is None or not owner.emitted_start + pc - owner.body_start < main_limit:
            raise ValueError(f'insertion {pc} is outside the main program; insertion into the prelude, an overlay prefix or trampoline, or the continuation is not supported')
        if pc in delayed:
            branch_pc, mnemonic = delayed[pc]
            raise ValueError(f'insertion {pc} is in the {delay}-bundle delay window of {mnemonic} at {branch_pc}; insert before that branch or after bundle {branch_pc + delay}')
        fragment = parse_assembly(insertion.source, filename=f'<insertion:{pc}>', fragment=True)
        if insertion.delete < 0 or fragment.hardware != hardware or not (fragment.bundles or insertion.delete):
            raise ValueError('insertion needs nonempty assembly for the executable target, or a positive delete count')
        for gone in range(pc, pc + insertion.delete):
            if not gone < min(owner.body_limit, owner.body_start + main_limit - owner.emitted_start):
                raise ValueError(f'deletion at {pc} runs past the main program or its overlay')
            if gone in delayed or any(i.mnemonic.startswith(('sbr.', 'scall.')) or i.mnemonic == 'shalt' for i in original.bundles[gone].instructions):
                raise ValueError(f'cannot delete bundle {gone}: it holds a branch, call or shalt, or lies in a branch delay window')
        for local, extra in enumerate(fragment.bundles):
            for instruction in extra.instructions:
                # 程序 overlay 装入指令存储的位置在运行时确定，片段中的绝对地址没有意义。
                if instruction.mnemonic in ('sbr.abs', 'scall.abs'):
                    raise ValueError(f'insertion {pc}: {instruction.mnemonic} at fragment bundle {local} needs an absolute address, but program overlays are loaded at a run-time address; use the .rel form')
                # A delay window reaching past the fragment would turn the following original bundles into delay slots.
                if instruction.mnemonic.startswith(('sbr.', 'scall.')) and local + delay >= len(fragment.bundles):
                    raise ValueError(f'insertion {pc}: {instruction.mnemonic} at fragment bundle {local} needs its {delay}-bundle delay window inside the fragment; append empty bundles')
        fragments[pc], policies[pc], deletes[pc], owners[pc], emitted[pc] = fragment, insertion.branch_target, insertion.delete, owner, owner.emitted_start + pc - owner.body_start
    deleted = {gone for pc, count in deletes.items() for gone in range(pc, pc + count)}
    if sum(deletes.values()) != len(deleted) or any(pc in deleted and deletes[pc] == 0 for pc in fragments) or any(pc != at and at < pc < at + count for pc in fragments for at, count in deletes.items()):
        raise ValueError('deleted ranges must not overlap each other or contain another insertion position')
    counts = {pc: len(f.bundles) for pc, f in fragments.items()}
    # 每个 overlay 单独按块装载：各自补齐到整块，其后的 overlay 整块后移。
    growth: dict[int, tuple[int, int]] = {}
    for overlay in overlays:
        added = sum(counts[pc] - deletes[pc] for pc, owner in owners.items() if owner == overlay)
        growth[overlay.index] = added, -added % hardware.bundles_per_block
    # An overlay's alignment padding follows its last original bundle; keyed by the image position where that overlay ends.
    paddings = {overlay.body_limit + overlay.suffix_size: growth[overlay.index][1] for overlay in overlays}
    starts = [overlay.image_start for overlay in overlays]

    def shifted(pc: int, *, target: bool = False) -> int:
        # 被删除的 bundle 只有作为分支目标、且是删除区间的第一个时才有新位置：片段开头，或片段之后（即区间后的第一个原 bundle）。
        if pc in deleted and not (target and deletes.get(pc)):
            raise ValueError(f'a direct branch targets deleted bundle {pc}')
        inserted = sum(count for at, count in counts.items() if at < pc or (at == pc and (not target or policies[at] == 'original')))
        return pc + inserted - sum(1 for gone in deleted if gone < pc) + sum(padding for limit, padding in paddings.items() if limit <= pc)

    # Keep the original suffix, including a final continuation branch and its delay slot.
    # Alignment padding is fresh halt bundles, never copies of a possibly live final instruction.
    halt = parse_assembly(f'.target {hardware.identifier}\n{{ s0: shalt }}\n', fragment=True)
    halt_word = BundleSolver(ISA, halt.bundles[0], halt, 0).solve()
    # The loader names an overlay by its encoded_word_offset in blocks, which exceeds the image position when the first overlay does not start at offset 0, and takes its size in blocks.
    # Every trampoline passes the overlay to enter in (s0, s1) and the one to prefetch, if any, in (s2, s3); find them all, because overlays after an edited one move.
    block_words = hardware.image_block_size // _ENCODED_WORD_BYTES[hardware.identifier]
    capacity = _overlay_capacity(original, routines)
    loaded: dict[tuple[int, int], tuple[int, int]] = {}
    moved_blocks = 0
    for overlay in overlays:
        blocks = (overlay.body_limit + overlay.suffix_size - overlay.image_start) // hardware.bundles_per_block
        new_blocks = blocks + sum(growth[overlay.index]) // hardware.bundles_per_block
        if new_blocks > capacity:
            raise ValueError(f'overlay {overlay.index} would need {new_blocks} blocks of instruction memory, but the overlay loader has room for {capacity}')
        loaded[overlay.encoded_word_offset // block_words, blocks] = overlay.encoded_word_offset // block_words + moved_blocks, new_blocks
        moved_blocks += new_blocks - blocks
    branch_routine = _routine(routines, 'routine:branch')
    loaders: dict[int, dict[str, int]] = {}
    for pc, bundle in enumerate(original.bundles):
        if not any(branch_target(i.mnemonic, i.operands, pc) == (int(i.mnemonic.startswith('scall')), branch_routine) and i.mnemonic.endswith('.abs') for i in bundle.instructions):
            continue
        for at, registers, required in ((pc + 1, ('s0', 's1'), True), (pc - 1, ('s2', 's3'), False)):
            immediates = {i.operands[0]: integer(i.operands[1]) for i in original.bundles[at].instructions if i.mnemonic == 'simm.s32' and i.predicate == 15}
            pair = immediates.get(registers[0]), immediates.get(registers[1])
            if pair == (0, 0) and not required:
                continue
            if pair not in loaded:
                raise ValueError(f'cannot identify the overlay loader immediates at bundle {at}')
            loaders[at] = dict(zip(registers, loaded[cast(tuple[int, int], pair)]))
    if not any(pc < starts[0] for pc in loaders):
        raise ValueError('cannot identify the overlay loader immediates')
    modulus = 1 << _DISPLACEMENT_BITS[hardware.identifier]
    output: list[tuple[int, tuple[Form, ...]]] = []
    for pc, bundle in enumerate(original.bundles):
        if pc in fragments:
            fragment = fragments[pc]
            first = len(output)
            relocated = replace(fragment, labels={name: first + at for name, at in fragment.labels.items()})
            for local, extra in enumerate(fragment.bundles):
                instructions = []
                for instruction in extra.instructions:
                    if instruction.mnemonic in ('sbr.rel', 'scall.rel'):
                        index = int(instruction.mnemonic.startswith('scall'))
                        operand = instruction.operands[index]
                        destination = fragment.target(operand, local, False, instruction.location)
                        if operand not in fragment.labels:
                            destination += local
                        if not 0 <= destination <= len(fragment.bundles):
                            raise ValueError('inserted branches must stay within their fragment')
                        instruction = replace(instruction, operands=instruction.operands[:index] + (str(destination - local),) + instruction.operands[index + 1:])
                    instructions.append(instruction)
                output.append(BundleSolver(ISA, replace(extra, instructions=tuple(instructions)), relocated, first + local).solve())
        if pc not in deleted:
            instructions = []
            for instruction in bundle.instructions:
                branch = branch_target(instruction.mnemonic, instruction.operands, pc)
                if branch is not None and instruction.mnemonic.endswith('.abs'):
                    # 只有前缀的位置固定；程序 overlay 之间、以及指向程序 overlay 的跳转都经 trampoline 由装载例程完成。
                    if not 0 <= branch[1] < starts[0]:
                        raise ValueError(f'cannot relocate the absolute branch at bundle {pc}: its target is outside the prelude')
                elif branch is not None:
                    # v4 的位移按 65536 回绕：超过 32767 个 bundle 的前向分支写成负位移。目标取位移所指的、落在本 overlay 内的那一个位置。
                    index = branch[0]
                    region = bisect_right(starts, pc)
                    low = starts[region - 1] if region else 0
                    destination = low + (branch[1] - low) % modulus
                    if destination >= (starts[region] if region < len(starts) else len(original.bundles)):
                        raise ValueError(f'cannot relocate the direct branch at bundle {pc}: its target is outside the overlay')
                    value = (shifted(destination, target=True) - shifted(pc) + modulus // 2) % modulus - modulus // 2
                    if value != integer(instruction.operands[index]):
                        instruction = replace(instruction, operands=instruction.operands[:index] + (str(value),) + instruction.operands[index + 1:])
                if instruction.mnemonic == 'simm.s32' and instruction.operands[0] in loaders.get(pc, ()):
                    instruction = replace(instruction, operands=(instruction.operands[0], str(loaders[pc][instruction.operands[0]])))
                instructions.append(instruction)
            if tuple(instructions) == bundle.instructions:
                output.append(encoded[pc])
            else:
                output.append(BundleSolver(ISA, replace(bundle, instructions=tuple(instructions)), original, shifted(pc)).solve())
        output.extend([halt_word] * paddings.get(pc + 1, 0))
    image = encode_program(output)
    _verify_image(image, target=hardware.identifier)
    trap_halts = {pc for pc, block in enumerate(original.bundles) if any(i.mnemonic == 'shalt' for i in block.instructions)} if hardware == TPU_V4_TC else set()
    metadata = _relocate_metadata(program.metadata, overlays, growth, emitted, counts, deletes, hardware.bundles_per_block, block_words, len(program.image), len(image), trap_halts)
    return image, metadata

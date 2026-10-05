"""TC 单程序 overlay 内插入 bundle，迁移机器分支与编译器元数据。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from ._protobuf import message, replace_fields
from .assembly_model import Form
from .assembly_syntax import branch_target, integer, parse_assembly
from .printer import _verify_image
from .targets import HardwareTarget, TPU_V4_TC
from .tc_solver import BundleSolver
from .tc_source_mapping import _ENCODED_WORD_BYTES, Overlay, _Program, _one, _source_map, _values

@dataclass(frozen=True)
class BundleInsertion:
    """在原映像的 image_pc 之前插入汇编片段。

    source 含 .target 声明和一个或多个 bundle，无需块对齐。片段中的直接分支只能引用片段内标签或片段局部编号（允许指向末尾以继续执行原程序）。branch_target='inserted' 使原程序中指向 image_pc 的直接分支先执行片段；'original' 则跳过片段。fallthrough 总会执行片段。多个插入点的编号均相对于输入映像。
    """
    image_pc: int
    source: str
    branch_target: Literal['inserted', 'original'] = 'inserted'

def _set(data: bytes, number: int, value: int | bytes) -> bytes:
    return replace_fields(data, {number: [value]})

def _main_limit(metadata: bytes) -> int:
    ranges = [r for table in _values(metadata, 8) for r in _values(table, 2) if _one(r, 4) == 1]
    if not ranges:
        raise ValueError('bundle insertion requires TensorCore symbol ranges')
    return max(_one(r, 2) for r in ranges)

def _relocate_metadata(metadata: bytes, overlay: Overlay, counts: Mapping[int, int], padding: int, old_size: int, new_size: int, trap_halts: set[int]) -> bytes:
    offset = overlay.body_start - overlay.emitted_start
    edits = {pc - offset: count for pc, count in counts.items()}
    main_limit = _main_limit(metadata)

    def shifted(pc: int) -> int:
        return pc + sum(count for at, count in edits.items() if at <= pc)

    def trap_location(location: bytes) -> bytes:
        # v4 TagAndPc identifies the bundle following shalt, relative to the overlay image start.
        # Validate the actual halt instead of assuming every metadata PC uses this coordinate system.
        pc = _one(location, 2)
        halt = overlay.image_start + pc - 1
        if _one(location, 1) != overlay.index or _one(location, 3) != 1 or halt not in trap_halts:
            raise ValueError('unsupported trap tag, sequencer or halt PC coordinate')
        return _set(location, 2, pc + sum(count for at, count in counts.items() if at <= halt))

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
        cuts = [start] + sorted(at for at in edits if start < at < limit) + [limit]
        return [replace_fields(data, {1: [shifted(a)], 2: [shifted(b - 1) + 1]}) for a, b in zip(cuts, cuts[1:])]

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
            if key < main_limit and pc == key:
                annotation = _set(annotation, 1, shifted(pc))
            elif not (main_limit <= key < overlay.emitted_limit and pc == key - main_limit):
                raise ValueError('unsupported annotation key/PC coordinate relation')
            entries.append(replace_fields(entry, {1: [shifted(key)], 2: [annotation]}))
        # New instructions have no original source or symbol owner.
        for at, count in sorted(edits.items()):
            first = shifted(at) - count
            for pc in range(first, first + count):
                entries.append(message([(1, pc), (2, message([(1, pc), (3, b'tpuasm: inserted bundle')]))]))
        return replace_fields(table, {1: entries})

    tables = _values(metadata, 10)
    raw_overlays = _values(tables[0], 1)
    raw_overlays[overlay.index] = replace_fields(raw_overlays[overlay.index], {3: [overlay.emitted_limit + sum(counts.values())], 4: [overlay.suffix_size + padding]})
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

def insert_program(program: _Program, insertions: list[BundleInsertion], hardware: HardwareTarget) -> tuple[bytes, bytes]:
    if hardware == TPU_V4_TC:
        from .tpu_v4_tc_assembler import ISA, _decoded_source as decode_v4
        from .tpu_v4_tc_codec import encode_program
    else:
        from .tpu_v6e_tc_assembler import ISA, _decoded_source as decode_v6e
        from .tpu_v6e_tc_codec import encode_program
    source = _source_map(program, hardware)
    if program.diagnostic or len(source.overlays) != 2 or len(_values(program.metadata, 10)) != 1:
        raise ValueError('insertion requires one code image with a prelude and one program overlay')
    prelude, overlay = source.overlays
    if prelude.image_start != 0 or prelude.emitted_start != prelude.emitted_limit or prelude.body_limit + prelude.suffix_size != overlay.image_start:
        raise ValueError('unsupported prelude overlay layout')
    if overlay.emitted_start != 0 or overlay.hlo_function_overlay or overlay.image_start % hardware.bundles_per_block or overlay.body_limit + overlay.suffix_size != source.bundle_count:
        raise ValueError('unsupported program overlay layout')
    original, encoded_raw = decode_v4(program.image) if hardware == TPU_V4_TC else decode_v6e(program.image)
    encoded: list[tuple[int, tuple[Form, ...]]] = [(word, tuple(forms)) for word, forms in encoded_raw]
    main_limit = _main_limit(program.metadata)
    if not 0 < main_limit <= overlay.emitted_limit:
        raise ValueError('TensorCore symbol ranges exceed the program overlay')
    if any(_one(entry, 1) >= overlay.image_start for table in _values(program.metadata, 30) for entry in _values(table, 1)):
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
    for insertion in insertions:
        pc = insertion.image_pc
        if pc in fragments:
            raise ValueError(f'duplicate insertion position {pc}; combine its bundles in one fragment')
        if insertion.branch_target not in ('inserted', 'original'):
            raise ValueError('branch_target must be inserted or original')
        if not overlay.body_start <= pc < overlay.body_start + main_limit:
            raise ValueError(f'insertion {pc} is outside the main program; prelude and continuation insertion is not supported')
        if pc in delayed:
            branch_pc, mnemonic = delayed[pc]
            raise ValueError(f'insertion {pc} is in the {delay}-bundle delay window of {mnemonic} at {branch_pc}; insert before that branch or after bundle {branch_pc + delay}')
        fragment = parse_assembly(insertion.source, filename=f'<insertion:{pc}>', fragment=True)
        if fragment.hardware != hardware or not fragment.bundles:
            raise ValueError('insertion needs nonempty assembly for the executable target')
        # A delay window reaching past the fragment would turn the following original bundles into delay slots.
        for local, extra in enumerate(fragment.bundles):
            for instruction in extra.instructions:
                if instruction.mnemonic.startswith(('sbr.', 'scall.')) and local + delay >= len(fragment.bundles):
                    raise ValueError(f'insertion {pc}: {instruction.mnemonic} at fragment bundle {local} needs its {delay}-bundle delay window inside the fragment; append empty bundles')
        fragments[pc], policies[pc] = fragment, insertion.branch_target
    counts = {pc: len(f.bundles) for pc, f in fragments.items()}
    added = sum(counts.values())
    padding = -added % hardware.bundles_per_block
    new_count = source.bundle_count + added + padding
    # XDB GetEmittedBundleNumber takes PC modulo overlay_slot_size, then subtracts prefix_size;
    # the unit is bundles for both TensorCore targets, not the encoded_word_offset unit.
    slot_size_bundles = _one(_values(program.metadata, 10)[0], 2)
    if not slot_size_bundles or new_count - overlay.image_start > slot_size_bundles:
        raise ValueError('inserted program exceeds the overlay slot capacity in bundles')

    def shifted(pc: int, *, target: bool = False) -> int:
        return pc + sum(count for at, count in counts.items() if at < pc or (at == pc and (not target or policies[at] == 'original')))

    # Keep the original suffix, including a final continuation branch and its delay slot.
    # Alignment padding is fresh halt bundles, never copies of a possibly live final instruction.
    halt = parse_assembly(f'.target {hardware.identifier}\n{{ s0: shalt }}\n', fragment=True)
    halt_word = BundleSolver(ISA, halt.bundles[0], halt, 0).solve()
    trampolines = [a.image_pc for a in source.annotations if a.coordinate_space == 'image' and a.text == 'trampoline:program-start']
    if len(trampolines) != 1:
        raise ValueError('insertion requires the program-start trampoline annotation')
    loaders = []
    # The loader addresses the overlay by its encoded_word_offset in blocks, which exceeds the image position when the first overlay does not start at offset 0.
    start_block = overlay.encoded_word_offset * _ENCODED_WORD_BYTES[hardware.identifier] // hardware.image_block_size
    for pc in range(trampolines[0], overlay.image_start):
        loader_instructions = original.bundles[pc].instructions
        registers = {i.operands[0]: i for i in loader_instructions if i.mnemonic == 'simm.s32' and i.predicate == 15}
        if set(registers) == {'s0', 's1'} and integer(registers['s0'].operands[1]) == start_block:
            loaders.append((pc, integer(registers['s1'].operands[1])))
    if len(loaders) != 1:
        raise ValueError('cannot identify the overlay loader immediates')
    loader_pc, old_blocks = loaders[0]
    actual_blocks = (source.bundle_count - overlay.image_start) // hardware.bundles_per_block
    if old_blocks != actual_blocks:
        raise ValueError('overlay loader block count does not match the code image')
    new_blocks = old_blocks + (added + padding) // hardware.bundles_per_block
    output: list[tuple[int, tuple[Form, ...]]] = []
    for pc, bundle in enumerate(original.bundles):
        if pc in fragments:
            fragment = fragments[pc]
            first = len(output)
            relocated = replace(fragment, labels={name: first + at for name, at in fragment.labels.items()})
            for local, extra in enumerate(fragment.bundles):
                instructions = []
                for instruction in extra.instructions:
                    if instruction.mnemonic in ('sbr.rel', 'sbr.abs', 'scall.rel', 'scall.abs'):
                        index = int(instruction.mnemonic.startswith('scall'))
                        operand = instruction.operands[index]
                        destination = fragment.target(operand, local, False, instruction.location)
                        if operand not in fragment.labels and instruction.mnemonic.endswith('.rel'):
                            destination += local
                        if not 0 <= destination <= len(fragment.bundles):
                            raise ValueError('inserted branches must stay within their fragment')
                        value = destination - local if instruction.mnemonic.endswith('.rel') else first + destination
                        instruction = replace(instruction, operands=instruction.operands[:index] + (str(value),) + instruction.operands[index + 1:])
                    instructions.append(instruction)
                output.append(BundleSolver(ISA, replace(extra, instructions=tuple(instructions)), relocated, first + local).solve())
        instructions = []
        for instruction in bundle.instructions:
            branch = branch_target(instruction.mnemonic, instruction.operands, pc)
            if branch is not None:
                index, destination = branch
                if not 0 <= destination < len(original.bundles):
                    raise ValueError('cannot relocate a direct branch outside the image')
                value = shifted(destination, target=True)
                if instruction.mnemonic.endswith('.rel'):
                    value -= shifted(pc)
                if value != integer(instruction.operands[index]):
                    instruction = replace(instruction, operands=instruction.operands[:index] + (str(value),) + instruction.operands[index + 1:])
            if pc == loader_pc and instruction.mnemonic == 'simm.s32' and instruction.operands[0] == 's1':
                instruction = replace(instruction, operands=('s1', str(new_blocks)))
            instructions.append(instruction)
        if tuple(instructions) == bundle.instructions:
            output.append(encoded[pc])
        else:
            output.append(BundleSolver(ISA, replace(bundle, instructions=tuple(instructions)), original, shifted(pc)).solve())
    output.extend([halt_word] * padding)
    image = encode_program(output)
    _verify_image(image, target=hardware.identifier)
    trap_halts = {pc for pc, block in enumerate(original.bundles) if any(i.mnemonic == 'shalt' for i in block.instructions)} if hardware == TPU_V4_TC else set()
    metadata = _relocate_metadata(program.metadata, overlay, counts, padding, len(program.image), len(image), trap_halts)
    return image, metadata

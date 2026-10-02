"""从 executable 保存的元数据恢复完整程序映像中的 PC / 物理槽来源；TC 各代际共用。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any

from ._protobuf import fields, read_varint
from .targets import HardwareTarget, TPU_V4_TC, TPU_V6E_TC
from .program_container import executable_records, resolve_executable_target

SCHEMA_VERSION = 3
# Native annotation keys are physical slots, not LLO instruction positions.
_SLOTS = {
    TPU_V4_TC.identifier: {
        'SLOT_SCALAR_ALU_0': 's0',
        'SLOT_SCALAR_ALU_1': 's1',
        'SLOT_VECTOR_ALU_0': 'va0',
        'SLOT_VECTOR_ALU_1': 'va1',
        'SLOT_VECTOR_STORE': 'vst',
        'SLOT_VECTOR_LOAD': 'vld',
        'SLOT_CMEM_LOAD': 'cld',
        'SLOT_VECTOR_EXTENDED_0': 'vx0',
        'SLOT_VECTOR_EXTENDED_1': 'vx1',
        'SLOT_VECTOR_RESULT_0': 'vr0',
        'SLOT_VECTOR_RESULT_1': 'vr1',
        'SLOT_MISC': 'misc',
    },
    TPU_V6E_TC.identifier: {
        'SLOT_SCALAR_ALU_0': 's0',
        'SLOT_SCALAR_ALU_1': 's1',
        'SLOT_DMA': 'dma',
        **{f'SLOT_VECTOR_ALU_{index}': f'va{index}' for index in range(4)},
        'SLOT_VECTOR_STORE': 'vst',
        'SLOT_VECTOR_LOAD_0': 'vld0',
        'SLOT_VECTOR_LOAD_1': 'vld1',
        'SLOT_VECTOR_MISC': 'misc',
        'SLOT_VECTOR_EXTENDED_0': 'vx0',
        'SLOT_VECTOR_EXTENDED_1': 'vx1',
        'SLOT_VECTOR_RESULT_0': 'vr0',
        'SLOT_VECTOR_RESULT_1': 'vr1',
    },
}
# Overlay.encoded_word_offset 的单位：v4 为 512 字节的块，v6e 为 32 字节。
_ENCODED_WORD_BYTES = {TPU_V4_TC.identifier: 512, TPU_V6E_TC.identifier: 32}
_RECORD = re.compile(r'\[\[tpuasm:v1:([0-9a-f]+)\]\]')
_LOCATION = re.compile(r'loc\(("(?:[^"\\]|\\.)*"):(\d+):(\d+)(?: to (\d*):(\d+))?\)')

def _values(data: bytes, number: int) -> list[Any]:
    return [value for field, _, value in fields(data) if field == number]

def _one(data: bytes, number: int, default: Any = 0) -> Any:
    values = _values(data, number)
    return values[-1] if values else default

def _text(data: bytes, number: int) -> str:
    return _one(data, number, b'').decode('utf-8')

def _integers(data: bytes, number: int) -> tuple[int, ...]:
    result = []
    for value in _values(data, number):
        if isinstance(value, int):
            result.append(value)
        else:
            pos = 0
            while pos < len(value):
                item, pos = read_varint(value, pos)
                result.append(item)
    return tuple(result)

@dataclass(frozen=True)
class SourceFrame:
    path: str
    line_start: int
    line_end: int
    col_start: int
    col_end: int
    function_name: str = ''

@dataclass(frozen=True)
class SourceLocation:
    """Pallas 原始位置；汇编源码中的诊断位置是 :class:`AssemblyLocation`。

    Attributes:
        frames: 原始来源的位置栈。
        primitive: 保存的 Pallas primitive 名称。
        scope_stack: 保存的 scope 栈。
        ordinals: 与该位置相关的 ordinals。
    """
    frames: tuple[SourceFrame, ...]
    primitive: str
    scope_stack: tuple[str, ...]
    ordinals: tuple[int, ...]

@dataclass(frozen=True)
class InstructionOrigin:
    """一条指令的独立来源，保存 HLO/module 身份、最终 LLO ordinal 和所有原始位置。

    Attributes:
        hlo_name: HLO 名称。
        hlo_module_name: HLO module 名称。
        hlo_module_id: HLO module ID。
        llo_ordinal: 最终 LLO ordinal。
        locations: 该来源保存的所有 SourceLocation。
    """
    hlo_name: str
    hlo_module_name: str
    hlo_module_id: int
    llo_ordinal: int
    locations: tuple[SourceLocation, ...]

@dataclass(frozen=True)
class Overlay:
    """一个 overlay 在发射坐标与程序映像坐标之间的换算；image_start 由 encoded_word_offset 按目标的编码字大小换算。"""
    index: int
    emitted_start: int
    emitted_limit: int
    encoded_word_offset: int
    image_start: int
    prefix_size: int
    suffix_size: int
    hlo_function_overlay: bool

    @property
    def body_start(self) -> int:
        return self.image_start + self.prefix_size

    @property
    def body_limit(self) -> int:
        return self.body_start + self.emitted_limit - self.emitted_start

    def translate(self, emitted: int) -> int:
        if not self.emitted_start <= emitted < self.emitted_limit:
            raise ValueError('emitted PC is outside the overlay body')
        return self.body_start + emitted - self.emitted_start

@dataclass(frozen=True)
class SourceRange:
    image_start: int
    image_limit: int
    emitted_start: int
    emitted_limit: int
    overlay: int

@dataclass(frozen=True)
class FunctionSource:
    """程序映像内的函数身份及其半开区间列表。

    函数由本程序映像内的 symbol_id 标识，display_name 可以重复。ranges 保存多个半开区间，保留函数范围中的空洞。
    """
    symbol_id: int
    hlo_name: str
    deduplicated_name: str
    display_name: str
    parent_symbols: tuple[int, ...]
    ranges: tuple[SourceRange, ...]

@dataclass(frozen=True)
class SlotSource:
    """一个已解码物理槽的来源集合、编译器注释及函数归属。

    origins 保存独立的 InstructionOrigin 集合。空来源表示未知，不能据此推断该指令一定由编译器生成。function_symbols 引用本程序映像内的函数 symbol ID。

    source_frames 是供调用者直接消费的统一源码位置：它合并捕获的 SourceMap frame 与原编译器 ``loc(...)`` 并按坐标去重。source_kind 区分 ``captured``、``compiler_location`` 和 ``unknown``。compiler_annotation、annotation_locations 和 origins 保留原始编译器证据；不从机器码数据依赖推断源码。
    """
    image_pc: int
    slot: str
    native_slot: str
    annotation_key: int
    annotation_pc: int
    coordinate_space: str
    overlay: int
    compiler_annotation: str
    annotation_locations: tuple[SourceFrame, ...]
    origins: tuple[InstructionOrigin, ...]
    function_symbols: tuple[int, ...]
    source_frames: tuple[SourceFrame, ...]
    source_kind: str

@dataclass(frozen=True)
class BundleAnnotation:
    image_pc: int
    annotation_key: int
    annotation_pc: int
    coordinate_space: str
    overlay: int
    text: str

@dataclass(frozen=True)
class ProgramSourceMap:
    """一份程序映像的来源映射，身份由记录、程序映像与 segment 共同限定。

    hash、fingerprint 和元数据 ID 分开保存。overlays 保存坐标映射，functions 保存函数身份及半开区间列表，slots 保存逐槽来源，annotations 保存 bundle 注释，diagnostics 保存来源恢复时的诊断。

    status 为 ``'captured'`` 表示保存了 tpuasm 记录，不表示每条指令都有完整来源；``'absent'`` 表示没有 tpuasm 来源记录，仍可包含编译器原生注释或函数归属。空来源表示未知，不能据此判定指令一定由编译器生成。

    :meth:`to_dict` 和 :meth:`to_json` 导出版本 3 的结构化记录；
    :func:`source_maps_json` 将多份映射导出到同一个 JSON 文档。版本 3 相对版本 2 增加统一来源 source_frames 与 source_kind，并列出函数范围中未保存注释的已占用槽。
    """
    target: str
    record: int
    image_index: int
    segment_set_index: int
    segment_index: int
    image_offset: int
    image_hash: str
    program_fingerprint: str
    compilation_id: int | None
    metadata_record: int | None
    metadata_program_id: int | None
    module_name: str
    bundle_count: int
    status: str
    overlays: tuple[Overlay, ...]
    functions: tuple[FunctionSource, ...]
    slots: tuple[SlotSource, ...]
    annotations: tuple[BundleAnnotation, ...]
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """返回含 ``schema_version=3`` 和全部映射字段的字典，不写文件。

        嵌套数据类递归转换为字典，tuple 集合仍为 tuple；字段和状态含义见 :class:`ProgramSourceMap`。
        """
        return {'schema_version': SCHEMA_VERSION, **asdict(self)}

    def to_json(self) -> str:
        """返回 :meth:`to_dict` 的 JSON 文本，不写文件。

        保留非 ASCII 字符，以两个空格缩进并以换行结尾；tuple 集合序列化为 JSON 数组。
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + '\n'

@dataclass(frozen=True)
class _Program:
    record: int
    image_index: int
    segment_set_index: int
    segment_index: int
    image_offset: int
    image_hash: bytes
    fingerprint: bytes
    compilation_id: int | None
    image: bytes
    metadata_record: int | None
    metadata: bytes
    diagnostic: str = ''

def _programs(serialized: bytes) -> list[_Program]:
    records = executable_records(serialized)
    result = []
    for record, raw in enumerate(records):
        try:
            outer = fields(raw)
        except ValueError:
            continue
        tensor_core = _one(raw, 5, b'')
        if not isinstance(tensor_core, bytes):
            continue
        try:
            sequencer = _one(tensor_core, 1, b'')
            if not isinstance(sequencer, bytes) or _one(sequencer, 3) != 1:
                continue
        except ValueError:
            continue
        index = 0
        segment_set_index = -1
        for number, wire, nested in outer:
            if number != 8 or wire != 2 or not isinstance(nested, bytes):
                continue
            segment_set_index += 1
            initialized = _one(nested, 3, b'')
            # These two container versions serialize each core program followed
            # by its CompilerMetadata (including separate SparseCore records).
            # MemorySegmentSet field 4 is a hash, NOT a program ID.
            metadata_record = record + 1 if record + 1 < len(records) else None
            metadata = records[metadata_record] if metadata_record is not None else b''
            if not _values(metadata, 10):
                metadata_record, metadata = None, b''
            for segment_index, segment in enumerate(_values(nested, 2)):
                if _one(segment, 1) != 1:  # TPU_SEGMENT_TYPE_CODE
                    continue
                if _one(nested, 5) != 0:
                    raise ValueError('compressed executable code segments are not supported')
                data_range = _one(segment, 3, b'')
                offset, size = _one(data_range, 1), _one(data_range, 2)
                if size <= 0 or offset + size > len(initialized):
                    raise ValueError('TPU ISA code segment exceeds initialized data')
                image = initialized[offset:offset + size]
                result.append(_Program(
                    record,
                    index,
                    segment_set_index,
                    segment_index,
                    offset,
                    _one(nested, 4, b''),
                    _one(raw, 3, b''),
                    _one(raw, 4) if _values(raw, 4) else None,
                    image,
                    metadata_record,
                    metadata,
                ))
                index += 1
        if index > 1:
            # No image identifier exists in CompilerMetadata. Without a verified
            # split-image layout it cannot safely be assigned to every segment.
            for i in range(len(result) - index, len(result)):
                result[i] = replace(result[i], diagnostic='multiple code images in one core record: metadata/image association is ambiguous')
    if not result:
        raise ValueError('no TPU program images found in the executable')
    return result

def _origins(annotation: str) -> tuple[str, tuple[InstructionOrigin, ...]]:
    result = []
    for match in _RECORD.finditer(annotation):
        record = bytes.fromhex(match[1])
        source_map = _one(record, 2, b'')
        strings = [value.decode('utf-8') for value in _values(source_map, 2)]
        def string(index: int) -> str:
            if not 0 <= index < len(strings):
                raise ValueError('source string-table index out of range')
            return strings[index]
        locations = []
        for info in _values(source_map, 1):
            frames = []
            for frame in _values(info, 1):
                function_name = string(_one(frame, 6)) if _values(frame, 6) else ''
                frames.append(SourceFrame(string(_one(frame, 1)), _one(frame, 2), _one(frame, 3), _one(frame, 4), _one(frame, 5), function_name))
            primitive = string(_one(info, 2)) if _values(info, 2) else ''
            locations.append(SourceLocation(tuple(frames), primitive, tuple(string(i) for i in _integers(info, 3)), _integers(info, 4)))
        origin = InstructionOrigin(_text(record, 3), _text(record, 4), _one(record, 5), _one(record, 1), tuple(locations))
        if origin not in result:
            result.append(origin)
    return _RECORD.sub('', annotation).strip(), tuple(result)

def _annotation_frame(match: re.Match[str]) -> SourceFrame | None:
    path, line, col, end_line, end_col = match.groups()
    try:
        decoded_path = json.loads(path)
    except json.JSONDecodeError:
        return None  # Keep unrecognized MLIR escapes in compiler_annotation.
    return SourceFrame(decoded_path, int(line), int(end_line or line), int(col), int(end_col or col))

def _annotation_locations(text: str) -> tuple[SourceFrame, ...]:
    frames: list[SourceFrame] = []
    for match in _LOCATION.finditer(text):
        frame = _annotation_frame(match)
        if frame is None:
            continue
        if frame not in frames:
            frames.append(frame)
    return tuple(frames)

def _frame_position(frame: SourceFrame) -> tuple[str, int, int, int, int]:
    return frame.path, frame.line_start, frame.line_end, frame.col_start, frame.col_end

def _append_frame(frames: list[SourceFrame], frame: SourceFrame) -> None:
    """按坐标去重；同一坐标优先保留带函数名的 frame。"""
    for index, existing in enumerate(frames):
        if _frame_position(existing) != _frame_position(frame):
            continue
        if not existing.function_name and frame.function_name:
            frames[index] = frame
        return
    frames.append(frame)

def _source_frames(origins: tuple[InstructionOrigin, ...], annotation_locations: tuple[SourceFrame, ...]) -> tuple[SourceFrame, ...]:
    """合并两类编译器明确保存的源码 frame，不推断缺失位置。"""
    frames: list[SourceFrame] = []
    for origin in origins:
        for location in origin.locations:
            for frame in location.frames:
                _append_frame(frames, frame)
    for frame in annotation_locations:
        _append_frame(frames, frame)
    return tuple(frames)

def _decode(image: bytes, hardware: HardwareTarget) -> list[set[str]]:
    """每个 bundle 实际占用的物理槽；不从寄存器或邻近指令推断来源。"""
    if hardware == TPU_V6E_TC:
        from .tpu_v6e_tc_codec import decode_program as decode_v6e
        return [{form.slot for form, _ in forms} for _, forms in decode_v6e(image)]
    from .tpu_v4_tc_codec import decode_program
    return [{form.slot for form, _ in forms} for _, forms in decode_program(image)]

def _source_map(program: _Program, hardware: HardwareTarget) -> ProgramSourceMap:
    occupied = _decode(program.image, hardware)
    native_slots = _SLOTS[hardware.identifier]
    physical_slots = {slot: native for native, slot in native_slots.items()}
    count = len(occupied)
    metadata = b'' if program.diagnostic else program.metadata
    diagnostics = [program.diagnostic] if program.diagnostic else []
    overlays: list[Overlay] = []
    for table in _values(metadata, 10):
        if _one(table, 6) != 1:
            continue
        raws = _values(table, 1)
        # encoded_word_offset 以第一个 overlay 为基准：程序映像从第一个 overlay 开始，其偏移不一定为 0（例如 XLA scatter 程序中为 33）。
        base = _one(raws[0], 5) if raws else 0
        for raw in raws:
            offset = _one(raw, 5)
            image_start = (offset - base) * _ENCODED_WORD_BYTES[hardware.identifier] * hardware.bundles_per_block // hardware.image_block_size
            overlay = Overlay(len(overlays), _one(raw, 2), _one(raw, 3), offset, image_start, _one(raw, 1), _one(raw, 4), bool(_one(raw, 6)))
            if overlay.emitted_limit < overlay.emitted_start or overlay.body_limit + overlay.suffix_size > count:
                raise ValueError('overlay lies outside the program image')
            overlays.append(overlay)
    symbols = {}
    tables = _values(metadata, 8)
    for table in tables:
        for entry in _values(table, 1):
            symbols[_one(entry, 1)] = _one(entry, 2, b'')
    functions = []
    for symbol_id, symbol in symbols.items():
        hlo = _one(symbol, 70, b'')
        if _text(hlo, 3) != 'custom-call':
            continue
        parents = _integers(symbol, 21)
        display = [_text(symbols.get(parent, b''), 3) for parent in parents if _values(symbols.get(parent, b''), 60)]
        ranges = []
        # child_instructions preserves holes and handles deduplicated parents.
        for raw in _values(symbol, 23):
            if _one(raw, 4) != 1:
                continue
            start, limit = _one(raw, 1), _one(raw, 2)
            for overlay in overlays:
                left, right = max(start, overlay.emitted_start), min(limit, overlay.emitted_limit)
                if left < right:
                    ranges.append(SourceRange(overlay.translate(left), overlay.translate(right - 1) + 1, left, right, overlay.index))
        functions.append(FunctionSource(symbol_id, _text(hlo, 4), _text(hlo, 8), ' / '.join(display) or _text(hlo, 4), parents, tuple(ranges)))
    slots = []
    annotations = []
    for field, space in ((4, 'emitted'), (30, 'image')):
        for table in _values(metadata, field):
            for entry in _values(table, 1):
                key, annotation = _one(entry, 1), _one(entry, 2, b'')
                inner_pc = _one(annotation, 1)
                positions = []
                for overlay in overlays:
                    if space == 'emitted' and overlay.emitted_start <= key < overlay.emitted_limit:
                        positions.append((overlay.translate(key), overlay.index))
                    elif space == 'image' and overlay.image_start <= key < overlay.body_limit + overlay.suffix_size:
                        positions.append((key, overlay.index))
                if not positions:
                    diagnostics.append(f'{space} annotation {key} (local pc {inner_pc}) has no matching overlay')
                for pc, overlay_index in positions:
                    if text := _text(annotation, 3):
                        annotations.append(BundleAnnotation(pc, key, inner_pc, space, overlay_index, text))
                    owners = tuple(f.symbol_id for f in functions if any(r.image_start <= pc < r.image_limit for r in f.ranges))
                    for item in _values(annotation, 2):
                        native_slot = _text(item, 1)
                        value = _one(item, 2, b'')
                        slot = native_slots.get(native_slot)
                        if slot is None or _text(value, 1) != native_slot or slot not in occupied[pc]:
                            diagnostics.append(f'annotation {space}:{key}/{native_slot} does not match an occupied physical slot at image PC {pc}')
                            continue
                        text, origins = _origins(_text(value, 2))
                        annotation_locations = _annotation_locations(text)
                        source_frames = _source_frames(origins, annotation_locations)
                        slots.append(SlotSource(
                            pc,
                            slot,
                            native_slot,
                            key,
                            inner_pc,
                            space,
                            overlay_index,
                            text,
                            annotation_locations,
                            origins,
                            owners,
                            source_frames,
                            'captured' if any(location.frames for origin in origins for location in origin.locations) else 'compiler_location' if annotation_locations else 'unknown',
                        ))
    existing = {(slot.image_pc, slot.slot) for slot in slots}
    for pc, bundle in enumerate(occupied):
        owners = tuple(f.symbol_id for f in functions if any(r.image_start <= pc < r.image_limit for r in f.ranges))
        if not owners:
            continue
        overlay_index = next((overlay.index for overlay in overlays if overlay.image_start <= pc < overlay.body_limit + overlay.suffix_size), -1)
        for slot in hardware.slots:
            if slot not in bundle or (pc, slot) in existing:
                continue
            slots.append(SlotSource(
                pc,
                slot,
                physical_slots[slot],
                pc,
                pc,
                'image',
                overlay_index,
                '',
                (),
                (),
                owners,
                (),
                'unknown',
            ))
            existing.add((pc, slot))
    status = 'captured' if any(slot.origins for slot in slots) else 'absent'
    if status != 'captured':
        diagnostics.append('no tpuasm source records saved; recompile inside compiler_source_mapping with caches cleared')
    return ProgramSourceMap(
        hardware.identifier,
        program.record,
        program.image_index,
        program.segment_set_index,
        program.segment_index,
        program.image_offset,
        program.image_hash.hex(),
        program.fingerprint.hex(),
        program.compilation_id,
        program.metadata_record,
        _one(program.metadata, 9) if _values(program.metadata, 9) else None,
        _text(program.metadata, 20),
        count,
        status,
        tuple(overlays),
        tuple(functions),
        tuple(sorted(slots, key=lambda item: (item.image_pc, hardware.slots.index(item.slot)))),
        tuple(annotations),
        tuple(dict.fromkeys(diagnostics)),
    )

def executable_source_maps(serialized: bytes) -> list[ProgramSourceMap]:
    """读取 executable 保存的来源元数据，并核对实际解码的物理槽。

    无需 TPU、编译补丁或 final dump，但需要匹配的 libtpu decoder；不写文件。本函数只恢复已有来源，不会补齐编译时未保存的信息。

    Args:
        serialized: ``bytes(compiled.runtime_executable().serialize())`` 得到的字节。

    Returns:
        每份程序映像对应一个 ProgramSourceMap，与 :func:`executable_programs` 的顺序相同。每份映射包含程序身份、overlay、函数范围、逐槽来源和诊断。``status='captured'`` 不保证来源完整，``'absent'`` 仍可含原生注释或函数归属；客户端应直接读取 SlotSource.source_frames，详细原始证据见 :class:`SlotSource` 和 :class:`InstructionOrigin`。可用 ProgramSourceMap.to_dict() / .to_json() 导出版本 3 的记录，或用 :func:`source_maps_json` 合并为一个 JSON 文档。

    Raises:
        ValueError: 容器或来源元数据无效，或程序映像包含不支持的指令形式。
        RuntimeError: 匹配的原生后端不可用或原生解码失败。
    """
    hardware = resolve_executable_target(serialized, None)
    return [_source_map(program, hardware) for program in _programs(serialized)]

def source_maps_json(maps: list[ProgramSourceMap]) -> str:
    """将多份程序映像的来源映射序列化为一个 JSON 文档，不写文件。

    Args:
        maps: 待导出的 ProgramSourceMap 列表，例如 :func:`executable_source_maps` 的返回值。

    Returns:
        含 ``schema_version=3`` 和 ``programs`` 数组的 JSON 文本。数组按输入顺序保存每份映射的 :meth:`ProgramSourceMap.to_dict` 结果。保留非 ASCII 字符，以两个空格缩进并以换行结尾。
    """
    return json.dumps({'schema_version': SCHEMA_VERSION, 'programs': [source.to_dict() for source in maps]}, ensure_ascii=False, indent=2) + '\n'

def _comment(text: str) -> str:
    return text.replace('\r', r'\r').replace('\n', r'\n')

def _frame_text(frame: SourceFrame, previous: SourceFrame | None = None) -> str:
    """与前一 frame 同文件时省略路径。"""
    text = f'{frame.line_start}:{frame.col_start}-{frame.line_end}:{frame.col_end}'
    if previous is None or previous.path != frame.path:
        text = f'{frame.path}:{text}'
    return text + f' ({frame.function_name})' if frame.function_name else text

def _chain_text(frames: list[SourceFrame], scopes: dict[str, list[int]]) -> str:
    """``path:inner <- caller [scope, ...; LLO n, ...]``；共享同一组 ordinal 的 scope 合并为一项。"""
    groups: dict[tuple[int, ...], list[str]] = {}
    for scope, ordinals in scopes.items():
        groups.setdefault(tuple(ordinals), []).append(scope)
    labels = '; '.join(
        ', '.join(scope for scope in names if scope) + ('; ' if any(names) else '') + 'LLO ' + ', '.join(map(str, ordinals))
        for ordinals, names in groups.items()
    )
    return ' <- '.join(_frame_text(frame, frames[index - 1] if index else None) for index, frame in enumerate(frames)) + (' ' if frames else '') + f'[{labels}]'

def source_comments(source: ProgramSourceMap) -> tuple[dict[int, list[str]], dict[tuple[int, str], str]]:
    """逐槽来源按调用链去重，保留编译器解释性注释及无法解析的位置文本。"""
    outside: dict[int, list[str]] = {}
    inline: dict[tuple[int, str], str] = {}
    outside[0] = [f'source mapping: {source.status}; image {source.record}:{source.image_index}; metadata program {source.metadata_program_id}']
    for diagnostic in source.diagnostics:
        outside[0].append(_comment(diagnostic))
    for function in source.functions:
        for region in function.ranges:
            label = _comment(f'{function.display_name} [symbol {function.symbol_id}, {function.hlo_name}]')
            outside.setdefault(region.image_start, []).append(f'function {label}; image bundles [{region.image_start}, {region.image_limit})')
            outside.setdefault(region.image_limit, []).append(f'end function {label}')
    for annotation in source.annotations:
        outside.setdefault(annotation.image_pc, []).append(_comment(annotation.text))
    records: dict[tuple[int, str], list[SlotSource]] = {}
    for slot in source.slots:
        records.setdefault((slot.image_pc, slot.slot), []).append(slot)
    for key, slots in records.items():
        # 同一调用链只输出一次；frame 从内到外排列，合并该链上的全部 scope 与 LLO ordinal。
        chains: dict[tuple[tuple[str, int, int, int, int], ...], tuple[list[SourceFrame], dict[str, list[int]]]] = {}
        for slot in slots:
            for origin in slot.origins:
                for location in origin.locations:
                    frames = location.frames[::-1]
                    chain = tuple(_frame_position(frame) for frame in frames)
                    if chain not in chains:
                        chains[chain] = list(frames), {}
                    shown, scopes = chains[chain]
                    for index, frame in enumerate(frames):
                        if not shown[index].function_name and frame.function_name:
                            shown[index] = frame
                    ordinals = scopes.setdefault('/'.join((*location.scope_stack, location.primitive)), [])
                    if origin.llo_ordinal not in ordinals:
                        ordinals.append(origin.llo_ordinal)
        parts = [_chain_text(frames, scopes) for frames, scopes in chains.values()]
        # 编译器 loc(...) 只在坐标未出现在任何调用链中时单独列出，不带标签。
        displayed = {position for chain in chains for position in chain}
        for slot in slots:
            for frame in slot.annotation_locations:
                if _frame_position(frame) not in displayed:
                    displayed.add(_frame_position(frame))
                    parts.append(_frame_text(frame))
        for slot in slots:
            remaining = _LOCATION.sub(lambda match: match[0] if _annotation_frame(match) is None else '', slot.compiler_annotation)
            remaining = ' :: '.join(part.strip() for part in remaining.split(' :: ') if part.strip())
            if remaining:
                parts.append(remaining)
        if parts:
            inline[key] = _comment(' | '.join(dict.fromkeys(parts)))
    return outside, inline

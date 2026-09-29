"""识别程序容器中明确保存的 TPU 代际与执行单元。"""
from __future__ import annotations

from ._protobuf import fields, read_varint
from .targets import HardwareTarget, TPU_V4_TC, TPU_V4_BCS, TPU_V6E_TC, hardware_target

def record_spans(serialized: bytes) -> list[tuple[int, int]]:
    """每条记录在 serialized 中的半开区间，不含长度前缀。"""
    spans = []
    pos = 0
    while pos < len(serialized):
        size, pos = read_varint(serialized, pos)
        if pos + size > len(serialized):
            raise ValueError('executable record exceeds file size')
        spans.append((pos, pos + size))
        pos += size
    return spans

def executable_records(serialized: bytes) -> list[bytes]:
    return [serialized[start:limit] for start, limit in record_spans(serialized)]

def _message(data: bytes, number: int) -> bytes | None:
    values = [value for key, wire, value in fields(data) if key == number and wire == 2 and isinstance(value, bytes)]
    return values[0] if len(values) == 1 else None

def core_program_target(core: bytes) -> HardwareTarget | None:
    """按 program oneof 或 ABI version 识别目标；缺少证据时返回 None。"""
    outer = fields(core)
    kinds = [key for key, wire, value in outer if key in (5, 6, 7) and wire == 2]
    if kinds not in ([5], [6]):
        return None
    body = _message(core, kinds[0])
    sequencer = _message(body, 1) if body is not None else None
    if sequencer is None:
        return None
    seq = fields(sequencer)
    expected = 1 if kinds == [5] else 2
    if [value for key, wire, value in seq if key == 3 and wire == 0] != [expected]:
        return None
    alternatives = [key for key, wire, value in seq if 7 <= key <= 23 and wire == 2]
    semantic = 9 if kinds == [5] else 10
    abi = _message(core, 12) or _message(sequencer, 31)
    versions = [value for key, wire, value in fields(abi) if key == 1 and wire == 0] if abi else []
    # TpuCoreProgramAbiProto.version: 3 is TPU_VERSION_PUFFERFISH, 5 is TPU_VERSION_GHOSTLITE.
    # platform_type=1 only means HARDWARE and says nothing about generation.
    if versions == [5] and kinds == [5] and alternatives == [16]:
        return TPU_V6E_TC
    if versions and versions != [3]:
        return None
    if alternatives != [semantic] and not (alternatives == [16] and versions == [3]):
        return None
    return TPU_V4_TC if kinds == [5] else TPU_V4_BCS

def resolve_executable_target(serialized: bytes, target: str | None) -> HardwareTarget:
    if target is not None:
        return hardware_target(target)
    candidates = set()
    for raw in executable_records(serialized):
        try:
            inferred = core_program_target(raw)
        except ValueError:
            continue
        if inferred is not None:
            candidates.add(inferred)
    if len(candidates) != 1:
        available = ', '.join(sorted(item.identifier for item in candidates)) or 'none'
        raise ValueError(f'executable target is not unambiguous ({available}); specify target')
    return candidates.pop()

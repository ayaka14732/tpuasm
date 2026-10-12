"""识别程序容器中明确保存的 TPU 代际与执行单元。"""
from __future__ import annotations

from collections.abc import Sequence

from ._protobuf import encode_varint, fields, message, read_varint
from ._riegeli import is_riegeli, read_records, write_records
from .targets import HardwareTarget, TPU_V4_TC, TPU_V4_BCS, TPU_V6E_TC, hardware_target

# split 容器中 SplitProtoManifest 之后的六条记录各是 xla.TpuExecutableProto 的一部分，这里是每条记录的内容在其中的字段路径：
# ds_executable(1) 的 core_program(1)、compiler_metadata(2)、sparse_core_program(3)、sparse_core_compiler_metadata(4)，hlo_module_with_config(2)，最后一条是其余字段本身。
_SPLIT_PATHS: tuple[tuple[int, ...], ...] = ((1, 1), (1, 2), (1, 3), (1, 4), (2,), ())

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

def _split(serialized: bytes) -> tuple[bytes, bytes, list[bytes]] | None:
    """split 容器的 IFRT 头、manifest 和其后的各条记录；serialized 不是 split 容器时返回 None。"""
    try:
        size, pos = read_varint(serialized, 0)
    except ValueError:
        return None
    if not is_riegeli(serialized[pos + size:]):
        return None
    manifest, *parts = read_records(serialized[pos + size:])
    described = fields(manifest)
    # 每条记录都是 proto_merge_record：读取方把它当作结果类型的一部分合并进去。
    if [value for number, _, value in described if number == 1] != [b'xla.TpuExecutableProto'] or [value for number, _, value in described if number == 2] != [message([(2, b'')])] * len(_SPLIT_PATHS) or len(parts) != len(_SPLIT_PATHS):
        raise ValueError('unsupported split executable: expected six merge records of xla.TpuExecutableProto')
    return serialized[pos:pos + size], manifest, parts

def executable_records(serialized: bytes) -> list[bytes]:
    """容器中的记录：IFRT 头、TensorCore core program、CompilerMetadata、SparseCore core program 及其 CompilerMetadata、HLO module、其余字段。

    libtpu 0.0.48 与 0.0.49 的容器把这些记录各加一个 varint 长度前缀依次排列。0.0.50 的容器中，IFRT 头之后是 XLA 的 split proto：一个 Riegeli/records 流，第一条记录是 SplitProtoManifest，其后每条记录是 xla.TpuExecutableProto 的一部分。两种容器在这里给出相同的记录和编号。
    """
    split = _split(serialized)
    if split is None:
        return [serialized[start:limit] for start, limit in record_spans(serialized)]
    header, _, parts = split
    records = [header]
    for part, path in zip(parts, _SPLIT_PATHS):
        for number in path:
            if not part:
                break
            # 沿路径的每一层都只有这一个字段，否则这条记录还带着别的内容，不能只取出其中一部分。
            (field, wire, value), = fields(part)
            if field != number or wire != 2 or not isinstance(value, bytes):
                raise ValueError('unsupported split executable: unexpected record layout')
            part = value
        records.append(part)
    return records

def join_executable_records(serialized: bytes, records: Sequence[bytes]) -> bytes:
    """把 :func:`executable_records` 给出的记录（内容可以已被修改）按 serialized 所用的容器重新写出。"""
    split = _split(serialized)
    if split is None:
        return b''.join(encode_varint(len(record)) + record for record in records)
    header, manifest, parts = split
    wrapped = []
    for record, part, path in zip(records[1:], parts, _SPLIT_PATHS, strict=True):
        if part:
            for number in reversed(path):
                record = message([(number, record)])
        wrapped.append(record)
    return encode_varint(len(records[0])) + records[0] + write_records([manifest, *wrapped])

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

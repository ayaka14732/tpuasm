"""在 serialized executable 中找出 TPU v6e SparseCore TEC 程序映像。

Pallas SparseCore kernel 编译后，SparseCore 代码由 TC 程序记录携带：记录的 ``has_concurrent_program``（字段 10）为 1，代码放在一个 DATA segment 中，依次是 SCS 段、TEC 段和补到 segment 末尾的 0。容器不记录 TEC 段的起点，SCS 代码用链接符号 ``_scs_section_size`` 计算它。TEC 程序的任何以 bundle 为界的后缀仍是合法的 TEC 程序，而 SCS 段不能按 TEC 解码，所以 tpuasm 从 DATA segment 开头按 4 KiB 边界依次尝试，取第一个能把从该处到 segment 中非零数据末尾的字节完整通过 TEC 原生编解码校验的位置作为 TEC 程序映像的起点。
"""
from __future__ import annotations

from dataclasses import replace

from ._protobuf import fields
from .printer import _verify_image
from .program_container import executable_records
from .targets import TPU_V6E_TEC
from .tc_source_mapping import _Program, _one, _programs, _values

TARGET = TPU_V6E_TEC.identifier
# The TEC section starts on a 4 KiB boundary of the SparseCore code.
_ALIGNMENT = 4096
_BUNDLE_BYTES = 64

def _decodes(image: bytes) -> bool:
    try:
        _verify_image(image, target=TARGET)
    except (ValueError, RuntimeError):
        return False
    return True

def executable_tec_programs(serialized: bytes) -> list[_Program]:
    """每份 TEC 程序映像及其所在记录的身份信息，顺序与容器相同。"""
    records = executable_records(serialized)
    # The TC program of the same record supplies the fingerprint and the CompilerMetadata record.
    owners = {program.record: program for program in _programs(serialized)}
    result = []
    for record, raw in enumerate(records):
        if record not in owners or _one(raw, 10) != 1:
            continue
        index = 0
        segment_set_index = -1
        for number, wire, nested in fields(raw):
            if number != 8 or wire != 2 or not isinstance(nested, bytes):
                continue
            segment_set_index += 1
            initialized = _one(nested, 3, b'')
            for segment_index, segment in enumerate(_values(nested, 2)):
                if _one(segment, 1) != 2:  # TPU_SEGMENT_TYPE_DATA
                    continue
                data_range = _one(segment, 3, b'')
                start, size = _one(data_range, 1), _one(data_range, 2)
                data = initialized[start:start + size]
                end = -(-len(data.rstrip(b'\0')) // _BUNDLE_BYTES) * _BUNDLE_BYTES
                offset = next((offset for offset in range(0, end, _ALIGNMENT) if _decodes(data[offset:offset + _BUNDLE_BYTES]) and _decodes(data[offset:end])), None)
                if offset is None:
                    continue
                result.append(replace(owners[record], image_index=index, segment_set_index=segment_set_index, segment_index=segment_index, image_offset=start + offset, image_hash=_one(nested, 4, b''), image=data[offset:end]))
                index += 1
    if not result:
        raise ValueError('no tpu-v6e-tec program found in the executable')
    return result

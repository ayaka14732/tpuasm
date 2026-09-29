"""原生后端内部使用的 protobuf wire 编解码，不接受汇编字段文本。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

def encode_varint(value: int) -> bytes:
    if value < 0:
        value += 1 << 64
    if not 0 <= value < 1 << 64:
        raise ValueError('protobuf integer exceeds 64 bits')
    encoded = bytearray()
    while value >= 128:
        encoded.append((value & 127) | 128)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)

def read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if pos == len(data):
            break
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            if value >= 1 << 64:
                raise ValueError('protobuf integer exceeds 64 bits')
            return value, pos
    raise ValueError('invalid protobuf varint')

def field_spans(data: bytes, start: int = 0, end: int | None = None) -> list[tuple[int, int, int, int]]:
    """逐字段返回 (编号, wire 类型, 值起点, 值终点)，偏移相对于 data；varint 的区间是其编码字节。"""
    end = len(data) if end is None else end
    spans = []
    pos = start
    while pos < end:
        tag, pos = read_varint(data, pos)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError('protobuf field zero is invalid')
        if wire == 0:
            _, limit = read_varint(data, pos)
        elif wire == 2:
            size, pos = read_varint(data, pos)
            limit = pos + size
            if limit > end:
                raise ValueError('truncated protobuf field')
        elif wire in (1, 5):
            limit = pos + (8 if wire == 1 else 4)
            if limit > end:
                raise ValueError('truncated protobuf fixed field')
        else:
            raise ValueError(f'unsupported protobuf wire type {wire}')
        spans.append((number, wire, pos, limit))
        pos = limit
    if pos != end:
        raise ValueError('truncated protobuf varint')
    return spans

def fields(data: bytes) -> list[tuple[int, int, int | bytes]]:
    return [(number, wire, read_varint(data, start)[0] if wire == 0 else data[start:limit]) for number, wire, start, limit in field_spans(data)]

def message(values: Sequence[tuple[int, int | bytes]]) -> bytes:
    result = bytearray()
    for number, value in values:
        wire = 2 if isinstance(value, bytes) else 0
        result.extend(encode_varint(number << 3 | wire))
        result.extend(encode_varint(len(value)) + value if isinstance(value, bytes) else encode_varint(value))
    return bytes(result)

def replace_fields(data: bytes, replacements: Mapping[int, Sequence[int | bytes]]) -> bytes:
    """替换指定字段的全部出现，保留其他字段的原始 wire bytes；空序列删除字段。"""
    result = bytearray()
    remaining = dict(replacements)
    cursor = 0
    for number, _, _, limit in field_spans(data):
        if number in replacements:
            if number in remaining:
                result.extend(message([(number, value) for value in remaining.pop(number)]))
        else:
            result.extend(data[cursor:limit])
        cursor = limit
    for number, values in remaining.items():
        result.extend(message([(number, value) for value in values]))
    return bytes(result)

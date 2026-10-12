"""Riegeli/records 文件的读写，只覆盖 libtpu 序列化 executable 时用到的部分：不压缩的 simple chunk。

文件由 64 KiB 的块组成，每个块的开头有一个 24 字节的块头；chunk 首尾相接，遇到块头时被它从中间打断。每个 chunk 有 40 字节的头，其中的两个 hash 是 HighwayHash-64。格式见 Riegeli 仓库的 ``doc/riegeli_records_file_format.md``。
"""
from __future__ import annotations

from collections.abc import Sequence
import ctypes
from pathlib import Path
import struct
import subprocess
import tempfile

from ._protobuf import encode_varint, read_varint

_BLOCK = 1 << 16
_BLOCK_HEADER = 24
_CHUNK_HEADER = 40
_MASK = (1 << 64) - 1
# 'Riegeli/', 'records\n', 'Riegeli/', 'records\n' as little-endian words.
_KEY = (0x2f696c6567656952, 0x0a7364726f636572, 0x2f696c6567656952, 0x0a7364726f636572)
_INIT0 = (0xdbe6d5d5fe4cce2f, 0xa4093822299f31d0, 0x13198a2e03707344, 0x243f6a8885a308d3)
_INIT1 = (0x3bd39e10cb0ef593, 0xc0acf169b5f18a8c, 0xbe5466cf34e90c6c, 0x452821e638d01377)

def _zipper(v1: int, v0: int) -> tuple[int, int]:
    add0 = (((v0 & 0xff000000) | (v1 & 0xff00000000)) >> 24) + (((v0 & 0xff0000000000) | (v1 & 0xff000000000000)) >> 16) + (v0 & 0xff0000) + ((v0 & 0xff00) << 32) + ((v1 & 0xff00000000000000) >> 8) + ((v0 << 56) & _MASK)
    add1 = (((v1 & 0xff000000) | (v0 & 0xff00000000)) >> 24) + (v1 & 0xff0000) + ((v1 & 0xff0000000000) >> 16) + ((v1 & 0xff00) << 24) + ((v0 & 0xff000000000000) >> 8) + ((v1 & 0xff) << 48) + (v0 & 0xff00000000000000)
    return add0 & _MASK, add1 & _MASK

_native: list[tuple[tempfile.TemporaryDirectory, ctypes.CDLL]] = []

def _native_hash(data: bytes) -> int:
    """用随包分发的 riegeli_hash.cc 计算同一个 hash；首次使用时由 g++ 编译，供大块数据使用。"""
    if not _native:
        directory = tempfile.TemporaryDirectory(prefix='tpuasm-', dir='/tmp')
        output = Path(directory.name) / 'riegeli_hash.so'
        subprocess.run(['g++', '-std=c++17', '-O2', '-fPIC', '-shared', '-Wall', '-Wextra', str(Path(__file__).parent / 'riegeli_hash.cc'), '-o', str(output)], check=True)
        library = ctypes.CDLL(str(output))
        library.tpuasm_highway_hash.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        library.tpuasm_highway_hash.restype = ctypes.c_uint64
        _native.append((directory, library))
    return _native[0][1].tpuasm_highway_hash(data, len(data))

def highway_hash(data: bytes) -> int:
    """以 Riegeli 的固定密钥计算 HighwayHash-64；与参考实现的可移植版本逐步对应。

    chunk 头和块头只有几十字节，直接在 Python 中计算；Python 实现每 MiB 约需 0.4 秒，chunk 的内容超过 64 KiB 时改用原生实现。
    """
    if len(data) > _BLOCK:
        return _native_hash(data)
    v0 = [_INIT0[i] ^ _KEY[i] for i in range(4)]
    v1 = [_INIT1[i] ^ ((_KEY[i] >> 32) | (_KEY[i] << 32) & _MASK) for i in range(4)]
    mul0 = list(_INIT0)
    mul1 = list(_INIT1)

    def update(lanes: Sequence[int]) -> None:
        for i in range(4):
            v1[i] = (v1[i] + mul0[i] + lanes[i]) & _MASK
            mul0[i] ^= (v1[i] & 0xffffffff) * (v0[i] >> 32)
            v0[i] = (v0[i] + mul1[i]) & _MASK
            mul1[i] ^= (v0[i] & 0xffffffff) * (v1[i] >> 32)
        for i in (0, 2):
            add0, add1 = _zipper(v1[i + 1], v1[i])
            v0[i] = (v0[i] + add0) & _MASK
            v0[i + 1] = (v0[i + 1] + add1) & _MASK
        for i in (0, 2):
            add0, add1 = _zipper(v0[i + 1], v0[i])
            v1[i] = (v1[i] + add0) & _MASK
            v1[i + 1] = (v1[i + 1] + add1) & _MASK

    whole = len(data) // 32 * 32
    for lanes in struct.iter_unpack('<4Q', data[:whole]):
        update(lanes)
    remainder = len(data) - whole
    if remainder:
        tail = data[whole:]
        for i in range(4):
            v0[i] = (v0[i] + ((remainder << 32) + remainder)) & _MASK
        # Rotate each 32-bit half of v1 left by the remainder size.
        for i in range(4):
            halves = []
            for half in (v1[i] & 0xffffffff, v1[i] >> 32):
                halves.append(((half << remainder) | (half >> (32 - remainder))) & 0xffffffff)
            v1[i] = halves[0] | (halves[1] << 32)
        packet = bytearray(32)
        size4 = remainder & ~3
        packet[:size4] = tail[:size4]
        if remainder & 16:
            packet[28:32] = tail[remainder - 4:remainder]
        elif remainder & 3:
            rest = tail[size4:]
            packet[16] = rest[0]
            packet[17] = rest[(remainder & 3) >> 1]
            packet[18] = rest[(remainder & 3) - 1]
        update(struct.unpack('<4Q', packet))
    for _ in range(4):
        update((((v0[2] >> 32) | (v0[2] << 32)) & _MASK, ((v0[3] >> 32) | (v0[3] << 32)) & _MASK, ((v0[0] >> 32) | (v0[0] << 32)) & _MASK, ((v0[1] >> 32) | (v0[1] << 32)) & _MASK))
    return (v0[0] + v1[0] + mul0[0] + mul1[0]) & _MASK

def _content(data: bytes, position: int, size: int) -> tuple[bytes, int]:
    """从 position 起读 size 字节的 chunk 内容，跳过途中的块头；返回内容和读完后的位置。"""
    parts = []
    while size:
        if position % _BLOCK < _BLOCK_HEADER:
            position += _BLOCK_HEADER - position % _BLOCK
        take = min(size, _BLOCK - position % _BLOCK)
        if position + take > len(data):
            raise ValueError('truncated Riegeli/records file')
        parts.append(data[position:position + take])
        position += take
        size -= take
    return b''.join(parts), position

def is_riegeli(data: bytes) -> bool:
    """data 是否以 Riegeli/records 的文件签名 chunk 开头。"""
    if len(data) < _BLOCK_HEADER + _CHUNK_HEADER:
        return False
    header = data[_BLOCK_HEADER:_BLOCK_HEADER + _CHUNK_HEADER]
    return header[8:] == struct.pack('<QQ8sQ', 0, highway_hash(b''), b's' + bytes(7), 0) and header[:8] == struct.pack('<Q', highway_hash(header[8:]))

def read_records(data: bytes) -> list[bytes]:
    """读出全部记录。只支持不压缩的 simple chunk；不校验 hash。"""
    if not is_riegeli(data):
        raise ValueError('missing Riegeli/records file signature')
    records = []
    position = 0
    while position < len(data):
        begin = position
        header, position = _content(data, position, _CHUNK_HEADER)
        data_size, _, kind, decoded_size = struct.unpack('<8xQQ8sQ', header)
        count = int.from_bytes(kind[1:], 'little')
        body, position = _content(data, position, data_size)
        # 一个 chunk 至少占 num_records 个字节，使每条记录都有各自的文件位置。
        position = max(position, begin + count)
        if kind[:1] in (b's', b'p', b'm'):
            continue
        if kind[:1] != b'r':
            raise ValueError(f'unsupported Riegeli chunk type {kind[:1]!r}; only simple chunks are supported')
        if body[:1] != b'\0':
            raise ValueError(f'unsupported Riegeli compression {body[:1]!r}; only uncompressed records are supported')
        sizes_size, start = read_varint(body, 1)
        values = start + sizes_size
        for _ in range(count):
            size, start = read_varint(body, start)
            records.append(body[values:values + size])
            values += size
        if start != 1 + len(encode_varint(sizes_size)) + sizes_size or values != len(body):
            raise ValueError('invalid Riegeli simple chunk')
    return records

def _append_chunk(output: bytearray, kind: bytes, count: int, decoded_size: int, body: bytes) -> None:
    """在 output 末尾写一个 chunk，途中每到块边界先写块头。"""
    fields = struct.pack('<QQ8sQ', len(body), highway_hash(body), kind + count.to_bytes(7, 'little'), decoded_size)
    content = struct.pack('<Q', highway_hash(fields)) + fields + body
    begin = len(output)
    # 先算出 chunk 的结束位置：块头要记录从块的开头到被它打断的 chunk 末尾的距离。
    end = begin
    remaining = len(content)
    while remaining:
        if end % _BLOCK == 0:
            end += _BLOCK_HEADER
        take = min(remaining, _BLOCK - end % _BLOCK)
        end += take
        remaining -= take
    offset = 0
    while len(output) < end:
        at = len(output)
        if at % _BLOCK == 0:
            links = struct.pack('<QQ', at - begin, end - at)
            output += struct.pack('<Q', highway_hash(links)) + links
        take = min(len(content) - offset, _BLOCK - len(output) % _BLOCK)
        output += content[offset:offset + take]
        offset += take

def write_records(records: Sequence[bytes]) -> bytes:
    """文件签名 chunk 之后，把每条记录写成一个不压缩的 simple chunk。"""
    output = bytearray()
    _append_chunk(output, b's', 0, 0, b'')
    for record in records:
        sizes = encode_varint(len(record))
        _append_chunk(output, b'r', 1, len(record), b'\0' + encode_varint(len(sizes)) + sizes + record)
    return bytes(output)

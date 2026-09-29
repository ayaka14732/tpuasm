"""离线检查 BCS 汇编、编码歧义、semantic 互操作与拒绝路径。

在仓库根目录运行：PYTHONPATH=src python tests/reproduce_tpu_v4_bcs.py 需要匹配的 libtpu，不使用 TPU。
"""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import tempfile

from tpuasm import assemble_listing, decode_tpu_v4_bcs_program, dump_executable, encode_tpu_v4_bcs_program, executable_programs, extract_tpu_v4_bcs_program, format_assembly
from tpuasm._protobuf import encode_varint
from tpuasm._protobuf import message

TARGET = 'tpu-v4-bcs'

def one(instructions: str, constraints: str = '') -> bytes:
    encoding = ' ; .encoding { ' + constraints + ' }' if constraints else ''
    return assemble_listing('.target tpu-v4-bcs\n{ ' + instructions + encoding + ' }\n.align 16\n')

def verify(image: bytes) -> None:
    assert assemble_listing(format_assembly(image, target=TARGET)) == image
    canonical = format_assembly(image, target=TARGET, encoding='canonical')
    assert format_assembly(assemble_listing(canonical), target=TARGET, encoding='canonical') == canonical
    assert encode_tpu_v4_bcs_program(decode_tpu_v4_bcs_program(image)) == image

def reject(source: str) -> None:
    try:
        assemble_listing(source)
    except ValueError:
        return
    raise AssertionError('invalid source accepted: ' + source)

def main() -> None:
    root = Path(__file__).resolve().parent
    for name in ('basic', 'encoding'):
        source = root / 'data' / 'tpu_v4_bcs' / f'{name}.tpuasm'
        verify(assemble_listing(source.read_text(encoding='utf-8'), filename=str(source)))
    pairs = (
        ('s0: smov s0, 0x1234', 's0.sy = 32', 's0.sy = 33'),
        ('s0: smov s0, 0x12345678', 's0.sy = 44', 's0.sy = 45'),
        ('s0: smov s0, sy.zero', 'imm0 = 0', 'imm0 = 21'),
    )
    for instruction, left, right in pairs:
        a, b = one(instruction, left), one(instruction, right)
        assert a != b
        assert format_assembly(a, target=TARGET, encoding='canonical') == format_assembly(b, target=TARGET, encoding='canonical')
        verify(a)
        verify(b)
    # Golden machine bytes independently obtained from the native codec.
    assert one('s0: shalt')[:32].hex() == '000000000000000000000000e00300000f000000000000000000000000000000'
    assert one('s0: srdreg s4, gtc0 ; s1: srdreg s5, gtc1')[:32].hex() == '000000000000000000803280ee1101740f000000000000000000000000000000'
    assert one('s0: smov s1, 0x12345678 ; s1: smov s2, 0xdeadbeef') == one('s1: smov s2, 0xdeadbeef ; s0: smov s1, 0x12345678')
    verify(one('s0: sbr.rel 7 ; s1: smov s1, 0x12345678'))
    invalid = (
        '.target tpu-v4-bcs\n{}\n',
        '.target tpu-v4-bcs\n{ va0: shalt }\n.align 16\n',
        '.target tpu-v4-bcs\n{ s1: sbr.rel 0 }\n.align 16\n',
        '.target tpu-v4-bcs\n{ s0: smov s32, 0 }\n.align 16\n',
        '.target tpu-v4-bcs\n{ s0: smov s0, 0x100000000 }\n.align 16\n',
        '.target tpu-v4-bcs\n{ s0: smov s0, 7 ; .encoding { s0.sy = 32 ; imm0 = 8 } }\n.align 16\n',
        '.target tpu-v4-bcs\n{ s0: sbr.rel 7 ; s1: smov s1, 8 ; .encoding { s1.sy = 32 } }\n.align 16\n',
        '.target tpu-v4-bcs\n{ s0: smov s0, 7 ; .encoding { a1 = 32 } }\n.align 16\n',
    )
    for invalid_source in invalid:
        reject(invalid_source)
    proto = decode_tpu_v4_bcs_program(one('s0: shalt'))
    # A synthetic record exercises the container adapter; it is not a captured
    # executable and must never be presented as device execution evidence.
    wrapper = message([(2, 1), (3, 2), (10, proto)])
    core = message([(2, 1), (6, message([(1, wrapper)]))])
    assert extract_tpu_v4_bcs_program(core) == proto
    container = b'\x03abc' + encode_varint(len(core)) + core
    assert executable_programs(container, target=TARGET) == [(1, 0, one('s0: shalt'))]
    with tempfile.TemporaryDirectory(prefix='tpuasm-bcs-', dir='/tmp') as output:
        paths = dump_executable(container, Path(output), target=TARGET)
        assert len(paths) == 1 and paths[0].name == 'program-tpu-v4-bcs-1-0.tpuasm'
        assert assemble_listing(paths[0].read_text()) == one('s0: shalt')
    for invalid_proto in (b'', b'\x0a\x00' * 15, message([(2, b'')]) * 16):
        try:
            encode_tpu_v4_bcs_program(invalid_proto)
        except ValueError:
            continue
        raise AssertionError('invalid semantic program accepted')
    print(f'{version("libtpu")}: BCS fixtures, 3 collision pairs, shared lanes, container and rejection checks OK')

if __name__ == '__main__':
    main()

"""离线检查编码歧义、精确源码往返和普通源码的编码稳定性。

在仓库根目录运行：PYTHONPATH=src python tests/reproduce_tpu_v4_tc.py 需要匹配的 libtpu，不使用 TPU；输入保存在 tests/data/tpu_v4_tc/。
"""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from tpuasm import assemble_listing, format_assembly

def one(instruction: str, constraint: str = '') -> bytes:
    suffix = f' ; .encoding {{ {constraint} }}' if constraint else ''
    return assemble_listing('.target tpu-v4-tc\n{ ' + instruction + suffix + ' }\n.empty 9\n')

def main() -> None:
    cases = [
        ('s0: simm.s32 s0, 0', '', 'imm0 = 21'),
        ('s0: simm.s32 s0, 0', '', 's0.y = lo(imm0)'),
        ('va0: vadd.8x128.s32 v2, v3, v1', '', 'imm0 = 21'),
        ('cld: cld.8x128 crf, [cmem:0]', '', 'vs0 = s1'),
        ('cld: cld.8x128 crf, [cmem:0]', 'cld.offset = imm2', 'cld.offset = imm3'),
        ('vx1: vmatmul.8x128.f32 mrf0, v0', 'vx1.read = port.va0.x', 'vx1.read = port.va0.y'),
        ('vr1: vpop.8x128 v0, mrf0', 'vr1.write = port.va0.dst', 'vr1.write = port.va1.dst'),
        ('vr1: vpop.8x128 v0, mrf0', 'vr1.write = port.va0.dst', 'vr1.write = port.va0.dst ; port.va1.dst = 1'),
        ('misc: vdelay 1', 'misc.delay_count = const(1)', 'misc.delay_count = imm2'),
    ]
    for index, (instruction, left, right) in enumerate(cases, 1):
        a, b = one(instruction, left), one(instruction, right)
        assert a != b, (index, 'expected distinct program image bytes')
        assert format_assembly(a, encoding='canonical', target='tpu-v4-tc') == format_assembly(b, encoding='canonical', target='tpu-v4-tc'), index
        exact_a, exact_b = format_assembly(a, target='tpu-v4-tc'), format_assembly(b, target='tpu-v4-tc')
        assert exact_a != exact_b, index
        assert assemble_listing(exact_a) == a, index
        assert assemble_listing(exact_b) == b, index
    print(f'{len(cases)} encoding collision pairs: OK', flush=True)

    data = Path(__file__).resolve().parent / 'data' / 'tpu_v4_tc'
    for name in ('normal', 'xpose', 'slots'):
        source_path = data / f'{name}.tpuasm'
        source = source_path.read_text(encoding='utf-8')
        image = assemble_listing(source, filename=str(source_path))
        assert assemble_listing(format_assembly(image, target='tpu-v4-tc')) == image, source_path
        canonical = format_assembly(image, encoding='canonical', target='tpu-v4-tc')
        canonical_image = assemble_listing(canonical)
        assert assemble_listing(format_assembly(canonical_image, target='tpu-v4-tc')) == canonical_image, source_path
        assert format_assembly(canonical_image, encoding='canonical', target='tpu-v4-tc') == canonical, source_path
        if name in ('normal', 'xpose'):
            assert len(image) == 34816, source_path
            assert format_assembly(image, target='tpu-v4-tc') == source, source_path
        else:
            assert len(image) == 1024, source_path
        print(f'{name}: {len(image)} program image bytes, exact/canonical checks OK', flush=True)
    print(version('libtpu'), f'{len(cases)} pairs and 3 source files: OK')

if __name__ == '__main__':
    main()

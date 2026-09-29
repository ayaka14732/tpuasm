"""离线检查 TPU v6e TC 的编码歧义、全部指令形式的往返、样例源码与拒绝路径。

在仓库根目录运行：PYTHONPATH=src python tests/reproduce_tpu_v6e_tc.py 需要匹配的 libtpu，不使用 TPU；输入保存在 tests/data/tpu_v6e_tc/。
"""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import random

from tpuasm import assemble_listing, format_assembly
from tpuasm.tpu_v6e_tc_isa_data import EMPTY_WORD, ENUMS, FORMATTER_ABORTS
from tpuasm.tpu_v6e_tc_model import FORMS, SHARED, GlForm

TARGET = 'tpu-v6e-tc'
EMPTY = EMPTY_WORD.to_bytes(64, 'little')

def one(instruction: str, constraint: str = '') -> bytes:
    suffix = f' ; .encoding {{ {constraint} }}' if constraint else ''
    return assemble_listing(f'.target {TARGET}\n{{ ' + instruction + suffix + ' }\n.empty 7\n')

def random_word(form: GlForm, rng: random.Random) -> int:
    """按字段表构造一个形式的机器字：随机的共享立即数、标量槽和非保留字段值。"""
    word = EMPTY_WORD
    for field in SHARED.values():
        word = (word & ~field.mask) | (rng.getrandbits(field.width) << field.start)
    for start, width, value in form.fixed_bits:
        mask = ((1 << width) - 1) << start
        word = (word & ~mask) | (value << start)
    for name in form.layout:
        field = form.fields[name]
        enum = form.enums.get(name)
        if enum:
            value = rng.choice([value for value, label in ENUMS[enum].items() if value not in FORMATTER_ABORTS.get(enum, ()) and 'RESERVED' not in label and 'INVALID' not in label])
        else:
            value = rng.getrandbits(field.width)
        word = (word & ~field.mask) | (value << field.start)
    return (word & ~form.predicate_field.mask) | (14 << form.predicate_field.start)

def rejected(action: object, message: str) -> None:
    assert callable(action)
    try:
        action()
    except ValueError as error:
        assert message in str(error), error
    else:
        raise AssertionError(f'expected rejection: {message}')

def main() -> None:
    cases = [
        ('s0: simm.s32 s0, 0', '', 'imm0 = 21'),
        ('s0: simm.s32 s0, 0', '', 's0.y = zero_imm0'),
        ('s0: simm.s32 s0, 0x7ffffff', '', 'imm0 = 1048575'),
        ('va0: vimm.8x128.s32 v10, 0xb2461357', '', 'imm0 = 660311 ; imm1 = 373318'),
        ('va0: vadd.8x128.s32 v2, v3, v1', '', 'imm0 = 21'),
        ('vst: vst.8x128 [vmem:0x40], v1', 'vst.offset = imm0', 'vst.offset = imm1'),
        ('vx1: vmatmul.8x128.f32 mrf1, v0', 'vx1.vex_source = vst_source', 'vx1.vex_source = v0_x'),
        ('s0: por p0, 0, 0', '', 's0.px = inverted_always'),
        ('va0: vunpack.vsel.c.bf16 v10, v0, s20', 'va0.vs = 0', 'va0.vs = 3'),
        ('misc: vdelay 1', 'misc.form = vector_delay ; misc.operand = one', 'misc.form = vector_delay ; misc.operand = imm_0'),
        # The current vtrace formatter model still allows the low field's high four bits to overlap the high field.
        ('misc: vtrace 0xd0010001', 'misc.operand = imm0 ; misc.upper_operand_field = imm1 ; imm0 = 1 ; imm1 = 53249', 'misc.operand = imm0 ; misc.upper_operand_field = imm1 ; imm0 = 65537 ; imm1 = 53248'),
    ]
    for index, (instruction, left, right) in enumerate(cases, 1):
        a, b = one(instruction, left), one(instruction, right)
        assert a != b, (index, 'expected distinct program image bytes')
        assert format_assembly(a, encoding='canonical', target=TARGET) == format_assembly(b, encoding='canonical', target=TARGET), index
        exact_a, exact_b = format_assembly(a, target=TARGET), format_assembly(b, target=TARGET)
        assert exact_a != exact_b, index
        assert assemble_listing(exact_a) == a, index
        assert assemble_listing(exact_b) == b, index
    print(f'{len(cases)} encoding collision pairs: OK', flush=True)

    # One random instance of every registered form, in every physical slot.
    rng = random.Random(0)
    for form in FORMS:
        image = random_word(form, rng).to_bytes(64, 'little') + EMPTY * 7
        try:
            exact = format_assembly(image, target=TARGET)
        except ValueError as error:
            # vtrace halves from different scalar lanes have no formatter text either.
            assert form.name == 'trace', (form.slot, form.name, error)
            continue
        assert assemble_listing(exact) == image, (form.slot, form.name)
        canonical = format_assembly(image, encoding='canonical', target=TARGET)
        assert format_assembly(assemble_listing(canonical), encoding='canonical', target=TARGET) == canonical, (form.slot, form.name)
    print(f'{len(FORMS)} slot forms: exact and canonical roundtrips OK', flush=True)

    data = Path(__file__).resolve().parent / 'data' / 'tpu_v6e_tc'
    for name in ('slots', 'matmul'):
        source_path = data / f'{name}.tpuasm'
        source = source_path.read_text(encoding='utf-8')
        image = assemble_listing(source, filename=str(source_path))
        assert assemble_listing(format_assembly(image, target=TARGET)) == image, source_path
        canonical = format_assembly(image, encoding='canonical', target=TARGET)
        canonical_image = assemble_listing(canonical)
        assert format_assembly(canonical_image, encoding='canonical', target=TARGET) == canonical, source_path
        if name == 'matmul':
            uncommented = ''.join(line for line in source.splitlines(keepends=True) if not line.startswith('#'))
            assert format_assembly(image, target=TARGET) == uncommented, source_path
        print(f'{name}: {len(image)} program image bytes, exact/canonical checks OK', flush=True)

    rejected(lambda: one('va0: @!p0 vadd.8x128.s32 v2, v3, v1'), 'va0 has no inverted predicate')
    rejected(lambda: one('s0: @p14 sfence'), 'predicate register must be p0..p13')
    rejected(lambda: assemble_listing(f'.target {TARGET}\n{{ va0: vmul.8x128.u32.u64 v1, v2, v3, v4 ; va1: vadd.8x128.s32 v5, v6, v7 }}\n.empty 7\n'), 'cannot issue in the same bundle as va1')
    rejected(lambda: assemble_listing(f'.target {TARGET}\n{{ s0: sfence ; dma: dma.simple [vmem:s1], [hbm:s2], length=s3, dst_flag=[sflag:52] }}\n.empty 7\n'), 'cannot issue in the same bundle as dma')
    # A reserved shuffle selector aborts the libtpu formatter; decoding rejects it first.
    shuffled = next(form for form in FORMS if form.slot == 'vld0' and form.name == 'vector_load_shuffled')
    word = random_word(shuffled, rng) & ~shuffled.fields['shuffle'].mask
    rejected(lambda: format_assembly(word.to_bytes(64, 'little') + EMPTY * 7, target=TARGET), 'reserved vld0 shuffle encoding')
    print(version('libtpu'), f'{len(cases)} pairs, {len(FORMS)} forms, 2 source files and rejection checks: OK')

if __name__ == '__main__':
    main()

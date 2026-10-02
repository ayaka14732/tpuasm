"""离线检查 TPU v6e TEC 的编码歧义、全部指令形式的往返、样例源码与拒绝路径。

在仓库根目录运行：PYTHONPATH=src python tests/reproduce_tpu_v6e_tec.py 需要匹配的 libtpu，不使用 TPU；输入保存在 tests/data/tpu_v6e_tec/。
"""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import random

from tpuasm import assemble_listing, format_assembly
from tpuasm.tpu_v6e_tec_isa import FALLBACKS
from tpuasm.tpu_v6e_tec_isa_data import EMPTY_WORD, ENUMS
from tpuasm.tpu_v6e_tec_model import ALWAYS, FORMS, SHARED, TecForm

TARGET = 'tpu-v6e-tec'

def one(instruction: str, constraint: str = '') -> bytes:
    suffix = f' ; .encoding {{ {constraint} }}' if constraint else ''
    return assemble_listing(f'.target {TARGET}\n{{ ' + instruction + suffix + ' }\n')

def random_word(form: TecForm, rng: random.Random) -> int:
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
            value = rng.choice([value for value, label in ENUMS[enum].items() if 'RESERVED' not in label and 'INVALID' not in label])
        else:
            value = rng.getrandbits(field.width)
        word = (word & ~field.mask) | (value << field.start)
    word = (word & ~form.inversion_field.mask)
    return (word & ~form.predicate_field.mask) | (ALWAYS << form.predicate_field.start)

def rejected(action: object, message: str, kind: type[Exception] = ValueError) -> None:
    assert callable(action)
    try:
        action()
    except kind as error:
        assert message in str(error), error
    else:
        raise AssertionError(f'expected rejection: {message}')

def main() -> None:
    cases = [
        ('misc: simm.s32 s5, 0x2', '', 'misc.y = zero_imm0'),
        ('s0: sadd.s32 s0, 0x4e, s0', '', 's0.y = zero_imm1'),
        ('vld: vld v1, [tilespmem:0x80]', '', 'vld.offset = imm4'),
        ('s1: sand.u32 s0, 0xfffffff, s31', '', 's1.y = imm3_imm2'),
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
    named = 0
    fallbacks = {(signature.form.slot, signature.form.name): signature.mnemonic for signature in FALLBACKS}
    for form in FORMS:
        image = random_word(form, rng).to_bytes(64, 'little')
        exact = format_assembly(image, target=TARGET)
        assert assemble_listing(exact) == image, (form.slot, form.name)
        named += f'{form.slot}: {fallbacks[(form.slot, form.name)]} ' in exact
        canonical = format_assembly(image, encoding='canonical', target=TARGET)
        assert format_assembly(assemble_listing(canonical), encoding='canonical', target=TARGET) == canonical, (form.slot, form.name)
    print(f'{len(FORMS)} slot forms: exact and canonical roundtrips OK; {named} random instances use the named-field syntax', flush=True)

    data = Path(__file__).resolve().parent / 'data' / 'tpu_v6e_tec'
    for name in ('slots', 'add_one'):
        source_path = data / f'{name}.tpuasm'
        source = source_path.read_text(encoding='utf-8')
        image = assemble_listing(source, filename=str(source_path))
        assert assemble_listing(format_assembly(image, target=TARGET)) == image, source_path
        canonical = format_assembly(image, encoding='canonical', target=TARGET)
        canonical_image = assemble_listing(canonical)
        assert format_assembly(canonical_image, encoding='canonical', target=TARGET) == canonical, source_path
        if name == 'add_one':
            uncommented = ''.join(line for line in source.splitlines(keepends=True) if not line.startswith('#'))
            assert format_assembly(image, target=TARGET) == uncommented, source_path
        print(f'{name}: {len(image)} program image bytes, exact/canonical checks OK', flush=True)

    rejected(lambda: one('s0: @p14 sfence'), 'predicate register must be p0..p13')
    rejected(lambda: assemble_listing(f'.target {TARGET}\n{{ s0: sfence ; dma: dma.local [timem:s3], [sflag:s2], [hbm:s0], s1 }}\n'), 'cannot issue in the same bundle as dma')
    rejected(lambda: assemble_listing(f'.target {TARGET}\n{{ s0: sfence ; stream: stream.linear.gather [tilespmem:s0], [sflag:0x1], [hbm4b:s1+s0], 0x400, 0x38 }}\n'), 'cannot issue in the same bundle as stream')
    rejected(lambda: format_assembly(b'\0' * 32, target=TARGET), 'not block-aligned', RuntimeError)
    print(version('libtpu'), f'{len(cases)} pairs, {len(FORMS)} forms, 2 source files and rejection checks: OK')

if __name__ == '__main__':
    main()

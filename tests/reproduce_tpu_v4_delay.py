"""在本机 TPU v4 上比较 Delay selector 0、立即数 0、1、2、16 与 DelayFixed 一周期形式。

复用 v6e 探针的输入 DMA、xor 标记和输出 DMA 载体。以 insert_executable_bundles 插入计数和 delay；计数前后均从 vector FIFO 读回，避免把输入 DMA 等待计入周期。
多 host 环境仅使用本 host 四芯片：TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 PYTHONPATH=src python tests/reproduce_tpu_v4_delay.py
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
from typing import cast

import jax
from jaxlib.xla_client import LoadedExecutable
import numpy as np

from reproduce_tpu_v6e_execution import kernel, op, place
from tpuasm import BundleInsertion, assemble_listing, executable_programs, format_assembly, insert_executable_bundles, load_executable, replace_executable_programs
from tpuasm.assembly_syntax import parse_assembly
from tpuasm.backends import select_backend
from tpuasm.tpu_v4_tc_codec import EMPTY_WORD, encode_program
from tpuasm.tpu_v4_tc_model import FORMS

TARGET = 'tpu-v4-tc'

def delay_source(operand: str | int) -> str:
    form = next(form for form in FORMS if form.slot == 'misc' and form.name == ('DelayFixed' if operand == 'fixed' else 'Delay'))
    fixed = form.fixed(15)
    word = (EMPTY_WORD & ~fixed.mask) | fixed.value
    values = {'delay_count': 0} if operand == 'fixed' else {'delay_count': 0 if operand == 'builtin' else 4, 'imm2': 0 if operand == 'builtin' else int(operand)}
    for key, value in values.items():
        field = form.fields[key]
        word = (word & ~field.mask) | (value << field.start)
    encoded = encode_program([(word, (form,))] + [(EMPTY_WORD, ())] * 9)
    return '\n'.join(line for line in format_assembly(encoded, target=TARGET).splitlines()[1:] if line != '{}' and not line.endswith(':')) + '\n'

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpuasm-v4-delay'))
    args = parser.parse_args()
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    device = jax.local_devices()[0]
    assert device.device_kind == 'TPU v4'
    assert version('libtpu') == '0.0.49'
    host = np.arange(1024, dtype=np.uint32).reshape(8, 128)
    x = place(host)
    compiled = jax.jit(kernel, compiler_options={'xla_msa_enable': 'false', 'xla_tpu_vmem_scavenging_mode': 'NONE'}).lower(x).compile()
    np.testing.assert_array_equal(np.asarray(compiled(x)), host ^ 0x13579bdf)
    raw = bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize())
    (record, index, image), = executable_programs(raw)
    source = format_assembly(image, target=TARGET)
    program = parse_assembly(source)
    markers = [(pc, item) for pc, bundle in enumerate(program.bundles) for item in bundle.instructions if item.mnemonic.startswith('vxor.8x128') and '0x13579bdf' in item.operands]
    assert len(markers) == 1
    pc, marker = markers[0]
    assert len(program.bundles[pc].instructions) == 1
    old = marker.slot + ': ' + marker.mnemonic + ' ' + ', '.join(marker.operands)
    assert source.count(old) == 1
    source = source.replace(old, f'{marker.slot}: vmov.8x128 {marker.operands[0]}, v10')
    raw = replace_executable_programs(raw, {(record, index): assemble_listing(source)})
    records = []
    minima = {}
    for count in (1, 16, 64, 128):
        for operand in ('fixed', 'builtin', 0, 1, 2, 16):
            body = op('misc: vsyncset.s32 [sflag:100], 0') + op('misc: vsyncmov v2sf, [sflag:100]') + op('s0: spop s23, v2sf')
            body += op('s0: srdreg.lcclo s20') + delay_source(operand) * count
            body += op('misc: vsyncmov v2sf, [sflag:100]') + op('s0: spop s22, v2sf') + op('s0: srdreg.lcclo s21')
            body += op('s0: ssub.s32 s20, s21, s20') + op('va0: vmov.8x128 v10, s20')
            fragment = f'.target {TARGET}\n.empty 16\n' + body + '\n.empty 16\n'
            patched = insert_executable_bundles(raw, {(record, index): [BundleInsertion(pc, fragment)]})
            function = load_executable(patched, compiled)
            cycles = []
            for _ in range(20):
                actual = np.asarray(function(x))
                assert np.all(actual == actual[0, 0])
                cycles.append(int(actual[0, 0]))
            records.append({'count': count, 'operand': operand, 'cycles': cycles})
            minima[count, operand] = min(cycles)
            (output / f'delay-{count}-{operand}.tpuasm').write_text(fragment)
            print(count, operand, min(cycles), cycles, flush=True)
    backend, _ = select_backend(TARGET)
    environment = {
        'jax': jax.__version__,
        'libtpu': version('libtpu'),
        'device': str(device),
        'kind': device.device_kind,
        'python': f'{sys.version_info.major}.{sys.version_info.minor}{sys.abiflags}',
        'jaxlib': version('jaxlib'),
        'libtpu_build_id': backend.build_id,
    }
    for count in (64, 128):
        assert abs(minima[count, 'fixed'] - minima[count, 1]) <= 16
        assert abs(minima[count, 'builtin'] - minima[count, 1]) <= 16
        for operand in (1, 2, 16):
            assert abs(minima[count, operand] - minima[count, 0] - count * operand) <= 16
    np.testing.assert_array_equal(np.asarray(compiled(x)), host ^ 0x13579bdf)
    report = {'environment': environment, 'measurements': records, 'original_executable_baseline_passed': True}
    (output / 'delay-timings.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(records)} delay comparisons passed; original executable still returns its baseline result.', flush=True)

if __name__ == '__main__':
    main()

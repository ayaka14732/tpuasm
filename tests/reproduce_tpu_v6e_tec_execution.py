"""在单芯片 TPU v6e 上核对 TEC 的 selector、标量操作数槽和具名字段写法。

在仓库根目录运行：PYTHONPATH=src python tests/reproduce_tpu_v6e_tec_execution.py --output /tmp/tpuasm-v6e-tec-execution
需要已登记 tpu-v6e-tec 后端的 libtpu。只使用真实设备，不进入 CPU CI。

载体是一个 SparseCore kernel：把 i32[8,8] 的每个元素加上 0x7f3d1。每个案例导出 TEC 程序的 canonical 清单，把其中全部 ``vadd.s32 vD, 0x7f3d1, vS`` 换成待测写法，按需在前面的指令包中设置标量寄存器，并用 ``.encoding`` 固定待测的 selector；重新汇编、写回 executable 后执行，比较结果与 tpuasm 对该 selector 的数值模型。输出目录保存每个案例的清单和 JSONL 摘要。
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib.metadata import version
import json
from pathlib import Path
import re
from typing import Any, cast

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
from jaxlib.xla_client import LoadedExecutable
import numpy as np

from tpuasm import assemble_listing, executable_programs, format_assembly, load_executable, replace_executable_programs
from tpuasm.tpu_v6e_tec_constraints import value_names
from tpuasm.tpu_v6e_tec_isa import source
from tpuasm.tpu_v6e_tec_model import FORMS_BY_NAME

TARGET = 'tpu-v6e-tec'
MARKER = 0x7F3D1
# Shared immediate values for the selectors that read them.
IMMEDIATES = (0xA1357, 0x5B246, 0xC369A, 0x7D48B, 0xE5ABC, 0x9F6DE)

def body(x_hbm: Ref, out_hbm: Ref, tile: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, tile, sem).wait()
    tile[...] = tile[...] + MARKER
    pltpu.async_copy(tile, out_hbm, sem).wait()

def kernel(x: jax.Array) -> jax.Array:
    sc = plsc.VectorSubcoreMesh(core_axis_name='core', subcore_axis_name='tile', num_cores=1, num_subcores=1)
    return pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((8, 8), jnp.int32),
        mesh=sc,
        scratch_types=(pltpu.VMEM((8, 8), jnp.int32), pltpu.SemaphoreType.DMA),
        name='tec_probe',
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
    )(x)

def value(name: str) -> int | None:
    """tpuasm 对一个 selector 取值的数值模型；共享立即数取 IMMEDIATES。"""
    expression = source(name, 'hex')
    if expression is None:
        return None
    _, _, _, parts, base, _ = expression
    result = base
    for field, shift, limit in parts:
        result |= (IMMEDIATES[int(field[3:])] & ((1 << (20 if limit is None else limit)) - 1)) << shift
    return result & 0xFFFFFFFF

# A replacement maps (destination, source register) to the new instruction text; pins are added to the same bundle.
Replacement = Callable[[str, str], str]

class Probe:
    def __init__(self, output: Path) -> None:
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        assert jax.devices()[0].device_kind == 'TPU v6 lite'
        self.host = (np.arange(64, dtype=np.int64).reshape(8, 8) * 0x1234567 - 0x40000000).astype(np.int32)
        self.x = jax.device_put(self.host)
        self.compiled = jax.jit(kernel).lower(self.x).compile()
        np.testing.assert_array_equal(np.asarray(self.compiled(self.x)), self.expected(MARKER))
        self.serialized = bytes(cast(LoadedExecutable, self.compiled.runtime_executable()).serialize())
        (record, index, image), = executable_programs(self.serialized, target=TARGET)
        self.key = record, index
        self.source = format_assembly(image, target=TARGET, encoding='canonical')
        self.bundles = re.findall(r'\{[^{}]*\}', self.source)
        self.markers = [position for position, bundle in enumerate(self.bundles) if f'{MARKER:#x}' in bundle]
        assert self.markers and not re.search(r'\bs2[0-3]\b', self.source)
        # Bundles before the first marker with free scalar slots carry the scalar setup.
        self.free = [position for position, bundle in enumerate(self.bundles[1:self.markers[0]], 1) if not re.search(r'\b(?:s0|s1|dma|stream):', bundle)]
        self.log = (output / 'results.jsonl').open('a', buffering=1)
        self.log.write(json.dumps({'environment': {'jax': jax.__version__, 'libtpu': version('libtpu'), 'device': str(jax.devices()[0])}}) + '\n')
        self.failures: list[str] = []

    def expected(self, addend: int) -> np.ndarray:
        return ((self.host.astype(np.int64) + addend) & 0xFFFFFFFF).astype(np.uint32).view(np.int32)

    def run(self, name: str, replacement: Replacement, pins: Callable[[str], str], addend: int, setup: tuple[tuple[str, str], ...] = ()) -> None:
        """setup 的每项是 (槽中指令, 该指令包的约束)，依次放进 self.free 中的指令包。"""
        bundles = list(self.bundles)
        for position, (instruction, constraint) in zip(self.free, setup):
            bundles[position] = _add(bundles[position], instruction, constraint)
        for position in self.markers:
            slot = None

            def substitute(match: re.Match[str]) -> str:
                nonlocal slot
                slot = match[1]
                return f'{match[1]}: {match[2]}' + replacement(match[3], match[4])

            text = re.sub(rf'\b(va[0-2]): ((?:@!?p[0-9]+ )?)vadd\.s32 (v[0-9]+), {MARKER:#x}, (v[0-9]+)', substitute, bundles[position])
            assert slot is not None
            bundles[position] = _add(text, '', pins(slot))
        listing = self.source
        for old, new in zip(self.bundles, bundles):
            listing = listing.replace(old, new, 1)
        (self.output / f'{name}.tpuasm').write_text(listing)
        patched = replace_executable_programs(self.serialized, {self.key: assemble_listing(listing)}, target=TARGET)
        result = np.asarray(load_executable(patched, self.compiled)(self.x))
        okay = bool(np.array_equal(result, self.expected(addend)))
        self.log.write(json.dumps({'name': name, 'addend': addend, 'match': okay, 'sample': [int(v) for v in result.ravel()[:4]]}) + '\n')
        print(name, 'OK' if okay else 'MISMATCH', hex(addend), flush=True)
        if not okay:
            self.failures.append(name)

def _add(bundle: str, instruction: str, constraint: str) -> str:
    """在指令包开头加一条指令，并把约束并入指令包的 .encoding。"""
    if instruction:
        bundle = '{ ' + instruction + ' ;\n  ' + bundle[1:].lstrip()
    if constraint:
        if '.encoding {' in bundle:
            bundle = bundle.replace('.encoding {', f'.encoding {{ {constraint} ;', 1)
        else:
            bundle = bundle[:-1].rstrip() + f' ;\n  .encoding {{ {constraint} }} }}'
    return bundle

def shared(names: tuple[str, ...]) -> str:
    return ' ; '.join(f'imm{index} = {IMMEDIATES[index]}' for index in sorted({int(match) for name in names for match in re.findall(r'imm([0-5])', name)}))

def vector_y(probe: Probe) -> None:
    """vadd.s32 的 y 源取每个立即数与常量 selector。"""
    form = FORMS_BY_NAME['va0', 'vector_add_s32']
    for name in value_names(form.enums['y_src']).values():
        addend = value(name)
        if addend is None:
            continue
        constraint = ' ; '.join(item for item in (f'{{slot}}.y_src = {name}', shared((name,))) if item)
        probe.run(f'vector-y-{name}', lambda dest, src: f'vadd.s32 {dest}, {addend:#x}, {src}', lambda slot: constraint.format(slot=slot), addend)

def scalar_y(probe: Probe) -> None:
    """simm.s32 的 y 取每个立即数与常量 selector，结果经标量操作数槽加到向量上。"""
    form = FORMS_BY_NAME['s0', 'move_y']
    for name in value_names(form.enums['y']).values():
        addend = value(name)
        if addend is None:
            continue
        constraint = ' ; '.join(item for item in ('s0.y = ' + name, shared((name,))) if item)
        probe.run(f'scalar-y-{name}', lambda dest, src: f'vadd.s32 {dest}, s20, {src}', lambda slot: '', addend, ((f's0: simm.s32 s20, {addend:#x}', constraint),))

def lanes(probe: Probe) -> None:
    """vadd.s32 的 y 源依次经 vs0..vs3 读取标量寄存器。"""
    addends = (0x11111, 0x2222222, 0x33333, 0x44444)
    setup = tuple((f's0: simm.s32 s{20 + lane}, {addend:#x}', '') for lane, addend in enumerate(addends))
    for lane, addend in enumerate(addends):
        probe.run(f'lane-vs{lane}', lambda dest, src: f'vadd.s32 {dest}, s{20 + lane}, {src}', lambda slot: f'{slot}.y_src = vs{lane}', addend, setup)

def named(probe: Probe) -> None:
    """具名字段写法：vector_add_s32 的 y 源取常量 two。"""
    probe.run('named-field-vector-add', lambda dest, src: f'vector_add_s32 dest={int(dest[1:]):#x}, x={int(src[1:]):#x}, y_src=0x2, y_vreg=0x0', lambda slot: '', 2)

GROUPS: dict[str, Callable[[Probe], Any]] = {'vector-y': vector_y, 'scalar-y': scalar_y, 'lanes': lanes, 'named': named}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('all', *GROUPS), default='all')
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpuasm-v6e-tec-execution'))
    args = parser.parse_args()
    probe = Probe(args.output)
    for group in GROUPS if args.group == 'all' else (args.group,):
        GROUPS[group](probe)
    np.testing.assert_array_equal(np.asarray(probe.compiled(probe.x)), probe.expected(MARKER))
    probe.log.close()
    assert not probe.failures, probe.failures
    print('All selected device checks passed; original executable still returns its baseline result.', flush=True)

if __name__ == '__main__':
    main()

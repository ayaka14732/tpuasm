"""在本机 TPU v4 上检查来源集合、物理槽、缓存、离线导出和补丁恢复。

在仓库根目录运行：
    PYTHONPATH=src python tests/reproduce_tpu_v4_tc_source_mapping.py --output-dir /tmp/tpuasm-source-reproduction

需要匹配的 JAX/jaxlib、libtpu 和本地四芯片，脚本在导入 JAX 前配置可见范围。产物包括 executable、带来源清单、JSON 和 results.json，不生成 LLO/Mosaic dump。--offline-only --output-dir PATH 复核已有产物和补丁恢复，不初始化 TPU。--trace-only 在 TPU 上检查启用 custom-call tracing 时的展开循环。
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import ctypes
from dataclasses import replace
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, cast
import warnings

os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '2,2,1'
os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'
os.environ['TPU_VISIBLE_CHIPS'] = '0,1,2,3'
os.environ['LIBTPU_INIT_ARGS'] = '--xla_enable_custom_call_region_trace=false --xla_xprof_enable_custom_call_tracing=false --xla_jf_auto_assign_mxu=false'

import jax
from jax import Ref
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import lowering as mosaic_lowering
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P
from jaxlib.mlir import ir
from jaxlib.xla_client import LoadedExecutable

from tpuasm import assemble_listing, compiler_source_mapping, dump_compiled, executable_programs, executable_source_maps, format_assembly
from tpuasm.tc_compiler import CompilerSourceMapping, _FLAGS, _maps
from tpuasm.tpu_v4_tc_assembler import _decoded_source
from tpuasm._protobuf import encode_varint, fields, read_varint
from tpuasm.targets import TPU_V4_TC
from tpuasm.tc_source_mapping import ProgramSourceMap, _programs, _source_map

PARAMS = pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True)
TC = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
SHAPE = (40, 128)

def arithmetic(kind: str, unroll: int | bool = 1) -> Callable[[jax.Array, jax.Array], jax.Array]:
    scratch = (pltpu.VMEM(SHAPE, jnp.float32),) * 2 + (pltpu.SemaphoreType.DMA, pltpu.SMEM((1,), jnp.int32))

    @pl.kernel(out_type=jax.ShapeDtypeStruct(SHAPE, jnp.float32), mesh=TC, scratch_types=scratch, name='same_name', compiler_params=PARAMS)
    def kernel(x_hbm: Ref, count_hbm: Ref, y_hbm: Ref, x: Ref, y: Ref, sem: Ref, count: Ref) -> None:
        pltpu.async_copy(x_hbm, x, sem).wait()
        pltpu.async_copy(count_hbm, count, sem).wait()
        y[...] = x[...]
        with jax.named_scope('iterations'):
            @pl.loop(0, count[0] if kind == 'dynamic' else 5, unroll=unroll)
            def body(i: jax.Array) -> None:
                with jax.named_scope('producer'):
                    intermediate = jnp.maximum(x[pl.ds(i * 8, 8), :], -1.0)
                with jax.named_scope('consumer'):
                    result = jnp.minimum(intermediate, 1.0)
                y[pl.ds(i * 8, 8), :] = result
        pltpu.async_copy(y, y_hbm, sem).wait()
    return kernel

def parallel() -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    shape = (8, 128)
    output = jax.ShapeDtypeStruct(shape, jnp.float32)
    scratch = (pltpu.VMEM(shape, jnp.float32),) * 3 + (pltpu.SemaphoreType.DMA,)

    @pl.kernel(out_type=(output, output), mesh=TC, scratch_types=scratch, name='same_bundle', compiler_params=PARAMS)
    def kernel(x_hbm: Ref, y_hbm: Ref, z_hbm: Ref, x: Ref, y: Ref, z: Ref, sem: Ref) -> None:
        pltpu.async_copy(x_hbm, x, sem).wait()
        data = x[...]
        y[...] = data + 0.5
        z[...] = data * 1.25
        pltpu.async_copy(y, y_hbm, sem).wait()
        pltpu.async_copy(z, z_hbm, sem).wait()
    return kernel

def fifo(sharding: jax.NamedSharding) -> jax.stages.Wrapped:
    @jax.jit(out_shardings=sharding, compiler_options={'xla_msa_enable': 'false', 'xla_tpu_vmem_scavenging_mode': 'NONE'})
    @jax.shard_map(out_specs=P(None, None), check_vma=False)
    def run(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        lhs_hbm = jax.new_ref(lhs, memory_space=pltpu.HBM)
        rhs_hbm = jax.new_ref(rhs, memory_space=pltpu.HBM)
        out_hbm = jax.empty_ref(jax.ShapeDtypeStruct((32, 128), jnp.float32), memory_space=pltpu.HBM)

        @pl.kernel(mesh=TC, name='pallas_v4_fifo_rhs_reuse', compiler_params=PARAMS)
        def kernel() -> None:
            sem_type = jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype)
            lhs_sem = jax.empty_ref(sem_type, memory_space=pltpu.SEMAPHORE)
            rhs_sem = jax.empty_ref(sem_type, memory_space=pltpu.SEMAPHORE)
            out_sem = jax.empty_ref(sem_type, memory_space=pltpu.SEMAPHORE)
            x = jax.empty_ref(jax.ShapeDtypeStruct(lhs.shape, lhs.dtype), memory_space=pltpu.VMEM @ TC)
            w = jax.empty_ref(jax.ShapeDtypeStruct(rhs.shape, rhs.dtype), memory_space=pltpu.VMEM @ TC)
            y = jax.empty_ref(jax.ShapeDtypeStruct((32, 128), jnp.float32), memory_space=pltpu.VMEM @ TC)
            lhs_dma = pltpu.async_copy(lhs_hbm, x, lhs_sem)
            rhs_dma = pltpu.async_copy(rhs_hbm, w, rhs_sem)
            lhs_dma.wait()
            rhs_dma.wait()
            pltpu.matmul_push_rhs(w[...], staging_register=0, mxu_index=0)
            # 保留 Python 展开与显式 RHS 复用语义。
            for tile in range(2):
                pltpu.matmul_lhs_fifo(x[tile * 16:(tile + 1) * 16, :], mxu_index=0, load_staged_rhs=0 if tile == 0 else None)
                y[tile * 16:(tile + 1) * 16, :] = pltpu.matmul_pop_fifo(shape=(16, 128), dtype=jnp.float32, mxu_index=0)
            pltpu.async_copy(y, out_hbm, out_sem).wait()
        kernel()
        return jax.freeze(out_hbm)
    return run

dwg_p = jax_core.Primitive('tpuasm_demo_dwg')
dwg_p.multiple_results = True

@dwg_p.def_effectful_abstract_eval
def _dwg_abstract_eval() -> tuple[list[jax_core.AbstractValue], jax_core.Effects]:
    return [], {tpu_primitives.mxu_effect}

def _dwg_lowering(ctx: mosaic_lowering.LoweringRuleContext) -> list[ir.Value]:
    del ctx
    cast(ir.Context, ir.Context.current).allow_unregistered_dialects = True
    mxu_id = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
    ir.Operation.create('llo.vdwg', attributes={'mxu_id': mxu_id})
    return []

mosaic_lowering.lowering_rules[tpu_core.CoreType.TC][dwg_p] = _dwg_lowering

def make_dwg_runner(mesh: jax.sharding.Mesh, sharding: jax.NamedSharding, transpose: bool) -> tuple[str, jax.stages.Wrapped]:
    name = 'tpuasm_xpose' if transpose else 'tpuasm_normal'

    @jax.jit(out_shardings=sharding, compiler_options={'xla_msa_enable': 'false'})
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(None, None), P(None, None)),
        out_specs=P(None, None),
        check_vma=False,
    )
    def run(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        @pl.kernel(out_type=jax.ShapeDtypeStruct((16, 128), jnp.float32), mesh=TC, name=name, compiler_params=PARAMS)
        def kernel(lhs_hbm: Ref, rhs_hbm: Ref, out_hbm: Ref) -> None:
            lhs_vmem = jax.empty_ref(jax.ShapeDtypeStruct((16, 128), jnp.bfloat16), memory_space=pltpu.VMEM @ TC)
            rhs_vmem = jax.empty_ref(jax.ShapeDtypeStruct((128, 128), jnp.bfloat16), memory_space=pltpu.VMEM @ TC)
            out_vmem = jax.empty_ref(jax.ShapeDtypeStruct((16, 128), jnp.float32), memory_space=pltpu.VMEM @ TC)
            lhs_sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
            rhs_sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
            out_sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
            pltpu.async_copy(lhs_hbm, lhs_vmem, lhs_sem).wait()
            pltpu.async_copy(rhs_hbm, rhs_vmem, rhs_sem).wait()
            pltpu.matmul_push_rhs(rhs_vmem[...], staging_register=0, mxu_index=0, transpose=transpose)
            dwg_p.bind()
            pltpu.matmul_lhs_fifo(lhs_vmem[...], mxu_index=0, load_staged_rhs=None)
            out_vmem[...] = pltpu.matmul_pop_fifo(shape=(16, 128), dtype=jnp.float32, mxu_index=0)
            pltpu.async_copy(out_vmem, out_hbm, out_sem).wait()
        return kernel(lhs, rhs)
    return name, run

def snapshot(state: CompilerSourceMapping) -> tuple[tuple[bytes, ...], tuple[tuple[int, int, str], ...]]:
    calls = tuple(ctypes.string_at(state.base + site, len(raw)) for site, (raw, _, _) in state.backend.calls.items())
    pages = tuple((site, site + len(raw), next(perms for start, end, perms, _, _ in _maps() if start <= state.base + site < end)) for site, (raw, _, _) in state.backend.calls.items())
    return calls, pages

def check_case(name: str, runner: jax.stages.Wrapped, inputs: tuple[jax.Array, ...], output: Path, warmup: int = 0) -> tuple[Any, list[ProgramSourceMap], dict[str, Any]]:
    jax.clear_caches()
    baseline = runner.lower(*inputs).compile()
    before = bytes(cast(LoadedExecutable, baseline.runtime_executable()).serialize())
    for _ in range(warmup):
        jax.block_until_ready(baseline(*inputs))
    baseline_value = jax.device_get(baseline(*inputs))
    jax.clear_caches()
    with compiler_source_mapping() as state:
        installed = snapshot(state)
        with compiler_source_mapping() as nested:
            assert nested is state and snapshot(state) == installed
            compiled = runner.lower(*inputs).compile()
        assert snapshot(state) == installed
    assert state._original()
    assert all(perms == 'r-xp' for _, _, perms in snapshot(state)[1])
    after = bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize())
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'executable.bin').write_bytes(after)
    (directory / 'baseline.bin').write_bytes(before)
    assert executable_programs(before) == executable_programs(after), name
    for _ in range(warmup):
        jax.block_until_ready(compiled(*inputs))
    result = jax.device_get(compiled(*inputs))
    for actual, expected in zip(jax.tree.leaves(result), jax.tree.leaves(baseline_value)):
        np.testing.assert_array_equal(actual, expected)
    maps = executable_source_maps(after)
    assert all(source.status == 'captured' and not source.diagnostics for source in maps), name
    paths = dump_compiled(compiled, directory, source_map_json=True)
    for path, (_, _, image) in zip(paths, executable_programs(after)):
        assert assemble_listing(path.read_text()) == image
        canonical = format_assembly(image, encoding='canonical', target='tpu-v4-tc')
        assert format_assembly(assemble_listing(canonical), encoding='canonical', target='tpu-v4-tc') == canonical
    if name in ('tpuasm_normal', 'tpuasm_xpose'):
        source, = maps
        stores = [slot for slot in source.slots if slot.slot == 'vst' and any(loc.primitive == 'swap' for origin in slot.origins for loc in origin.locations)]
        assert len(stores) == 2, name
        path, = paths
        lines = path.read_text().splitlines()
        assert any('vdwg.' in line and 'gmr0, gsfn0' in line for line in lines), name
        push_source = 'gsft0,' if name == 'tpuasm_xpose' else 'gsfn0,'
        assert any('vmatpush.' in line and push_source in line for line in lines), name
    origins = [origin for source in maps for slot in source.slots for origin in slot.origins]
    if name in ('fifo_reuse', 'tpuasm_normal', 'tpuasm_xpose'):
        expected = {'vld': 'get', 'vst': 'swap', 'vmatpush': 'matmul_push_rhs', 'vmatmul': 'matmul_lhs_fifo', 'vpop': 'matmul_pop_fifo'}
        checked = set()
        for source, (_, _, image) in zip(maps, executable_programs(after)):
            program, _ = _decoded_source(image)
            kernel_name = 'pallas_v4_fifo_rhs_reuse' if name == 'fifo_reuse' else name
            regions = [region for function in source.functions if function.hlo_name.startswith(kernel_name) for region in function.ranges]
            for pc, bundle in enumerate(program.bundles):
                if not any(region.image_start <= pc < region.image_limit for region in regions):
                    continue
                for inst in bundle.instructions:
                    family = inst.mnemonic.split('.')[0]
                    if family not in expected:
                        continue
                    entries = [slot for slot in source.slots if slot.image_pc == pc and slot.slot == inst.slot]
                    locations = [location for slot in entries for origin in slot.origins for location in origin.locations]
                    assert any(location.primitive == expected[family] and location.frames for location in locations), (name, pc, inst)
                    checked.add(family)
        assert checked == expected.keys(), (name, checked)
    report = {'counters': state.counters(), 'images': len(maps), 'mapped_slots': sum(len(source.slots) for source in maps), 'source_records': len(origins), 'isa_equal': True, 'numerical_equal': True}
    print(name, report, flush=True)
    return result, maps, report

def fifo_probe(mode: str, output: Path) -> None:
    mesh = jax.make_mesh((1,), ('device',), devices=jax.local_devices()[:1])
    jax.set_mesh(mesh)
    sharding = jax.NamedSharding(mesh, P(None, None), memory_kind='device')
    lhs = jax.device_put(((np.arange(32 * 128) % 17) - 8).reshape(32, 128).astype(jnp.bfloat16), sharding)
    rhs = jax.device_put(((np.arange(128 * 128) % 13) - 6).reshape(128, 128).astype(jnp.bfloat16), sharding)
    with compiler_source_mapping() if mode == 'captured' else nullcontext():
        compiled = fifo(sharding).lower(lhs, rhs).compile()
    values = np.stack([np.asarray(compiled(lhs, rhs)) for _ in range(4)])
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / f'fifo-{mode}.npy', values)
    (output / f'fifo-{mode}.bin').write_bytes(bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize()))

def records(data: bytes) -> list[bytes]:
    result = []
    pos = 0
    while pos < len(data):
        size, pos = read_varint(data, pos)
        result.append(data[pos:pos + size])
        pos += size
    return result

def message(number: int, value: int | bytes) -> bytes:
    return encode_varint(number << 3 | (2 if isinstance(value, bytes) else 0)) + (encode_varint(len(value)) + value if isinstance(value, bytes) else encode_varint(value))

def container(items: list[bytes]) -> bytes:
    return b''.join(encode_varint(len(item)) + item for item in items)

def check_offline(output: Path) -> None:
    left = (output / 'rolled/executable.bin').read_bytes()
    right = (output / 'multiple/executable.bin').read_bytes()
    a, b = records(left), records(right)
    original = executable_source_maps(left) + executable_source_maps(right)
    # Parser fixture made of independently compiled core/metadata pairs. It is
    # never executed and does not pretend to be a newly compiled multi-core image.
    core = _programs(left)[0].record
    a[core] += message(8, message(2, message(1, 2)) + message(3, b'non-code data') + message(4, b'data hash'))
    joined = executable_source_maps(container(a + b))
    assert len(joined) == 2 and joined[0].record != joined[1].record
    for actual, expected in zip(joined, original):
        assert actual.slots == expected.slots and actual.functions == expected.functions
        assert actual.image_hash == expected.image_hash and actual.metadata_program_id == expected.metadata_program_id
    code_set = next(value for number, _, value in fields(a[core]) if number == 8)
    assert isinstance(code_set, bytes)
    a[core] += message(8, code_set)
    ambiguous = executable_source_maps(container(a))
    assert len(ambiguous) == 2 and all(not source.slots and 'ambiguous' in source.diagnostics[0] for source in ambiguous)

    program = _programs(left)[0]
    source = original[0]
    for slot in source.slots:
        overlay = source.overlays[slot.overlay]
        expected_pc = overlay.translate(slot.annotation_key) if slot.coordinate_space == 'emitted' else slot.annotation_key
        assert slot.image_pc == expected_pc
        assert all(origin.llo_ordinal in location.ordinals for origin in slot.origins for location in origin.locations)
    image_annotations = [a for a in source.annotations if a.coordinate_space == 'image']
    assert any(a.image_pc < source.overlays[a.overlay].body_start for a in image_annotations)
    assert any(a.image_pc >= source.overlays[a.overlay].body_limit and a.annotation_pc != a.annotation_key for a in image_annotations)
    tail = [s for s in source.slots if s.annotation_key != s.annotation_pc and s.coordinate_space == 'emitted']
    assert tail and all(s.image_pc != source.overlays[s.overlay].body_start + s.annotation_pc for s in tail)
    # A physical-slot mismatch must be diagnosed and excluded, even with plausible
    # text and coordinates. The actual decoder remains the authority.
    overlay = next(o for o in source.overlays if o.emitted_limit > o.emitted_start)
    empty_pc = next(pc for pc in range(overlay.body_start, overlay.body_limit) if not any(s.image_pc == pc and s.slot == 'cld' for s in source.slots))
    key = empty_pc - overlay.body_start + overlay.emitted_start
    name = b'SLOT_CMEM_LOAD'
    slot_annotation = message(1, name) + message(2, b'fixture')
    entry = message(1, key) + message(2, message(2, message(1, name) + message(2, slot_annotation)))
    rejected = _source_map(replace(program, metadata=program.metadata + message(4, message(1, entry))), TPU_V4_TC)
    assert any('does not match an occupied physical slot' in diagnostic for diagnostic in rejected.diagnostics)
    print('multi-image container, ambiguity, physical-slot and overlay checks: OK', flush=True)

def check_lifecycle() -> None:
    original_lowering = mosaic_lowering.jaxpr_subcomp, mosaic_lowering._lower_jaxpr_to_for_loop
    environment = os.environ.get('LIBTPU_INIT_ARGS')
    probe = CompilerSourceMapping()
    flags = {name: probe._flag(name) for name in _FLAGS}
    initial = snapshot(probe)
    write = probe._write
    writes = 0
    def fail_once(site: int, raw: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError('exercise partial installation rollback')
        write(site, raw)
    probe._write = fail_once  # type: ignore[method-assign]
    try:
        probe._install()
        raise AssertionError('expected installation failure')
    except OSError:
        pass
    finally:
        probe._write = write  # type: ignore[method-assign]
        assert snapshot(probe) == initial
        probe._unmap()
    try:
        with compiler_source_mapping():
            with compiler_source_mapping():
                raise ValueError('exercise exception restoration')
    except ValueError:
        pass
    assert snapshot(probe) == initial
    assert flags == {name: probe._flag(name) for name in _FLAGS}
    assert os.environ.get('LIBTPU_INIT_ARGS') == environment
    assert (mosaic_lowering.jaxpr_subcomp, mosaic_lowering._lower_jaxpr_to_for_loop) == original_lowering
    print('partial installation, nested exception, flags, lowering and RX restoration: OK', flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--fifo-probe', choices=('baseline', 'captured'))
    parser.add_argument('--offline-only', action='store_true')
    parser.add_argument('--trace-only', action='store_true')
    args = parser.parse_args()
    output = args.output_dir or Path(tempfile.mkdtemp(prefix='tpuasm-source-reproduction-', dir='/tmp'))
    jax.config.update('jax_enable_compilation_cache', False)
    if args.offline_only:
        check_offline(output)
        check_lifecycle()
        return
    if args.fifo_probe:
        fifo_probe(args.fifo_probe, output)
        return
    if args.trace_only:
        os.environ['LIBTPU_INIT_ARGS'] = os.environ['LIBTPU_INIT_ARGS'].replace('trace=false', 'trace=true').replace('tracing=false', 'tracing=true')
    else:
        for mode in ('baseline', 'captured'):
            subprocess.run([sys.executable, __file__, '--fifo-probe', mode, '--output-dir', str(output)], check=True)
        cold = np.load(output / 'fifo-baseline.npy')
        np.testing.assert_array_equal(cold, np.load(output / 'fifo-captured.npy'))
        assert executable_programs((output / 'fifo-baseline.bin').read_bytes()) == executable_programs((output / 'fifo-captured.bin').read_bytes())
    mesh = jax.make_mesh((1,), ('device',), devices=jax.local_devices()[:1])
    sharding = jax.NamedSharding(mesh, P(None, None), memory_kind='device')
    host = np.linspace(-8, 8, np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    x = jax.device_put(host, jax.local_devices()[0])
    count = jax.device_put(np.array([3], np.int32), jax.local_devices()[0])
    if args.trace_only:
        value, _, _ = check_case('tracing', jax.jit(arithmetic('static', 2)), (x, count), output)
        np.testing.assert_array_equal(value, np.clip(host, -1, 1))
        return
    report: dict[str, Any] = {'versions': {package: version(package) for package in ('jax', 'jaxlib', 'libtpu')}, 'fifo_cold_sequence_equal': True, 'cases': {}}
    for name, kind, unroll in (('rolled', 'static', 1), ('unroll2', 'static', 2), ('unrolled', 'static', True), ('dynamic', 'dynamic', 1)):
        runner: jax.stages.Wrapped = jax.jit(arithmetic(kind, unroll))
        value, maps, case = check_case(name, runner, (x, count), output)
        expected = host.copy()
        expected[:24 if kind == 'dynamic' else 40] = np.clip(expected[:24 if kind == 'dynamic' else 40], -1, 1)
        np.testing.assert_array_equal(value, expected)
        fused = [slot for source in maps for slot in source.slots if {'max', 'min'} <= {loc.primitive for origin in slot.origins for loc in origin.locations}]
        assert fused, name
        assert all(loc.scope_stack[0] == 'iterations' for slot in fused for origin in slot.origins for loc in origin.locations if loc.primitive in ('max', 'min'))
        case['fused_slots'] = len(fused)
        if kind == 'dynamic':
            for steps in (0, 1, 5):
                actual = np.asarray(runner(x, jax.device_put(np.array([steps], np.int32))))
                expected = host.copy()
                expected[:steps * 8] = np.clip(expected[:steps * 8], -1, 1)
                np.testing.assert_array_equal(actual, expected)
        report['cases'][name] = case
    value, maps, case = check_case('parallel', jax.jit(parallel()), (x[:8],), output)
    np.testing.assert_array_equal(value[0], host[:8] + 0.5)
    np.testing.assert_array_equal(value[1], host[:8] * 1.25)
    by_pc: dict[int, set[str]] = {}
    for slot in maps[0].slots:
        by_pc.setdefault(slot.image_pc, set()).update(loc.primitive for origin in slot.origins for loc in origin.locations)
    assert any({'add', 'mul'} <= primitives for primitives in by_pc.values()), by_pc
    report['cases']['parallel'] = case

    first, second = arithmetic('static', 1), arithmetic('static', 2)
    @jax.jit
    def multiple(value: jax.Array, steps: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return first(value, steps), second(value, steps), first(value + 0.25, steps)
    _, maps, case = check_case('multiple', multiple, (x, count), output)
    assert len(maps[0].functions) >= 2
    report['cases']['multiple'] = case

    jax.set_mesh(mesh)
    lhs = jax.device_put(((np.arange(32 * 128) % 17) - 8).reshape(32, 128).astype(jnp.bfloat16), sharding)
    rhs = jax.device_put(((np.arange(128 * 128) % 13) - 6).reshape(128, 128).astype(jnp.bfloat16), sharding)
    value, maps, case = check_case('fifo_reuse', fifo(sharding), (lhs, rhs), output, warmup=3)
    reference = np.asarray(lhs, dtype=np.float32) @ np.asarray(rhs, dtype=np.float32)
    np.testing.assert_array_equal(value, reference)
    np.testing.assert_array_equal(cold[0, :16], reference[:16])
    case['reused_tile_nonzero'] = int(np.count_nonzero(value[16:]))
    case['reference_reused_tile_nonzero'] = int(np.count_nonzero(reference[16:]))
    case['cold_reused_tile_nonzero'] = int(np.count_nonzero(cold[0, 16:]))
    case['warmup'] = 3
    primitives = {loc.primitive for source in maps for slot in source.slots for origin in slot.origins for loc in origin.locations}
    assert {'get', 'matmul_push_rhs', 'matmul_lhs_fifo', 'matmul_pop_fifo'} <= primitives
    report['cases']['fifo_reuse'] = case

    for transpose in (False, True):
        name, runner = make_dwg_runner(mesh, sharding, transpose)
        _, _, case = check_case(name, runner, (lhs[:16], rhs), output, warmup=3)
        report['cases'][name] = case

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with compiler_source_mapping() as state:
            cached = runner.lower(lhs[:16], rhs).compile()
    assert state.counters()['annotated'] == 0 and caught
    assert executable_source_maps(bytes(cast(LoadedExecutable, cached.runtime_executable()).serialize()))[0].status == 'captured'
    baseline_sources = executable_source_maps((output / 'rolled/baseline.bin').read_bytes())
    assert all(source.status == 'absent' and source.diagnostics for source in baseline_sources)
    try:
        with compiler_source_mapping() as state:
            with compiler_source_mapping():
                raise ValueError('exercise restoration')
    except ValueError:
        pass
    assert state._original() and all(perms == 'r-xp' for _, _, perms in snapshot(state)[1])
    for name in report['cases']:
        directory = output / name
        command = [sys.executable, '-m', 'tpuasm', str(directory / 'executable.bin'), '--input-format', 'executable', '--output-dir', str(directory / 'offline'), '--source-map-json']
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        for listing in directory.glob('*.tpuasm'):
            assert listing.read_bytes() == (directory / 'offline' / listing.name).read_bytes()
    report['offline_new_process'] = True
    report['nested_exception_cache_checks'] = True
    check_offline(output)
    check_lifecycle()
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print('full outputs:', output)

if __name__ == '__main__':
    main()

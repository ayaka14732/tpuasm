"""在标量循环前后插入独立的周期计数器读数，检查数值和计数随循环次数增长；运行方式见 common.py。

原 kernel 用两条加法预留输出寄存器。先将加法改为寄存器自身加零，再在它们之前插入 srdreg.lcclo；计数器值经原来的 store 和 DMA 返回。循环内部另插入 16 个空 bundle，观察循环成本增加。新增指令不需要重新运行 Pallas 编译或调度。
"""
from pathlib import Path
from typing import cast

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxlib.xla_client import LoadedExecutable
import numpy as np

import common
from tpuasm import BundleInsertion, assemble_listing, executable_programs, insert_executable_bundles, format_assembly, load_executable, parse_assembly, replace_executable_programs

@pl.kernel(
    out_type=jax.ShapeDtypeStruct((8,), jnp.uint32),
    mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
    scratch_types=(pltpu.SMEM((8,), jnp.uint32), pltpu.SMEM((8,), jnp.uint32), pltpu.SMEM((1,), jnp.uint32), pltpu.SemaphoreType.DMA),
    name='insert_program',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, out_hbm: Ref, seed: Ref, out: Ref, state: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, seed, sem).wait()
    state[0] = seed[1]
    out[0] = seed[2] + jnp.uint32(12345)
    @pl.loop(0, seed[0].astype(jnp.int32))
    def work(_: jax.Array) -> None:
        state[0] = state[0] * jnp.uint32(1664525) + jnp.uint32(1013904223)
    out[1] = seed[3] + jnp.uint32(23456)
    out[2] = state[0]
    for i in range(3, 8):
        out[i] = jnp.uint32(0xcdefabcd)
    pltpu.async_copy(out, out_hbm, sem).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)

    def inputs(steps: int) -> jax.Array | jax.ShapeDtypeStruct:
        return common.place(np.array([steps, 13, 7, 11, 0, 0, 0, 0], np.uint32), sharding)

    compiled = common.compile(kernel, inputs(4))
    serialized = bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize())
    (record, index, image), = executable_programs(serialized)
    source = format_assembly(image, target=common.TARGET)
    program = parse_assembly(source)
    reads = []
    loop_positions = []
    for pc, bundle in enumerate(program.bundles):
        for instruction in bundle.instructions:
            if instruction.mnemonic == 'smul.u32' and {'1664525', '0x19660d'}.intersection(instruction.operands):
                loop_positions.append(pc)
            if instruction.mnemonic != 'sadd.s32' or not {'12345', '23456'}.intersection(instruction.operands):
                continue
            register = instruction.operands[0]
            old = f'{instruction.slot}: sadd.s32 ' + ', '.join(instruction.operands)
            assert source.count(old) == 1
            source = source.replace(old, f'{instruction.slot}: sadd.s32 {register}, 0, {register}')
            reads.append(BundleInsertion(pc, f'.target {common.TARGET}\n{{ {instruction.slot}: srdreg.lcclo {register} }}'))
    assert len(reads) == 2 and len(loop_positions) == 1
    carrier = replace_executable_programs(serialized, {(record, index): assemble_listing(source)})
    measured = insert_executable_bundles(carrier, {(record, index): reads})
    loop_pc = loop_positions[0]
    delay = program.hardware.branch_delay_bundles
    assert delay is not None
    # v6e schedules the multiply in the loop branch's delay window. Keep that window intact.
    for pc, bundle in enumerate(program.bundles):
        if pc < loop_pc <= pc + delay and any(i.mnemonic.startswith(('sbr.', 'scall.')) for i in bundle.instructions):
            loop_pc = pc
    padding = BundleInsertion(loop_pc, f'.target {common.TARGET}\n.empty 16')
    patched = insert_executable_bundles(carrier, {(record, index): [*reads, padding]})
    # 删除是插入的逆操作：在 patched 中把那 16 个空 bundle 删掉，程序映像应与只插入读数的 measured 逐字节相同。
    shifted = loop_pc + sum(read.image_pc <= loop_pc for read in reads)
    restored = insert_executable_bundles(patched, {(record, index): [BundleInsertion(shifted, f'.target {common.TARGET}\n', delete=16)]})
    assert [image for _, _, image in executable_programs(restored)] == [image for _, _, image in executable_programs(measured)]

    def check() -> None:
        functions = [load_executable(raw, compiled) for raw in (measured, patched)]
        deltas: list[list[int]] = [[], []]
        steps = (0, 1, 4, 31)
        for n in steps:
            x = inputs(n)
            value = 13
            for _ in range(n):
                value = (value * 1664525 + 1013904223) & 0xffffffff
            expected = np.array([12352, 23467, value] + [0xcdefabcd] * 5, np.uint32)
            np.testing.assert_array_equal(np.asarray(compiled(x)), expected)
            for i, function in enumerate(functions):
                output = np.asarray(function(x))
                np.testing.assert_array_equal(output[2:], expected[2:])
                deltas[i].append((int(output[1]) - int(output[0])) & 0xffffffff)
            np.testing.assert_array_equal(np.asarray(compiled(x)), expected)
        print('循环次数：', steps, '独立读数：', deltas[0], '再插入 16 个空 bundle：', deltas[1], flush=True)
        for values in deltas:
            slope = values[1] - values[0]
            assert values[0] > 0 and slope > 0
            assert values == [values[0] + n * slope for n in steps]
        assert deltas[1][0] == deltas[0][0] and deltas[1][1] > deltas[0][1]
        for raw, function in zip((measured, patched), functions):
            reserialized = bytes(cast(LoadedExecutable, function.runtime_executable()).serialize())
            assert executable_programs(raw) == executable_programs(reserialized)

    common.finish(patched, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

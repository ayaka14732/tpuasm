"""四块 f32[8,128] 的串行与输入、输出双缓冲对照；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
from pathlib import Path

import jax
from jax import Ref
from jax._src.pallas.mosaic.primitives import AsyncCopyDescriptor
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

import common

tc = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

@pl.kernel(
    out_type=jax.ShapeDtypeStruct((32, 128), jnp.float32),
    mesh=tc,
    scratch_types=(pltpu.VMEM((8, 128), jnp.float32), pltpu.SemaphoreType.DMA),
    name='serial',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def serial(x_hbm: Ref, out_hbm: Ref, tile: Ref, sem: Ref) -> None:
    @pl.loop(0, 4)
    def body(index: jax.Array) -> None:
        rows = pl.ds(index * 8, 8)
        pltpu.async_copy(x_hbm.at[rows, :], tile, sem).wait()
        tile[...] = tile[...] * 2.0 + 1.0
        pltpu.async_copy(tile, out_hbm.at[rows, :], sem).wait()

@pl.kernel(
    out_type=jax.ShapeDtypeStruct((32, 128), jnp.float32),
    mesh=tc,
    scratch_types=(
        pltpu.VMEM((2, 8, 128), jnp.float32),
        pltpu.VMEM((2, 8, 128), jnp.float32),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    ),
    name='double_buffer',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def pipelined(x_hbm: Ref, out_hbm: Ref, x_vmem: Ref, out_vmem: Ref, load_sem: Ref, store_sem: Ref) -> None:
    def load(index: int | jax.Array, buffer: int | jax.Array) -> AsyncCopyDescriptor:
        return pltpu.make_async_copy(x_hbm.at[pl.ds(index * 8, 8), :], x_vmem.at[buffer], load_sem.at[buffer])

    def store(index: int | jax.Array, buffer: int | jax.Array) -> AsyncCopyDescriptor:
        return pltpu.make_async_copy(out_vmem.at[buffer], out_hbm.at[pl.ds(index * 8, 8), :], store_sem.at[buffer])

    # 先预取第 0 块；稳态计算当前块时，另一 buffer 接收下一块。
    load(0, 0).start()

    @pl.loop(0, 4)
    def body(index: jax.Array) -> None:
        buffer = index % 2
        load(index, buffer).wait()

        @pl.when(index < 3)
        def prefetch() -> None:
            load(index + 1, 1 - buffer).start()

        @pl.when(index >= 2)
        def release_output() -> None:
            store(index - 2, buffer).wait()

        out_vmem[buffer] = x_vmem[buffer] * 2.0 + 1.0
        store(index, buffer).start()

    # 最后两笔写回完成后，kernel 才能退出。
    store(2, 0).wait()
    store(3, 1).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = np.arange(32 * 128, dtype=np.float32).reshape(32, 128)
    x = common.place(host, sharding)
    for name, kernel in (('serial', serial), ('pipelined', pipelined)):
        compiled = common.compile(kernel, x)

        def check() -> None:
            np.testing.assert_array_equal(np.asarray(compiled(x)), host * 2.0 + 1.0)

        common.finish(compiled, f'{Path(__file__).stem}_{name}', check)

if __name__ == '__main__':
    main()

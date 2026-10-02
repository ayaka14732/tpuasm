"""SparseCore 向量子核上 i32[8,8] 逐元素加一；检查数值并导出 TEC 机器清单，运行方式见 common.py。"""
from pathlib import Path

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np

import common

def body(x_hbm: Ref, out_hbm: Ref, tile: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, tile, sem).wait()
    tile[...] = tile[...] + 1
    pltpu.async_copy(tile, out_hbm, sem).wait()

def kernel(x: jax.Array) -> jax.Array:
    # The SparseCore mesh reads the TPU generation from the abstract mesh that common.compile sets.
    sc = plsc.VectorSubcoreMesh(core_axis_name='core', subcore_axis_name='tile', num_cores=1, num_subcores=1)
    return pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((8, 8), jnp.int32),
        mesh=sc,
        scratch_types=(pltpu.VMEM((8, 8), jnp.int32), pltpu.SemaphoreType.DMA),
        name='sc_add_one',
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
    )(x)

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = np.arange(64, dtype=np.int32).reshape(8, 8) - 32
    x = common.place(host, sharding)
    compiled = common.compile(kernel, x)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x)), host + 1)

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

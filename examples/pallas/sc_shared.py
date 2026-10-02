"""SparseCore 的 16 个向量子核各算一行 2 * x + 子核号，经 VMEM_SHARED（SPMEM）汇总，屏障后由子核 0 写回 i32[16,128]；检查数值并导出 TEC 机器清单，运行方式见 common.py。"""
from pathlib import Path

import jax
from jax import Ref, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np

import common

def body(x_hbm: Ref, out_hbm: Ref, row: Ref, shared: Ref) -> None:
    tile = lax.axis_index('tile')
    pltpu.sync_copy(x_hbm.at[tile], row)
    row[...] = row[...] * 2 + tile
    pltpu.sync_copy(row, shared.at[tile])
    plsc.subcore_barrier()

    @pl.when(tile == 0)
    def write() -> None:
        pltpu.sync_copy(shared, out_hbm)

def kernel(x: jax.Array) -> jax.Array:
    # The SparseCore mesh reads the TPU generation from the abstract mesh that common.compile sets.
    sc = plsc.VectorSubcoreMesh(core_axis_name='core', subcore_axis_name='tile', num_cores=1, num_subcores=16)
    return pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((16, 128), jnp.int32),
        mesh=sc,
        scratch_types=(pltpu.VMEM((128,), jnp.int32), pltpu.VMEM_SHARED((16, 128), jnp.int32)),
        name='sc_shared',
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
    )(x)

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = np.arange(16 * 128, dtype=np.int32).reshape(16, 128) - 1000
    x = common.place(host, sharding)
    compiled = common.compile(kernel, x)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x)), host * 2 + np.arange(16, dtype=np.int32)[:, None])

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

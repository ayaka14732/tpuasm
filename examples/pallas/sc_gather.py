"""SparseCore 向量子核按下标从 HBM 收集 f32[64,128] 的 8 行（indirect stream）；检查数值并导出 TEC 机器清单，运行方式见 common.py。"""
from pathlib import Path

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np

import common

def body(x_hbm: Ref, indices_hbm: Ref, out_hbm: Ref, indices: Ref, rows: Ref) -> None:
    pltpu.sync_copy(indices_hbm, indices)
    pltpu.sync_copy(x_hbm.at[indices], rows)
    pltpu.sync_copy(rows, out_hbm)

def kernel(x: jax.Array, indices: jax.Array) -> jax.Array:
    # The SparseCore mesh reads the TPU generation from the abstract mesh that common.compile sets.
    sc = plsc.VectorSubcoreMesh(core_axis_name='core', subcore_axis_name='tile', num_cores=1, num_subcores=1)
    return pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        mesh=sc,
        scratch_types=(pltpu.VMEM((8,), jnp.int32), pltpu.VMEM((8, 128), jnp.float32)),
        name='sc_gather',
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
    )(x, indices)

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = np.arange(64 * 128, dtype=np.float32).reshape(64, 128) / 4
    host_indices = np.array([5, 63, 0, 17, 42, 8, 31, 2], dtype=np.int32)
    x = common.place(host, sharding)
    indices = common.place(host_indices, sharding)
    compiled = common.compile(kernel, x, indices)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x, indices)), host[host_indices])

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

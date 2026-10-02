"""SparseCore 向量子核从 i32[16,512] 取第 128..255 列（strided stream），用不展开的 pl.loop 逐行累加成 i32[128]；检查数值并导出 TEC 机器清单，运行方式见 common.py。"""
from pathlib import Path

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np

import common

def body(x_hbm: Ref, out_hbm: Ref, columns: Ref, total: Ref) -> None:
    pltpu.sync_copy(x_hbm.at[:, pl.ds(128, 128)], columns)
    total[...] = jnp.zeros_like(total)

    @pl.loop(0, 16, unroll=False)
    def row(i: jax.Array) -> None:
        total[...] += columns[i]

    pltpu.sync_copy(total, out_hbm)

def kernel(x: jax.Array) -> jax.Array:
    # The SparseCore mesh reads the TPU generation from the abstract mesh that common.compile sets.
    sc = plsc.VectorSubcoreMesh(core_axis_name='core', subcore_axis_name='tile', num_cores=1, num_subcores=1)
    return pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((128,), jnp.int32),
        mesh=sc,
        scratch_types=(pltpu.VMEM((16, 128), jnp.int32), pltpu.VMEM((128,), jnp.int32)),
        name='sc_strided',
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
    )(x)

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = (np.arange(16 * 512, dtype=np.int32).reshape(16, 512) * 7) % 1000 - 500
    x = common.place(host, sharding)
    compiled = common.compile(kernel, x)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x)), host[:, 128:256].sum(axis=0))

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

"""f32[8,256] Top-8，检查跨 tile 索引与同值候选；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
from pathlib import Path

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

import common

tc = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

@pl.kernel(
    out_type=(jax.ShapeDtypeStruct((8, 8), jnp.float32), jax.ShapeDtypeStruct((8, 8), jnp.int32)),
    mesh=tc,
    scratch_types=(pltpu.VMEM((8, 256), jnp.float32), pltpu.VMEM((8, 8), jnp.float32), pltpu.VMEM((8, 8), jnp.int32), pltpu.SemaphoreType.DMA),
    name='top_k',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, values_hbm: Ref, indices_hbm: Ref, x_vmem: Ref, values_vmem: Ref, indices_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    values, indices = jax.lax.top_k(x_vmem[...], 8, is_stable=False)
    values_vmem[...] = values
    indices_vmem[...] = indices
    pltpu.async_copy(values_vmem, values_hbm, sem).wait()
    pltpu.async_copy(indices_vmem, indices_hbm, sem).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    rng = np.random.default_rng(12)
    host = np.stack([rng.permutation(256) for _ in range(8)]).astype(np.float32)
    tied = rng.integers(-4, 5, size=(8, 256)).astype(np.float32)
    x = common.place(host, sharding)
    compiled = common.compile(kernel, x)

    def check() -> None:
        values, indices = jax.device_get(compiled(x))
        expected_indices = np.argsort(-host, axis=1)[:, :8]
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(values, np.take_along_axis(host, expected_indices, axis=1))
        # 同值时只要求合法的 Top-8，不要求稳定索引顺序。
        values, indices = jax.device_get(compiled(common.place(tied, sharding)))
        np.testing.assert_array_equal(values, np.sort(tied, axis=1)[:, -8:][:, ::-1])
        np.testing.assert_array_equal(values, np.take_along_axis(tied, indices, axis=1))
        assert np.all(np.diff(np.sort(indices, axis=1), axis=1) > 0)

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

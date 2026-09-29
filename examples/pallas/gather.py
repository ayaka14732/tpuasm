"""f32[8,128] 动态 lane gather，包含负索引；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
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
    out_type=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    mesh=tc,
    scratch_types=(pltpu.VMEM((8, 128), jnp.float32), pltpu.VMEM((8, 128), jnp.int32), pltpu.VMEM((8, 128), jnp.float32), pltpu.SemaphoreType.DMA),
    name='gather',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, indices_hbm: Ref, out_hbm: Ref, x_vmem: Ref, indices_vmem: Ref, out_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    pltpu.async_copy(indices_hbm, indices_vmem, sem).wait()
    out_vmem[...] = jnp.take_along_axis(x_vmem[...], indices_vmem[...], axis=1)
    pltpu.async_copy(out_vmem, out_hbm, sem).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = np.arange(8 * 128, dtype=np.float32).reshape(8, 128)
    host_indices = (np.arange(8 * 128, dtype=np.int32).reshape(8, 128) * 17 + 3) % 128
    host_indices[:, ::2] -= 128
    x = common.place(host, sharding)
    indices = common.place(host_indices, sharding)
    compiled = common.compile(kernel, x, indices)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x, indices)), np.take_along_axis(host, host_indices, axis=1))

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

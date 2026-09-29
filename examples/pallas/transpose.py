"""两个独立 i32[128,128] 转置 task；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
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
    out_type=jax.ShapeDtypeStruct((2, 128, 128), jnp.int32),
    mesh=tc,
    scratch_types=(pltpu.VMEM((2, 128, 128), jnp.int32), pltpu.VMEM((2, 128, 128), jnp.int32), pltpu.SemaphoreType.DMA),
    name='transpose',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, out_hbm: Ref, x_vmem: Ref, out_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    out_vmem[...] = jnp.transpose(x_vmem[...], (0, 2, 1))
    pltpu.async_copy(out_vmem, out_hbm, sem).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = np.arange(2 * 128 * 128, dtype=np.int32).reshape(2, 128, 128)
    x = common.place(host, sharding)
    compiled = common.compile(kernel, x)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x)), host.transpose(0, 2, 1))

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

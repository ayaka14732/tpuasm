"""两个 f32[8,129] 沿 lane 拼接，观察重排与 masked store；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
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
    out_type=jax.ShapeDtypeStruct((8, 258), jnp.float32),
    mesh=tc,
    scratch_types=(pltpu.VMEM((8, 129), jnp.float32), pltpu.VMEM((8, 129), jnp.float32), pltpu.VMEM((8, 258), jnp.float32), pltpu.SemaphoreType.DMA),
    name='concatenate',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, y_hbm: Ref, out_hbm: Ref, x_vmem: Ref, y_vmem: Ref, out_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    pltpu.async_copy(y_hbm, y_vmem, sem).wait()
    out_vmem[...] = jnp.concatenate((x_vmem[...], y_vmem[...]), axis=1)
    pltpu.async_copy(out_vmem, out_hbm, sem).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host_x = np.arange(8 * 129, dtype=np.float32).reshape(8, 129)
    host_y = -host_x - 1.0
    x = common.place(host_x, sharding)
    y = common.place(host_y, sharding)
    compiled = common.compile(kernel, x, y)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x, y)), np.concatenate((host_x, host_y), axis=1))

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

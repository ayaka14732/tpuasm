"""两个 f32[9,128] 直接 DMA 到输出 TC VMEM 的相邻行窗口；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
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
    out_type=jax.ShapeDtypeStruct((18, 128), jnp.float32),
    mesh=tc,
    scratch_types=(pltpu.VMEM((18, 128), jnp.float32), pltpu.SemaphoreType.DMA),
    name='concatenate_dma',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, y_hbm: Ref, out_hbm: Ref, out_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, out_vmem.at[pl.ds(0, 9), :], sem).wait()
    pltpu.async_copy(y_hbm, out_vmem.at[pl.ds(9, 9), :], sem).wait()
    pltpu.async_copy(out_vmem, out_hbm, sem).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host_x = np.arange(9 * 128, dtype=np.float32).reshape(9, 128)
    host_y = -host_x - 1.0
    x = common.place(host_x, sharding)
    y = common.place(host_y, sharding)
    compiled = common.compile(kernel, x, y)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x, y)), np.concatenate((host_x, host_y), axis=0))

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

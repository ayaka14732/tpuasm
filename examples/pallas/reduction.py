"""f32[8,128] 的三层 sublane tree 与跨 lane max/sum；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
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
    out_type=(jax.ShapeDtypeStruct((8, 128), jnp.float32), jax.ShapeDtypeStruct((8, 128), jnp.float32)),
    mesh=tc,
    scratch_types=(pltpu.VMEM((8, 128), jnp.float32), pltpu.VMEM((8, 128), jnp.float32), pltpu.VMEM((8, 128), jnp.float32), pltpu.SemaphoreType.DMA),
    name='reduction',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, max_hbm: Ref, sum_hbm: Ref, x_vmem: Ref, max_vmem: Ref, sum_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    maximum = x_vmem[...]
    maximum = jnp.maximum(maximum, pltpu.roll(maximum, shift=7, axis=0))
    maximum = jnp.maximum(maximum, pltpu.roll(maximum, shift=6, axis=0))
    maximum = jnp.maximum(maximum, pltpu.roll(maximum, shift=4, axis=0))
    max_vmem[...] = jnp.broadcast_to(jnp.max(maximum, axis=1, keepdims=True), (8, 128))
    total = x_vmem[...]
    total += pltpu.roll(total, shift=7, axis=0)
    total += pltpu.roll(total, shift=6, axis=0)
    total += pltpu.roll(total, shift=4, axis=0)
    sum_vmem[...] = jnp.broadcast_to(jnp.sum(total, axis=1, keepdims=True), (8, 128))
    pltpu.async_copy(max_vmem, max_hbm, sem).wait()
    pltpu.async_copy(sum_vmem, sum_hbm, sem).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = ((np.arange(8 * 128, dtype=np.int32) % 17) - 8).reshape(8, 128).astype(np.float32)
    x = common.place(host, sharding)
    compiled = common.compile(kernel, x)

    def check() -> None:
        maximum, total = jax.device_get(compiled(x))
        np.testing.assert_array_equal(maximum, np.full((8, 128), host.max(), dtype=np.float32))
        np.testing.assert_array_equal(total, np.full((8, 128), host.sum(), dtype=np.float32))

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

"""f32[8,128] 稳定 Softmax，观察指数、倒数与跨 lane 归约；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
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
    scratch_types=(pltpu.VMEM((8, 128), jnp.float32), pltpu.SemaphoreType.DMA),
    name='softmax',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, out_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    value = x_vmem[...]
    exp = jnp.exp(value - jnp.max(value, axis=1, keepdims=True))
    x_vmem[...] = exp / jnp.sum(exp, axis=1, keepdims=True)
    pltpu.async_copy(x_vmem, out_hbm, sem).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = (((np.arange(8 * 128, dtype=np.int32) * 17) % 101) - 50).reshape(8, 128).astype(np.float32) / 16
    x = common.place(host, sharding)
    compiled = common.compile(kernel, x)

    def check() -> None:
        exp = np.exp(host - host.max(axis=1, keepdims=True))
        np.testing.assert_allclose(np.asarray(compiled(x)), exp / exp.sum(axis=1, keepdims=True), rtol=2e-6, atol=2e-6)

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

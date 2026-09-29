"""f32[8,128] RMSNorm，含 (1 + weight) 广播与平方根倒数；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
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
    scratch_types=(pltpu.VMEM((8, 128), jnp.float32), pltpu.VMEM((1, 128), jnp.float32), pltpu.SemaphoreType.DMA),
    name='rms_norm',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, weight_hbm: Ref, out_hbm: Ref, x_vmem: Ref, weight_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    pltpu.async_copy(weight_hbm, weight_vmem, sem).wait()
    value = x_vmem[...]
    mean_square = jnp.sum(value * value, axis=1, keepdims=True) * (1 / 128)
    x_vmem[...] = value * jax.lax.rsqrt(mean_square + 1e-6) * (1.0 + weight_vmem[...])
    pltpu.async_copy(x_vmem, out_hbm, sem).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = (((np.arange(8 * 128, dtype=np.int32) * 17) % 101) - 50).reshape(8, 128).astype(np.float32) / 16
    host_weight = (((np.arange(128, dtype=np.int32) * 7) % 31) - 15).reshape(1, 128).astype(np.float32) / 256
    x = common.place(host, sharding)
    weight = common.place(host_weight, sharding)
    compiled = common.compile(kernel, x, weight)

    def check() -> None:
        expected = host / np.sqrt(np.mean(host * host, axis=1, keepdims=True) + 1e-6) * (1.0 + host_weight)
        np.testing.assert_allclose(np.asarray(compiled(x, weight)), expected, rtol=2e-6, atol=2e-6)

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

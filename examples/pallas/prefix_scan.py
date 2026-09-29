"""bf16[16,128] 七轮 Hillis–Steele scan，f32 累加后舍入；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
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
    out_type=jax.ShapeDtypeStruct((16, 128), jnp.bfloat16),
    mesh=tc,
    scratch_types=(pltpu.VMEM((16, 128), jnp.bfloat16), pltpu.SemaphoreType.DMA),
    name='prefix_scan',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, out_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    lanes = jnp.arange(128, dtype=jnp.int32)[None, :]

    @pl.loop(0, 7, init_carry=x_vmem[...].astype(jnp.float32), unroll=True)
    def scan(level: jax.Array, prefix: jax.Array) -> jax.Array:
        distance = 1 << level
        left = pltpu.roll(prefix, shift=distance, axis=1)
        return jnp.where(lanes >= distance, prefix + left, prefix)

    x_vmem[...] = scan.astype(jnp.bfloat16)
    pltpu.async_copy(x_vmem, out_hbm, sem).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = ((np.arange(16 * 128, dtype=np.int32) % 17) - 8).reshape(16, 128).astype(jnp.bfloat16)
    x = common.place(host, sharding)
    compiled = common.compile(kernel, x)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.cumsum(host.astype(np.float32), axis=1).astype(jnp.bfloat16))

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

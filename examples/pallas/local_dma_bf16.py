"""bf16[64,128] 的 DMA、unpack、乘二与 pack；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
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
    out_type=jax.ShapeDtypeStruct((64, 128), jnp.bfloat16),
    mesh=tc,
    scratch_types=(pltpu.VMEM((64, 128), jnp.bfloat16), pltpu.SemaphoreType.DMA),
    name='local_dma_bf16',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, out_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    x_vmem[...] = x_vmem[...] * 2.0
    pltpu.async_copy(x_vmem, out_hbm, sem).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = (np.arange(64 * 128, dtype=np.float32) % 127).reshape(64, 128).astype(jnp.bfloat16)
    x = common.place(host, sharding)
    compiled = common.compile(kernel, x)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x), dtype=np.float32), host.astype(np.float32) * 2.0)

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

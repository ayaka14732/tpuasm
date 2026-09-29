"""bf16[128,128] 矩阵乘法，f32 输出；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
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
    out_type=jax.ShapeDtypeStruct((128, 128), jnp.float32),
    mesh=tc,
    scratch_types=(pltpu.VMEM((128, 128), jnp.bfloat16), pltpu.VMEM((128, 128), jnp.bfloat16), pltpu.VMEM((128, 128), jnp.float32), pltpu.SemaphoreType.DMA),
    name='matmul',
    compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(lhs_hbm: Ref, rhs_hbm: Ref, out_hbm: Ref, lhs_vmem: Ref, rhs_vmem: Ref, out_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(lhs_hbm, lhs_vmem, sem).wait()
    pltpu.async_copy(rhs_hbm, rhs_vmem, sem).wait()
    out_vmem[...] = jnp.dot(lhs_vmem[...], rhs_vmem[...], preferred_element_type=jnp.float32)
    pltpu.async_copy(out_vmem, out_hbm, sem).wait()

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host_lhs = ((np.arange(128 * 128, dtype=np.int32) % 17) - 8).reshape(128, 128).astype(jnp.bfloat16)
    host_rhs = ((np.arange(128 * 128, dtype=np.int32) % 19) - 9).reshape(128, 128).astype(jnp.bfloat16)
    lhs = common.place(host_lhs, sharding)
    rhs = common.place(host_rhs, sharding)
    compiled = common.compile(kernel, lhs, rhs)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(lhs, rhs)), host_lhs.astype(np.float32) @ host_rhs.astype(np.float32))

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

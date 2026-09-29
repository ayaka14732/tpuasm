"""一个 Megacore 的两个 TensorCore 交换 f32[8,128]，只适用于 TPU v4；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
from pathlib import Path

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

import common

tc = pltpu.TensorCoreMesh(axis_name='tc', num_cores=2)

@pl.kernel(
    out_type=jax.ShapeDtypeStruct((2, 8, 128), jnp.float32),
    mesh=tc,
    name='remote_dma',
    compiler_params=pltpu.CompilerParams(collective_id=1, disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, out_hbm: Ref) -> None:
    core = jax.lax.axis_index('tc')
    destination = {'tc': 1 - core}
    send = jax.empty_ref(jax.ShapeDtypeStruct((8, 128), jnp.float32), memory_space=pltpu.VMEM @ tc)
    recv = jax.empty_ref(jax.ShapeDtypeStruct((8, 128), jnp.float32), memory_space=pltpu.VMEM @ tc)
    load_sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
    send_sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
    recv_sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
    store_sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
    pltpu.async_copy(x_hbm.at[core], send, load_sem).wait()

    # 所有接收端及其 semaphore 就绪后，才允许写入远端。
    ready = pltpu.get_barrier_semaphore()

    @pl.loop(0, 2, unroll=True)
    def signal_ready(rank: jax.Array) -> None:
        pl.semaphore_signal(ready, 1, device_id={'tc': rank})

    pl.semaphore_wait(ready, 2)

    transfer = pltpu.make_async_remote_copy(send, recv, send_sem, recv_sem, device_id=destination, device_id_type=pl.DeviceIdType.MESH)
    transfer.start()
    transfer.wait_send()
    transfer.wait_recv()
    pltpu.async_copy(recv, out_hbm.at[core], store_sem).wait()

def main() -> None:
    if common.CORES == 1:
        print(f'跳过：{common.KIND} 的一个 device 只有一个 TensorCore')
        return
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = np.arange(2 * 8 * 128, dtype=np.float32).reshape(2, 8, 128)
    x = common.place(host, sharding)
    compiled = common.compile(kernel, x)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x)), host[::-1])

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

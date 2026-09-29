"""四颗芯片的三步 ring all-reduce，转发原始 partial；检查数值并导出带源码注释的机器清单，运行方式见 common.py。"""
from pathlib import Path

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
import numpy as np

import common

tc = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

@pl.kernel(
    out_type=jax.ShapeDtypeStruct((1, 8, 128), jnp.float32),
    mesh=tc,
    name='all_reduce',
    compiler_params=pltpu.CompilerParams(collective_id=1, disable_bounds_checks=True, disable_semaphore_checks=True),
)
def kernel(x_hbm: Ref, out_hbm: Ref) -> None:
    device = jax.lax.axis_index('device')
    destination = {'device': (device + 1) % 4, 'tc': 0}
    send = jax.empty_ref(jax.ShapeDtypeStruct((8, 128), jnp.float32), memory_space=pltpu.VMEM @ tc)
    recv = jax.empty_ref(jax.ShapeDtypeStruct((8, 128), jnp.float32), memory_space=pltpu.VMEM @ tc)
    load_sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
    send_sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
    recv_sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
    store_sem = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.DMA.dtype), memory_space=pltpu.SEMAPHORE)
    acc = jax.empty_ref(jax.ShapeDtypeStruct((8, 128), jnp.float32), memory_space=pltpu.VMEM @ tc)
    capacity = jax.empty_ref(jax.ShapeDtypeStruct((), pltpu.SemaphoreType.REGULAR.dtype), memory_space=pltpu.SEMAPHORE)
    pltpu.async_copy(x_hbm.at[0], send, load_sem).wait()
    acc[...] = send[...]

    # 所有接收端及其 semaphore 就绪后，才允许写入远端。
    ready = pltpu.get_barrier_semaphore()

    @pl.loop(0, 4, unroll=True)
    def signal_ready(rank: jax.Array) -> None:
        pl.semaphore_signal(ready, 1, device_id={'device': rank, 'tc': 0})

    pl.semaphore_wait(ready, 4)

    @pl.loop(0, 3)
    def step(index: jax.Array) -> None:
        # 下游释放接收 buffer 后，本轮才可覆盖它。
        pl.semaphore_signal(capacity, 1, device_id={'device': (device + 3) % 4, 'tc': 0})
        pl.semaphore_wait(capacity, 1)
        transfer = pltpu.make_async_remote_copy(send, recv, send_sem, recv_sem, device_id=destination, device_id_type=pl.DeviceIdType.MESH)
        transfer.start()
        transfer.wait_send()
        transfer.wait_recv()
        acc[...] += recv[...]
        send[...] = recv[...]

    pltpu.async_copy(acc, out_hbm.at[0], store_sem).wait()

def main() -> None:
    mesh = jax.make_mesh((4,), ('device',), devices=common.devices(4))
    sharding = jax.NamedSharding(mesh, P('device', None, None))

    @jax.shard_map(
        mesh=mesh,
        in_specs=P('device', None, None),
        out_specs=P('device', None, None),
        check_vma=False,
    )
    def run(x: jax.Array) -> jax.Array:
        return kernel(x)

    host = np.arange(4 * 8 * 128, dtype=np.float32).reshape(4, 8, 128)
    x = common.place(host, sharding)
    compiled = common.compile(run, x, mesh=mesh)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.broadcast_to(host.sum(axis=0), host.shape))

    common.finish(compiled, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

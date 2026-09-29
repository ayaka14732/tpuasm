"""示例共用的目标选择、设备选择、编译、数值检查与清单导出。

环境变量 ``TPUASM_EXAMPLES_TARGET`` 选择目标（``tpu-v4-tc`` 或 ``tpu-v6e-tc``），清单写入 ``examples/pallas/<目标>/``，注释中的源码路径改为相对仓库根目录。默认在本机该目标的 TPU 上编译、运行并检查数值。设置 ``TPUASM_EXAMPLES_AOT=1`` 时改为按参考拓扑离线编译：不需要 TPU，也不运行 kernel，只导出清单，并在输出中注明未检查数值。

参考拓扑是编写清单所用的设备：v4 为四颗 Megacore 芯片（2x2x1），v6e 为 ``v6e:2x2``。按名称创建的 ``v4:2x2x1`` 拓扑把每个 TensorCore 当作一个 device，编译出的程序与 Megacore 不同，所以 v4 使用 ``v4_2x2x1_megacore.topology``：它是在 TPU v4 上只让本 host 四颗芯片可见时，``topologies.get_attached_topology().serialize()`` 的结果。
"""
from __future__ import annotations

from collections.abc import Callable
import functools
import os
from pathlib import Path
from typing import Any

import jax
from jax.experimental import topologies
from jax.stages import Compiled
import numpy as np

from tpuasm import compiler_source_mapping, dump_compiled, dump_executable

EXAMPLES = Path(__file__).resolve().parent
ROOT = EXAMPLES.parents[1]
TARGET = os.environ['TPUASM_EXAMPLES_TARGET']
AOT = os.environ.get('TPUASM_EXAMPLES_AOT') == '1'
KIND = {'tpu-v4-tc': 'TPU v4', 'tpu-v6e-tc': 'TPU v6 lite'}[TARGET]
# 每个 JAX device 的 TensorCore 数：TPU v4 的 Megacore device 包含一颗芯片的两个 TensorCore。
CORES = 2 if TARGET == 'tpu-v4-tc' else 1
OPTIONS = {'xla_msa_enable': 'false', 'xla_tpu_vmem_scavenging_mode': 'NONE'}
if CORES == 2:
    # Megacore 两个 TensorCore 之间的远程 DMA 需要此选项；它不改变单核示例的机器码。
    OPTIONS['xla_mosaic_unsafe_allow_multicore_remote_dma'] = 'true'

@functools.cache
def _reference_devices() -> list[Any]:
    if TARGET == 'tpu-v6e-tc':
        return list(topologies.get_topology_desc(platform='tpu', topology_name='v6e:2x2').devices)
    # 反序列化需要已初始化的 TPU PJRT 插件，按名称创建任一拓扑即可完成初始化。
    topologies.get_topology_desc(platform='tpu', topology_name='v4:2x2x1')
    return list(topologies.TopologyDescription.deserialize((EXAMPLES / 'v4_2x2x1_megacore.topology').read_bytes()).devices)

def devices(count: int = 1) -> list[Any]:
    """返回 count 个目标 device；离线编译时返回参考拓扑中的 compile-only device。"""
    if AOT:
        return _reference_devices()[:count]
    found = jax.local_devices()[:count]
    assert len(found) == count and all(device.device_kind == KIND and device.num_cores == CORES for device in found), f'需要 {count} 个本机 {KIND} device；没有时设置 TPUASM_EXAMPLES_AOT=1 离线编译'
    return found

def place(host: np.ndarray, sharding: jax.sharding.Sharding) -> Any:
    """离线编译时只给出形状、类型与分片，不传输数据。"""
    if AOT:
        return jax.ShapeDtypeStruct(host.shape, host.dtype, sharding=sharding)
    return jax.device_put(host, sharding)

def compile(function: Callable[..., Any], *arguments: Any, mesh: jax.sharding.Mesh | None = None) -> Compiled:
    """在来源映射上下文中编译；mesh 是 shard_map 使用的 mesh，单设备示例省略。"""
    jax.config.update('jax_enable_compilation_cache', False)
    if mesh is None:
        mesh = jax.sharding.Mesh(np.array(devices()), ('device',))
    # 离线编译时 Pallas lowering 从 abstract mesh 读取 TPU 代际。设备上也设置，因为它影响 JAX jit 追踪缓存是否命中：不设置时，kernel 中重复调用的 jnp.maximum 等 jit 函数复用第一次追踪的 jaxpr，来源注释会带上第一次调用的行号。
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh), compiler_source_mapping():
        return jax.jit(function, compiler_options=OPTIONS).lower(*arguments).compile()

def finish(compiled: Compiled | bytes, name: str, check: Callable[[], None]) -> None:
    """在设备上检查数值后导出清单 ``<目标>/<name>.tpuasm``；离线编译时跳过检查。compiled 也可以是 serialized executable。"""
    if AOT:
        state = '离线编译，未检查数值'
    else:
        check()
        state = '数值检查通过'
    output = EXAMPLES / TARGET.replace('-', '_') / f'{name}.tpuasm'
    path, = dump_executable(compiled, output.parent) if isinstance(compiled, bytes) else dump_compiled(compiled, output.parent)
    text = path.read_text(encoding='utf-8')
    path.unlink()
    output.write_text(text.replace(f'{ROOT}/', ''), encoding='utf-8')
    print(f'{state}；产物：', output.relative_to(ROOT))

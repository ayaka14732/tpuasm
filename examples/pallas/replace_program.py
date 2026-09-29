"""把 clamp 示例的 vclamps 边界从 1.0 改为 0.5，回灌并执行；检查数值并导出修改后的机器清单，运行方式见 common.py。

在设备上先执行原程序，再在同一进程中执行修改后的程序：runtime 按程序身份复用已装载的程序，若替换后的 executable 沿用原身份，第二次执行仍会得到原结果。离线编译时只替换程序映像并导出清单。
"""
from pathlib import Path
from typing import cast

import jax
from jaxlib.xla_client import LoadedExecutable
import numpy as np

import clamp
import common
from tpuasm import assemble_listing, executable_programs, format_assembly, load_executable, replace_executable_programs

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = ((np.arange(8 * 128, dtype=np.int32) % 65) - 32).reshape(8, 128).astype(np.float32) / 8
    x = common.place(host, sharding)
    compiled = common.compile(clamp.kernel, x)
    serialized = bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize())
    (record, index, image), = executable_programs(serialized)
    source = format_assembly(image, target=common.TARGET)
    original, edited = 'vclamps.8x128.f32 v1, v0, 1.0', 'vclamps.8x128.f32 v1, v0, 0.5'
    assert source.count(original) == 1
    patched = replace_executable_programs(serialized, {(record, index): assemble_listing(source.replace(original, edited))})

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.clip(host, -1.0, 1.0))
        np.testing.assert_array_equal(np.asarray(load_executable(patched, compiled)(x)), np.clip(host, -0.5, 0.5))
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.clip(host, -1.0, 1.0))

    common.finish(patched, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

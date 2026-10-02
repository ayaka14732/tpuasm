"""把 sc_add_one 的加数从 1 改为 2，写回 TEC 程序并执行；检查数值并导出修改后的 TEC 机器清单，运行方式见 common.py。

在设备上先执行原程序，再在同一进程中执行修改后的程序，最后再执行一次原程序；写回时更新程序身份，否则 runtime 会复用已装载的原程序。离线编译时只替换程序映像并导出清单。
"""
from pathlib import Path
from typing import cast

import jax
from jaxlib.xla_client import LoadedExecutable
import numpy as np

import common
import sc_add_one
from tpuasm import assemble_listing, executable_programs, format_assembly, load_executable, replace_executable_programs

def main() -> None:
    device, = common.devices()
    sharding = jax.sharding.SingleDeviceSharding(device)
    host = np.arange(64, dtype=np.int32).reshape(8, 8) - 32
    x = common.place(host, sharding)
    compiled = common.compile(sc_add_one.kernel, x)
    serialized = bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize())
    (record, index, image), = executable_programs(serialized, target=common.TARGET)
    source = format_assembly(image, target=common.TARGET)
    assert source.count(', 0x1, v') == 8
    patched = replace_executable_programs(serialized, {(record, index): assemble_listing(source.replace(', 0x1, v', ', 0x2, v'))}, target=common.TARGET)

    def check() -> None:
        np.testing.assert_array_equal(np.asarray(compiled(x)), host + 1)
        np.testing.assert_array_equal(np.asarray(load_executable(patched, compiled)(x)), host + 2)
        np.testing.assert_array_equal(np.asarray(compiled(x)), host + 1)

    common.finish(patched, Path(__file__).stem, check)

if __name__ == '__main__':
    main()

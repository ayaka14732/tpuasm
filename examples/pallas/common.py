"""示例共用的目标选择、设备选择、编译、数值检查与清单导出。

环境变量 ``TPUASM_EXAMPLES_TARGET`` 选择目标（``tpu-v4-tc``、``tpu-v6e-tc`` 或 ``tpu-v6e-tec``），清单写入 ``examples/pallas/<目标>/``，注释中的源码路径改为相对仓库根目录。默认在本机该目标的 TPU 上编译、运行并检查数值。设置 ``TPUASM_EXAMPLES_AOT=1`` 时改为按参考拓扑离线编译：不需要 TPU，也不运行 kernel，只导出清单，并在输出中注明未检查数值。

参考拓扑是编写清单所用的设备：v4 为四颗 Megacore 芯片（2x2x1），v6e 为 ``v6e:2x2``。按名称创建的 ``v4:2x2x1`` 拓扑把每个 TensorCore 当作一个 device，编译出的程序与 Megacore 不同，所以 v4 使用 ``v4_2x2x1_megacore.topology``：它是在 TPU v4 上只让本 host 四颗芯片可见时，``topologies.get_attached_topology().serialize()`` 的结果。

``tpu-v6e-tec`` 的示例是文件名以 ``sc_`` 开头的 SparseCore kernel。导出编译结果的 TEC 清单后，再与编译器 SparseCore bundle dump 中 LLVM TPU printer 的文本逐条核对（见 ``_compare_with_dump``）。
"""
from __future__ import annotations

from collections.abc import Callable
import contextlib
import functools
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from typing import Any

import jax
from jax.experimental import topologies
from jax.stages import Compiled
import numpy as np

from tpuasm import compiler_source_mapping, dump_compiled, dump_executable, parse_assembly

EXAMPLES = Path(__file__).resolve().parent
ROOT = EXAMPLES.parents[1]
TARGET = os.environ['TPUASM_EXAMPLES_TARGET']
AOT = os.environ.get('TPUASM_EXAMPLES_AOT') == '1'
KIND = {'tpu-v4-tc': 'TPU v4', 'tpu-v6e-tc': 'TPU v6 lite', 'tpu-v6e-tec': 'TPU v6 lite'}[TARGET]
TEC = TARGET == 'tpu-v6e-tec'
if TEC:
    # libtpu 初始化时读取这个选项，所以在第一次使用 JAX backend 之前设置。
    SPARSECORE_DUMP = Path(tempfile.mkdtemp(prefix='tpuasm-sparsecore-', dir='/tmp'))
    os.environ['LIBTPU_INIT_ARGS'] = f"{os.environ.get('LIBTPU_INIT_ARGS', '')} --xla_sc_dump_bundles_to={SPARSECORE_DUMP}".strip()
# 每个 JAX device 的 TensorCore 数：TPU v4 的 Megacore device 包含一颗芯片的两个 TensorCore。
CORES = 2 if TARGET == 'tpu-v4-tc' else 1
OPTIONS = {'xla_msa_enable': 'false', 'xla_tpu_vmem_scavenging_mode': 'NONE'}
if CORES == 2:
    # Megacore 两个 TensorCore 之间的远程 DMA 需要此选项；它不改变单核示例的机器码。
    OPTIONS['xla_mosaic_unsafe_allow_multicore_remote_dma'] = 'true'

@functools.cache
def _reference_devices() -> list[Any]:
    if TARGET != 'tpu-v4-tc':
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
    """在来源映射上下文中编译（TEC 不支持来源映射）；mesh 是 shard_map 使用的 mesh，单设备示例省略。"""
    jax.config.update('jax_enable_compilation_cache', False)
    if mesh is None:
        mesh = jax.sharding.Mesh(np.array(devices()), ('device',))
    # 离线编译时 Pallas lowering 从 abstract mesh 读取 TPU 代际。设备上也设置，因为它影响 JAX jit 追踪缓存是否命中：不设置时，kernel 中重复调用的 jnp.maximum 等 jit 函数复用第一次追踪的 jaxpr，来源注释会带上第一次调用的行号。
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh), contextlib.nullcontext() if TEC else compiler_source_mapping():
        return jax.jit(function, compiler_options=OPTIONS).lower(*arguments).compile()

def finish(compiled: Compiled | bytes, name: str, check: Callable[[], None]) -> None:
    """在设备上检查数值后导出清单 ``<目标>/<name>.tpuasm``；离线编译时跳过检查。compiled 也可以是 serialized executable。"""
    if AOT:
        state = '离线编译，未检查数值'
    else:
        check()
        state = '数值检查通过'
    output = EXAMPLES / TARGET.replace('-', '_') / f'{name}.tpuasm'
    # SparseCore code travels in the TC record, so the TEC target must be named explicitly.
    target = TARGET if TEC else None
    path, = dump_executable(compiled, output.parent, target=target) if isinstance(compiled, bytes) else dump_compiled(compiled, output.parent, target=target)
    text = path.read_text(encoding='utf-8')
    path.unlink()
    if TEC:
        # A serialized executable passed in may have been edited, so only the compiler's own output is compared.
        if not isinstance(compiled, bytes):
            _compare_with_dump(text)
        shutil.rmtree(SPARSECORE_DUMP)
    output.write_text(text.replace(f'{ROOT}/', ''), encoding='utf-8')
    print(f'{state}；产物：', output.relative_to(ROOT))

# Tokens that only name what the instruction writes; tpuasm omits them.
_IMPLIED = ('_', '(pc)', '(tag)', '(tm)')
_NUMBER = re.compile(r'(?<![\w.])-?(?:0x[0-9a-f]+|[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)(?![\w.])')

def _split(text: str) -> list[str]:
    """按不在括号内的逗号拆分操作数。"""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        depth += char in '[('
        depth -= char in '])'
        if char == ',' and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]

def _number(match: re.Match[str]) -> str:
    text = match[0]
    if re.fullmatch(r'-?[0-9]+(?:\.[0-9]+)?e[+-]?[0-9]+|-?[0-9]+\.[0-9]+', text):
        return hex(struct.unpack('<I', struct.pack('<f', float(text)))[0])
    return hex(int(text, 0) & 0xffffffff)

def _token(text: str) -> str:
    """操作数的比较形式：去掉空白与 ``$``，数值统一为 32 位十六进制，单个隐含寄存器去掉括号和编号。"""
    text = re.sub(r'\s+', '', text).replace('$', '').lower()
    if match := re.fullmatch(r'\(([a-z0-9]+?)(?:_?[0-9]+)?\)', text):
        text = match[1]
    return _NUMBER.sub(_number, text)

def _printer_target(operand: str, pc: int, labels: dict[str, int]) -> str:
    """相对分支的目标 bundle 编号；printer 把目标写成标签之差（如 ``.LBB2_2-.Ltmp0``）或相对位移。"""
    if match := re.fullmatch(r'([\w.]+)-([\w.]+)', operand):
        return str(pc + labels[match[1]] - labels[match[2]])
    if operand.startswith('$'):
        return str(pc + int(operand[1:], 0))
    return operand

def _printer_bundles(text: str) -> list[list[tuple[str, str, tuple[str, ...]]]]:
    """dump 中 TEC 段的 bundle，每条指令为 (谓词, 助记符, 目的在前的操作数)；相对分支写成目标 bundle 编号。"""
    lines = text.split('\ntec\n', 1)[1].splitlines()
    labels: dict[str, int] = {}
    bundles: list[str] = []
    current = ''
    for line in lines:
        if re.fullmatch(r'[\w.]+:', line):
            labels[line[:-1]] = len(bundles)
            continue
        current += re.sub(r'^0x[0-9a-f]+: \{', '', line) + '\n'
        if line.endswith('}'):
            bundles.append(current.strip().removesuffix('}'))
            current = ''
    result = []
    for pc, body in enumerate(bundles):
        instructions = []
        for text in body.split(';\n'):
            destinations, _, source = text.partition('=')
            mnemonic, _, rest = source.strip().partition(' ') if ' ' in source.strip() else source.strip().partition('\t')
            predicate = ''
            if match := re.match(r'\s*(@!?p[0-9]+)\s*(.*)', rest, re.S):
                predicate, rest = match[1], match[2]
            operands = [item for item in _split(destinations) if item not in _IMPLIED] + _split(rest)
            if mnemonic.endswith('.rel'):
                operands = [_printer_target(item, pc, labels) for item in operands]
            instructions.append((predicate, mnemonic, tuple(_token(item) for item in operands)))
        result.append(sorted(instructions))
    return result

def _listing_bundles(listing: str) -> list[list[tuple[str, str, tuple[str, ...]]]]:
    """tpuasm 清单中的 bundle，形式同 ``_printer_bundles``。"""
    program = parse_assembly(listing)
    result = []
    for pc, bundle in enumerate(program.bundles):
        instructions = []
        for instruction in bundle.instructions:
            predicate = '' if instruction.predicate == 15 else f'@!p{instruction.predicate - 16}' if instruction.predicate >= 16 else f'@p{instruction.predicate}'
            operands = list(instruction.operands)
            if instruction.mnemonic.endswith('.rel'):
                operands = [str(program.labels[item]) if item in program.labels else str(pc + int(item, 0)) if _NUMBER.fullmatch(item) else item for item in operands]
            instructions.append((predicate, instruction.mnemonic, tuple(_token(item) for item in operands)))
        result.append(sorted(instructions))
    return result

def _compare_with_dump(listing: str) -> None:
    """核对导出的 TEC 清单与编译器 SparseCore bundle dump 中 LLVM TPU printer 的文本逐条一致。"""
    dump, = [path for path in SPARSECORE_DUMP.glob('*_bundles.txt') if '\ntec\n' in path.read_text(encoding='utf-8')]
    expected = _printer_bundles(dump.read_text(encoding='utf-8'))
    actual = _listing_bundles(listing)
    assert len(actual) == len(expected)
    for pc, (left, right) in enumerate(zip(actual, expected)):
        assert left == right, f'bundle {pc:#x}: tpuasm {left} != LLVM TPU printer {right}'

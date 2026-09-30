# tpuasm

tpuasm 是 TPU 指令包的汇编器与反汇编器。

## 许可证

MPL-2.0。

## tpuasm 解决了什么问题？

调试和优化 TPU 程序时，经常需要核对编译器实际生成了哪些指令。在过去，常用的方法是查看 LLO dump 的最终指令包（final bundles）。它虽然接近机器程序的反汇编，但仍然属于较高的抽象层级，与实际程序映像不能完全对应。这导致了以下四个问题：

**(1) 最终指令包中仍然包含伪指令**

最终指令包的名称中虽然含有“最终”二字，但并非所有指令都会被包括在程序映像中。例如，最终指令包中包含 `sphi`、`compiler-scheduling-barrier` 和 `inlined_call_operand.hbm` 这样的伪指令，会影响对程序执行周期的精确判读。tpuasm 从机器字节解码，只列出实际占用的指令槽。

**(2) 最终指令包没有显式表达指令槽信息**

在对 TPU 程序进行性能分析时，常常需要通过参考一条指令所属的指令槽判断其具体执行过程，例如判断指令是否需要向量发射、预测是否产生 VIF 背压等。然而，最终指令包中不含有 `s0:`、`vx0:` 等槽标签。tpuasm 显式标出物理槽，使同一指令包内各条指令的发射位置和资源使用能够直接核对。

**(3) 最终指令包不能无歧义地映射到机器指令**

最终指令包使用的指令格式不足以完整地表达机器指令的所有信息。例如，在实际调试中发现，Pallas 生成的 `vdwg` 指令本应选择 GSFT，却错误地选择了 GSFN，导致计算结果错误。而在最终指令包中，两者都被表示为 `vdwg.mxu0`，没有包含 GSFN 或 GSFT 的选择字段，掩盖了这一错误，增加了调试的困难。tpuasm 的指令格式写出了这类选择字段，例如前述指令被写为 `vdwg.128x128.f16 gmr0, gsfn0`，消除了这种歧义。

**(4) 最终指令包无法直接修改并重新汇编**

在进行 TPU 程序的性能分析时，常常需要在一段特定的指令包前后精准插入或删除特定的指令（如 `vtrace`、`sfence`、`srdreg.lcclo` 等）。然而，最终指令包并不支持编辑后直接重新汇编为程序映像。过去，开发者往往需要在更高层级上修改代码，但这容易触发编译器的重新调度，改变其他指令的布局，导致难以构造单变量对照实验。而若直接通过二进制解析来改写字节，又会面临 PC 相对地址偏移等调整难题。tpuasm 完整支持程序的导出、修改、重新汇编与回灌，解决了这一痛点。

## 安装

首先确保系统为 Linux x86-64，且已安装支持 C++17 的 g++。本项目不声明依赖，JAX 与 libtpu 需自行安装。

受支持的 libtpu 版本仅限以下列表，用户需自行确保 libtpu 版本兼容：

**支持 TPU v4 TC 与 BCS 编解码；编译来源捕获限 TC**

- CPython 3.14t，libtpu `0.0.48.dev20260912+nightly`
- CPython 3.14t，libtpu `0.0.48`
- CPython 3.14t，libtpu `0.0.49`

**支持 TPU v6e TC 编解码与编译来源捕获**

- CPython 3.14t，libtpu `0.0.49`

本项目在 JAX `0.12.0.dev20260926+886d2370c1` 上测试通过。编译来源捕获会核对所适配 JAX 函数源码字节的 SHA-256，源码相同即可接受，不限定 JAX 版本号。

从 PyPI 安装本项目：

```sh
python -m pip install tpuasm
```

首次汇编或反汇编会将随包分发的原生桥接编译到 `/tmp`。

离线汇编、反汇编不需要 TPU。在没有对应 TPU 的机器上，也可以按目标拓扑离线编译 kernel 并导出清单，写法见 `examples/pallas/common.py`。

## 使用

### 在命令行使用

命令行在 serialized executable、程序映像和 `.tpuasm` 清单之间转换，本身不执行程序。serialized executable 是把 `bytes(compiled.runtime_executable().serialize())` 写成的文件。常见用法：

```sh
# 把 executable 中的每个程序导出为清单，注释中附带编译来源
tpuasm executable.bin --input-format executable --output-dir asm

# 导出便于手工编辑的清单：重新分配共享资源，不保证与原程序字节相同
tpuasm executable.bin --input-format executable --output-dir asm --encoding canonical

# 反汇编单个程序映像；程序映像不含目标标记，须给出 --target
tpuasm program.bin --input-format image --target tpu-v4-tc --output program.tpuasm

# 把清单汇编为程序映像
tpuasm program.tpuasm --input-format listing --output program.bin

# 用编辑后的清单等长替换程序 1:0，写出新的 executable
tpuasm executable.bin --input-format executable --output-format executable \
  --replace 1:0 asm/program-tpu-v4-tc-1-0.tpuasm --output patched.bin

# 在程序 1:0 编号为 525 的 bundle 之前插入汇编片段
tpuasm executable.bin --input-format executable --output-format executable \
  --insert 1:0:525 probe.tpuasm --output patched.bin
```

`1:0` 即导出文件名 `program-<target>-<record>-<index>.tpuasm` 中的 record 和 index。写出的 executable 须在 Python 中用 `load_executable` 装载执行，见下节。

完整用法见 `tpuasm --help`。

### 在 JAX 中使用

在已有 `kernel` 和 `example_inputs` 的 JAX/TPU 程序中，于 `compiler_source_mapping()` 内编译，导出的清单即带有 Pallas 源码注释：

```python
from pathlib import Path
import jax
from tpuasm import compiler_source_mapping, dump_compiled

with compiler_source_mapping():
    compiled = jax.jit(kernel).lower(*example_inputs).compile()
paths = dump_compiled(compiled, Path('/tmp/tpuasm-output'), source_map_json=True)
```

每个程序写出一份 `.tpuasm`，`source_map_json=True` 时另写同名的 `.sources.json`。

清单是 runtime 实际装载的完整程序映像，Pallas kernel 前后的初始化、同步与退出代码也在其中，裁掉后就不能再回灌。只想浏览 kernel 时，按 `ProgramSourceMap.functions[*].ranges` 给出的编译器函数区间折叠显示。逐槽的源码位置读 `ProgramSourceMap.slots[*].source_frames`，它合并了 tpuasm 捕获的来源与编译器自带的 `loc(...)`；没有来源的指令保持未知，不代表它属于 runtime。设计见[来源映射](docs/design/tc_source_mapping.md)。

要在设备上观察修改后的程序，常用做法是在 kernel 中放一条带唯一立即数的运算作为标记（例如 `x ^ 0x13579bdf`），在清单中找到它所在的 bundle，在此插入手写片段，再借用 `compiled` 的调用约定执行：

```python
from tpuasm import BundleInsertion, executable_programs, format_assembly, insert_executable_bundles, load_executable, parse_assembly

serialized = bytes(compiled.runtime_executable().serialize())
(record, index, image), = executable_programs(serialized)
source = format_assembly(image, target='tpu-v4-tc')  # TPU v6e 为 'tpu-v6e-tc'
pc, = [pc for pc, bundle in enumerate(parse_assembly(source).bundles) if any('0x13579bdf' in instruction.operands for instruction in bundle.instructions)]
patched = insert_executable_bundles(serialized, {(record, index): [BundleInsertion(pc, '.target tpu-v4-tc\n{ s0: srdreg.lcclo s20 }')]})
result = load_executable(patched, compiled)(*example_inputs)
```

片段插在编号为 `pc` 的原 bundle 之前，可以包含多个 bundle，也可以同时在多处插入。插入点不能落在分支的延迟窗口内；片段使用的寄存器须由调用者确认空闲，读数也须由调用者安排带回，完整例子见 `examples/pallas/insert_program.py`。

只修改已有指令时，编辑清单后重新汇编，用 `replace_executable_programs` 做等长替换，例如去掉标记的效果：

```python
from tpuasm import assemble_listing, replace_executable_programs

edited = assemble_listing(source.replace('vxor.8x128.u32 v1, 0x13579bdf, v0', 'vxor.8x128.u32 v1, 0x0, v0'))
patched = replace_executable_programs(serialized, {(record, index): edited})
```

等长替换不改变 bundle 编号，替换结果可以继续交给 `insert_executable_bundles`，仍按原编号插入。两者都会为修改后的程序生成新的程序身份，否则 runtime 可能继续执行已装载的原程序。

## 文档

[文档站点](https://ayaka14732.github.io/tpuasm/)包括以下内容。格式参考说明清单怎么写，设计文档说明实现方式、依据和验证。

- **格式参考**
  - [TPU v4 TC](https://ayaka14732.github.io/tpuasm/references/tpu_v4_tc.html)：各目标共用的源码语法，以及 v4 TC 的操作数与约束。
  - [TPU v4 BCS](https://ayaka14732.github.io/tpuasm/references/tpu_v4_bcs.html)：BCS 的写法、semantic protobuf 转换与容器提取。
  - [TPU v6e TC](https://ayaka14732.github.io/tpuasm/references/tpu_v6e_tc.html)：v6e 与 v4 TC 写法的差异。
  - 三份指令索引：各目标每个槽的助记符与操作数签名，由 `tools/generate_isa_reference.py` 在构建文档时生成。
- [**API 参考**](https://ayaka14732.github.io/tpuasm/api.html)：公开函数和类型的签名与说明。
- **设计文档**
  - [总体架构](https://ayaka14732.github.io/tpuasm/design/architecture.html)：设计原则、模块划分、数据通路、目标识别与程序容器、扩展点。
  - [原生编解码后端](https://ayaka14732.github.io/tpuasm/design/native_backend.html)：后端选择、C ABI 与原生校验。
  - [汇编与可逆导出](https://ayaka14732.github.io/tpuasm/design/assembly.html)：位模型、指令包求解、命名约束、导出，以及 v4 TC ISA 表的维护。
  - [TPU v4 BCS 目标](https://ayaka14732.github.io/tpuasm/design/tpu_v4_bcs.html)：BCS 在实现上与 TC 的差异。
  - [TPU v6e TC 目标](https://ayaka14732.github.io/tpuasm/design/tpu_v6e_tc.html)：字段表的生成、签名规则与 formatter 的限制。
  - [TPU v6e 指令执行语义核对](https://ayaka14732.github.io/tpuasm/design/tpu_v6e_execution.html)：selector、延迟常量等指令语义的设备证据，含 v4 的 Delay 对照。
  - [TC 编译来源映射](https://ayaka14732.github.io/tpuasm/design/tc_source_mapping.html)：编译期捕获来源与离线恢复。
  - [回灌与执行](https://ayaka14732.github.io/tpuasm/design/executable_replacement.html)：替换、插入 bundle、程序身份与装载。

## 测试

GitHub Actions 会自动测试无需 TPU 的部分。

测试需要 TPU v4 的部分，在 TPU v4 机器上运行：

```sh
PYTHONPATH=src python tests/reproduce_tpu_v4_tc_source_mapping.py --output-dir /tmp/tpuasm-source-reproduction
examples/pallas/run_all.sh tpu-v4-tc
```

测试需要 TPU v6e 的部分，在 TPU v6e 机器上运行：

```sh
examples/pallas/run_all.sh tpu-v6e-tc
```

`all_reduce`、`remote_dma_devices` 与 `remote_dma_ring` 需要 2 或 4 颗芯片。

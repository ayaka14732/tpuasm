# 总体架构

tpuasm 在 Python 中维护 ISA 位模型、源码语法和指令包资源求解。程序映像与已解码程序对象之间的转换则交给匹配版本 libtpu 内部的编解码器。本文说明模块划分、汇编和反汇编两条数据通路、硬件目标的确定方式和程序容器的解析方式，并列出扩展时需要改动的位置。

## 设计原则

- **以机器字节为准。** 反汇编只解码程序映像，不读取 LLO dump 或 final bundles。libtpu formatter 输出的文本只用于内部交叉校验，以及离线生成 v6e 字段表时取助记符和操作数顺序；它不作为清单输出，也不作为汇编输入。
- **硬件目标与 libtpu 后端是两个独立的维度。** 硬件目标由 TPU 代际和执行单元确定，决定程序映像格式、物理槽、ISA 表和求解器。libtpu 后端对应某个 release、Python ABI、平台和 build-id，记录函数地址与对象布局。为已有硬件增加 libtpu 版本时不改硬件定义；也不根据 libtpu 版本推断硬件目标。
- **没有默认目标。** raw image 不带目标标记，而 TC 与 BCS 都以 512 字节分块，无法凭长度区分。raw image 的目标由调用方指定，listing 的目标由 `.target` 声明，executable 的目标从容器推断，推断不唯一时报错。
- **不支持就报错。** 未登记的指令形式、无法恢复的编码、不匹配的运行环境都直接报错。不退回到原字节附件、任意 bit patch 或完整 protobuf 输入。

## 模块

| 层 | 模块 | 职责 |
|---|---|---|
| 公开入口 | [printer.py](../../src/tpuasm/printer.py) | API 与 CLI、executable 导出为文件；另含原生桥接的加载与调用（`_load_native`、`_program_proto`、`_verify_image`）。 |
| 目标分派 | [assembler.py](../../src/tpuasm/assembler.py) | `assemble_listing` 按 `.target`、`format_assembly` 按 `target` 参数分派到 v4 TC、v4 BCS 或 v6e TC 实现。 |
| 硬件目标 | [targets.py](../../src/tpuasm/targets.py) | `HardwareTarget`：块大小、每块 bundle 数和物理槽顺序。 |
| 源码与数值 | [assembly_syntax.py](../../src/tpuasm/assembly_syntax.py)、[assembly_printer.py](../../src/tpuasm/assembly_printer.py)、[assembly_values.py](../../src/tpuasm/assembly_values.py) | 带源码位置的 AST、标签、`.empty` / `.align`；清单渲染；整数与 float32 位型。 |
| 共用位模型 | [assembly_model.py](../../src/tpuasm/assembly_model.py)、[assembly_expressions.py](../../src/tpuasm/assembly_expressions.py)、[assembly_operands.py](../../src/tpuasm/assembly_operands.py) | `Field`、`Bits`、`Form`、`Signature`；操作数表达式；操作数文本与候选位约束的双向转换。 |
| 可逆导出 | [assembly_export.py](../../src/tpuasm/assembly_export.py) | 各目标共用的 exact 约束删减、分支标签恢复和重汇编核对。 |
| TC 求解 | [tc_solver.py](../../src/tpuasm/tc_solver.py) | 各代 TC 共用的指令包求解；代际差异由 `TcIsa` 描述。 |
| v4 TC | `tpu_v4_tc_isa_data.py`、`tpu_v4_tc_model.py`、`tpu_v4_tc_isa.py`、`tpu_v4_tc_constraints.py`、`tpu_v4_tc_codec.py`、`tpu_v4_tc_assembler.py` | 字段与形式表、助记符签名、命名约束、与 libtpu 交换字段、汇编与导出流程。 |
| v6e TC | `tpu_v6e_tc_isa_data.py`（由 [generate_tpu_v6e_tc_isa.py](../../tools/generate_tpu_v6e_tc_isa.py) 生成）、`tpu_v6e_tc_model.py`、`tpu_v6e_tc_isa.py`、`tpu_v6e_tc_constraints.py`、`tpu_v6e_tc_codec.py`、`tpu_v6e_tc_assembler.py` | 同上。 |
| BCS | `tpu_v4_bcs_isa_data.py`、`tpu_v4_bcs_isa.py`、`tpu_v4_bcs_assembler.py`、`tpu_v4_bcs_program.py` | 同上，另含 semantic protobuf 互操作和容器提取。 |
| 容器 | [program_container.py](../../src/tpuasm/program_container.py)、[_protobuf.py](../../src/tpuasm/_protobuf.py) | 切分 executable 记录、推断目标；最小的 protobuf wire 读写，不依赖 protobuf 运行时。 |
| 写回与装载 | [executable_replacement.py](../../src/tpuasm/executable_replacement.py) | 等长替换及按插入点变长写回 TC 程序映像并更新程序身份；借用已编译 JAX 对象的调用约定装载 executable。 |
| 原生后端 | [backends.py](../../src/tpuasm/backends.py)、[native.cc](../../src/tpuasm/native.cc)、[native_backends/](../../src/tpuasm/native_backends/) | 后端登记与选择；C ABI 桥接；各版本常量。 |
| TC 来源映射 | `tc_compiler.py`、`tc_source_lowering.py`、`tc_source_backend.py`、`tc_source_native.cc`、`source_backends/`、`tc_source_mapping.py` | 编译期捕获来源，离线从 executable 恢复。 |

## 数据通路

汇编和反汇编使用同一个 AST 和同一个求解器。反汇编的结果先转换为与源码解析结果结构相同的 `AssemblyProgram`，精确导出的正确性也由实际重汇编来检验。因此，打印和汇编之间不存在两套各自维护、可能不一致的编码规则。

**汇编**（`assemble_listing`）：

1. 解析源码：读取 `.target`，建立 bundle、标签、`.empty` / `.align`，检查 bundle 总数是否为每块 bundle 数的正整数倍。
2. 逐个 bundle 求解：将指令匹配到签名，生成候选位约束，与 `.encoding` 合并后选择确定的解，得到这个 bundle 的机器字。
3. 用各 bundle 的机器字构造 libtpu ISA program protobuf，交给原生 encoder 生成程序映像。
4. 核对：重新解码后的机器字和已占用槽集合与求解结果一致，程序映像通过原生往返校验。

**反汇编**（`format_assembly`）：

1. 原生校验：程序映像块对齐，解码出的 bundle 数正确，decode→encode 逐字节一致。TC 还检查逐槽 formatter 的输出。
2. 原生解码得到 ISA program protobuf，在 Python 中还原为每个 bundle 的机器字和已占用的指令形式。
3. 每个形式按签名把字段读成操作数文本；为程序内的直接分支目标生成标签。
4. `export_program`：exact 模式为需要的 bundle 补充命名约束，渲染清单，再调用 `assemble_listing` 重汇编并与原程序映像比较。

## 硬件目标与程序容器

serialized executable（`compiled.runtime_executable().serialize()` 的结果）是若干条用 varint 长度前缀分隔的记录。已支持的 libtpu 版本在每条 core program（`TpuCoreProgramProto`）记录之后紧接着序列化它的 `CompilerMetadata` 记录，v4 与 v6e 的 executable 都是如此。

`program_container.core_program_target()` 按以下规则识别单条记录的目标，缺少任何一项证据都返回 `None`：

- program oneof 中恰好有 `tensor_core(5)` 或 `barna_core(6)` 之一。
- 其 `sequencer(1).sequencer_type(3)` 为 1（TC）或 2（BCS）。
- program alternative 唯一，并且是该目标已核对的字段。
- 若有 ABI 信息，则 `version` 为 3（`TPU_VERSION_PUFFERFISH`）时是 v4；为 5（`TPU_VERSION_GHOSTLITE`）、且 TC 的 alternative 是程序映像字段 16 时是 v6e TC。枚举值取自 libtpu 内嵌的 descriptor。外层的 `platform_type=1` 只表示 HARDWARE，不能用来判断代际。

`resolve_executable_target()` 遍历所有记录，只有恰好识别出一个目标时才推断成功。

TC 与 BCS 从容器取得程序映像的方式不同：

- **TC**（`tc_source_mapping._programs`，v4 与 v6e 共用）：读取 `memory_segments(8)` 中类型为 CODE 的 segment，按其 range 从 `initialized_data` 中切出程序映像，拒绝压缩的代码段。同一条 core 记录可以有多份程序映像，编号从 0 开始。提取逻辑放在来源映射模块里，因为来源映射还需要同时保存 segment 索引、hash、fingerprint 等身份信息。
- **BCS**（`tpu_v4_bcs_program.executable_bcs_programs`）：容器中保存的是 semantic protobuf 而非机器字节，所以先提取 semantic body，再调用原生 codec 编码为程序映像，每条记录的索引固定为 0。

导出文件命名为 `program-<target>-<record>-<index>.tpuasm`，record 在整个容器内编号。两种提取都只读取需要的字段，既不是完整的 executable 验证器，也不改写 executable。

反方向由 `replace_executable_programs` 完成：以同样的 `(record, index)` 为键，把等长的新 TC 程序映像写回 `initialized_data`，并为内容有变化的记录生成新的 segment set hash 和 fingerprint，否则 runtime 可能复用已装载的原程序。`insert_executable_bundles` 则通过 [tc_relocation.py](../../src/tpuasm/tc_relocation.py) 迁移直接分支、装载块数、overlay、符号与注释，并重新编码容器长度。修改后的 executable 由 `load_executable` 按原调用约定装载执行。

## 扩展点

| 需求 | 需要改动的位置 |
|---|---|
| 为已有硬件增加 libtpu 版本 | `backends.LIBTPU_RELEASES` 与 `native_backends/` 下的版本源文件，步骤见 skill [add-libtpu-backend](../../.agents/skills/add-libtpu-backend/SKILL.md)。编译来源捕获另有独立的来源后端，见[来源映射](tc_source_mapping.md#版本相关的部分)。 |
| 增加指令形式 | 目标的 `*_isa_data.py` 与签名规则，见[汇编与可逆导出](assembly.md#维护-isa-表)。 |
| 增加硬件目标 | 在 `targets.py` 中定义目标；编写 ISA 表、签名、约束和导出函数，TC 可复用 `tc_solver`；登记原生后端；在 `assembler.py` 中分派；在 `program_container.py` 中补充目标识别规则；来源映射需要登记槽名与 overlay 单位。CLI 的 `--target` 选项直接取自 `TARGETS`。 |

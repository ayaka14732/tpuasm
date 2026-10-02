# 汇编与可逆导出

本文说明 `.tpuasm` 源码如何变成机器字，以及程序映像如何导出为可重新汇编的源码。所述机制由各目标共用，资源模型以 v4 TC 为例；BCS 的差异见 [BCS 目标](tpu_v4_bcs.md)。

## 设计动机

整体设计由以下三个事实决定。

**一、同一 bundle 的多个槽共用一部分字段。** TC 的标量槽与向量槽共用 6 个 16 位立即数 lane `imm0..imm5`（标量指令只能使用其中的 `imm0..imm3`）、3 个标量读取字段 `vs0..vs2`，以及若干 TC VREG 读写通路字段。libtpu 的 encoder 按固定的槽顺序，把各槽 protobuf 中存在的字段写进同一个 bundle 缓冲区，后写入的值覆盖先写入的值，显式写入的零也会覆盖。例如 `s0` 和 `s1` 的两条 `simm` 都选择 `imm0`，分别写入 7 和 21，结果两条指令都读到 21。所以不能先逐槽编码、再把结果拼起来。

**二、可见的文本不能决定全部机器位。** 以下几种情况中，两份程序映像字节不同，却有相同的规范汇编文本：没有指令读取的立即数 lane 中仍保存着值；两种 selector 产生同一个值，例如 ScalarY 的“常量零”和“零扩展 `imm0`”在 `imm0=0` 时结果相同；两条读通路读到编号相同的寄存器；没有被选中的目的字段中仍编码着寄存器号。[tests/reproduce_tpu_v4_tc.py](../../tests/reproduce_tpu_v4_tc.py) 收录了九对这样的程序映像。

**三、ISA protobuf 中有些字段不对应机器位。** 槽级的 `latency`、`resource_usage` 和 `bit_width` 由 decoder 生成，encoder 不读取它们。删除或改写这些字段后，程序映像不变。

这三点分别对应以下设计：以整个 bundle 为单位求解共享字段；exact 导出用命名的 `.encoding` 约束保存可见指令以外的编码信息，约束使用硬件名称，不使用 protobuf 字段号；普通源码不必写约束，由确定的规则选出规范编码。第三类辅助字段不进入语法。

汇编器不做以下事情：选择发射槽、重排指令、插入流水线等待、把一条指令展开为多条。`{ … }` 的边界和 `slot:` 都由作者决定。

## 位模型

[assembly_model.py](../../src/tpuasm/assembly_model.py) 定义了与目标无关的四个类型：

- **`Field(number, start, width)`**：`start` 和 `width` 是该字段在 bundle 机器字中的位置。机器字是一个 Python 整数，v4 TC 为 408 位（51 字节），BCS 为 256 位，v6e TC 为 512 位（64 字节）。`number` 是该字段在 libtpu ISA protobuf 中的字段号，只在与原生后端交换数据时使用。
- **`Form`**：某个槽的一个 oneof 分支，包括槽名、分支字段号 `branch`、操作名和字段表。`fixed(predicate)` 绑定 opcode、谓词和其他固定位。空槽的编码是谓词 NEVER（31）；`EMPTY_WORD` 在所有槽的谓词位置都填入 NEVER。
- **`Bits(mask, value, immediates, scalars)`**：部分赋值，以及本次赋值实际读取的共享资源集合（立即数 lane 与 `vs` 各用一个位集表示）。两个 `Bits` 只有在重叠的已赋值位上取值相同时才能合并。
- **`Signature`**：形式、助记符、操作数表达式、固定字段和具名参数。

冲突检测只看位，不看名字。某个形式的 `imm0` 字段和全局的 `imm0` 覆盖相同的位，DMA 的若干字段互为别名，这些别名关系都通过 `Bits.merge` 自动保持一致，不需要额外的别名表。

`Form.bind(..., consumed=True)` 表示某个操作数确实读取了这个共享字段。TC 按位置识别立即数 lane 和 `vs` 字段，并把它们记入 `immediates` / `scalars`。`.encoding` 约束写入的值不算读取，因此不计入求解代价。

## 签名与操作数

一个 `(slot, mnemonic)` 可以对应多个签名，一个形式也可以有多个签名。例如，`ScalarMove` 形式有 `smov`（寄存器源）和 `simm.s32`（立即数源）两个签名；DMA 形式借助固定字段 `trace` 派生出 `.trace` 变体；`vst.8x128` 同时对应 `VmemStore` 和 `VmemStoreNoOffset` 两个形式。反汇编时，按顺序选取第一个固定字段匹配、且所有操作数都能解码的签名。

操作数表达式定义在 [assembly_expressions.py](../../src/tpuasm/assembly_expressions.py)，是带标签的元组：`literal`、`register`、`number`、`choice`、`memory`、`table`、`pattern`，以及 v4 专用的 `dma_address` 和 `trace`。数值表达式还可以限制每个来源字段的有效位宽：v6e 的 20 位立即数字段在双槽拼接时只读取低 16 位，忽略的位由 `.encoding` 保留；v6e TEC 的配置字由若干字段按位拼成一个数。`memory` 的地址修饰（`sm=`、`ss=`）由签名逐项给出修饰名、字段和操作数，省略修饰时字段取 0。`table` 把一段固定文本对应到若干字段的取值，用于由单元号决定的目的寄存器组，例如 v6e 的 `(mrf1, gmr1, msra1)`。`pattern` 是嵌有若干操作数的固定文本，例如 v6e TEC 的 `[hbm4b:{0}+{1}]`，汇编时忽略空白。`choice(field, alternatives)` 用一个 selector 字段在几种来源之间选择，所以同一段文本可以产生多个候选。例如 ScalarY 位置上的 `7` 可以来自内置常量 selector，也可以来自某个立即数 lane 的零扩展。

[assembly_operands.py](../../src/tpuasm/assembly_operands.py) 中的 `encode_operand` 把文本转换为候选 `Bits` 列表，`decode_operand` 从机器字读回文本。二者必须互逆：`decode_operand` 输出的文本重新编码时，候选中必须有一个与它读取的字段取值相同。新增表达式类型时，两个方向都要实现。

数值由若干立即数字段拼接而成。字段中超出操作数位宽或有效位宽的高位不影响数值，编码时保持未约束，由 exact 导出的 `.encoding` 保留。v6e 的 ScalarY、VectorY 和 VectorShuffle 双立即数 selector 只读取两个 20 位字段各自的低 16 位，不存在重叠。`vtrace` 的立即数表达式目前仍按 formatter 的 20 位按位或模型登记，尚未通过设备 trace 记录核对，因此求解器仍需枚举其重叠位的分配，不能直接删掉该分支。

数值带有类型（`s32`、`u32`、`hex`、`f32`）和位宽。十进制数检查数值范围；非负十六进制数还可以表示该位宽下的原始位型。浮点十进制数以精确有理数计算，一次舍入到 binary32（ties-to-even）。打印时使用 Python 的最短往返表示，NaN 和 −0.0 打印为 `f32bits(...)`。这样打印出的文本重新解析后必然得到相同的位型，与 libtpu formatter 的显示精度无关。

一条指令的候选集是 `fixed(predicate)`、签名的固定字段和各操作数候选的笛卡尔积合并结果，去掉冲突和重复的组合。

## 指令包求解

TC 的求解在 [tc_solver.py](../../src/tpuasm/tc_solver.py) 中完成，v4 与 v6e 共用。代际之间的差异由 `TcIsa` 给出：机器字字节数、`EMPTY_WORD`、始终由 encoder 写出的位（v6e 的共享立即数与标量操作数）、全局约束、签名表、形式的可写位、形式不能同时占用的槽，以及是否优先使用编号小的共享槽。

1. 先把全局约束（`imm*`、`vs*`、`port.*`）合并为一个 `Bits`。再把槽级约束（`<slot>.<name>`）合并到该槽的每个候选上，丢弃冲突的候选。某个槽的候选全部被丢弃时，报错并列出涉及的约束。
2. 各槽的候选集按大小升序排列后做深度优先搜索，用分支定界剪枝。比较键依次为：实际读取的立即数 lane 数、实际读取的 `vs` 数、（仅 v6e）所用立即数与标量槽的编号集合、机器字的 little-endian 字节串。v6e 加入编号集合，是为了让规范编码像编译器一样先用 `imm0`、`vs0`；否则字节串比较会随 selector 编码的位置随意挑选共享槽。计算下界时，任何候选都不写的位固定为 `EMPTY_WORD` 中的值。
3. 最终机器字在所选形式可写的位上取求解值，没有赋值的位取 0；其余位取 `EMPTY_WORD`。
4. 全局约束只能落在所选形式可写的位上。encoder 只写入存在的槽的字段，不属于任何存在的形式的位无法编码。

比较键与源码中槽的书写顺序、字典遍历顺序和 protobuf 字段顺序都无关，所以结果是确定的。先最小化实际读取的共享资源，一是让规范编码为后续编辑留出空间，二是优先使用不占资源的内置常量。例如 `simm.s32 s0, 7` 选择常量 selector，不占用 `imm0`。

## 与 libtpu 交换字段

[tpu_v4_tc_codec.py](../../src/tpuasm/tpu_v4_tc_codec.py) 把求解结果写成 libtpu 的 ISA program protobuf：`Program` 的 field 1 是重复的 bundle；bundle 的 field `槽序号+1` 是槽消息；槽消息的 field 1 是谓词，field `branch` 是操作消息；操作消息的各字段取 `Field.number`，值从求解出的机器字中读取。

每个形式的所有字段都显式写出，零值也写。这些值全部读自同一个机器字，所以重叠的字段一定携带相同的位，encoder 按什么顺序写入不再影响结果。解码方向相反：从各字段值重建机器字，同时检查重叠字段的取值是否一致。

TC 的机器字不能直接与程序映像字节比较，因为块内编排由 libtpu 处理。isa_data 中登记的位位置如果有误，是在下面这一步被发现的：编码后重新解码，要求机器字和已占用槽集合都与求解结果一致。位置登记错误时，求解器没有发现真实的字段重叠，encoder 的覆盖会使重新解码的机器字与求解结果不同，汇编随即报错。

## 命名约束

[tpu_v4_tc_constraints.py](../../src/tpuasm/tpu_v4_tc_constraints.py) 定义约束的名称和取值：

- 全局名称包括 `imm0..imm5`、`vs0..vs2`、`port.<槽>.<x|y|src|dst>` 和 `port.aux.dst`。形式中某个字段的位范围如果与全局字段完全相同，就以全局名称公开。
- 槽级名称为 `<slot>.<字段>`，部分字段使用更易读的别名（如 `sy`→`y`、`read_port`→`read`、`base_address`→`base`）。向量结果槽的目的通路写作 `<vrN>.write`。此外每个形式都公开 `<slot>.opcode` 和额外的固定字段。
- selector 字段取符号值，如 `lo(imm0)`、`const(0)`、`port.va0.x`、`vs1`、`s5`；其他字段取整数。`selector_values` 是编码值与符号名之间唯一的对照表。导出时遇到表中没有的编码值会报错（“no named encoding”），不会退回为打印裸数字。

约束与操作数候选一样以 `Bits` 的形式参与搜索。约束不能绕过操作数检查：与可见操作数冲突的约束会导致汇编失败。

## 导出

两个目标共用 [assembly_export.py](../../src/tpuasm/assembly_export.py) 中的 `export_program`。

**exact 模式**逐个 bundle 处理：

1. 不带约束求解。结果等于原机器字时，这个 bundle 不需要约束。
2. 否则，从**解码得到的形式**列出全部命名约束，并确认它们足以重建原机器字。不足时报错“named fields do not cover the original encoding”，说明约束登记表缺少某个字段。这里必须使用解码得到的形式，而不是规范求解选出的形式。原因是同一助记符可能对应多个形式（如 `VmemStore` 与 `VmemStoreNoOffset`），只有 `<slot>.opcode` 等约束才能把求解固定到原来的形式上。
3. 按名称顺序逐项尝试删除约束，删除后仍能重建原机器字才真正删除，反复进行直到没有约束可删。得到的约束集没有冗余，也就是说其中任何一项都不能单独删除，但不一定是全局最少的约束集。值为零的约束也可能被保留，只要它限制了资源分配。

最后渲染整份清单并调用 `assemble_listing`，要求结果与原程序映像逐字节相同。这一步使 exact 导出的正确性只依赖公开的汇编器，而不依赖导出过程的中间结果。

**canonical 模式**不输出约束，重汇编成功即可返回。BCS 还额外核对重汇编后的可见操作数保持不变。canonical 模式不承诺字节相同。

两种模式共同的规则如下：

- PC 0 标为 `entry`，程序内的直接分支目标标为 `L_xxxx`，目标超出程序范围时保留数值。
- 带 NEVER 谓词、却占用了槽的指令在清单中不可见，因此拒绝导出。
- 来源注释由 `render_program` 插入 `#` 注释中，不参与编码。

## 验证层次

| 检查 | 位置 | 证明的内容 |
|---|---|---|
| 解码 bundle 数、原生往返、逐槽 formatter | `native.cc` | decoder 接受了完整映像，解码对象没有丢失机器信息；逐槽 formatter 核对槽名映射（仅 TC）。 |
| 编码后重新解码的机器字与槽集合等于求解结果 | `tpu_v4_tc_assembler._verify` | 后端没有改变求解结果，显式约束和所选槽依然成立，位位置登记与 encoder 一致。 |
| encoder 输出的字节等于求解出的机器字，重新解码后槽集合不变 | `tpu_v6e_tc_codec.encode_program`、`tpu_v6e_tc_assembler.assemble_program` | v6e 的机器字就是映像字节，所以直接逐字节比较；槽集合的比较能发现被 encoder 静默丢弃的槽。 |
| `assemble_listing(format_assembly(image)) == image`（exact） | `export_program` | 导出的源码不依赖其他信息即可恢复原程序映像。 |

这些检查不证明程序的调度正确，也不证明设备执行效果。不能命名的字段、未登记的形式和无法往返的输入都会报错。

## 维护 ISA 表

本节说明 v4 TC 的表。v6e 的表由工具从 libtpu 生成，维护方法见 [TPU v6e TC 目标](tpu_v6e_tc.md#字段表的生成)。

v4 TC 的表位于 [tpu_v4_tc_isa_data.py](../../src/tpuasm/tpu_v4_tc_isa_data.py)：

- `OPCODES`：各槽 opcode 的位置；谓词紧接在 opcode 之上，占 5 位。
- `FIELD_LAYOUTS`：多个形式共用的字段组，每项为 (名称, protobuf 字段号, 起始位, 位宽)。
- `INSTRUCTION_FORMS`：槽、分支号、ISA 名、opcode、字段组编号、助记符、固定操作数骨架和额外固定字段。

字段名、字段号和 oneof 结构来自 libtpu 内嵌的 ISA descriptor；位位置来自 encoder 实际写入的位（通过反汇编 encoder，或只改变一个字段后比较编码差异得到）。两个已支持的 libtpu 版本在这些方面完全一致。[tpu_v4_tc_isa.py](../../src/tpuasm/tpu_v4_tc_isa.py) 按槽族的规则从形式生成签名，操作数顺序的约定（例如部分 ALU 指令 Y 源在前）也写在这里。

### 未登记的形式

descriptor 中除 Noop 外有 582 个槽内形式，目前登记了 573 个。以下 9 个未登记：`s1` 的 `ReadDone`、`WriteDone`；`vst` 的 `VmemStoreIndexedNoOffset`；`misc` 的 `DelayUntilNotDone`、`ClearResultFifo`、`AtomicRemoteWriteSetDone`、`AtomicRemoteWriteSetDoneInverted`、`AtomicRemoteAddSetDone`、`AtomicRemoteAddSetDoneInverted`。这几个形式用默认操作数做编码探针时失败，还没有找到合法的非零操作数，但这并不证明它们无法编码。遇到这些形式时，解码报 “unsupported instruction form”。

### 新增形式的步骤

1. 在 `INSTRUCTION_FORMS` 中加入形式，必要时新增字段组。
2. 确认 `tpu_v4_tc_isa.py` 生成了预期的助记符与操作数；新的 selector 字段需要在 `tpu_v4_tc_constraints.selector_values` 中补充符号值。
3. 用该形式的若干操作数变体检查：汇编→解码往返，以及 exact 导出能否逐字节恢复（包括只有 `.encoding` 才能区分的变体）。
4. 运行 [tests/reproduce_tpu_v4_tc.py](../../tests/reproduce_tpu_v4_tc.py)，用 `tools/generate_isa_reference.py` 重新生成指令索引并检查。

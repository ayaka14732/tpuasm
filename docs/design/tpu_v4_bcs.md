# TPU v4 BCS 目标

`tpu-v4-bcs`（BarnaCore Sequencer）是独立的硬件目标。它复用公共的源码语法、`Bits` 位模型、操作数表达式和 `export_program`，但有自己的 ISA 表、求解器和原生校验方式。TC 标量槽的字段布局不能套用到 BCS。本文说明 BCS 与 TC 在实现上的不同之处；两者共用的机制见[汇编与可逆导出](assembly.md)。

## 与 TC 的差异

| 方面 | TC | BCS |
|---|---|---|
| bundle 与程序映像 | 51 字节 bundle，每 512 字节块 10 个，另有块内分隔字节；Python 机器字不直接对应映像字节 | 32 字节 little-endian bundle，每块 16 个，无分隔字节；机器字就是映像中对应的 32 个字节 |
| 物理槽 | 12 个 | `s0`、`s1` |
| 共享字段 | `imm0..imm5`、`vs0..vs2`、读写通路 | 4 个 16 位立即数 lane `imm0..imm3`，直接分支的地址也占用 `imm0` |
| 形式的固定位 | opcode 字段加少量额外字段 | 任意的固定 mask / value（`BcsForm.fixed_mask` / `fixed_value`），因为子操作由 opcode 之外的位选择 |
| 约束取值 | selector 用符号名 | 所有字段都用无符号整数，名称为 `imm0..imm3` 和 `<slot>.<字段>` |
| 求解 | 分支定界搜索 | 两个槽候选的笛卡尔积，比较键为（实际读取的 lane 数, 32 字节 little-endian 字节串） |
| 原生校验 | 往返校验加逐槽 formatter | 仅 codec 往返（`TPUASM_CODEC_ONLY`） |
| 编译来源映射 | 支持 | 不支持，`source_map` 和 `--source-map-json` 被拒绝 |

## 硬件表示

S0 的谓词与 opcode 起始位分别为 128 与 122，S1 的为 101 与 95。空槽的谓词为 NEVER，因此规范空 bundle 为 `(31 << 128) | (31 << 101)`。四个立即数 lane 的起始位为 63、47、31、15。

ScalarY 是 6 位 selector：0–31 选择寄存器；32–45 选择 inline lane 的编码方式，包括零扩展、高位全一扩展、取高半字和两个 lane 拼成的 32 位值；46–63 是内置 selector，按当前 descriptor 的枚举名写作 `sy.NAME`。既往 BCS 研究对部分内置常量的数值说法不一，所以汇编器不把数值自动替换为这些 selector，也不从 selector 名称推出数值。

DMA 的字段与 S1 的部分编码位重叠，destination memory / core 与 outfeed queue 也共用同一段位。这些关系都由 `Bits.merge` 按位处理，不需要为 DMA 写特殊规则。

## 位布局的来源

[tpu_v4_bcs_isa_data.py](../../src/tpuasm/tpu_v4_bcs_isa_data.py) 中的字段名、字段号、oneof 结构和枚举，来自 libtpu 内嵌的 BCS program、bundle、scalar_0、scalar_1 和 isa_base descriptor。descriptor 只给出这些信息，不给出机器位布局。位布局通过原生 codec 探测得到：

1. 对每个槽内形式，先编码所有参数为零的版本作为基线；
2. 每次只改变一个字段，整数取 `1`、`2`、`-1`、`0x55555555`、`0xaaaaaaaa`，枚举遍历全部取值；
3. 与基线做 XOR，确认差异是一段连续位域，并且截断后与字段值直接对应；
4. 谓词单独探测，其余始终固定的位记为该形式的 fixed mask / value。

两个已支持的 libtpu 版本独立探测的结果一致。descriptor 中有 114 个槽内形式，其中 2 个是 Noop，其余 112 个全部登记（S0 57 个，S1 55 个）。这个数字是编解码的覆盖范围，不是设备执行的覆盖范围。仓库没有保留探测脚本；新版本的 descriptor 若有变化，需要按上述方法重新探测，并与现有表逐项比较。

## 汇编与导出

[tpu_v4_bcs_assembler.py](../../src/tpuasm/tpu_v4_bcs_assembler.py) 中，每个 `(slot, mnemonic)` 只对应一个签名。求解得到机器字后，依次进行三项核对：

1. 原生 encoder 生成的程序映像与求解出的机器字逐字节相同；
2. 程序映像通过原生往返校验；
3. 重新解码后，已占用的槽集合与源码相同。

第三项检查是必要的：某些组合编码后，某个槽会在解码时消失，例如 DMA 占用了 S1 的位，或者指令的谓词为 NEVER。这类情况必须报错，不能静默接受。

解码时，从原生 protobuf 读出的每个字段值都必须等于从映像字节中按登记位置读出的值。这项检查持续核对位布局表与 libtpu 是否一致。

exact 导出的约束候选是全部 `imm*` 和两个槽的所有字段，删减方法与 TC 相同。canonical 导出重汇编后，核对每个 bundle 的 (槽, 助记符, 操作数, 谓词) 保持不变。

## semantic protobuf 互操作

[tpu_v4_bcs_program.py](../../src/tpuasm/tpu_v4_bcs_program.py) 在 semantic protobuf 与程序映像之间提供两个方向的转换。它们处理的是已经完成 lowering、调度和链接的 `BarnaCoreSequencerProgram`。

- **`encode_tpu_v4_bcs_program`**：先在 Python 中解析 semantic protobuf，把其中**显式出现**的字段转换为要求的位，同时检查位宽、别名冲突和跨槽冲突。没有出现的字段交给原生 codec 处理，不当作要求为零。随后调用原生编码，检查每一个要求的位都保持原值，最后做往返校验。bundle 数必须是 16 的正整数倍，不替调用方补齐。Noop 与槽级辅助字段（latency、resource_usage、bit_width）可以出现，但不参与编码。
- **`decode_tpu_v4_bcs_program`**：原生解码得到 protobuf，并检查它再次编码后得到原程序映像。只保证机器映像是这对转换的固定点，不保留原 protobuf 的字段顺序、presence 或辅助元数据。

内存分配、装载、生命周期等运行时职责仍由调用方负责。本工具不限制程序长度，只要求 bundle 数是 16 的正整数倍。

## 程序容器

`extract_tpu_v4_bcs_program` 从单个 `TpuCoreProgramProto` 中沿 `barna_core(6) → sequencer(1) → pufferfish_barna_core_sequencer(10)` 提取 semantic body，要求 `sequencer_type(3)=2`，且 program alternative 唯一。在 serialized executable 中，候选记录的条件是 `platform_type(2)=1` 并含有 `barna_core(6)`。field 2 是平台类型，不是 core 种类。找到后同样沿上述路径提取，再编码为程序映像。

提取不涉及 Channel Controller，也不改写 executable。普通的 JAX 编译产物中不一定有 BCS 程序，找不到 BCS 记录时报错。

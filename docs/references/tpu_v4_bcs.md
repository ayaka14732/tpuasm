# TPU v4 BCS 汇编与互操作

`.target tpu-v4-bcs` 选择 TPU v4 BarnaCore Sequencer，与 `.target tpu-v4-tc` 并列。它有 `s0`、`s1` 两个物理槽，每个 bundle 32 字节，完整程序映像由 16-bundle／512-byte 块组成。BCS 不包括六槽的 Channel Controller。

两个目标均支持 `assemble_listing()`、`format_assembly()` 的 `exact/canonical` 模式，以及 API/CLI 程序导出。BCS 额外公开 semantic protobuf 转换和单个 core-program 提取入口。支持的 libtpu 版本与 v4 TC 相同；安装和运行不依赖 TPU。

## 最小使用

```sh
tpuasm tests/data/tpu_v4_bcs/basic.tpuasm --input-format listing --output /tmp/bcs.bin
tpuasm /tmp/bcs.bin --target tpu-v4-bcs --input-format image --output /tmp/bcs.tpuasm
tpuasm /tmp/bcs.tpuasm --input-format listing --output /tmp/bcs-restored.bin
cmp /tmp/bcs.bin /tmp/bcs-restored.bin
```

```python
from tpuasm import assemble_listing, format_assembly

source = '''.target tpu-v4-bcs
{ s0: smov s1, 0x12345678 ; s1: smov s2, 0xdeadbeef }
{ s0: srdreg s4, gtc0 ; s1: srdreg s5, gtc1 }
{ s0: shalt }
.align 16
'''
image = assemble_listing(source)
listing = format_assembly(image, target='tpu-v4-bcs')
assert assemble_listing(listing) == image
```

这些是离线表示与编码样例，没有验证调度或构造运行时服务。汇编器保留 bundle 和槽，不插入分支延迟槽或流水线等待。程序作者负责执行语义。

raw image 没有 target 标记，不能由 512 字节对齐推断执行单元。`format_assembly(image, target='tpu-v4-bcs')` 和 raw-image CLI 的 `--target tpu-v4-bcs` 必须明确提供。listing 从首行 `.target` 推断，core-program/executable 从容器中的 oneof 与 ABI 推断；有多个目标或信息不足时要求显式选择，不回落到默认目标。若为 listing/core-program 指定 `--target`，必须与内容一致。

## 语法与编码

BCS 使用公共的花括号 bundle、`slot:`、`@pN`／`@!pN`、标签、`.empty N`、`.align N` 和注释语法，见[格式参考](tpu_v4_tc.md)。完整源码的 bundle 数须为 16 的正整数倍；可以显式写 `.align 16`。标签绑定 bundle PC，相对分支以当前 bundle 为基准。Noop 通过省略槽表示，`{}` 占一个空 bundle。

[指令索引](tpu_v4_bcs_isa.md)列出 112 个非 Noop 槽内形式，覆盖两槽的 66 种非 Noop operation。常用形式包括 `smov`、整数和浮点 ALU、条件比较、分支/call、SMEM load/store、GTC、sync、DMA 与 `issue.fsm`。它们使用明确操作数和具名字段，不接受 protobuf 字段号写法、原始机器字或隐藏的原字节附件。

源操作数规则按 BCS 定义：二源 `ssub.s32`／`ssub.f32` 写 Y、X，其余二源 ALU 写 X、Y。`smov` 同时接受 `sN` 和立即数，目的操作数在前。DMA 采用显式具名字段，endpoint 枚举不替代 runtime 的地址域、allocation 和 completion 约束。SMEM 语法为 `[smem:address]`，这里指 BCS SMEM。

ScalarY 的数值立即数使用 32-bit 位型，包括零扩展、全一扩展、高半字及两个 32-bit lane pair；浮点 ALU 同样显式写位型，例如 `0x3f800000`。两个槽共享四个 16-bit lane，直接 branch/call 的地址也占用 `imm0`。求解器联合分配所有操作数，优先最少被消费的 lane，再按完整 32 字节编码的字典序选择，不依赖槽的书写顺序。

内置 selector 保留为 `sy.zero`、`sy.one`、`sy.hex_100` 等当前 descriptor 名称。descriptor 只给出这些 selector 的名称，不给出数值，所以名称只选择该编码，汇编器不将其自动换成数值。`smov s0, 1` 使用 inline immediate；`smov s0, sy.one` 选择具名 selector，两者不能仅凭名字当作同一设备数值。

`exact` 默认导出必须实际重汇编后逐字节一致；`canonical` 重新分配共享资源并核对所有可见操作数。必要的编码选择通过 bundle 内 `.encoding` 保留：

```text
{ s0: smov s0, 0x12345678 ;
  .encoding { s0.sy = 45 ; imm0 = 21 } }
```

全局约束名是 `imm0..imm3`；槽级约束是当前指令的硬件字段名，如 `s0.sy`、`s1.address`。取值是该字段的无符号编码值。不能用任意位偏移、protobuf 字段号或未登记名称。约束与可见操作数冲突时拒绝汇编；exact 导出按名称顺序反复删去仍可精确重建的冗余约束。

DMA 使用与 S1 重叠的字段，不意味着可以任意与 S1 指令组合。汇编器检查物理位冲突，并拒绝原生编码后被消隐或改变的指令；成功汇编仍不证明硬件调度合法。

## semantic protobuf 接口

```python
from tpuasm import assemble_listing, decode_tpu_v4_bcs_program, encode_tpu_v4_bcs_program, format_assembly

# semantic_bytes 是已经完成 lowering、调度和链接的 BarnaCoreSequencerProgram。
image = encode_tpu_v4_bcs_program(semantic_bytes)
source = format_assembly(image, target='tpu-v4-bcs')

# 手写或编辑后的清单也可以转换回 semantic protobuf。
edited_image = assemble_listing(edited_source)
edited_semantic = decode_tpu_v4_bcs_program(edited_image)
assert encode_tpu_v4_bcs_program(edited_semantic) == edited_image
```

`encode_tpu_v4_bcs_program(bytes) -> bytes` 接受完整的 `BarnaCoreSequencerProgram`，即 repeated field-1 bundle，不接受外层 core-program。要求正整数个 16-bundle 块；不替调用方补齐 fragment。它检查已提供机器字段的位宽、跨槽共享位和 codec 往返，不允许 encoder 静默覆盖冲突值。

`decode_tpu_v4_bcs_program(bytes) -> bytes` 返回当前后端的 semantic protobuf，并验证重新编码恢复原机器映像。它不保证原 semantic protobuf 的字段顺序、presence、latency、resource map 或其他辅助元数据保持不变。运行时的 program identity 若依赖 protobuf bytes，需要按新产物重新计算。

接口保证的是机器字节往返。内存分配、装载和生命周期等运行时信息不在机器清单中，仍由调用方保留并核对。程序长度只须是 16 个 bundle 的正整数倍，运行时对长度的限制不属于 ISA 编码，本工具不引入。

命令行对应入口：

```sh
tpuasm /tmp/bcs.pb --target tpu-v4-bcs --input-format semantic-proto --output /tmp/bcs.tpuasm
tpuasm /tmp/bcs.tpuasm --input-format listing --output-format semantic-proto --output /tmp/edited.pb
tpuasm /tmp/bcs.bin --target tpu-v4-bcs --input-format image --output-format semantic-proto --output /tmp/decoded.pb
```

## 程序容器提取

`extract_tpu_v4_bcs_program(core_program: bytes) -> bytes` 从单个 `TpuCoreProgramProto` 提取 semantic bytes。已核对路径是 `barna_core(6) → sequencer(1) → pufferfish_barna_core_sequencer(10)`，并检查 `sequencer_type=2` 和唯一 program alternative。它只解析容器，不加载 libtpu。不能把单个 protobuf 当作长度分隔的 serialized executable：

```sh
tpuasm /tmp/core-program.pb --target tpu-v4-bcs --input-format core-program --output /tmp/core-program.tpuasm
tpuasm /tmp/executable.bin --target tpu-v4-bcs --input-format executable --output-dir /tmp/bcs-programs
```

`executable_programs(serialized, target='tpu-v4-bcs')` 返回 `(record, index, image)`；每个已识别 BCS core record 的 index 为 0，record 按整个容器编号。BCS semantic body 需要调用目标 codec 才能得到 image，这与 TC 直接提取 code segment 的行为不同。`dump_executable()`／`dump_compiled()` 接受相同 target，输出 `program-tpu-v4-bcs-<record>-0.tpuasm`；TC 使用对等的 `program-tpu-v4-tc-<record>-<index>.tpuasm`。容器能够唯一确定目标时，这些 API 可以省略 target。

容器提取覆盖包含该 semantic body 的记录，不是完整 executable 验证器，也不重写 executable。普通 JAX 编译产物不一定包含 BCS 程序；没有 BCS 记录时明确报错。BCS 编译来源捕获和 source map 尚未适配：默认只导出机器清单，`--source-map-json` 明确拒绝，既有 TC 来源功能保留。

## 验证与复现

从仓库根目录在匹配环境中运行：

```sh
python tests/reproduce_tpu_v4_bcs.py
python tests/reproduce_tpu_v4_tc.py
```

BCS 入口使用 `tests/data/tpu_v4_bcs/` 中的两份文本样例，检查 exact/canonical、三对编码歧义、共享立即数、分支、双槽 GTC、semantic 固定点、合成容器提取和非法输入拒绝。

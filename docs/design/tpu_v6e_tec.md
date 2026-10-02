# TPU v6e TEC 目标

`tpu-v6e-tec` 是 TPU v6e SparseCore 向量子核（TEC）的程序。它复用 v6e TC 的源码语法、共用位模型、指令包求解器和可逆导出，硬件表示与 v6e TC 相近。不同的是，libtpu 没有 TEC 的 formatter，助记符和操作数写法改从编译 SparseCore kernel 所用的 LLVM TPU printer 取得。本文说明与 v6e TC 的差异、字段表与指令语法如何从 libtpu 生成，以及哪些结论已经核对。

目前只支持 libtpu 0.0.49（CPython 3.14t）。字段表的生成与离线核对在没有 v6e 设备的机器上完成；示例和 selector 探针另在 v6e-1 上执行，见[验证](#验证)。

## 硬件表示

| 项 | v6e TensorCore | v6e TEC |
|---|---|---|
| 程序映像 | 512 字节块，8 个连续的 64 字节 bundle | 64 字节块，每块 1 个 bundle，bundle 数不限 |
| 物理槽 | 15 个 | 12 个：`s0 s1 dma misc va0 va1 va2 vld vst stream vr vx` |
| 共享立即数 | `imm0..imm5`，各 20 位 | 相同 |
| 标量操作数槽 | `vs0..vs3`，各 5 位 | 相同，另有每槽 1 位的使用标记 |
| 谓词 | 4 位；只有 `s0`、`s1`、`dma` 有取反位 | 4 位；每个槽都有取反位 |
| 互斥 | `dma` 与 `s0`、`s1` | `dma` 与 `stream` 互斥，二者都与 `s0`、`s1` 互斥 |

libtpu 的 ISA protobuf 为 `asic_sw.deepsea.gxc.glc.isa.SparseCoreTecBundle`：`s0`、`s1` 是标量子 bundle 中的两个字段，与 `dma`、`stream` 同属一个 oneof；共享立即数与标量操作数是 bundle 级的两条消息。编解码入口是 `tpu::DecodeProgram<…SparseCoreTecProgram, ghostlite::isa::DecoderGlSparseCoreTec>` 和对应的 `EncodeProgram`，program 对象布局与 TC 相同，公共桥接 `native.cc` 只需换常量；没有 formatter，所以后端定义 `TPUASM_CODEC_ONLY`。

TEC 的 encoder 拒绝超出字段位宽的整数，而 TC 的 encoder 静默截断；有些字段被零扩展写入更宽的位段，例如 VALU 的 `x` 读取 5 位掩码寄存器号时写满 6 位。字段探测把这些始终写 0 的高位记为固定位。

Pallas SparseCore kernel 的 SCS 与 TEC 代码由 TC 程序记录携带，放在一个 DATA segment 中，TEC 段的起点只由 SCS 代码给出，容器中没有记录。tpuasm 用原生编解码校验定位 TEC 段（见[总体架构](architecture.md#硬件目标与程序容器)），支持从 executable 导出 TEC 程序和等长写回，不支持插入 bundle。

## 字段表的生成

[tools/generate_tpu_v6e_tec_isa.py](../../tools/generate_tpu_v6e_tec_isa.py) 从已安装的 libtpu 生成 [tpu_v6e_tec_isa_data.py](../../src/tpuasm/tpu_v6e_tec_isa_data.py)，不要手工修改生成的文件。字段位置、固定位、互斥槽与随机核对和 v6e TC 相同，由两个目标共用的 [ghostlite_isa.py](../../tools/ghostlite_isa.py) 完成（见 [v6e TC 的字段表生成](tpu_v6e_tc.md#字段表的生成)）：965 个“槽×形式”组合登记，另有 encoder 以任何操作数都拒绝的组合写入 `REJECTED`。

指令语法来自编译器自身的两个部件。libtpu 编译 SparseCore kernel 时先得到 LLVM MCInst，再在 `xla::tpu::sparse_core::isa_emitter::utils::ConsumeBundle` 中把每个 bundle 的 MCInst 转成 `SparseCoreTecBundle`；编译器的文本 dump（`--xla_sc_dump_bundles_to`）由同一批 MCInst 经 `TPUInstPrinter` 打印。[tools/tec_llvm.cc](../../tools/tec_llvm.cc) 在子进程中构造 MCInst，同时交给 printer 和 emitter，得到“打印文本 ↔ 写入的字段”的对应。所用函数地址登记在 TEC 原生后端文件中，桥接层本身不使用它们；MCInst、MCOperand、`TPUMCImmExpr` 和 LLVM 表的布局写在 `tec_llvm.cc` 开头。emitter 遇到不支持的操作数时会以 CHECK 失败终止进程，工具在 fork 出的子进程中逐批处理，并从下一条继续。

工具依次做以下几步：

1. **找出每个 opcode 可用的槽。** 对每个 LLVM opcode 先以 `s0` 尝试，emitter 的报错指出所需槽位或不支持的操作数，工具据此改换槽位、寄存器、立即数与 selector 表达式，直到 emitter 接受且只占一个槽。stream 与 DMA 指令的操作数 0 是模式字（两端的地址空间、方向、circular buffer 等），从 0 出发搜索不到编译器的用法，所以以编译器写出的模式字为起点，它们取自编译 SparseCore kernel 时交给 emitter 的 MCInst。每类指令有两个起点（stream 为 TileSPMEM 与带偏移寄存器的 HBM、TileSPMEM 与 SPMEM；DMA 为 HBM 到 TIMEM、SPMEM 到 HBM），各自搜索其余操作数。
2. **展开改变写法的立即数。** 某个立即数操作数若改变打印文本的样子（助记符、显示位置的个数或地址空间名），就按取值分成独立的指令，例如 `stream.linear.gather.add.f32` 与 `dma.local` 的各种地址空间组合。模式字只逐段改变单独翻转时会改变写法的相邻位。以下取值丢弃：写入的字段值超出字段位宽的，因为 encoder 会拒绝它们；改变了 emitter 所选形式的，例如 `stream.strided` 的 opcode 配上 indirect 的模式位，助记符与编码不一致；只在地址空间名后加编号的，这个编号来自存储器编号操作数，emitter 把它加到 2 位的存储器类型上，结果没有意义。
3. **探测操作数。** 逐个改变每个操作数：寄存器换成同类的其他寄存器并改变经由的 `vsN`，selector 表达式扫描编码与取值，立即数取若干数值与全部单个 bit，记录 emitter 写入的字段与打印文本。由此判定操作数是寄存器号、`vsN` 中的标量寄存器、selector、共享立即数、直接数值，还是按位拆入多个字段的配置字（例如 stream 最后的配置操作数）。改变取值时打印文本不变的操作数不出现在语法中，它写入的字段由编码约束表达；emitter 接受的取值都不改变字段的立即数（例如只能为 0 的 HBM 编号）同样不出现。
4. **对齐打印文本。** 给每个操作数一个互不相同的取值再打印一次，在文本中找到各操作数的位置，得到助记符、目的在前的操作数顺序，以及地址等复合写法的模板（如 `[hbm4b:{0}+{1}]`）。
5. **整理。** 操作数不控制、但由 opcode 决定的字段作为固定字段；枚举字段（地址空间、selector 等）在不由操作数控制时一律固定，避免一种写法匹配另一种地址空间的编码。编码相同的多种写法中，与编译器 MCInst 最接近、数值按整数显示的写法排在前面，解码取第一个；不同 LLVM opcode 给出的同编码写法（如 `simm.s32` 与 `simm.f32`、`vimm.s32` 与 `vimm.bf16`）都保留供汇编使用，32 位整数类型的写法排在前面；形状展开得到的写法只在带来新编码时保留，所以不影响编码的地址空间名等不会成为别名。

当前 2227 条指令覆盖 856 个“槽×形式”组合。其余 109 个组合没有 printer 写法，它们对应的 LLVM opcode 都没有找到 emitter 接受的 MCInst：工具没有找到的 LLVM opcode 中，1641 个在每个槽都被 emitter 报告不支持（例如 u16 比较、bf16 与 8 位或 4 位整数之间的转换、`call`），99 个是 LLVM 伪指令，DMA 的 strided、general 与 host 形式则对工具能构造的操作数都报告 HBM 编号非零。另有 `yieldable_sync_*` 等形式按名称在 LLVM 中找不到对应的指令。这些组合使用具名字段写法，见[签名规则](#签名规则)。工具运行约七十五分钟。libtpu 换版本后需要重新运行并比较生成结果，步骤见 skill [add-libtpu-backend](../../.agents/skills/add-libtpu-backend/SKILL.md)。

## 共享字段与命名约束

与 v6e TC 相同，共享立即数和标量操作数槽每个指令包都写出，求解器把这些位视为始终可写，selector 读取它们时计入求解代价。标量操作数槽的使用标记 `vsN_used` 随读取该槽的操作数一起写入，源码不单独写。

约束名为 `imm0..imm5`、`vs0..vs3`、`<slot>.<descriptor 字段名>` 和 `<slot>.form`，取值规则与 v6e TC 相同。TEC 不像 TC 那样为跨槽共享的字段另起 `port.*` 名称：`vx` 等槽读取其他槽源寄存器的字段（如 `v0_x`、`vst_source`）与那些槽的字段同位，约束名仍是 `<slot>.<字段名>`，同一指令包中各槽对同位字段的取值必须一致，由求解器按位合并检查。

规范编码在读取的共享槽数相同时，优先使用编号小的立即数和标量槽。示例中编译器把 TileSPMEM 地址偏移放在 `imm4`、`imm5`，exact 导出用约束保留这种分配。

## 签名规则

[tpu_v6e_tec_isa.py](../../src/tpuasm/tpu_v6e_tec_isa.py) 把生成的描述转为签名。助记符与操作数顺序取 printer 的写法，与 v6e TC 一致的转换有：目的在前，目的与源合并为一个操作数列表；`(pc)`、`(tag)`、`(tm)` 这类由助记符确定的目的不写出；单个隐含寄存器去掉括号与编号；数值去掉 `$` 前缀；谓词写在槽名之后，而 printer 写在助记符之后。

selector 的数值含义沿用 v6e TC：`zero_immN` 为零扩展的 20 位立即数，`ones_immN` 为 `0xfff00000 | immN`，`immN_zero` 为 `immN << 12`，`immH_immL` 拼接两个字段的低 16 位，其余为内置常量。printer 把浮点运算的立即数显示为浮点数，tpuasm 同样按 `f32` 显示。

寄存器号字段的取值范围可能大于实际的寄存器数，多出的编码只能用具名字段写法表示。

每个形式另有一种具名字段写法，助记符是 descriptor 中的形式名，各字段按字段顺序写成 `字段=值`，例如 `misc: yieldable_sync_done x=0x1`。selector 能写成数值时写数值，否则写取值名。没有 printer 写法的组合，以及 printer 写法表达不了的编码（例如保留编码、超出寄存器数的寄存器号），解码时都使用这种写法。

## 限制

- 设备上核对了 VALU 与标量 ALU 的全部立即数和常量 selector、四个标量操作数槽和一条具名字段写法（见[验证](#验证)）；stream 配置字各 bit、其余形式的具名字段写法和 stream/DMA 的其他地址空间组合没有在设备上执行。
- 不支持插入 bundle 和编译来源映射；executable 中 TEC 段的定位依赖“SCS 段不能按 TEC 解码”，已在本文提到的全部编译样例上成立。
- stream 对端的存储器类型由 emitter 从模式字和存储器编号操作数共同算出。编译样例中出现过 `hbm4b` 与 `spmem`，二者有编译器写法；其他存储器类型（例如带编号的 TileSPMEM）没有编译样例，按具名字段写法表示。
- DMA 的 strided、general 与 host 形式，emitter 对工具能构造的操作数都报告 HBM 编号非零，没有 printer 写法。
- 生成工具依赖 libtpu 内部 LLVM 对象的布局与 emitter 的报错文本，换版本后可能需要调整搜索规则。

## 验证

[tests/reproduce_tpu_v6e_tec.py](../../tests/reproduce_tpu_v6e_tec.py) 离线检查：只有 `.encoding` 才能区分的编码对；每个槽中每个形式的一个随机实例能 exact 往返，canonical 输出稳定；样例源码和一个编译得到的完整程序映像能往返；以及 `p14`、互斥槽等拒绝路径。

[examples/pallas/](../../examples/pallas/) 中的 `sc_*.py` 是 SparseCore kernel，覆盖 linear、strided 与 indirect stream，HBM 与 SPMEM 两种对端，16 个子核经 SPMEM 汇总后屏障同步，不展开的循环，以及 TEC 程序的写回（`sc_replace_program`）。`run_all.sh --aot tpu-v6e-tec` 按 `v6e:2x2` 拓扑离线编译，用 `dump_compiled(..., target='tpu-v6e-tec')` 导出为 [tpu_v6e_tec/](../../examples/pallas/tpu_v6e_tec/) 中的清单，再逐个 bundle 与编译器 SparseCore bundle dump 中 LLVM TPU printer 的文本比较：比较时去掉 `$` 和空白，数值统一为 32 位十六进制，分支目标统一为 bundle 编号，同一 bundle 内的指令不计顺序。这项比较是 tpuasm 写法与编译器写法一致的直接证据；导出过程本身也包含重汇编与逐字节比较。

七个 SparseCore 示例在 v6e-1 上运行，数值检查通过；在设备上编译得到的 TEC 清单与按 `v6e:2x2` 离线编译的逐字节相同，与编译器 dump 的比较同样通过。`sc_replace_program` 在同一进程中依次执行原程序、写回后的程序和原程序，得到加一、加二、加一，说明写回时更新的程序身份让 runtime 装载了新程序。

[tests/reproduce_tpu_v6e_tec_execution.py](../../tests/reproduce_tpu_v6e_tec_execution.py) 在设备上核对 selector 的数值含义：以一个给每个元素加常数的 kernel 为载体，导出 TEC 程序，把其中的 `vadd.s32` 换成待测写法，用 `.encoding` 固定待测的 selector，写回后执行，并与 tpuasm 对该 selector 的数值模型比较。71 个案例全部一致：VALU y 源的全部立即数与常量 selector（`zero_immN`、`ones_immN`、`immN_zero`、`immH_immL` 与内置常量），标量 ALU y 的全部立即数与常量 selector（经 `simm.s32` 写入标量寄存器，再由向量加法读取），经 `vs0..vs3` 读取标量寄存器，以及一条具名字段写法。

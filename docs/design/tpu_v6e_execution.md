# TPU v6e 指令执行语义核对

本页记录 TPU v6e 指令的设备执行语义及其与 libtpu formatter 显示的差异。实验使用单芯片 TPU v6e、CPython 3.14t、jaxlib 0.11.2、JAX `886d2370c1` 和 libtpu 0.0.49（build-id `97e27df7268da25ab03e455e30dd86b0`）。结论适用于这个组合，不由 v4 的运行结果外推；[v4 的 Delay 对照](#v4-的-delay-对照)一节单独在本机 TPU v4 上测得。

## 实验方法

Pallas 只负责编译一个输入 DMA、向量运算、输出 DMA 的载体。先执行原程序，检查结果，再以 `insert_executable_bundles` 在向量运算前预留空 bundle；每个探针通过 `replace_executable_programs` 写入该区域，用 `load_executable` 装载执行。输出由原程序的 store 和 DMA 返回。各条有依赖的指令之间留空 bundle，不依赖编译器替探针重新调度。

探针区使用载体 kernel 内未占用的标量和向量寄存器；地址探针的 TC VMEM 由载体的 scratch Ref 分配。每个案例保存汇编片段、完整设备返回值及摘要。数值模型在 host 上计算；动态 unpack 另与已有静态 unpack 指令对照。机器字能经 encoder、decoder、formatter 往返，只作为编码检查，不作为执行语义的证据。

## 立即数拼接：formatter 与设备不同

ScalarY 和 VectorY 的 `immH_immL` 实际结果是：

```text
((immH & 0xffff) << 16) | (immL & 0xffff)
```

两个字段各有 20 位，但这个 selector 只读取各自低 16 位。FormatterGl 则显示 `((immH << 16) | immL) & 0xffffffff`，把低槽的高 4 位也按位或进结果。例如 `imm0=0xa1357`、`imm1=0x5b246`，formatter 显示 `0xb24e1357`，设备返回 `0xb2461357`。`imm3_imm2` 和向量独有的 `imm5_imm4` 也使用低 16 位拼接。

tpuasm 按设备语义解释这类操作数：数值表达式为每个来源字段登记有效位宽，拼接形式限定为 16 位。各字段的高 4 位不参与该操作数的数值计算，可由 `.encoding` 保留，exact 导出不丢失原字节。

零扩展 `zero_immN`、补一 `ones_immN`、左移 `immN_zero` 的结果分别为 `immN`、`0xfff00000 | immN`、`immN << 12`。ScalarY 与 VectorY 的所有内置常量分别经标量移动后广播、向量移动直接返回，避免只核对 formatter 文本。

## 动态 unpack 的标量槽

`dynamic_vector_unpack.vs` 的两位编码选择 `vs0..vs3`，再由该共享槽的五位字段指定标量寄存器。它不是直接读取 `s0..s3`。探针分别把四个槽指向 `s20..s23`，给寄存器写入 unpack 索引；六种可反汇编的格式、全部合法半部索引，均与相应静态 unpack 返回值逐位一致。

源码写作 `vunpack.vsel.c.bf16 v10, v0, s20`，三个操作数依次为目的、输入和实际标量寄存器。`packing_format=6,7` 会使 formatter 终止进程，不在本实验的覆盖范围内。

## encoder 拒绝的 81 个槽内形式

复核包含全零、逐字段非零值、枚举的全部非保留值、整数边界、全字段非零组合，以及每个形式 128 组固定种子的随机组合，共 13,101 个请求。81 个槽内形式全部仍被拒绝。对每个形式，另在 descriptor 提供的合法槽进行正对照，所有正对照都被接受；文末的 encoder 复现命令会将逐项结果写入 `/tmp/tpuasm-v6e-rejections.json`。

进一步检查当前二进制的六个 `TensorCore{Scalar,Vector}AluNEncoder::Encode`：它们先写谓词，再按消息 `+0x50` 的 oneof 分支编号分派。下表中的 81 个分支全部直接走错误路径，不进入操作数字段编码。错误路径调用 `proto2::ShortFormat` 构造诊断，再构造 InvalidArgument 状态。表中的地址是本构建的 ELF VA；读取跳表时按所属 `PT_LOAD` 换算文件偏移。

| 槽 | encoder 入口 | 跳表 VA（按分支编号范围） | 拒绝路径 VA |
|---|---|---|---|
| s0 | `0x1b56e530` | 0…73：`0x9c6bd70` | `0x1b56eb1c`；6 项均超过上界 |
| s1 | `0x1b57cd10` | 0…79：`0x9c6be98` | `0x1b57cfa7`；12 项均直接落入该路径 |
| va0 | `0x1b59faa0` | 0…131：`0x9c6bfd8`；132…211：`0x9c6c1e8` | `0x1b5a08d9`；19 项均直接落入该路径 |
| va1 | `0x1b5cb560` | 0…131：`0x9c6c328`；132…189：`0x9c6c538` | `0x1b5cc549`；22 项均超过上界 |
| va2 | `0x1b5f5aa0` | 0…131：`0x9c6c620`；132…211：`0x9c6c830` | `0x1b5f68d9`；19 项均直接落入该路径 |
| va3 | `0x1b623440` | 0…131：`0x9c6c970`；132…195：`0x9c6cb80`；196…208：`0x9c6cc80` | `0x1b624548`；3 项均超过上界 |

这证实它们是 **libtpu 0.0.49 encoder 的槽位限制**，不会因换一组操作数就被接受。encoder 没有为这些组合产生机器字，因而没有把“被拒绝”冒充设备执行结果，也不据此宣称硬件保留编码绝无其他含义。

## 其余 selector 的设备结果

以下 selector 的设备结果用于定义操作数签名。表中的槽号指共享操作数槽，不等同于标量寄存器号。

| selector | 探针与结果 |
|---|---|
| Operand | `set_sync` 的 0、六个立即数槽、四个标量槽分别写入 sflag 100，再读回。20 位立即数 `0xabcde` 返回 `0xfffabcde`；标量路径保留完整的 `0x12345678`。 |
| VectorSource | `read_sync` 的 selector 0 与显式 sflag 0 比较；四个标量槽与六个立即数槽均指向 sflag 100，读回预先写入的 6543。 |
| VectorOffset、VectorBase | 六个立即数偏移和四个标量基址都指向 TC VMEM 第 64 行，返回第 64…71 行。 |
| SublaneMask | 内置 `0xff, 1, 0x0f, 0xf0, 3, 16` 分别控制读入的 sublane；未选中的目的 sublane 被清零，不保留旧向量寄存器内容。 |
| VectorStride | 单位、负一、二、四步长分别返回 `VMEM[64 + sublane * stride]`。八、十六步长另用单 sublane 掩码探测，避免多 sublane 同时访问同一 bank。 |
| VectorShuffle | 八个 `descending_pattern_N` 返回 `VMEM[64 + (sublane + N) % 8]`；三个双立即数 selector 均只拼接各字段的低 16 位。 |
| SourceSpecifier | lane rotate 的内置常量 1、64、48、32、16、8 与对应右旋结果一致。 |
| TransposeMatrixWidth | 内置宽度 8、16、32、64、128 分别与立即数宽度对照，逐次弹出转置结果，并与 host 转置比较。 |

直接以全部 sublane 启用的八步长做向量读取，会触发设备的 `bank_conflict_stride_vld0` 错误。这个失败不能用来把常量八解释成别的数值；它说明地址、步长和 sublane 掩码还必须共同满足 VMEM 访问约束。

### 延迟常量

VectorDelay 的 selector 0 在 descriptor 中叫 `OPERAND_ONE`，FormatterGl 却显示 0。用 `srdreg.lcclo` 读前后计数，并在后一次读数之前通过 vector FIFO 读回建立屏障，比较连续 1、16、64、128 条 delay。每组采样 20 次，取最小值减轻队列及运行时抖动。以下是最后一次完整对照的周期读数最小值；固定开销包含探针中的屏障与空 bundle，不能把绝对值当成 delay 本身的成本。

| delay 条数 | selector 0 | 立即数 0 | 立即数 1 | 立即数 2 | 立即数 16 |
|---|---|---|---|---|---|
| 64 | 1321 | 1257 | 1323 | 1385 | 2282 |
| 128 | 1451 | 1322 | 1450 | 1577 | 3370 |

selector 0 与立即数 1 接近；相对立即数 0 多出的周期数约等于 delay 条数。其他立即数也表现出相应的线性增量。tpuasm 把这个 selector 显示为 `vdelay 1`，与 descriptor 及设备行为一致；`vdelay 0` 通过立即数 selector 表达。编码歧义回归检查内置一和立即数一具有相同 canonical 文本，同时 exact 导出保留各自机器字。

周期检查允许差值有 16 周期的误差，目的是区分这里的常量语义，不是测量流水线延迟或给出调度模型。可用 `--group delays` 单独复现，完整 400 次读数写入输出目录的 `delay-timings.json`。

### v4 的 Delay 对照

本机 v4 也用相同的 Pallas 载体和 `insert_executable_bundles` 核对了 `Delay` 的 selector 0。计数起点前额外从 vector FIFO 读回：如果只在终点读回，标量计数器可能先于向量队列到达探针，把载体输入 DMA 的等待计入结果，抖动足以淹没要比较的差值。

加入起点同步后，各组 20 次读数相同。以下是较长序列的结果：

| delay 条数 | DelayFixed 1 | selector 0 | 立即数 0 | 立即数 1 | 立即数 2 | 立即数 16 |
|---|---|---|---|---|---|---|
| 64 | 174 | 174 | 115 | 174 | 238 | 1134 |
| 128 | 302 | 302 | 179 | 302 | 430 | 2222 |

selector 0 与立即数 1 完全一致，`DelayFixed` 的一周期形式也得到相同结果。序列从 64 条增至 128 条时，立即数 0、1、2、16 的增量分别为 64、128、192、1088 周期；selector 0 的增量也是 128。因此 v4 与 v6e 都写作 `vdelay 1`，v4 的命名约束为 `misc.delay_count = const(1)`。很短的序列仍可能被固定流水线开销遮住，单条 delay 的相同读数不能证明常量相同。

v4 复现脚本记录 24 组对照的原始 480 次读数与环境。本机为多 host v4 环境，复现只使用当前 host 的四颗芯片，不初始化分布式运行：

```bash
TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 \
PYTHONPATH=src python tests/reproduce_tpu_v4_delay.py --output /tmp/tpuasm-v4-delay
```

规范编码可以选择内置一；精确导出则用 `misc.opcode = 15` 保留编译器使用的固定延迟形式。

脚本将探针片段和 `delay-timings.json` 保存到 `/tmp/tpuasm-v4-delay`。检查过程先执行原始载体，再执行插桩程序，最后再次执行原始载体。

## formatter 无文本的 15 个形式

这 15 个形式均已实际放入载体并执行；它们仍使用 tpuasm 自定的名称。下面按可观察结果说明操作数。浮点探针比较原始 bit pattern，避免 host 对 NaN 的规范化掩盖设备行为。

### 重映射与 packed 算术

`vremap.8x128.f32 dest, x, y` 先按 `x` 的 bit pattern 分类，再从同一位置的 `y` 读取一个 4 位操作码。七类从低 nibble 到高 nibble 依次是：负无穷、负的正规有限数、负零或负 subnormal、正零或正 subnormal、正的正规有限数、正无穷、NaN。`y` 的最高 4 位不参与这个选择。bf16 形式对一个 32 位 word 中的两个 bf16 独立分类，两半共用该 word 的 `y`。

| 操作码 | 输出 |
|---|---|
| 0 | 保留输入位模式 |
| 1、2 | 分别为正零、负零 |
| 3、4 | 分别为 +1、−1 |
| 5、6 | 分别为负无穷、正无穷 |
| 7 | 翻转输入符号位 |
| 8 | quiet NaN：f32 为 `0x7fc00000`，bf16 为 `0x7fc0` |
| 9、10 | 分别为最负有限值、最大正有限值 |
| 11 | `sign_bit - 1`，即 f32 `0x7fffffff` 或 bf16 `0x7fff` |
| 12 | 负零，与操作码 2 相同 |
| 13 | 清除输入符号位 |
| 14 | 原始位模式 1 |
| 15 | 全一位模式 |

复现包含 `y=0`、32 个单 bit、全一、16 个重复 nibble，以及固定种子 4051 的随机输入与逐 word 随机控制值；随机用例覆盖 `va0..va3`，并显式加入正负 subnormal。两种格式都与 host 的整数位运算模型一致。

`vmul.8x128.u16 dest, x, y` 对两个 packed 无符号 16 位半字分别相乘，每半保留低 16 位。`vc.8x128.u16 mask, x, y` 对应每半无符号加法的进位；通过 `vsel` 把 mask 转换成每半 `0xffff` 或 0 读回。`vweird.8x128.bf16 mask, x` 对每半判断指数位是否全一，即无穷或 NaN，同样通过 `vsel` 读回。三者均与独立的 host 整数模型比较。

### packed lane 操作与 PCR 装载

`vrot.lane.packed.b8.8x128 x, amount` 和 `vbcast.lane.packed.b8.8x128 x, amount` 的结果进入对应 TRF，随后用 `vpop` 取出。设 sublane 为 `s`，word 内从低到高的 byte 为 `b`，则探测到的位移是：

```text
shift(s, b) = (amount & 255) + (4*s + b) * ((amount >> 8) & 255)
```

旋转时，目的 lane `l` 的该 byte 来自源 lane `(l - shift) % 128`；广播时来自源 lane `shift % 128`。11 个 amount 包含 0、1、3、8、127、128、255、511、`0x102`、`0x203`、`0x7f02`，从而同时区分旋转方向、模 128、两个字段与 sublane 相关的变化。未据此推断 amount 第 16 位以上的保留用法。

`vsetperm.half.u8 v11` 之后执行 `vperm.lane trf0, x`：在本次所有 lane 使用同一控制 word 的实验中，结果低 16 位取源 lane `control & 127`，高 16 位取源 lane `(control >> 16) & 127`。源数据的每个 sublane 和 byte 都不同，五种控制值分别为 0、1、`0x03020100`、`0x07060504`、`0x7f7e7d7c`。这确认了 half 形式的两个半字来源；没有把统一控制值实验扩展成对任意非均匀 PCR 装载规则的结论。本次 lane 探针使用 `vx0/trf0`。

### 七种带 yield 的等待形式

探针使用载体没有占用的 sflag 100，写入已满足条件的状态，执行 `vwait.eq/ne/gt/ge/lt/done/notdone.yield` 后返回输入。五种数值条件的 `(实际值, 阈值)` 分别为 `(7,7)`、`(7,8)`、`(7,6)`、`(7,7)`、`(7,8)`，与源码操作数顺序一致。

通过 `vsyncdonemov` 单独读回 done 状态：对 −1、0、1、7，`vsyncset.done.s32` 后读到 0，`vsyncset.doneinv.s32` 后读到 1。因而 done 等待的准备操作使用 `doneinv`，notdone 使用 `done`。

这七个案例确认编码可执行、flag 与阈值的读取路径，以及已满足条件后的继续执行。它们**没有验证条件不满足时的线程让出、其他线程唤醒和重新调度**，也不测量等待延迟。需要研究 scheduler 时应另建有界的多线程实验。

## 复现与证据

在上述版本的单芯片 v6e 环境中，从仓库根目录运行：

```bash
PYTHONPATH=src python tests/reproduce_tpu_v6e_execution.py --output /tmp/tpuasm-v6e-execution --archive /tmp/tpuasm-v6e-archive
```

`--group` 可单独选择 selector、算术、unpack、访存、XLU 或同步类案例。输入与返回值保存为 `.npy`，探针保存为 `.tpuasm`，`results.jsonl` 记录版本与逐例断言结果；最后再次执行未修改的载体，确认它仍返回原始基准结果。脚本直接操作真实设备，不使用 Pallas Interpret Mode，也不进入 CPU CI。

可选的 `--archive` 在成功运行后将输出打包为 `execution.json`、`execution.npz` 和 `execution-delay-timings.json`，保存到指定目录。JSON 的环境字段在设备运行时自动记录，包括 Python ABI、JAX、jaxlib、libtpu 版本和 backend build-id；打包时只读取这些原始记录，不把打包机器的版本补进设备结果。NPZ 按案例名存放输出，重复输入去重；周期 JSON 保留原始读数，只统一文件名。原始输出与打包文件均保留在 `/tmp`，不纳入仓库。

已有完整输出目录时，可在没有 TPU 的机器上重新打包：

```bash
PYTHONPATH=src python tests/reproduce_tpu_v6e_execution.py --output /tmp/tpuasm-v6e-execution --archive-only --archive /tmp/tpuasm-v6e-archive
```

打包不会重新执行案例，也不会把失败的 `match` 改成成功。最终基准检查由运行脚本写入末尾记录；没有这条记录时，归档中的 `original_executable_baseline_passed` 为 false。

完整数值运行包含 361 个案例，全部通过；随后补充的 20 组延迟周期对照也通过。两次运行最后均重新检查了未修改载体的基准。复现脚本保存原始输入、设备返回值及逐例汇编；若选择打包，NumPy 压缩包中以案例名索引输出，以 JSON 的 `input` 字段索引输入，共享的输入仅保存一份。

encoder 复核不需要设备：

```bash
PYTHONPATH=src python tools/reproduce_tpu_v6e_encoder_rejections.py --output /tmp/tpuasm-v6e-rejections.json
```

该检查只在 libtpu 0.0.49 的 CPU CI job 中运行。常规离线回归另验证 1214 个槽内形式、11 对编码歧义、完整程序映像和拒绝路径；其中一对 `vtrace` 编码核对按 formatter 登记的重叠位模型，不把它当成已经验证的设备 trace 语义。23 份 v6e 示例清单在离线导出时逐字节重汇编比较。

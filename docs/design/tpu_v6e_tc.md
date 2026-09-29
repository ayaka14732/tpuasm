# TPU v6e TC 目标

`tpu-v6e-tc` 是 TPU v6e（libtpu 内部代号 Ghostlite，缩写 GL）的 TensorCore 程序。它复用 v4 TensorCore 的源码语法、共用位模型、指令包求解器和可逆导出，只有硬件表示、字段表和签名规则不同。本文说明这些不同之处、字段表如何从 libtpu 生成，以及哪些结论已经核对、哪些仍是推断。

目前只支持 libtpu 0.0.49（CPython 3.14t）。编码、解码与字段表的核对都在 TPU v4 机器上完成，v6e 的 executable 由 JAX 按 `v6e:2x2` 拓扑离线编译得到；单芯片示例另在 v6e-1 上运行，见[验证](#验证)。

## 硬件表示

| 项 | v4 TensorCore | v6e TensorCore |
|---|---|---|
| 程序映像 | 512 字节块，10 个 51 字节 bundle 加分隔字节 | 512 字节块，8 个连续的 64 字节 bundle，没有分隔字节 |
| 机器字 | 408 位，由 libtpu 负责块内编排 | 512 位，就是映像中对应的 64 字节 |
| 物理槽 | 12 个 | 15 个：`s0 s1 dma va0 va1 va2 va3 vst vld0 vld1 misc vx0 vx1 vr0 vr1` |
| 共享立即数 | 在各形式中以别名字段出现，16 位 | bundle 级消息中的 `imm0..imm5`，各 20 位 |
| 标量操作数槽 | `vs0..vs2` | bundle 级消息中的 `vs0..vs3`，各 5 位 |
| 谓词 | 5 位，NEVER 为 31 | 4 位：0..13 为 `p0..p13`，14 为无条件，15 为 NEVER；`s0`、`s1`、`dma` 另有 1 位取反 |
| executable ABI | `version = 3`（PUFFERFISH） | `version = 5`（GHOSTLITE） |

libtpu 的 ISA protobuf 为 `asic_sw.deepsea.gxc.glc.isa.TensorCoreBundle`：`s0`、`s1` 是标量子 bundle 中的两个字段，与 `dma` 同属一个 oneof，所以 DMA 不能与标量槽同时出现；向量槽各是 bundle 的一个字段。编解码入口是 `tpu::DecodeProgram<…glc::isa::TensorCoreProgram, ghostlite::isa::DecoderGlTensorCore>` 和对应的 `EncodeProgram`，program 对象布局与 v4 相同，公共桥接 `native.cc` 只需换常量。

谓词为 NEVER 的槽，encoder 输出的字节与空槽相同，decoder 也不会报告这个槽，所以源码不能写 NEVER，`p14` 也不可用。汇编后若重新解码得到的槽集合与源码不同（例如被 encoder 丢弃），汇编报错。

## 字段表的生成

v4 的字段表是手工核对后写入的。v6e 有 476 种形式、1214 个“槽×形式”组合，全部由 [tools/generate_tpu_v6e_tc_isa.py](../../tools/generate_tpu_v6e_tc_isa.py) 从已安装的 libtpu 生成到 [tpu_v6e_tc_isa_data.py](../../src/tpuasm/tpu_v6e_tc_isa_data.py)，不要手工修改生成的文件。工具依次做以下几步：

1. **读取 descriptor。** 从 `libtpu.so` 中内嵌的 FileDescriptorProto 读出 bundle、各槽消息、形式（oneof 分支）、字段名、字段号和枚举。
2. **探测字段位置。** 每个形式先以全零字段编码一次，再逐字段、逐 bit（枚举则逐个取值）改变 encoder 输入，比较机器字。所有字段都是连续的位段，枚举值原样写入，没有出现一个输入 bit 影响多个机器位的情况。整数字段探测 32 位，encoder 会静默截断超出位宽的值，所以位宽就是有变化的 bit 数。
3. **求固定位。** 某个槽的“区域”是该槽任一形式写到的位。形式的固定位取区域中不与其他可同时出现的槽共享的部分，再加上本槽某些形式在共享位上写入的操作码：例如 `va0` 的一元运算把 `y_src` 的 5 位用作子操作码，而这 5 位也被 DMA 的向量源字段使用；`vst` 的操作码占用 `misc` 掩码存储读取的 `vmsk` 位。其余共享位只是其他槽的寄存器号字段，形式不写入它们。
4. **找出不能同时出现的槽。** 形式与其他槽的形式组合编码，encoder 总是拒绝的组合记为互斥。目前只有 `vmul.8x128.u32.u64`：它在 `va0` 时独占 `va1`，在 `va2` 时独占 `va3`，第二个目的寄存器写入相邻 VALU 的目的字段。
5. **随机核对。** 随机组合 2 万个指令包（随机的槽、形式、字段值、谓词和共享立即数），要求字段表预测的机器字与 encoder 输出逐字节相同。任何不一致都使工具失败。
6. **读取 formatter。** 在子进程中调用 FormatterGl，记录每个形式的助记符、目的与源的显示顺序、每个显示位置随哪些字段变化、由单元号决定的固定文本（如 `(mrf1, gmr1, msra1)`），以及只改变助记符的字段（如 `vmatpush1/2/3` 与 `.xpose`）。formatter 遇到部分保留编码会终止进程，工具逐个找出这些取值，写入 `FORMATTER_ABORTS`。

以全零操作数编码时被 encoder 拒绝的 81 个“槽×形式”组合写入 `REJECTED`，这些形式不在对应槽登记。按槽归纳：分支和 call 只能在 `s0`；smem 存取、`dma.desc` 和浮点加减只能在 `s1`；浮点乘法和整数乘除只能在 `s0`；超越函数和 `eup_push` 只能在 `va3`；向量无符号整数乘法（u32、u16 和 u32→u64）不能在 `va1`、`va3`。另以 13,101 组操作数重试并检查 encoder 分派，证实这些组合在读取操作数前就被拒绝；这是当前 encoder 的槽位限制。

libtpu 换版本后需要重新运行工具，并比较生成结果的差异；步骤见 skill [add-libtpu-backend](../../.agents/skills/add-libtpu-backend/SKILL.md)。

## 共享字段与命名约束

v6e 的共享立即数和标量操作数槽是 bundle 级的独立消息，encoder 每个指令包都写出它们，所以求解器把这些位视为始终可写。操作数通过 selector 读取它们，读取时计入求解代价。

跨槽共享的寄存器号字段以拥有它的 VALU 或 `vst` 命名为全局端口：`port.va0.x`、`port.va0.y`（`y_vreg`）、`port.va0.ysrc`（DMA 的向量源 selector 与 `va0.y_src` 同位）、`port.va1.dst`（`va0` 的 64 位乘法写入）、`port.vst.src`、`port.vst.vmsk` 等。`misc` 的向量存储和 `vx0/vx1` 通过 selector 从这些端口读取源寄存器，源码只写寄存器，端口选择是编码细节。DMA 的若干寄存器号字段与 `vs0..vs3` 同位，约束名就是 `vsN`。

其余字段的约束名为 `<slot>.<descriptor 字段名>`，枚举字段的值是 descriptor 枚举名去掉公共前缀后的小写形式，例如 `vst.offset = imm1`、`s0.px = inverted_always`。`<slot>.form = <descriptor 形式名>` 固定所用的形式，只在同一文本可由多个形式编码时保留。使 formatter 终止进程的取值不开放为约束值。

规范编码在读取的共享槽数相同时，优先使用编号小的立即数和标量槽，与编译器的常见分配一致；编译器没有这样分配时，exact 导出用约束保留原分配。23 份示例清单中约 11% 的指令包带 `.encoding`（1836/16672），主要用于保留立即数槽分配、未读取的立即数高位和端口选择。

## 签名规则

[tpu_v6e_tc_isa.py](../../src/tpuasm/tpu_v6e_tc_isa.py) 按槽族生成签名：助记符取 formatter 的写法（去掉 `<aluN>` 后缀），操作数按 formatter 的显示顺序排列，写法由字段类型决定。与 v4 一致的地方：目的在前；标量寄存器移动与立即数分别写作 `smov` 和 `simm.s32`，向量写作 `vmov.8x128` 与 `vimm.8x128.s32`；`(pc)`、`(tag)`、`(tm)` 这类由助记符确定的目的不写出；单个隐含寄存器去掉括号，如 `v2sf`、`erf`、`sfrf`；`vmmov.8x128.u1 vm0, vm0` 解码为 `vnop`，与 formatter 一致。

selector 的名称来自 descriptor，数值含义采用[设备语义核对](tpu_v6e_execution.md)的结果：

| selector | 取值 |
|---|---|
| ScalarY、VectorY | 寄存器；`zero_immN` 为零扩展的 20 位立即数；`ones_immN` 为 `0xfff00000 \| immN`；`immN_zero` 为 `immN << 12`；`immH_immL` 为 `((immH & 0xffff) << 16) \| (immL & 0xffff)`；内置常量；VectorY 另有 `vsN`。 |
| Operand、VectorSource | 0 或内置常量、`vsN` 中的标量寄存器、20 位立即数。同步类指令把立即数显示为有符号数，sflag 地址显示为无符号数。 |
| VectorOffset、SublaneMask、VectorStride | 地址偏移与 `sm=`、`ss=` 修饰；省略修饰表示全部 sublane 与单位步长。 |
| VectorBase | `vsN` 中的标量寄存器作为地址基址。 |
| VectorShuffle、SourceSpecifier、TransposeMatrixWidth | 标量寄存器、立即数或按枚举名给出的常量（shuffle 的降序模式、宽度 8…128 等）。 |

DMA 写作 `dma.simple|strided|general[.trace] 目的, 源, 具名操作数…`。端点的空间名由 core id 与 memory id 决定，名称取自 formatter（`vmem`、`smem`、`imem`、`hbm`、`host`、`vmem_all`、`vmem0`、`spmem0` 等），formatter 不显示的组合写作 `coreC_memM`。地址寄存器可以是标量寄存器，也可以是向量寄存器（`dest_in_vreg` / `source_in_vreg`）。formatter 不显示的 `opcode`、`sync`、`thread`、`relaxed`、`host_upper`、`src_operand` 是可省略的具名操作数：省略时取 0，解码时取 0 的项不打印。

formatter 对 15 个形式没有输出，助记符按同族写法命名，操作数按 descriptor 字段顺序排列：`vremap.8x128.f32/bf16`、`vc.8x128.u16`、`vweird.8x128.bf16`、`vmul.8x128.u16`、`vsetperm.half.u8`、`vrot.lane.packed.b8.8x128`、`vbcast.lane.packed.b8.8x128`，以及 `vwait.<条件>.yield` 七个形式。这些名称是 tpuasm 自己的写法，没有 formatter 可对照；它们的设备执行结果已与 host 数值模型核对，其中 wait 仅验证条件已经满足的路径，不包含线程让出与重新调度。

## formatter 的限制

- 对 `FORMATTER_ABORTS` 中的取值（VectorShuffle 0、VectorOffset 7、TransposeMatrixWidth 15、VectorSource 16..63、VunpackSel 6..7），FormatterGl 以 CHECK 失败终止进程。decoder 本身接受这些编码，但原生校验要调用 formatter，所以 v6e 的解码在调用原生校验之前拒绝它们。因此 `vunpack.vsel` 的 `c.s4.bf16`、`c.u4.bf16` 两种打包格式目前无法反汇编。
- `vtrace` 的高、低 16 位来自两个 selector。两半都取立即数，或两半读同一个标量槽（写作该标量寄存器）时，formatter 有输出，tpuasm 也支持；两半读不同标量槽等其他组合 formatter 没有输出，tpuasm 解码时报错。
- formatter 把 `vunpack.vsel` 的 2 位 `vs` 字段显示为 `s<字段值>`。设备核对确认它选择 `vs0..vs3` 中指定的标量寄存器，因此 tpuasm 只输出该实际寄存器，写作三个操作数，不重复输出 formatter 的槽编号。
- VectorDelay 的 selector 0 在 descriptor 中叫 `OPERAND_ONE`，formatter 显示为 0；设备周期对照证实它与立即数 1 等效，因此 tpuasm 显示为 `vdelay 1`。本机 v4 的 `Delay` selector 0 也经周期对照确认是内置一，两代使用相同写法。

## 验证

[tests/reproduce_tpu_v6e_tc.py](../../tests/reproduce_tpu_v6e_tc.py) 离线检查：只有 `.encoding` 才能区分的编码对；每个槽中每个形式的一个随机实例能 exact 往返，canonical 输出稳定；样例源码和一个完整程序映像能往返；以及取反谓词、`p14`、互斥槽和保留编码等拒绝路径。[examples/pallas/](../../examples/pallas/) 中适用于 v6e 的示例在 `run_all.sh --aot tpu-v6e-tc` 下按 `v6e:2x2` 拓扑离线编译，精确导出为 [tpu_v6e_tc/](../../examples/pallas/tpu_v6e_tc/) 中的 23 份清单，导出过程本身包含重汇编与逐字节比较。普通示例核对来源捕获打补丁与不打补丁编译出的机器码相同，另有 `replace_program` 和 `insert_program` 专门检查机器码编辑与写回。

18 份单芯片示例清单对应的 kernel 在 v6e-1 上运行，数值检查通过。设备上编译出的 kernel 函数与离线编译的相同，只有 kernel 之前的包装代码随拓扑（`1x1` 与 `2x2`）不同；仓库中的清单统一保留 `v6e:2x2` 离线编译的结果。按 `v6e:1x1` 拓扑（需另传 `chips_per_host_bounds=[1,1,1]`）离线编译时，其中 17 份与 v6e-1 上编译的清单逐字节相同；`reduction` 只有来源注释不同，原因是设备上编译时没有设置 abstract mesh，JAX 的 jit 追踪缓存命中情况随之不同。需要 2 或 4 个芯片的 `remote_dma_devices`、`remote_dma_ring` 和 `all_reduce` 只在离线编译下导出过。

普通示例只能说明编译器用到的编码被正确解码与重汇编。修改后的机器码可以用 `replace_executable_programs` 写回 executable，或用 `insert_executable_bundles` 增加独立 bundle，再用 `load_executable` 执行。这两种写回已在 TPU v4 和单芯片 v6e 上通过数值检查；v6e 的变长示例也验证了循环前后独立读数和增加循环内空 bundle 后的周期关系。[独立指令探针](../../tests/reproduce_tpu_v6e_execution.py)进一步核对 selector 常量和立即数拼接、15 个 formatter 无文本形式以及动态 unpack，覆盖范围、数值结果和限制见[设备语义核对](tpu_v6e_execution.md)。

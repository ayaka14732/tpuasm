# TPU v4 TC 汇编格式参考

本文说明公共源码结构，并以 `.target tpu-v4-tc` 展开操作数与约束规则。其他目标沿用本文的公共源码结构，只在 ISA、对齐单位和约束上不同。助记符、操作数和可选的命名编码约束共同描述完整程序，汇编结果为机器程序映像。

## 读写方式

用花括号表示同一指令包（bundle），用 `slot:` 指明发射槽；助记符在前，目的操作数在源操作数之前。默认打印每个物理槽一行，非末行在指令后、注释前放置 `;`，后续行缩进两个空格，最后一项后关闭 `}`。下面是转置样例中相邻两个指令包的写法：

```text
.target tpu-v4-tc

{ s0: simm.s32 s1, 72 ;
  vld: vld.8x128 v0, [vmem:0x40] ;
  vx0: vdwg.128x128.f16 gmr0, gsfn0 ;
  misc: vsyncadd.s32 [sflag:53], -64 }
{ vld: vld.8x128 v1, [vmem:0x38] ;
  vx0: vmatpush.packed.xpose.8x128.f16 gsft0, v0 }
```

这里的 `gsfn0` 与 `gsft0` 是实际操作数，不能因相邻指令属于同一计算而自动替换。上述片段只展示两个指令包，未满足完整程序映像的指令包数量要求。

常见指令不需要附加编码说明：

```text
.target tpu-v4-tc

entry:
{ s0: simm.s32 s0, 0 }
loop:
{ s0: sadd.s32 s0, 1, s0 ; s1: sst [smem:s1], s2 }
{ s0: slt.s32 p0, s0, 21 }
{ s0: @p0 sbr.rel loop }
{ s0: sfence }
{ s0: shalt }
.align 10
```

这个例子说明语法和标签解析，不证明示例的调度、数据依赖或设备执行效果。汇编器不重排指令、不插入流水线等待，也不把一条机器指令展开为几条指令。

## 文件、bundle 与标签

| 写法 | 规定 |
|---|---|
| `.target tpu-v4-tc` | 文件首个有效语句，必需且只能出现一次。选择硬件 ISA，不锁定 libtpu 的 Python ABI 或 release。 |
| `{ s0: sfence }` | 一个指令包。花括号决定指令包边界，缩进不参与语义。 |
| `{ s0: sfence ; s1: sfence }` | 同一指令包内的多条指令用 `;` 或换行分隔。每槽最多一条，源码排列顺序不影响编码。默认按槽换行。 |
| `{}` | 显式空指令包，占一个 PC。没有填写的物理槽按目标的空槽编码处理。 |
| `.empty N` | 在指令包外插入恰好 N 个空指令包，N 为正整数常量。 |
| `.align N` | 在指令包外插入最少空指令包，使下一个指令包 PC 为 N 的倍数；已对齐时不插入。单位是 bundle，N 为正整数，不要求是二的幂。 |
| `loop:` | 指令包外的标签，绑定当前下一个指令包的 PC；自身不占 PC。可前向引用，允许不同名字绑定同一 PC，不允许重名。 |
| `# …` | 到行末的注释；空行、注释不占 PC。 |

物理槽仍为 `s0/s1/va0/va1/vst/vld/cld/vx0/vx1/vr0/vr1/misc`。不做自动选槽：同名指令在哪个槽发射，可能改变共享资源约束。`s0:` 是槽标识，操作数中的 `s0` 是标量寄存器。

文件中指令包总数必须为 10 的正整数倍，符合 TPU v4 TensorCore 程序映像的分块编码格式。作者可在结尾写 `.align 10`；汇编器不隐式补齐指令包。反汇编生成的完整文件保留实际指令包数，包括所有尾部空指令包、halt 和其他控制指令，不把尾部代码替换成对齐指令。

标签采用区分大小写的 `[A-Za-z_][A-Za-z_0-9]*`；指令、槽和资源名采用规定的小写拼写。连续标签、末尾标签均可表示当前位置；是否会跳出已有代码不由标签语法推断。无宏、include、任意表达式或隐式指令包：这些都不是本版格式的一部分。

### 语法骨架

文件使用 UTF-8，接受 LF 或 CRLF。下面用 EBNF 表示结构，`[...]` 表示可选，`{...}` 表示重复；引号内是字面字符。横向空白不参与语义，标识符和数值内部不能插入空白。相邻指令、约束赋值之间必须以换行或 `;` 分隔；右花括号可以直接结束最后一项。指令包外语句同样以换行或 `;` 分隔。圆括号和方括号内部换行作为空白处理，其余位置不能跨行拆开指令；不支持反斜线续行或末尾逗号。`#` 注释可以跟在有效语句后。

```text
file          = target, { top_item } ;
target        = ".target", ("tpu-v4-tc" | "tpu-v4-bcs") ;
top_item      = label | bundle | ".empty", integer | ".align", integer ;
label         = identifier, ":" ;
bundle        = "{", { instruction }, [ encoding ], "}" ;
instruction   = slot, ":", [ predicate ], mnemonic, [ arguments ] ;
predicate     = "@", [ "!" ], predicate_register ;
arguments     = argument, { ",", argument } ;
argument      = operand | keyword, "=", operand ;
operand       = register | "!", predicate_register | integer | float
              | float_bits | address | destination_tuple | identifier ;
destination_tuple = "(", register, ",", register, { ",", register }, ")" ;
float_bits    = "f32bits", "(", hex_integer, ")" ;
address       = "[", space, ":", address_base,
                { ",", address_key, "=", address_value }, "]" ;
address_base  = integer | register, [ "+", (register | integer) | "-", unsigned_integer ] ;
address_key   = "sm" | "ss" ;
address_value = register | integer ;
encoding      = ".encoding", "{", { assignment }, "}" ;
assignment    = resource, "=", encoding_value ;
encoding_value = integer | register | resource
               | "const", "(", integer, ")"
               | ("lo" | "ones_hi" | "hi"), "(", resource, ")"
               | "pair", "(", resource, ",", resource, ")" ;
```

这里省略了分隔符的重复项；它们遵守上段规则。空 bundle 和空 `.encoding {}` 均合法，后者打印时省略。`mnemonic` 是 `[a-z][a-z0-9_.]*`，`resource` 是 `[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*`；二者仍须在目标登记。`register`、`predicate_register`、`slot`、`space` 和关键字是目标签名中的枚举或编号名称。`identifier` 作为操作数时只可引用分支/call 目标标签，不是任意符号表达式。

整数由可选符号和十进制数字或 `0x` 后的十六进制数字组成；不支持下划线、二进制、八进制或字符常量。十进制前导零不改变基数。十六进制数字接受大小写。浮点有限值可写小数点或 `e/E` 指数，例如 `1.0`、`.5`、`1e-3`；可有正负号。浮点类型的位置也接受十进制整数，例如 `1` 表示浮点值 `1.0`。无穷写为 `inf`、`+inf`、`-inf`，NaN 必须使用 `f32bits(...)`，避免隐含 payload。操作数签名决定一种语法形式在该位置是否有效，包括元组仅用于多目的、地址修饰项的取值及具名参数的集合。

### 分支与 call

```text
{ s0: @p0 sbr.rel loop }
{ s0: sbr.abs handler }
{ s0: sbr.rel -7 }
{ s0: scall.rel s31, helper }
{ s0: scall.abs s31, 0x20 }
{ s0: sbr.ind s31 }
```

- `.rel` 的数值操作数是**相对当前指令包**的有符号 bundle 位移；标签转换为 `target_pc - current_pc`，没有额外的 `+1`。
- `.abs` 的数值操作数是 bundle PC；标签转换为标签 PC。`.ind` 使用寄存器。
- `scall` 的首个操作数是返回地址目的寄存器，目标操作数仍在其后。
- 标签只允许出现在指令签名声明为目标的操作数中；不把名字偷偷当作立即数或寄存器。未定义标签、重复标签及超出指令字段宽度的结果报错。
- 打印时，为图内直接目标生成 `L_0009` 这样的稳定名字；文件外的目标保留数值。

插入或删除指令包后，标签指向同一条源代码中的指令包，并重新计算位移；使用数值位移则保留作者写下的数值。两者的区别是显式的。

## 指令与操作数

一般形式为 `slot: [@pN 或 @!pN] mnemonic dst, src0, src1`，具体个数和类型由指令签名决定。无谓词即无条件。谓词位于助记符前；谓词操作数中的 `pN` / `!pN` 与条件前缀的 `@pN` / `@!pN` 分开。

| 指令类别 | 写法 | 说明 |
|---|---|---|
| 标量常量/移动 | `s0: simm.s32 s5, -1`；`s1: smov s5, s4` | 目的在前，不带 `$`。 |
| 标量 ALU/比较 | `s0: ssub.s32 s6, 6553, s4`；`s0: sne.s32 p0, s0, 0` | 源操作数保留原有逻辑顺序；不根据交换律任意重排。 |
| 谓词运算 | `s1: por p4, !p3, !p3` | `!p3` 是源操作数；条件执行另用 `@!p3`。 |
| 向量 ALU/选择 | `va0: vadd.8x128.s32 v5, v4, v2`；`va0: vsel.8x128 v25, vm0, v19, v23` | 类型、形状和掩码修饰符保留在助记符中。 |
| 标量存取 | `s1: sld s0, [smem:0x3fff5]`；`s1: sst [smem:s1], s0` | store 的首个操作数是目的地址。 |
| 向量存取 | `vld: vld.8x128 v0, [vmem:0x40]`；`vst: vst.8x128 [vmem:0x48], v9` | `vmem` 在本目标中指 TC VMEM。 |
| CMEM load | `cld: cld.8x128 crf, [cmem:0]` | `cmem` 指 Megacore Shared CMEM；结果 FIFO 保持显式。 |
| 矩阵/结果通路 | `vx1: vmatmul.8x128.f32 mrf0, v9`；`vr1: vpop.8x128 v0, mrf0` | 特殊寄存器或 FIFO 用名字，不包单元素括号。 |
| 多结果操作 | `vx0: vmatmul.dwg.8x128.f32 (gmr0, gsfn0, mrf0), v0` | 多目的组成一个有序元组，不混入源列表。 |
| 同步/控制 | `misc: vwait.ge [sflag:511], s4`；`misc: vsyncset.s32 [sflag:511], 0` | `sflag` 使用带空间名的操作数。 |
| 无数据目的的指令 | `s0: sfence`；`s0: shalt`；`misc: vtrace 0xd0010000` | 只写助记符与实际操作数。 |
| 隐式特殊目的 | `s0: ssettag 0`；`misc: vsettm 1`；`s0: sbr.rel loop` | `tag/tm/pc` 由这些助记符确定。其他特殊目的仍显式列出。 |

原 formatter 的 `vnop` 保留为 `misc: vnop`，不能直接删成空槽或 `{}`；这条可见指令不等同于 descriptor 中被 decoder 去掉的 Noop。formatter 把 mask 寄存器的移动（`MoveVmsk`）一律打印成 `vnop`，但只有 `vm0 ← vm0` 才是空操作；其余的写作 `misc: vmmov.8x128.u1 vm4, vm7`（`vm4 ← vm7`），与 v6e TC 的写法相同。ISA 助记符与后缀不随设计改名，软件层面的 `dma.hbm_to_vmem` 等伪指令不混入本格式。

### 数值与地址

整数支持十进制和 `0x` 十六进制，可带正负号。不允许静默截断或模回绕：十进制按指令的有符号/无符号数值范围检查，十六进制非负整数还可表达该宽度的原始位型，例如 `simm.s32 s0, 0xffffffff` 与 `simm.s32 s0, -1` 生成同一个 32 位数。范围和单位来自操作数签名，不能因为 `simm` 输入的是 32 位数，就允许分支、移位量或地址使用同样的范围。

浮点操作数使用十进制浮点值，以 round-to-nearest, ties-to-even 舍入到目标类型；有限字面量溢出报错，无穷必须显式书写。需要指定 float32 位型时使用 `f32bits(0x7fc00001)`，括号内限无符号 32 位十六进制整数。该字面量保留 NaN payload、正负零等信息，不由十进制格式化推断。只有签名接受浮点或相应 32 位位型的操作数才可用它。格式设计不声称现有 formatter 已经无损显示所有浮点值；精确导出必须从解码字段取位型。

地址写作 `[space:address]`，可组合 ISA 已支持的基址、偏移和修饰项：

```text
[smem:s1]
[vmem:s0 + 0x48]
[vmem:0x48, sm=s1, ss=s2]
[cmem:0]
[hbm:s6]
[sflag:511]
```

`sm=` / `ss=` 对应原 formatter 的 `sm:` / `ss:`；它们是地址内部的具名修饰项。允许组合哪些寄存器、立即数和修饰项由各条存取指令规定，不把这个语法当作通用整数表达式。地址、长度和 stride 都沿用 ISA 操作数的单位，不因显示成整数而统一解释成字节。寄存器、FIFO 和地址空间名必须在目标签名中存在。

### DMA 的具名操作数

本地 `dma.simple`、`dma.strided` 和 `dma.general` 端点支持 `[cmem:sN]`，表示 Megacore Shared CMEM；其硬件端点是 `core_id=1, memory_id=2`。`[hbm:sN]` 为 `core_id=1, memory_id=0`。两者必须同时解码 core 与 memory，不能把所有 `core_id=1` 都打印成 HBM。六条 HBM／TC VMEM／Megacore Shared CMEM 有向路径已在 libtpu 0.0.49 真机上核对完整 payload；TC0 专用程序直接使用的 CMEM 区域必须与其他活跃 core／程序隔离。

DMA 固定按**目的地址、源地址**排列，其余操作数具名，避免多输出赋值和长串难辨认的位置参数：

```text
{ s0: dma.simple [imem:s5], [hbm:s6], length=s1, dst_flag=[sflag:511] }
{ s0: dma.strided [vmem:s0], [hbm:s1], length=s2, dst_stride=s3, src_stride=s4, elements_per_stride=s5, dst_flag=[sflag:511] }
{ s0: dma.general [vmem:s0], [hbm:s1], length=s2, stride_descriptor=[smem:s3], stride_count=1, src_flag=[sflag:s4], dst_flag=[sflag:s5], ici_dest=s6 }
```

这些指令包展示 DMA 的操作数签名。`dst_flag`、`src_flag` 和 `stride_descriptor` 对应已有机器操作数；名称不额外承诺信号触发时机。`dma.desc [smem:s0]` 仍是另一条指令。具名操作数可以换序，解析后按上述顺序打印；必需项不省略，重复或未知项报错。普通指令的具名操作数只在签名明确规定时允许，不接受任意关键字透传。

### 延迟

`vdelay` 的操作数写周期数或标量寄存器。`Delay` 形式的 selector 0 是内置一，写作 `vdelay 1`，命名约束为 `misc.delay_count = const(1)`；与 v6e 一致，已由设备周期对照确认。

## 编码约束：普通源码可以不写

调查确认，源码相同不一定意味着原程序映像相同。为同时支持手写和精确导出，在**同一语法**中提供可选、最后出现的指令包级 `.encoding` 块。它描述额外的硬件字段约束；不包含 protobuf 字段号、消息层次、latency 或 resource map。

```text
{ s0: simm.s32 s0, 0 ; .encoding { imm0 = 21 } }
```

这正好表达调查中的反例：指令仍使用内置常量零，整个指令包中未使用的 `imm0` 保存 `21`。若要保留“从零值 imm0 取立即数”的另一个编码：

```text
{ s0: simm.s32 s0, 0 ; .encoding { s0.y = lo(imm0) } }
```

第二例中 `s0.y` 固定为选择 `imm0` 低 16 位并在高位补零的选择器；操作数 `0` 同时约束 `imm0=0`，无需再重复写一次。`s0.y` 是 ISA 的 Y 选择，不是 protobuf 中第几个字段。

### 约束名称与值

命名表由目标 ISA 定义，跨 libtpu 后端共享。名称必须登记，不能把任意 `slot.foo` 自动转成同名 protobuf 字段。

| 名称 | 值与含义 |
|---|---|
| `imm0` … `imm5` | 共享 16 位编码单元，值为 `0..0xffff`。写入未使用单元也保留机器信息。 |
| `vs0` … `vs2` | 共享标量寄存器选择，如 `vs0 = s4`。 |
| `port.va0.x/y`、`port.va1.x/y`、`port.vst.src` | 共享 TC VREG 源编号字段，例如 `port.va1.x = 9`。`y` 在这里指对应的寄存器输入端口，不是 ALU 的立即数选择器。 |
| `port.va0.dst`、`port.va1.dst`、`port.vld.dst`、`port.aux.dst` | 共享目的编号字段，例如 `port.va1.dst = 1`；编号解释取决于消费它的指令。 |
| `s0.y`、`s1.y` | 标量 ISA Y 来源：`sN`、`const(V)`、`lo(immN)`、`ones_hi(immN)`、`hi(immN)`、`pair(immH, immL)`，限该选择器实际支持的组合。 |
| `va0.y`、`va1.y` | 向量 ALU Y 来源：对应的 `port.vaN.y`、`vsN`、`const(V)` 或目标支持的立即数组合。 |
| `vx0.read`、`vx1.read` | `port.va0.x`、`port.va0.y`、`port.va1.x`、`port.va1.y`、`port.vst.src`。 |
| `vr0.write`、`vr1.write` | `port.va0.dst`、`port.va1.dst`、`port.vld.dst`、`port.aux.dst`。 |
| `vld.base/offset/sm/ss`、`vst.base/offset/sm/ss`、`cld.base/offset/sm/ss` | 地址内部相应选择器，值限定为目标支持的内置值、`vsN` 或 `immN`；例如 `cld.offset = imm3`。 |
| DMA 的 `s0.length/dst_flag/stride_descriptor`，misc 的 `misc.flag/value` | 仅在使用这些选择器的指令形式下可用，取值域由该形式登记；为多处可选来源提供独立约束。 |
| `<slot>.opcode` | 实际机器 opcode 数值，仅用于保留同一文本下的编码变体；必须与该条指令签名相容，不能覆盖成另一条指令。 |

字段登记位于 [tpu_v4_tc_isa_data.py](../../src/tpuasm/tpu_v4_tc_isa_data.py)，操作数签名位于 [tpu_v4_tc_isa.py](../../src/tpuasm/tpu_v4_tc_isa.py)，公开名称由 [tpu_v4_tc_constraints.py](../../src/tpuasm/tpu_v4_tc_constraints.py) 显式映射。每个字段记录硬件位位置和位宽；共享字段以物理范围合并，包含 DMA 控制位对其他槽区域的重叠。辅助 protobuf 字段完全不参与这个表。

除上表外，精确保留还可使用登记的 `<slot>.reg_value/address/smem_address` 标量选择器、`vld.shuffle`、`vxN.matrix_width/rotate_count`、`misc.operand0/operand1/delay_count/interrupt_number` 选择器，以及该指令形式中存在的具名硬件控制字段，如 DMA 的 `s0.source_memory_id/source_core_id/src_opcode/destination_memory_id/destination_core_id/outfeed_queue_id/destination_opcode/trace`、`misc.done_control` 和 `misc.vmsrc1/vmdest`。选择器值使用上表的命名来源，普通控制字段使用该硬件位宽的无符号整数。不能给缺少对应字段的指令附加这些约束。

登记表覆盖当前支持形式能够往返的字段；不把任意 408 位组合宣称为合法指令。原生 decoder/encoder、已登记字段和实际重汇编三者均须通过，精确导出才返回。未登记形式、无法命名的控制值或不能恢复的机器位都明确失败，不退化成 `raw51`、任意 bit patch 或隐藏 sidecar。

`lo(x)` 是 `x`；`hi(x)` 是 `x << 16`；`ones_hi(x)` 是 `0xffff0000 | x`，**不等于按 bit 15 符号扩展**；`pair(h,l)` 是 `(h << 16) | l`。是否把最终 32 位结果显示为负数或浮点数由操作数类型决定。`const(V)` 只允许硬件内置的常量选择器，不能假装任意 V 都有独立编码。

约束遵守三条规则：

1. 所有赋值同时成立，没有“后写覆盖前写”。与可见操作数、另一个槽或其他约束冲突时，报出双方位置和共享资源名。
2. 每个指令包至多一个 `.encoding` 块，同一名字只赋值一次。未消费的已登记共享字段允许保留，但指令包内必须有能够编码该字段的指令形式；空指令包不能仅靠约束创建编码字段。不存在对应指令的选择器约束不允许凭空创建隐藏指令。
3. 约束不会自动随指令编辑而失效。若改了操作数导致冲突，明确报错；作者删除不再需要的约束后，由汇编器重新分配。绝不静默丢弃约束。

比如 `vx1.read = port.va1.x` 会约束同一指令包中的 `va1` 和 `vx1` 对该端口使用相同寄存器编号；`vr1.write = port.va1.dst` 会影响同一指令包中 `va1` 的目的编号。即使两条指令的谓词相反，物理字段仍只有一份，不能给它们写两个不同值。

## 普通源码与精确导出

两者共用上述格式和同一个 assembler，没有文件外的原程序映像依赖：

| 使用方式 | 文件内容与保证 |
|---|---|
| 手写源码或普通导出 | 助记符和标签，通常没有 `.encoding`。保证产生按确定性规则选出的编码；不承诺等于某份原程序映像。 |
| 精确导出 | 同样的助记符和标签，加上恢复原编码必需的命名约束。文件本身足以重建原字节，实际 assemble 后逐字节比较通过才返回文本。 |

`assemble_listing(text, *, filename="<assembly>") -> bytes` 接受上述源码格式，`filename` 用于诊断；`format_assembly(image, *, target, encoding='exact') -> str` 要求显式传入 `target='tpu-v4-tc'`，默认精确导出；需要可自由重新分配的模板时显式选 `encoding='canonical'`。`dump_executable()` 和 `dump_compiled()` 同样接受 `encoding` 关键字，统一输出 `.tpuasm`。CLI 对应 `--encoding exact|canonical`，默认 `exact`；汇编输入只按文件内容处理，不设一个能忽略约束的开关。

打印顺序固定为本规范列出的 12 槽顺序，编码约束按名字 ASCII 字典序。多槽 bundle 每槽一行；`.encoding { … }` 在指令包内单独一行，内部赋值继续用 ` ; ` 分隔。单槽且没有约束时保持一行，空指令包打印 `{}`。生成标签使用 `entry` 表示 PC 0，其他内部目标用 `L_` 加至少四位的小写十六进制 PC。输入注释、标签拼写及 `.empty` / `.align` 简写不属于反汇编可恢复的信息；导出时以实际指令包和生成的标签表示。数值显示基数由操作数签名规定；浮点使用可往返的十进制值或精确位型，不能照抄可能损失精度的原生 formatter 字符串。

逐槽源码注释放在对应指令与分隔符之后：非末行使用 `;  #`，末行使用 `}  #`。函数区间标记放在 bundle 外，区间存在空洞时分别标出，不把最小到最大 PC 之间的全部代码归给一个函数。例如：

```text
{ va0: vadd.8x128.s32 v21, v20, v19 ;  # kernel.py:10:4
  va1: vshll.8x128.s32 v22, v13, 0x15 }  # kernel.py:11:4
```

来源展示来自 executable 元数据，结构化接口另保存多来源、调用帧、scope、ordinal、overlay 和原始注释坐标。独立程序映像没有这类元数据。编辑后汇编只产生机器字节，不生成或更新编译来源；不能把旧注释当作修改后的真实来源。

导出的清单始终是完整程序映像，包括 kernel 前后的 runtime 代码，因为精确导出要求整份清单可重新汇编并回灌。只读工具若只想展示 kernel，应按 `ProgramSourceMap.functions[*].ranges` 折叠视图，并保留区间之间的空洞，而不是裁出一份看似完整的清单。

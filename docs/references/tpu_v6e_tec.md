# TPU v6e TEC 汇编格式参考

`.target tpu-v6e-tec` 是 TPU v6e SparseCore 向量子核（TEC）的程序。它与 [TPU v6e TC](tpu_v6e_tc.md) 使用同一套源码语法：bundle、槽前缀、标签、谓词、数值与浮点写法、`.encoding` 块，以及精确导出和 canonical 导出的区别，都按 v4 与 v6e TC 的规定处理。本文只列出 TEC 不同的地方。各助记符的操作数签名见[指令索引](tpu_v6e_tec_isa.md)。

## 程序结构

```text
.target tpu-v6e-tec

entry:
{ s0: simm.s32 s0, 0x7 ;
  s1: simm.s32 s1, 0x12345678 ;
  misc: ssyncset.s32 [sflag:s5], 0x0 }
{ va0: vadd.s32 v2, v3, v1 ;
  va1: vmul.f32 v4, 1.5, v2 ;
  va2: vxor.u32 v5, 0x7, v2 ;
  vld: vld v6, [tilespmem:0x80] ;
  vst: vst [tilespmem:0x400], v4 }
{ vr: vpop v10, erf }
{ vr: vmovc v11, v10 }
{ stream: @!p0 stream.linear.gather [tilespmem:s0], [sflag:0x1], [hbm4b:s1+s0], 0x400, 0x38 }
{ dma: @!p0 dma.local [timem:s3], [sflag:s2], [hbm:s0], s1 }
{ misc: @!p0 swait.ge [sflag:s1], 0x400 }
{ misc: yieldable_sync_done x=0x1 }
{ s0: @p1 sbr.rel entry ;
  s1: sadd.s32 s5, 0x1, s5 }
{ s0: sfence }
{ misc: sbarrier.arrive [bflag:0x3], 0xffff }
{ s0: shalt }
```

这个样例覆盖全部 12 个物理槽，只用于核对语法和编码，没有在设备上执行。

- 物理槽按打印顺序为 `s0 s1 dma misc va0 va1 va2 vld vst stream vr vx`。`dma` 与 `stream` 不能出现在同一指令包，二者也都不能与 `s0`、`s1` 同时出现。
- 程序映像以 64 字节为一块，每块 1 个 bundle，指令包数不需要对齐，不写 `.align`。
- 谓词寄存器为 `p0..p13`，每个槽都能写 `@pN` 或 `@!pN`。与 v6e TC 一样，NEVER 谓词与空槽编码相同，不能写在源码中。
- 地址空间 `tilespmem` 指 SC TileSPMEM。stream 的另一端为 `hbm4b`（可带偏移寄存器，如 `[hbm4b:s1+s0]`）或 `spmem`（SparseCore 共享的 SPMEM）；DMA 端点有 `hbm`、`timem`、`smem`、`simem`、`spmem` 等名称。
- 某些形式只能在特定槽发射，例如分支只在 `s0`；在其他槽书写时报错。

## 与 v6e TC 写法的差异

| 项 | TEC 写法 |
|---|---|
| 助记符与操作数顺序 | 取自 libtpu 编译 SparseCore kernel 时 LLVM TPU printer 的输出（编译器 dump 中的文本）：目的在前，`$` 前缀与 `(pc)`、`(tag)` 这类隐含目的不写，谓词写在槽名之后。向量助记符没有 `.8x128` 后缀，例如 `vadd.s32`、`vmul.f32`。 |
| 地址 | 整段地址按 printer 的样子书写，例如 `[tilespmem:0x80]`、`[hbm4b:s1+s0]`、`[sflag:s5]`、`[sflag:s5@s12]`（`@` 之后的寄存器经由标量操作数槽读取）。 |
| stream | `stream.<linear\|strided\|indirect>.<gather\|scatter>[.cb][.add.<类型>]`：先写两端地址与同步标志，再写长度等操作数。最后一个操作数是配置字，各 bit 分别写入同步标志类型、完成位、TileSPMEM 步长等字段，例如 `0x38` 表示不跨步。 |
| DMA | `dma.local 目的, [sflag:…], 源, 长度`，两端的地址空间各自写在地址中。 |
| 没有 printer 写法的编码 | 每个形式都可以写成 `<descriptor 形式名> 字段=值, …`，字段按 descriptor 中的顺序全部给出，例如 `misc: yieldable_sync_done x=0x1`。selector 字段能表示为数值时写数值，否则写取值名。反汇编遇到 printer 写法表达不了的编码（例如没有对应 LLVM 指令的形式、保留编码、超出寄存器数的寄存器号）时使用这种写法。 |

## 编码约束

`.encoding` 的规则与 v6e TC 相同：所有赋值同时成立，每个指令包至多一个块，同一名字只赋值一次，约束不能凭空创建指令。名字和取值如下：

| 名称 | 取值 |
|---|---|
| `imm0` … `imm5` | 共享立即数，20 位无符号整数 `0..0xfffff`。 |
| `vs0` … `vs3` | 共享标量操作数槽，写 `sN`，例如 `vs1 = s5`。 |
| `<slot>.<字段名>` | 当前指令形式的字段，名称取自 libtpu ISA descriptor，例如 `vld.offset`、`s1.y`、`va1.y_src`、`dma.trace_en`。selector 字段写取值名，其余字段写无符号整数。`vx` 读取其他槽源寄存器的字段（如 `vx.v0_x`、`vx.vst_source`）与那些槽的字段同位，同一指令包中取值必须一致。 |
| `<slot>.form` | 固定使用的指令形式，值为 descriptor 中的形式名。只在同一文本能由多个形式编码、其他约束又不能区分时导出。 |

selector 的取值名与数值含义与 v6e TC 相同，例如 `s1.y` 可取 `sreg0..sreg31`、`zero_imm0..3`、`ones_imm0..3`、`imm0_zero..imm3_zero`、`imm1_imm0`、`imm3_imm2` 和 `zero`、`one`、`two` 等内置常量。

下面两个指令包文本相同，第二个用约束保留了编译器的立即数槽分配：

```text
{ vld: vld v1, [tilespmem:0x80] }
{ vld: vld v1, [tilespmem:0x80] ; .encoding { vld.offset = imm4 } }
```

没有约束时，汇编器在读取的共享槽数相同的候选中选编号最小的立即数槽，所以第一行使用 `imm0`。printer 不显示的字段（例如 DMA 的 `trace_en`、selector 选中立即数时闲置的寄存器号）同样由精确导出写成约束。

## 接口

`assemble_listing()`、`format_assembly(image, target='tpu-v6e-tec', encoding=...)`、`executable_programs()`、`dump_executable()`、`dump_compiled()`、`replace_executable_programs()` 与 `load_executable()` 与 v6e TC 用法相同，只是 executable 相关的函数必须显式传入 `target='tpu-v6e-tec'`：SparseCore 代码由 TC 记录携带，省略目标时推断为 TC。命令行对应 `--target tpu-v6e-tec`。TEC 只支持等长替换，不支持 `insert_executable_bundles()` 和编译来源注释。executable 中的 TEC 程序映像如何定位见[总体架构](../design/architecture.md#硬件目标与程序容器)。

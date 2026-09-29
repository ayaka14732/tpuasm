# TPU v6e TC 汇编格式参考

`.target tpu-v6e-tc` 与 [TPU v4 TC](tpu_v4_tc.md) 使用同一套源码语法：bundle、槽前缀、标签、谓词、数值与浮点写法、地址、DMA 的具名操作数、`.encoding` 块，以及精确导出和 canonical 导出的区别，都按 v4 的规定处理。本文只列出 v6e 不同的地方。各助记符的操作数签名见[指令索引](tpu_v6e_tc_isa.md)。

## 程序结构

```text
.target tpu-v6e-tc

entry:
{ s0: simm.s32 s0, 7 ;
  s1: simm.s32 s1, 0x12345678 }
{ va0: vadd.8x128.s32 v2, v3, v1 ;
  va1: vmul.8x128.f32 v4, 1.5, v2 ;
  va2: vshll.8x128.u32 v5, v2, 0x2 ;
  va3: vrcp.8x128.f32 erf, v4 }
{ vst: vst.8x128 [vmem:0x48], v4 ;
  vld0: vld.8x128 v5, [vmem:0x40] ;
  vld1: vld.8x128 v6, [vmem:s2 + 0x8, sm=0xf] ;
  misc: vwait.ge [sflag:511], s4 }
{ vx0: vmatpush1.8x128.f32 msra0, v5 ;
  vx1: vmatmul.8x128.f32 mrf1, v6 }
{ vr0: vpop.8x128 v10, erf ;
  vr1: vpop.8x128 v11, mrf1 }
{ dma: dma.simple [vmem:s1], [hbm:s2], length=s3, dst_flag=[sflag:52] }
{ s0: @!p0 sbr.rel entry ;
  s1: @p1 sadd.s32 s5, 1, s5 }
{ s0: sfence }
{ s0: shalt }
.align 8
```

这个样例覆盖全部 15 个物理槽，只用于核对语法和编码，没有在设备上执行。

- 物理槽按打印顺序为 `s0 s1 dma va0 va1 va2 va3 vst vld0 vld1 misc vx0 vx1 vr0 vr1`。`dma` 槽与 `s0`、`s1` 不能出现在同一指令包。
- 程序映像以 512 字节为一块，每块 8 个 bundle，所以指令包总数必须是 8 的正整数倍，常用 `.align 8` 结尾。
- 谓词寄存器为 `p0..p13`。只有 `s0`、`s1`、`dma` 能写 `@!pN`；向量槽只能写 `@pN` 或不写谓词。硬件的 NEVER 谓词与空槽编码相同，不能写在源码中。
- 地址空间 `vmem` 指 TC VMEM。DMA 端点另有 `hbm`、`host`、`smem`、`imem`、`vmem_all`、`vmem0`、`spmem0` 等名称，没有名称的端点写作 `coreC_memM`。
- `vmul.8x128.u32.u64` 写两个目的寄存器，在 `va0` 时占用 `va1`，在 `va2` 时占用 `va3`，同一指令包不能再使用被占用的槽。
- 某些形式只能在特定槽发射，例如分支只在 `s0`，smem 存取只在 `s1`，超越函数只在 `va3`；在其他槽书写时报错。

## 与 v4 写法的差异

| 项 | v6e 写法 |
|---|---|
| 向量读取与存储 | 两个读槽 `vld0`、`vld1`，`misc` 也能执行 `vld`/`vst`；没有 `cld` 槽。 |
| 矩阵单元 | `vmatpush1/2/3` 等助记符中的数字和 `.xpose` 由对应字段决定；MXU 结果名如 `mrf0`、`mrf1`，多结果目的仍写成有序元组。 |
| DMA | `dma.simple`、`dma.strided`、`dma.general` 及其 `.trace` 变体。除 v4 的具名操作数外，还有可省略的 `opcode`、`sync`、`thread`、`relaxed`、`host_upper`、`src_operand`，省略时取 0，导出时只打印非零值。地址寄存器可以是 `sN` 或 `vN`。 |
| `vtrace` | 操作数是 32 位值；立即数路径目前沿用 formatter 的 `((upper << 16) \| lower) & 0xffffffff` 模型，尚未核对设备 trace 记录。上下 selector 都来自同一个标量寄存器时写作该寄存器，例如 `vtrace s2`。 |
| `vdelay` | 与 v4 相同，数值写延迟周期数，也可以写标量寄存器。VectorDelay 的 selector 0 是内置一，写作 `vdelay 1`。 |

## 编码约束

`.encoding` 的规则与 v4 相同：所有赋值同时成立，每个指令包至多一个块，同一名字只赋值一次，约束不能凭空创建指令。名字和取值不同：

| 名称 | 取值 |
|---|---|
| `imm0` … `imm5` | 共享立即数，20 位无符号整数 `0..0xfffff`。 |
| `vs0` … `vs3` | 共享标量操作数槽，写 `sN`，例如 `vs1 = s5`。 |
| `port.va0.x/y/ysrc`、`port.va1.x/y/dst`、`port.va2.x/y`、`port.va3.x/y/dst`、`port.vst.src/vmsk` | 跨槽共享的向量寄存器号或 selector 字段。寄存器号写整数；`port.va0.ysrc` 这类 selector 写取值名。 |
| `<slot>.<字段名>` | 当前指令形式的字段，名称取自 libtpu ISA descriptor，例如 `s1.y`、`vst.offset`、`vld1.stride`、`vx0.vex_source`、`misc.sync_flag_number`、`s0.py`。selector 字段写取值名，其余字段写无符号整数。 |
| `<slot>.form` | 固定使用的指令形式，值为 descriptor 中的形式名，例如 `vld0.form = vector_load`。只在同一文本能由多个形式编码、其他约束又不能区分时导出。 |

selector 的取值名是 descriptor 枚举名去掉公共前缀后的小写形式，例如：

- `s0.y`、`s1.y`：`sreg0..sreg31`、`zero_imm0..3`（零扩展）、`ones_imm0..3`（`0xfff00000 | immN`）、`imm0_zero..imm3_zero`（`immN << 12`）、`imm1_imm0`、`imm3_imm2`（`((immH & 0xffff) << 16) | (immL & 0xffff)`，各字段只读取低 16 位），以及 `zero`、`one`、`negative_one`、`hex_100` 等内置常量。
- 地址修饰 `vst.offset`、`vld0.offset`、`vld1.stride`、`vld1.sublane_mask` 等：`zero`、`one`、`all_ones`、`negative_one` 等内置值、`vs0..vs3` 或 `imm0..imm5`。
- `vx0.vex_source`：`vst_source`、`v0_x`、`v0_y_vreg`、`v1_x` 等，即从哪个共享端口读取源寄存器。

使 libtpu formatter 终止进程的保留取值不接受为约束值，这类编码也不能反汇编，见[设计文档](../design/tpu_v6e_tc.md#formatter-的限制)。

下面两个指令包文本相同，第二个用约束保留了编译器的立即数槽分配：

```text
{ vst: vst.8x128 [vmem:0x40], v1 }
{ vst: vst.8x128 [vmem:0x40], v1 ; .encoding { vst.offset = imm1 } }
```

没有约束时，汇编器在读取的共享槽数相同的候选中选编号最小的立即数槽，所以第一行使用 `imm0`。精确导出只在原编码与这一规则的结果不同时写出约束。

## 接口

`assemble_listing()`、`format_assembly(image, target='tpu-v6e-tc', encoding=...)`、`dump_executable()` 和 `dump_compiled()` 与 v4 用法相同。v6e 的 executable 可以在没有 v6e 设备的机器上由 JAX 按拓扑离线编译得到。逐槽源码注释和函数区间与 v4 相同。

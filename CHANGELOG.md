# 更新日志

本文件记录 tpuasm 各版本面向用户的变化。各版本支持的 Python、libtpu 与 JAX 见[版本兼容性](docs/compatibility.md)。

## 0.3.0 - 2026-10-12

### 新增

- 支持 libtpu 0.0.50（CPython 3.14t）：登记 TPU v4 TC、TPU v4 BCS、TPU v6e TC 与 TPU v6e TEC 的编解码后端，以及 TensorCore 的来源后端。这个版本把 serialized executable 中 IFRT 头之后的内容改成 XLA 的 split proto，即一个 Riegeli/records 流（[#1](https://github.com/ayaka14732/tpuasm/issues/1)）；此前 `executable_programs` 等接口把它当作旧容器解析，报错 `executable record exceeds file size`。两种容器给出的记录编号相同，`replace_executable_programs` 和 `insert_executable_bundles` 按输入的容器形式写回。这个版本还把 SparseCore 的 TEC 段挪到了第一块 SCS 代码之后，TEC 段的定位改为以第一个全零 bundle 为界。v6e 的两个目标只做了离线验证。
- TPU v6e TEC 的指令语法改由 libtpu 0.0.50 的 LLVM TPU printer 生成，多出 123 条：`swait.*.yieldable`、`vcvt.*.bf16`、`vunpack.ic.*`、u16 比较等 76 个形式此前只能用字段名写法，现在有了助记符，另有 11 个形式多出一种写法。原有的写法不变，字段表的其余部分在两个版本上相同。
- `insert_executable_bundles` 支持有多个程序 overlay 的程序（[#2](https://github.com/ayaka14732/tpuasm/issues/2)）。编译器会把较大的直线代码拆成多个 overlay（例如一个 4.7 万个 bundle 的程序被拆成 4 个），此前这样的程序一律报错 `insertion requires one code image with a prelude and one program overlay`。现在插入点可以落在任何一个程序 overlay 的正文内，也可以紧接在正文末尾；每个 overlay 各自补齐到整块，其后的 overlay 后移，所有 trampoline 中的装载参数、overlay 表和 trap 元数据随之更新。

### 变更

- 来源捕获改为核对 JAX commit `cd454bd4d92de993c183c842f149cc1230a1c171` 的 Pallas lowering 源码，上游在此之前改动了 `_lower_jaxpr_to_for_loop`。更早的 JAX（例如 `0.12.0.dev20261002`）进入 `compiler_source_mapping` 时会报 `JAX _lower_jaxpr_to_for_loop source differs from the validated implementation`，需要升级 JAX。示例清单随之重新生成：远程 DMA 的三个示例和全部 SparseCore 示例的机器码因 JAX 的 lowering 变化而不同。
- `BundleInsertion` 的片段不再接受 `sbr.abs` 和 `scall.abs`。程序 overlay 装入指令存储的位置在运行时确定，与它在程序映像中的位置不同；此前按映像位置填写的目标实际指向别处（TPU v4 上相差 30 个 bundle）。片段内的分支改用 `sbr.rel`、`scall.rel`。

### 修复

- `insert_executable_bundles` 不再把元数据中的 `overlay_slot_size`（32760 个 bundle）当作单个 overlay 的容量上限（[#2](https://github.com/ayaka14732/tpuasm/issues/2)）。编译器不拆分循环，会生成比它更大的 overlay；此前对这样的程序即使只插入一个 bundle 也报错 `inserted program exceeds the overlay slot capacity in bundles`。容量上限改为从程序前缀的装载例程中读出：TPU v4 为 65360 个 bundle，TPU v6e 为 65168 个。
- TPU v4 上迁移相对分支时按 16 位回绕解释位移。超过 32767 个 bundle 的 overlay 中，编译器把跨度更大的前向分支写成负位移，此前这样的分支会被判为指向映像之外。
- `insert_executable_bundles` 只读取 overlay 表和前缀例程的名称，不再解码整份来源映射。在一个 2.4 万个 bundle 的程序中插入两处，由 28 秒降到 9 秒。
- `tpuasm.tools.hlo_bounds` 直接解码 bundle 来查找 `vtrace` 标记，不再生成整份清单；`KernelClock.time_ops` 靠它确定读数的位置。同一个程序上由 42 秒降到 5 秒。

## 0.2.1 - 2026-10-08

### 新增

- `BundleInsertion` 新增 `delete`：`insert_executable_bundles` 在插入片段的同时删除从插入点起的若干个原 bundle，片段可以为空。此前只能插入，把一段代码换成更短的写法时，多出来的 bundle 只能留作空 bundle 或用分支跳过。
- 新增模块 `tpuasm.tools`（TPU v4 TensorCore，依赖 JAX，需显式导入）：编译并保留源码映射、按 bundle 查找与改写清单、把改写后的 executable 装载运行，以及两种用设备上的 LCC 计时的工具。`LccProbe` 把手写片段插进载体 kernel，用来测指令的延迟和核对指令的语义；`KernelClock` 在任意已编译程序的指定 bundle 之前插入读数，读数留在 SMEM 中，程序运行之后再取回，被测程序不需要增加输出。

## 0.2.0 - 2026-10-05

### 新增

- 新增 `tpu-v6e-tec` 目标（libtpu 0.0.49）：v6e SparseCore TEC 程序映像的汇编与反汇编，助记符和操作数写法与 libtpu 编译器的 LLVM TPU printer 一致；executable 中的 TEC 程序可以导出和等长写回，需显式指定目标。
- `SlotSource` 新增 `source_frames` 与 `source_kind`。`source_frames` 合并 tpuasm 捕获的来源与编译器自带的 `loc(...)`，按文件和行列去重，是逐槽源码位置的统一视图；`source_kind` 标明位置来自 tpuasm 捕获（`captured`）、只来自编译器（`compiler_location`），还是没有位置（`unknown`）。函数区间内已占用但没有注释的槽也会列出。来源映射 JSON 的 `schema_version` 随之由 2 升为 3。
- libtpu 0.0.49 的来源捕获新增 hook，覆盖 `dma.done` 展开、MXU prep 改写、v6e matprep 发射、MLIR CSE 与 `FusedLoc` 解析。v6e matmul 示例的函数区间内只剩 `vtrace` 没有来源。
- 编译器注释中被 `loc(...)` 覆盖的 region builder 说明（例如取模的实现方式、循环退出判断、vreg 切片范围）得以保留，格式为 `loc(...) :: 原注释`。

### 变更

- 清单中的来源注释改为一条从内到外的调用链 `path:内层 <- 调用方 [scope; LLO n]`，同一槽上相同的调用链只显示一次。带方括号的是 tpuasm 捕获的来源，不带的是编译器的位置。
- 嵌套 lowering 的位置接上外层方程的 traceback，内层 op 的来源带有外层调用点。

### 移除

- 不再支持 libtpu `0.0.48.dev20260912+nightly`（CPython 3.14t）。仍需使用该构建时，安装 tpuasm 0.1.0。

### 修复

- TPU v4 TC 的 mask 寄存器移动（`MoveVmsk`）此前一律反汇编为 `misc: vnop`，目的与源寄存器只出现在精确清单的 `.encoding` 约束中；canonical 清单丢掉这两个字段，重新汇编后移动变成空操作，程序行为改变（真机上表现为挂起）。现在只有 `vm0 ← vm0` 写作 `vnop`，其余写作 `misc: vmmov.8x128.u1 vmD, vmS`，与 v6e TC 一致。
- 来源记录中每个 location 的 ordinals 只保留本指令一项。此前保留该 location 的全部 ordinals，完全展开的大运算使每条记录达数十 KB，注释表随指令数平方增长：约 1.8 万个 bundle 的 kernel 的元数据超过 2 GiB，`serialize()` 报 `Failed to serialize TpuExecutableProto, is it too large?`，读不到清单也无法改写。同一程序的元数据现在约 40 MB。`SourceLocation.ordinals` 因此只含所属指令的 ordinal。
- 发射 hook 只序列化与本指令有关的 location，不再对每条指令序列化整个 SourceMap；上述 kernel 在捕获来源时的编译时间由 53 秒降到 9 秒（不捕获时 3 秒）。
- `insert_executable_bundles` 按主程序 overlay 的 `encoded_word_offset` 识别装载指令。此前按映像位置识别，程序带有去重常量（第一个 overlay 的偏移不为 0，例如 XLA 的 `psum`、`scatter`）时报错 `cannot identify the overlay loader immediates`。
- overlay 的 `encoded_word_offset` 改为相对第一个 overlay 换算程序映像位置。此前第一个 overlay 的偏移不为 0 时（例如 XLA 为 `scatter` 生成的程序，偏移为 33），来源映射报错 `overlay lies outside the program image`。
- 捕获来源期间关闭 Pallas 的逐方程 lowering 缓存（[jax-ml/jax#41153](https://github.com/jax-ml/jax/issues/41153)）。此前相同运算第二次出现时会带上第一次出现的行号，例如同一 kernel 中多次 `pltpu.roll`。关闭前后机器码相同。

## 0.1.0 - 2026-09-29

首个发布。

- TPU v4 TC、TPU v4 BCS 与 TPU v6e TC 程序映像的汇编与反汇编；清单显式标出物理槽，写出 formatter 省略的选择字段。
- 命令行 `tpuasm` 在 serialized executable、程序映像和 `.tpuasm` 清单之间转换，支持逐字节可逆的精确导出和便于手工编辑的 canonical 导出。
- `replace_executable_programs` 等长替换 executable 中的程序，`insert_executable_bundles` 在指定 bundle 前插入片段，`load_executable` 借用已编译函数的调用约定装载执行修改后的 executable。
- `compiler_source_mapping` 在编译期捕获 Pallas 源码来源，`dump_compiled` 导出带来源注释的清单和来源映射 JSON（`schema_version` 为 2）。
- `encode_tpu_v4_bcs_program`、`decode_tpu_v4_bcs_program`、`extract_tpu_v4_bcs_program` 处理 BCS 程序与 semantic protobuf、容器之间的转换。

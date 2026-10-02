# 更新日志

本文件记录 tpuasm 各版本面向用户的变化。各版本支持的 Python、libtpu 与 JAX 见[版本兼容性](docs/compatibility.md)。

## 未发布

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

- overlay 的 `encoded_word_offset` 改为相对第一个 overlay 换算程序映像位置。此前第一个 overlay 的偏移不为 0 时（例如 XLA 为 `scatter` 生成的程序，偏移为 33），来源映射报错 `overlay lies outside the program image`。
- 捕获来源期间关闭 Pallas 的逐方程 lowering 缓存（[jax-ml/jax#41147](https://github.com/jax-ml/jax/issues/41147)）。此前相同运算第二次出现时会带上第一次出现的行号，例如同一 kernel 中多次 `pltpu.roll`。关闭前后机器码相同。

## 0.1.0 - 2026-09-29

首个发布。

- TPU v4 TC、TPU v4 BCS 与 TPU v6e TC 程序映像的汇编与反汇编；清单显式标出物理槽，写出 formatter 省略的选择字段。
- 命令行 `tpuasm` 在 serialized executable、程序映像和 `.tpuasm` 清单之间转换，支持逐字节可逆的精确导出和便于手工编辑的 canonical 导出。
- `replace_executable_programs` 等长替换 executable 中的程序，`insert_executable_bundles` 在指定 bundle 前插入片段，`load_executable` 借用已编译函数的调用约定装载执行修改后的 executable。
- `compiler_source_mapping` 在编译期捕获 Pallas 源码来源，`dump_compiled` 导出带来源注释的清单和来源映射 JSON（`schema_version` 为 2）。
- `encode_tpu_v4_bcs_program`、`decode_tpu_v4_bcs_program`、`extract_tpu_v4_bcs_program` 处理 BCS 程序与 semantic protobuf、容器之间的转换。

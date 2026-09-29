# TC 编译来源映射

来源映射为程序映像中每个 (PC, 物理槽) 关联三类信息：Pallas 源码位置、所属函数区间，以及编译器原生的注释。这里的 PC 是 bundle 在程序映像中的序号，不是设备 IMEM 地址。支持 `tpu-v4-tc` 与 `tpu-v6e-tc`，两者共用补丁和恢复流程，只有槽名和 overlay 坐标的单位不同。

功能分为两个阶段：

- **捕获**：在 `compiler_source_mapping()` 上下文中编译时进行，需要对 libtpu 打进程级补丁，补丁与 libtpu 版本绑定。
- **恢复**：`executable_source_maps()` 只读取 serialized executable。除匹配的编解码后端外，不需要 TPU、补丁或任何 dump。

本功能不做以下事情：不解析 final bundles 或 LLO dump；不根据 `vtrace` 的 PC 差或指令类别猜测来源；来源未知时保持未知，不能把“没有来源”解读为“由编译器生成”。

## 为什么借用 libtpu 的逐槽注释

libtpu 本身就会为每条 LLO 指令携带一段注释字符串，经过调度、组包和链接，最后按 bundle 和物理槽写进 `CompilerMetadata`。tpuasm 的做法是在指令发射前，把一条自包含的记录追加到这段注释里。此后由 libtpu 自己的 emitter 决定记录落到哪个槽，由它的 linker 重定位 PC。这样，tpuasm 不需要重新实现槽分配和重定位，而这两部分正是最容易随编译器版本变化出错的地方。

补丁有一条不变式：**不改变生成的机器码。** [tests/reproduce_tpu_v4_tc_source_mapping.py](../../tests/reproduce_tpu_v4_tc_source_mapping.py) 对每个案例分别在打补丁和不打补丁的情况下编译，要求两份程序映像相同，且数值结果一致。

### 记录格式 v1

追加的文本为 `[[tpuasm:v1:<hex>]]`，hex 编码一条独立的 protobuf wire 消息：

| 字段 | 类型与含义 |
|---|---|
| 1 | varint，最终的 LLO ordinal |
| 2 | bytes，筛选后的 `SourceMapProto` |
| 3 | string，HLO instruction 名 |
| 4 | string，HLO module 名 |
| 5 | varint，HLO module ID |

字段 2 只保留 ordinals 中含有本指令 ordinal 的 location，但保留完整的字符串表。因此一条记录可以单独解析，不依赖已经销毁的 LLO module。所用的 SourceMap 字段为：SourceMap 的 locations=1、strings=2；SourceInfo 的 frames=1、primitive=2、scope_stack=3、ordinals=4；Frame 的 path=1、line_start=2、line_end=3、col_start=4、col_end=5、function_name=6。

同一条注释中已有 v1 记录时不再追加。修改记录的字段需要使用新的版本标签，恢复端继续解析 v1。

## 捕获

### JAX lowering

Pallas 的 `jaxpr_subcomp` 用 `ctx.name_stack + eqn.source_info.name_stack` 生成每个方程的 MLIR location。lowering 规则递归调用 `jaxpr_subcomp` 处理子 jaxpr 时，传入的 `ctx.name_stack` 不一定包含外层方程的 scope，于是位置信息丢失了静态的父 scope。[tc_source_lowering.py](../../src/tpuasm/tc_source_lowering.py) 用一个 `ContextVar` 保存外层方程的完整 name stack，作为嵌套调用的父栈。改动只影响 location 参数，不改变 name stack 的 trace 操作。

第二处改动针对完全静态展开的循环（例如 `pl.loop(..., unroll=True)`）。JAX 的 `_lower_jaxpr_to_for_loop` 会把循环体 lower 成一个游离的 `func.func`，再用 `jax_mlir_ext.inlined_func_call` 逐次克隆到调用处。克隆时 JAX 把每个 op 的 location 改写为以循环方程为调用方的 `CallSiteLoc`，op 类型一律换成循环方程的类型，名称则拼接为调用方名称加原名称。libtpu 据此记录的 primitive 全部变成 `scan`，原来的 `max`、`min` 只剩 scope 的最后一段；原名称中已含 tpuasm 父栈，于是 scope 还会重复一次外层前缀。补丁在上下文内把完全展开的循环改回逐次调用 `jaxpr_subcomp`，每条展开出的 op 直接使用自己方程的 location。两条路径生成的是同一组 op；复现脚本中的 `unrolled` 案例在补丁前后编译出的程序映像逐字节相同，数值也一致。

兼容性以函数源码为准：先读取已安装 JAX 中 `jaxpr_subcomp` 和 `_lower_jaxpr_to_for_loop` 的源码字节，分别计算 SHA-256 并与 `tc_source_lowering.py` 中的 `_SOURCE_SHA256` 比较，不同就拒绝；然后做必须恰好匹配一次的文本替换（前者两处，后者一处）；最后以原文件名和原行号编译，使 traceback 仍指向真实位置。摘要覆盖 `inspect.getsourcelines()` 确定的函数行范围内的原始文件字节，包括空白、注释和换行，不做格式归一化。之所以不检查 JAX 版本号，是因为 nightly 的版本号不能标识函数内容，真正需要保证的是函数源码没有变化。

当前摘要来自 JAX commit `886d2370c1c959d210f522e352e3ddc6bcff7d6c` 的 [jaxpr_subcomp](https://github.com/jax-ml/jax/blob/886d2370c1c959d210f522e352e3ddc6bcff7d6c/jax/_src/pallas/mosaic/lowering.py#L1746) 和 [_lower_jaxpr_to_for_loop](https://github.com/jax-ml/jax/blob/886d2370c1c959d210f522e352e3ddc6bcff7d6c/jax/_src/pallas/mosaic/lowering.py#L4413)。仓库和安装包只保存摘要与固定版本链接；运行时直接读取已安装的 JAX，不下载或分发完整函数副本。JAX 修改其中任一函数后，需要先核对上游 diff、重新验证适配逻辑，再更新摘要和链接。只核对这两个函数并不能覆盖它们调用的其他 JAX 代码：`inlined_func_call` 的问题正是在 `jaxpr_subcomp` 源码不变时出现的，因此升级 JAX 后仍须运行复现脚本。

### JAX jit 追踪缓存

`jnp.maximum` 和 `+` 对应的 `jnp.add` 等函数本身是 `jax.jit` 函数。JAX 用 `pjit._infer_params_cached` 按函数和参数的 aval（含 sharding）缓存追踪出的 jaxpr；kernel 中以相同的 aval 再次调用时，直接复用第一次调用的 jaxpr，其中方程的 traceback 仍指向第一次调用。于是后一次调用生成的指令，来源同时带有自己的行号和第一次调用的行号。例如 [examples/pallas/reduction.py](../../examples/pallas/reduction.py) 第 25 至 27 行连续三次调用 `jnp.maximum(maximum, pltpu.roll(...))`。不设置 abstract mesh 编译时，第 26 行的 `max` 显示为 `reduction.py:26:14-26:72 <- reduction.py:25:14-25:72`，libtpu 的 `loc(...)` 也写成第 25 行。同一行 `roll` 生成的指令也带上了第 25 行，`roll` 本身不是 jit 函数，这一点的机制尚未查明。

缓存是否命中还取决于有没有 abstract mesh。从 Ref 读出的值，其 aval 的 sharding 使用空 mesh；在 `jax.sharding.use_abstract_mesh(...)` 中，jit 函数输出的 sharding 带有当前的 abstract mesh。因此第 25 行的输入来自 Ref，第 26 行的输入来自上一个 jit 输出，两者的缓存键不同，第 26 行被重新追踪；第 27 行的输入与第 26 行相同，仍然命中缓存。追踪整个 kernel 时，缓存未命中从 5 次变为 7 次，多出的是第 26 行的 `jnp.maximum` 和第 31 行的 `jnp.add`。这时第 26 行的来源只剩本行，机器码不变。离线编译必须设置 abstract mesh，Pallas lowering 才能读出 TPU 代际，所以 [examples/pallas/common.py](../../examples/pallas/common.py) 在设备上编译时也设置它，使两种方式导出的清单逐字节相同。这是 JAX 追踪层的行为，tpuasm 如实记录 JAX 给出的位置，不做修正。

### 编译器 flag

需要打开 `xla_jf_emit_annotations`、`xla_mosaic_enable_llo_source_annotations` 和 `xla_xprof_register_llo_debug_info` 三个 flag。其中最后一个决定是否创建 LLO SourceMap；只打开注释相关的 flag，并不会产生来源信息。libtpu 可能已经初始化，也可能还没有，所以这三个 flag 既通过原生 flag setter 设置，也追加到 `LIBTPU_INIT_ARGS`。退出上下文时两处都恢复原值。

### 原生 hook

[tc_source_native.cc](../../src/tpuasm/tc_source_native.cc) 替换了 libtpu 中若干处 `call` 指令，每个 hook 完成自己的工作后调用原函数：

- **发射 hook**：替换 bundle 发射过程中对单条指令发射函数的调用。它沿指令→region→module 找到 SourceMap，在原生代码中序列化并筛选，读出 HLO 名，把记录追加到指令注释中。追加通过原生 setter 完成，setter 会复制字符串，所以字符串的生命周期由 LLO 指令负责，能覆盖延迟发射和 bundle finalization。
- **替换 hook**（两处指令替换、一处 region 替换）：LLO 优化把一个值替换为新值时，把新值的 ordinal 加入所有含有被替换子图 ordinal 的 SourceInfo。子图的范围是旧值的操作数图中，不经过新旧表达式共同边界就能到达的节点。如果新值本来就在旧值的图中，说明优化是用已有的值消去了一个运算，这时不传播，否则该值的其他用途也会错误地继承这个来源。
- **合并 hook**：BF16 load/store 合并会把两个候选合成一个新值。hook 先调用原来的注释 setter，再把两个候选的来源传播给新值。第二个候选的指针位于调用方的寄存器中，由 trampoline 前缀代码放进第三个参数。
- **load 合并 hook**：load/store 优化器的 `SimplifyVectorCombineWithSublanesPerStrideInternal` 会把两个各取一半 sublane 的 load 合成一个新 load（注释为 `combined load` 或 `combine strided load`），原来的两个 load 被删除。不处理时，新 load 没有来源，Pallas 的 `get` 在清单中就没有记录。hook 同样先调用注释 setter，再把两个原 load 的来源传播给新值；两个原 load 的指针保存在调用方栈帧中，由 trampoline 前缀代码读入第三、第四个参数。目前只在 libtpu 0.0.49 登记了这两个调用点。
- **store 注释 hook**：store 的外层 annotator 已经分配了 store 槽，内层 `StoreCommon` 的 annotator 于是看不到新增的槽；但它析构时仍会清空 emitter 的当前注释，导致外层 annotator 也失去来源。hook 在这个析构调用前后保存并恢复 emitter 的注释视图，让外层 annotator 为实际新增的槽记录来源。四种 store（普通 / indexed × 有 / 无 offset）的调用点分别登记。

hook 维护五个计数器：emitted、annotated、failures、rewrites、propagated。退出上下文时，failures 不为零就抛出 `RuntimeError`；annotated 为零则发出警告，这通常意味着编译命中了缓存。

### 版本相关的部分

每个 libtpu 版本的来源后端由两部分组成：`source_backends/` 中的版本文件给出 hook 调用的各函数的 VA，包括发射、注释读写、SourceMap 序列化、HLO module 获取、操作数访问、ordinal 追加、两类替换、合并候选的访问、`ScopedAnnotator` 析构和 flag 读写；`tc_source_backend.py` 中的 `SourceBackend` 记录每个调用点的原始字节、hook 名和可选前缀（`calls`），以及安装前核对的字节（`signatures`，分类见下文）。

hook 读取的对象偏移不在版本文件中，而是直接写在公共的 `tc_source_native.cc` 中，因为已支持的各版本布局相同。它们涉及：指令的 region、ordinal 和操作数个数；region 的 module；module 的 SourceMap、root region 和 HLO 名；HLO module 的名称和 ID；SourceMap 与 SourceInfo 中的 repeated 字段；emitter 的当前注释；libc++ 字符串的表示。这些偏移由 `signatures` 中的第三类字节守护。若某个版本的布局不同，应把这些偏移改为按版本定义的常量，而不是修改公共文件，以免破坏已有版本。

记录的内容也随版本变化，因为 SourceInfo 由 libtpu 从 MLIR location 转换而来。使用同一 JAX 和同一补丁离线编译 v4 示例时，0.0.48 与 0.0.48 nightly 的 22 份清单相同，但与 0.0.49 的清单有 21 份不同。其中 20 份只有来源注释不同：0.0.48 系列的大部分记录在 scope 栈末尾多出一段方程自身的 primitive 名称，例如 0.0.49 的 `[get; …]`、`[top_k/argmax/top_k; …]` 在 0.0.48 中为 `[get/get; …]`、`[top_k/argmax/argmax/top_k; …]`。另外，`rms_norm` 的机器码也不同；两个版本各自打补丁与不打补丁编译出的机器码相同，所以这是编译器调度的差异，与来源捕获无关。仓库中的清单用 0.0.49 生成。

### 补丁的安装与恢复

[tc_compiler.py](../../src/tpuasm/tc_compiler.py) 中的 `CompilerSourceMapping` 负责安装和恢复补丁：

1. 先选出编解码后端，再按其标识在 `SOURCE_BACKENDS` 中查找来源后端。来源后端单独登记，有编解码后端不代表支持来源捕获。
2. 核对 `signatures` 中的每一段字节，磁盘上的 ELF 文件（经 `PT_LOAD` 把 VA 换算为文件偏移）和内存中已加载的代码都要核对，且所在映射必须是私有的 r-x 映射。`signatures` 包含三类字节：hook 调用的每个函数的入口、各个调用点本身，以及 libtpu 中访问同一批对象偏移的代码片段。第三类是对象布局的防护：hook 按固定偏移读取 emitter、module、SourceMap 等对象，libtpu 若改变了这些布局，相应代码片段的字节也会改变，安装就会被拒绝。
3. 从 `/proc/self/maps` 取得 libtpu 的加载基址。
4. 在距所有调用点 ±2 GiB 以内（`call rel32` 的可达范围）用 `MAP_FIXED_NOREPLACE` 申请一页，作为 gateway。每个调用点在 gateway 中对应一段 32 字节的 trampoline，内容是可选的前缀代码加上 `jmp [rip+0]` 和 hook 的绝对地址。写入后读回核对，再把该页设为 RX。
5. 安装时把每个 `call rel32` 改为调用对应的 trampoline。修改期间代码页暂时可写，改完恢复 RX；任何一处失败都回滚已改的调用点。
6. 退出时先确认补丁字节没有被改动，再恢复原调用、flag、环境变量和 lowering 函数，确认所有原字节都已复原，最后释放 gateway。如果恢复失败，gateway 可能仍被执行中的代码引用，因此保留状态对象、不释放 gateway，并拒绝在本进程中再次安装，要求重启。

嵌套的上下文复用同一个状态。锁只能协调通过本接口进行的调用，不能保护绕过接口、同时进行编译的其他线程。命中编译缓存时不会产生来源记录，使用时需要清除 JAX 缓存并禁用持久编译缓存。

## 恢复

[tc_source_mapping.py](../../src/tpuasm/tc_source_mapping.py) 的恢复流程只依赖 executable 中保存的数据和实际的解码结果。

### 程序身份

`ProgramSourceMap` 分别保存以下身份字段：record、映像索引、segment set / segment 索引、初始化数据中的偏移、segment set 的 hash、core program 的 fingerprint、可选的 compilation ID、metadata 所在 record 和 metadata 的 program ID。这些字段不要求彼此相等。不同程序映像之间不按名称或 ordinal 拼接。

metadata 取紧跟在 core 记录之后、且含有 overlay 信息的那条记录。如果一条 core 记录中有多份代码映像，而 metadata 中没有能区分映像的标识，就无法确定 metadata 属于哪份映像。这时每份映像都得到空映射和一条歧义诊断，不能把同一组来源复制给所有映像。

### 坐标

注释表在两种坐标下给出：field 4 以发射时（链接前）的 PC 为键；field 30 以程序映像 PC 为键，内容是链接器已经重定位的 overlay 前缀和后缀。发射 PC 按所在的 overlay 换算为程序映像 PC：

```text
image_start = encoded_word_offset * 编码字节数 * 每块 bundle 数 / 块字节数
body_start  = image_start + prefix_size
image_pc    = body_start + emitted_pc - start_bundle_number
```

`encoded_word_offset` 的单位随目标不同：v4 为 512 字节的块（每块 10 个 bundle，`image_start = encoded_word_offset * 10`），v6e 为 32 字节（每 512 字节 8 个 bundle，`image_start = encoded_word_offset / 2`）。v6e 的单位是从示例的元数据推断的：按 32 字节换算后，示例的注释都落在解码结果中被占用的槽上，没有诊断；按 64 字节或 512 字节换算，第二个 overlay 会落到程序映像之外。`Overlay` 同时保存原始的 `encoded_word_offset` 和换算后的 `image_start`。换算以 map 的键为准：continuation 尾部的 `Annotation.pc` 可能从零重新开始计数，不能代替键。

### 槽与来源

注释中的槽名是 libtpu 的枚举名（如 `SLOT_VECTOR_EXTENDED_0`），先按目标映射为物理槽名，再检查实际解码的 bundle 中该槽是否被占用；不符合时记录诊断，并丢弃这条注释。槽的归属以解码器为准。v6e 的枚举名为 `SLOT_SCALAR_ALU_0/1`、`SLOT_DMA`、`SLOT_VECTOR_ALU_0..3`、`SLOT_VECTOR_STORE`、`SLOT_VECTOR_LOAD_0/1`、`SLOT_VECTOR_MISC`、`SLOT_VECTOR_EXTENDED_0/1` 和 `SLOT_VECTOR_RESULT_0/1`，与物理槽一一对应。

注释中的 v1 记录解析为 `InstructionOrigin`，并去重；其余文本保留为 `compiler_annotation`。文本中 `loc(...)` 明确记录的位置单独解析为 `annotation_locations`，但不据此补造 primitive 或 scope。

### 函数区间

HLO 为 custom-call 的 symbol 视为函数。它的每段 `child_instructions` 按 overlay 分别裁切、换算，保留区间之间的空洞，不用最小和最大 PC 把区间填满。去重后共享的机器代码可以同时属于多个函数，它们的区间允许重叠；每个槽记录覆盖该 PC 的所有函数。

### 状态

至少一个槽有 v1 来源时，状态为 `captured`，但这不表示所有指令都有来源。否则状态为 `absent`，附带提示重新编译的诊断，同时仍保留编译器原生注释和函数归属。

已知的缺口：自动 MXU 分配会重新生成 push、matmul 等指令，这些指令没有来源记录。v4 的复现脚本为此关闭了自动 MXU 分配；v6e 的示例使用默认设置，所以矩阵乘法相关的槽大多没有来源。

### 依赖的元数据字段

代码按字段号读取以下消息：

| 消息 | 使用的字段 |
|---|---|
| `TpuCoreProgramProto` | fingerprint=3、compilation_id=4、tensor_core=5、memory_segments=8 |
| TensorCore / sequencer | sequencer=1；sequencer_type=3，值 1 表示 TensorCore |
| `TpuMemorySegmentSetProto` | segments=2、initialized_data=3、hash=4、compression=5 |
| memory segment / range | segment_type=1（CODE=1）、range=3；start_byte_offset=1、size=2 |
| `CompilerMetadata` | annotation_metadata=4、symbol_table=8、program_id=9、overlay_metadata=10、module_name=20、overlayer_annotation_metadata=30 |
| `AnnotationMetadata` / `Annotation` | annotations=1（map）；pc=1、slot_to_text=2、annotation=3 |
| `Overlay` | prefix_size=1、start_bundle_number=2、end_bundle_number=3、suffix_size=4、encoded_word_offset=5 |
| `Symbol` / HLO | parents=21、child_instructions=23、hlo_instruction=70；op=3、name=4、deduplicated_name=8 |

## 输出

这些 dataclass 定义在同一模块中。JSON 由 `dataclasses.asdict` 生成并加上 `schema_version`，所以修改任何 dataclass 字段都会改变 JSON schema，需要同时提高版本号。当前为版本 2，相对版本 1 增加了 `ProgramSourceMap.target` 和 `Overlay.image_start`。本模块的 `SourceLocation` 表示 Pallas 源码位置；汇编诊断位置是另一个类型 `AssemblyLocation`。

`source_comments` 把映射渲染为 `#` 注释：函数区间、bundle 注释和诊断放在 bundle 之外，逐槽来源放在对应指令行的末尾，换行符被转义。注释不参与编码。

## 验证

[tests/reproduce_tpu_v4_tc_source_mapping.py](../../tests/reproduce_tpu_v4_tc_source_mapping.py) 需要在 TPU 上运行，它检查以下内容：机器码与数值不受补丁影响；load/get、push、matmul、pop、store/swap 各自的来源关联；normal 与 xpose 两种布局下 DWG 和 push 的操作数；完全静态展开循环中每条指令保留自己的 primitive；安装中途失败时的回滚，以及嵌套上下文与异常退出后，调用点、flag、环境变量和两个 lowering 函数都能恢复原状。v6e 没有对应的设备测试：[examples/pallas/](../../examples/pallas/) 的示例在 `run_all.sh --aot tpu-v6e-tc` 下离线编译时捕获来源，每个示例在打补丁和不打补丁时编译出的机器码相同，但没有检查数值。同一批示例在 TPU v4 上编译运行得到的 23 份清单，与按 v4 参考拓扑离线编译的结果逐字节相同，包括来源注释。

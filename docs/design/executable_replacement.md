# 回灌与执行

导出的清单可以编辑后重新汇编为程序映像，但程序映像本身不能执行。要在设备上观察修改的效果，例如在指定程序段前后插入读周期计数器打点、核对某个 selector 的执行语义，或者在不触发编译器重新调度的前提下替换一条指令，还需要把新映像放回 serialized executable，再交给 runtime 装载。tpuasm 为此提供以下接口：

- `replace_executable_programs(serialized, images)`：用新的程序映像替换 executable 中的 TensorCore 程序映像，返回新的 executable 字节。它只处理字节，不需要 TPU。
- `insert_executable_bundles(serialized, insertions)`：按显式插入点增加 bundle，迁移机器码和元数据，返回新的 executable 字节。
- `load_executable(serialized, template)`：借用一个已编译 JAX 对象的调用约定，装载给定的 executable，返回可以直接调用的 `Compiled`。

等长替换的流程如下，可运行的版本见 [examples/pallas/replace_program.py](../../examples/pallas/replace_program.py)；插入的用法见下文[插入 bundle 与变长写回](#插入-bundle-与变长写回)：

```python
serialized = bytes(compiled.runtime_executable().serialize())
(record, index, image), = executable_programs(serialized)
source = format_assembly(image, target='tpu-v4-tc')
edited = assemble_listing(source.replace('vclamps.8x128.f32 v1, v0, 1.0', 'vclamps.8x128.f32 v1, v0, 0.5'))
patched = replace_executable_programs(serialized, {(record, index): edited})
result = load_executable(patched, compiled)(x)
```

命令行也可以完成写回这一步：先用 `--input-format executable --output-dir` 导出清单，编辑后用 `--output-format executable --replace RECORD:INDEX LISTING --output FILE` 写出新的 executable；插入用 `--insert`，见下文。清单的 `.target` 必须与 executable 的目标相同。命令行不能执行程序，写出的文件仍需在 Python 中用 `load_executable` 装载。

## 插入 bundle 与变长写回

`replace_executable_programs` 保持等长替换的语义。仅有修改前后的两份映像，不能确定重复空 bundle 的对应关系，也无法确定指向插入点的分支是否应该执行新增代码。变长写回因此使用 `insert_executable_bundles`，由调用者给出原映像中的插入位置和汇编片段：

```python
from tpuasm import BundleInsertion, insert_executable_bundles, load_executable

patched = insert_executable_bundles(serialized, {
    (record, index): [
        BundleInsertion(start_pc, '.target tpu-v4-tc\n{ s0: srdreg.lcclo s13 }'),
        BundleInsertion(end_pc, '.target tpu-v4-tc\n{ s0: srdreg.lcclo s20 }'),
    ],
})
result = load_executable(patched, compiled)(x)
```

`start_pc`、`end_pc` 是输入映像的 bundle 编号，均在修改前的坐标中指定。每个片段插在对应原 bundle **之前**，无需凑齐 512 B 块。示意中的寄存器必须由调用者确认可用；完整可运行的例子见 [insert_program.py](../../examples/pallas/insert_program.py)，它先以两条加法预留输出寄存器，再将加法改成寄存器自身加零，把独立读数插在前面，经原来的 store 和 DMA 返回读数。循环和数值计算仍沿用编译器生成的代码。

`BundleInsertion.branch_target` 默认是 `'inserted'`：原来的直接分支若指向该插入点，先执行新增片段。设为 `'original'` 时，这些分支跳过片段。顺序流入总会执行片段。不同插入点可以选择不同策略；同一位置的多个 bundle 写在同一个片段中。片段内的直接分支可以使用局部标签或局部编号，也可以跳到片段末尾，随后继续执行原 bundle。

插入点不能落在原分支的延迟窗口内。对原映像中的每条 `sbr` 或 `scall`（`.rel`、`.abs`、`.ind`），若其位置为 `b`，接口拒绝 `b < image_pc <= b + delay`：v4 的 `delay=1`，v6e 的 `delay=4`。分支自身之前，以及最后一个延迟 bundle 之后，都可以插入。该检查与分支目标策略无关，也不因分支带谓词或使用间接地址而省略。它保留原来的延迟指令序列。片段自己带分支时，每条 `sbr` 或 `scall` 的完整延迟窗口必须落在片段内，否则接口拒绝，需要在片段末尾补足空 bundle；不这样限制的话，紧随片段的原 bundle 会成为该分支的延迟槽，分支跳回片段内部时会被重复执行。窗口内放什么指令仍由调用者负责。

命令行的等价入口是：

```sh
python -m tpuasm original.bin --input-format executable --output-format executable \
  --insert 1:0:517 begin.tpuasm --insert 1:0:530 end.tpuasm --output patched.bin
```

这里的编号仅为语法示例，应从实际程序确定。`--insert` 可以重复；PC 接受十进制或 `0x` 十六进制。`--insert-branch-target original` 让该次命令中指向插入点的直接分支跳过片段。可以同时给出 `--replace` 和 `--insert`，此时先等长替换，再按原编号插入。

实现会同步更新以下内容：

- 直接分支和调用的目标、相对位移；程序前缀中装载主程序的块数。前缀和主程序的起始位置不变，因此起始块立即数保持原值。没有改变的 bundle 保留原机器字。
- 原有尾部内容完整保留，包括以 continuation 分支及其空 delay bundle 结束的程序。额外对齐填充由新汇编的无条件 halt 构成，不复制可能仍有语义的末条指令。片段中的 `{}` 或 `.empty N` 由汇编器生成真正的空 bundle，不会把 halt 当成空指令插入。
- overlay 的 `end_bundle_number` 增加实际插入数，`suffix_size` 增加额外对齐填充；`executable_size`、HBM summary、`OVERLAYS` block 和 `overlays` allocation 的代码大小同步增加。
- 符号表的 `symbol_instruction_ranges` 和每个 symbol 的 `child_instructions` 按插入点拆分。旧 bundle 的归属和空洞都保留，新增 bundle 不归属于原 HLO。主程序注释的 key 与 PC 一起移动；尾部 continuation 注释只移动 key，保留局部 PC。新增 bundle 仅标注 `tpuasm: inserted bundle`。原 metadata 没有 `annotation_metadata` 表时，新建一张仅包含新增位置记录的表。
- v4 未压缩的 trap 元数据逐条迁移。仅接受 `sequencer_type=1`、tag 等于主程序 overlay 编号，且 `overlay.image_start + pc - 1` 确实落在原程序 `shalt` 上的条目；按该 halt 之前的实际插入数移动 PC，保留消息及其他字段。这个 PC 是相对 overlay 映像起点的 halt 后续位置，不能直接当 emitted PC。其他坐标形式、压缩条目和 v6e trap 仍拒绝。
- 代码 segment 的范围和 `initialized_data`；同一数据区内位于代码之后的其他 segment 平移其数据偏移，内容不变；嵌套 protobuf 和 executable 记录的长度前缀重新编码。未知字段保留原 wire bytes。
- segment set hash 与三处 fingerprint。插入后的 fingerprint 也包含修改后的 metadata，避免相同机器码、不同插入归属共用身份。

### v4 与 v6e 的装载尺寸

libtpu 0.0.49 的独立长短编译对照中，前缀占用固定，主程序的实际 bundle 数、suffix 和装载块数一同变化。下表使用同一个标量载体，仅改变 Python 展开的计算次数：

| 目标 | 展开次数 | 全映像 / B | 主程序 body | prefix | suffix | 装载块数 |
|---|---:|---:|---:|---:|---:|---:|
| v4 | 0 | 33280 | 504 | 1 | 5 | 51 |
| v4 | 4 | 33792 | 514 | 1 | 5 | 52 |
| v4 | 16 | 35328 | 538 | 1 | 11 | 55 |
| v4 | 32 | 36864 | 570 | 1 | 9 | 58 |
| v6e | 0 | 43008 | 340 | 1 | 11 | 44 |
| v6e | 4 | 43520 | 350 | 1 | 9 | 45 |
| v6e | 16 | 45056 | 374 | 1 | 9 | 48 |
| v6e | 32 | 47104 | 406 | 1 | 9 | 52 |

v4 每块 10 个 bundle，前缀占 14 块；v6e 每块 8 个 bundle，前缀占 40 块。`encoded_word_offset` 的单位仍不同：v4 为 512 B，v6e 为 32 B。插入接口先检查原装载块数与映像一致，再增加 `(插入数 + 新填充数) / 每块 bundle 数`，不猜测隐含额外块。

装载指令是 program-start trampoline 中的一对立即数：`s0` 为主程序 overlay 的起始块，`s1` 为块数。起始块等于该 overlay 的 `encoded_word_offset` 换算成的块数，不是它在程序映像中的位置。两者在第一个 overlay 的偏移为 0 时相同；程序带有去重常量时，常量排在 overlay 之前，第一个 overlay 的偏移不为 0。例如 XLA 的 `psum` 程序有一块 512 B 的 `deduplicated constant`，前缀 overlay 的偏移为 1，主程序 overlay 的偏移为 15，在映像中却从第 14 块（bundle 140）开始，装载指令是 `simm.s32 s0, 15`、`simm.s32 s1, 142`。接口按 `encoded_word_offset` 识别这对立即数；插入不改变起始块。

旧交接记录曾把 v6e clamp 记为 664 个 bundle，并据此认为 `s1=44` 比映像多一块。当前仓库清单经解析与重新汇编实际为 **672 个 bundle、43008 B**，主程序是 `1 + 341 + 10 = 352` 个 bundle，即 44 块。重新编译得到相同机器码；编译期间在 `Overlay::PatchOverlay` 的尺寸路径观测到 352 个 bundle、22528 B、44 块，最终编码也使用 22528 B。旧记录中的计数不能用作 v6e 格式规则。

### 延迟数与容量单位的证据

以下证据来自当前 CPython 3.14t 的 libtpu 0.0.49，GNU build-id 为 `97e27df7268da25ab03e455e30dd86b0`。地址仅用于复核该二进制，不写进运行时实现。

- `PufferfishTarget` 构造函数在 ELF VA `0x19218711` 把目标对象 `+0xe34` 的 32 位字段设为 1；`GhostliteTarget` 在 `0x1921a12b` 向同一位置写入 `0x200000004`，该字段的低 32 位为 4。两者的 `SupportsFlexDelaySlots()` 都返回 false。
- `CodeGenerationHelper::bundle_for_delay_slots`（`0x15398c50`）从目标的 `+0xe34` 读取次数，逐次创建并标记延迟 bundle。对两个目标分别离线编译 clamp，在这个真实调用点读取到 1 和 4，与构造函数一致。因此目标表中的延迟数来自编译器配置及其消费路径，不是从清单中的空指令个数猜测。
- `WriteOverlayMetadata` 在 `0x153c6e3f` 将 `32760` 写入 `overlay_slot_size`。XDB 的 `GetEmittedBundleNumber(OverlayMetadata const&, long, OverlayInfo const&)`（`0x19c23f90`）先从 PC 减去 `overlay_slots_offset`，再对该字段取余；随后直接与 `prefix_size`、`end_bundle_number - start_bundle_number` 比较，并减去 prefix 得到 emitted bundle 编号。这条路径没有按目标乘除编码字大小，证明 `overlay_slot_size` 的单位是 **bundle**，包括 v6e。它与 `encoded_word_offset` 的 32 B 单位不同。

插入接口要求主 overlay 的总 bundle 数（含 prefix、原 suffix 和新增对齐填充）不超过该槽跨度。`32760` 是这里的 overlay 槽限制，不代表整颗 TensorCore 的全部 IMEM 容量。

## 程序身份

runtime 用程序身份识别已经装载的程序。只替换映像而不改变身份时，如果同一进程已经执行过原程序，runtime 会继续执行原程序，调用者看到的是旧结果，并且不会收到任何错误。

与身份有关的字段名取自 libtpu 内嵌的 protobuf descriptor（`tpu_core_program.proto`、`tpu_sequencer_program.proto` 和 xdb 的 `debugger.proto`）。在 serialized executable 中，fingerprint 的同一个值出现在三处，另有一个 segment set hash：

| 字段 | 位置 | 长度 |
|---|---|---|
| `TpuCoreProgramProto.fingerprint` | core 记录的 field 3 | 32 字节 |
| `TpuSequencerProgramProto.fingerprint` | core 记录的 `tensor_core(5) → sequencer(1) → field 4` | 32 字节 |
| `CompilerMetadata.executable_fingerprint` | 紧随 core 记录的 metadata 记录的 field 26 | 32 字节 |
| `TpuMemorySegmentSetProto.hash` | core 记录的 `memory_segments(8) → field 4` | 32 字节 |

v4 与 v6e 的 executable 都是这样。`compiled.runtime_executable().fingerprint` 返回的是第一行的值。

在 TPU v4、libtpu 0.0.49 上，用单设备的 clamp kernel 把 `vclamps` 的边界由 1.0 改为 0.5，然后分别改变不同的身份字段，再装载执行。新值是任意的 SHA-256 摘要。“先执行原程序”指在同一进程中装载修改版之前，已经执行过一次原 executable：

| 改变的字段 | 先执行原程序 | 未执行原程序 |
|---|---|---|
| 都不改 | 旧结果 | 新结果 |
| 只改 hash | 旧结果 | 新结果 |
| 只改 fingerprint（三处） | 旧结果 | 新结果 |
| hash 与三处 fingerprint | 新结果 | 新结果 |
| hash 与 core 记录的 fingerprint | 新结果 | — |
| hash 与 sequencer 的 fingerprint | 旧结果 | — |
| hash 与 metadata 的 fingerprint | 旧结果 | — |

由此可以得出以下结论：

- runtime 在首次执行时装载程序，而不是在编译时装载。在 core 记录的 fingerprint 和 segment set hash 两者中，任一项与已装载的程序相同，都会复用那个程序。
- 这两项都不根据内容校验。任意的新值都能正常装载执行，内容不变时沿用原值也不会报错。
- 修改后，原 executable 仍然执行原程序。

静态反汇编显示，libtpu 的 `SetProgramFingerprint` 在编译器内部的 program 表示上计算 SHA-256；segment set hash 也不等于 `initialized_data` 的 SHA-256。编译器给出的值无法从程序映像重新计算，因此 tpuasm 自行导出新值：

```text
新 hash        = SHA-256(原 hash + 新 initialized_data)
新 fingerprint = SHA-256(原 fingerprint + 该记录所有新 hash 的拼接 + 插入后的 metadata)
```

等长替换时上式的 metadata 部分为空；插入时使用更新身份之前的 metadata 字节。三处 fingerprint 同步改为同一个新值，改写前检查它们与原值相同。新值是确定的：同样的修改总是得到同样的身份，这时内容也相同，复用已装载的程序不会出错。映像与原来相同的替换不改变任何字节。`CompilerMetadata.program_id` 保持原值，它与 fingerprint 是不同的字段。

## 装载

`load_executable` 复用 JAX 的 `jax.experimental.serialize_executable`。`serialize(template)` 把 JAX 包装层连同 executable 一起 pickle：executable、device 和 client 都以 persistent id 表示，其中 executable 是 `('exec', 字节)`。tpuasm 继承 JAX 的 unpickler，只把 `'exec'` 中的字节换成给定的 executable，其余步骤与 `deserialize_and_load` 相同。因此参数 pytree、aval、分片、输出结构以及 device 的解析都来自 template。替换发生在 pickle 解析之后，所以装载本身不限制 executable 的长度。

实现依赖 JAX 的两个私有接口：`serialize_executable._JaxPjrtUnpickler` 和 `jax.stages.Compiled` 的构造参数。JAX 修改它们时会直接报错，不会静默装载出错误的对象。

执行设备默认取 `template.runtime_executable().local_devices()`，它的顺序与设备分配一致。例如 v4 上 `jax.make_mesh` 得到的顺序是 0、2、1、3，`local_devices()` 返回的顺序相同。离线编译得到的 template，其设备属于 compile-only 客户端，需要用 `devices=` 传入 ID 相同的真实设备。

## 验证

在 TPU v4（libtpu 0.0.49，本 host 四颗 Megacore 芯片）上，以下情形都在同一进程中先执行原程序，再执行替换后的程序，两者的结果都与 NumPy 逐项相等：

- 单设备的 clamp（[examples/pallas/replace_program.py](../../examples/pallas/replace_program.py)）。之后再次执行原 executable，仍然得到原结果。导出的清单与 `clamp` 示例只差被修改的这一行，来源注释不变，因为映像长度不变，原来的元数据仍然适用。
- 四个设备的 `shard_map`，每个设备的输出加上 10 倍的设备序号，说明默认的执行设备顺序正确。之后再次执行原 executable，仍然得到原结果。
- 同一颗 Megacore 芯片的两个 TensorCore 执行同一份程序，各自处理一半数据。
- 按 [v4_2x2x1_megacore.topology](../../examples/pallas/v4_2x2x1_megacore.topology) 离线编译得到 template，替换后通过 `devices=` 在真实设备上执行。原 executable 与替换后的 executable 都得到正确结果。

作为反向对照，在两个 TensorCore 的情形中，把替换后的 executable 除映像以外的字节恢复为原值（即保留原身份），runtime 执行的仍是原程序。

`replace_program.py` 在 `run_all.sh --aot` 中只做替换和导出，CI 借此检查 v4 与 v6e 的替换结果都能精确导出。它也已在单芯片 v6e 真机上通过原程序、修改版、原程序的数值对照。

### 变长验证

[insert_program.py](../../examples/pallas/insert_program.py) 在 v4 真机上对循环次数 `0、1、4、31` 检查原计算结果，并在同一进程中先后执行原程序、插桩程序和原程序。只有两条独立读数时周期差为 `9、16、37、226`，循环内部再插入 16 个空 bundle 后为 `9、30、93、660`，分别满足 `9 + 7N` 和 `9 + 21N`。额外的 32 空 bundle 临时对照满足 `9 + 37N`。这些是该载体的实测结果，空 bundle 可能与其他操作的等待重叠，不据此把任意新增 bundle 都解释为一周期。JAX 装载后重新序列化，程序映像逐字节保留。

v6e 将循环乘法排在回跳分支后的延迟窗口中，所以示例在该目标上把循环内空 bundle 插在回跳分支之前，保留原来的乘法、加法和 store 延迟序列。单芯片 v6e 真机上，同样的循环次数得到独立读数 `16、26、56、326`，增加 16 个空 bundle 后为 `16、39、108、729`，分别满足 `16 + 10N` 和 `16 + 23N`。计算值、哨兵值、原程序再次执行与装载后重新序列化均通过。严格线性和 `N=0` 两个版本相等的断言因此保留；示例在断言之前打印实际读数，便于后续目标或编译器变更时诊断。

v4 的该示例在设备上与离线编译的清单一致。v6e 的 `insert_program` 和 `replace_program` 按 `v6e:1x1` 临时离线编译，所得清单均与单芯片真机逐字节相同；仓库清单仍按 `v6e:2x2` 离线生成。两个目标均用多个插入长度和两种分支策略核对了全部直接分支、符号范围、注释坐标、代码大小、程序身份、尾随数据 segment 偏移和 CLI/API 一致性。分发检查也在仓库外运行安装包中的示例。

延迟槽拒绝检查覆盖两个目标的六种分支与调用形式、两种分支目标策略，以及延迟窗口的全部内部位置；分支之前和窗口之后的边界均接受。片段末尾带分支时，两个目标都在补足 1 个（v4）或 4 个（v6e）空 bundle 之前拒绝、之后接受。另检查了按 bundle 计算的容量边界和原注释表缺失时的新建行为。

## 不支持的情形

- **BCS**：executable 中保存的是 semantic protobuf，而不是机器字节。写回时需要重新生成 semantic body，并修改嵌套的长度前缀。
- **任意映像缩短或重排**：插入接口需要明确的旧到新 bundle 对应关系，不从两份映像猜测。
- **多个主程序 overlay、单 core 记录中多个代码映像、前缀或尾部 continuation 内部插入**：尚未校准这些布局，明确拒绝。需要可识别的 overlay、符号表和 program-start trampoline 注释。带需额外迁移的 fusion、breakpoint 或上述范围以外的 trap 元数据时也拒绝。
- **自动资源分配与调度**：不分配寄存器、缓冲区或信号量，不重新调度流水线。原分支延迟窗口由接口检查并拒绝插入，片段分支的延迟窗口必须留在片段内；窗口内的指令与数据依赖仍由调用者安排。间接跳转及存放在寄存器、内存中的代码地址由调用者维护。
- **压缩的代码段**：提取时就会拒绝。

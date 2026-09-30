---
name: add-libtpu-backend
description: 为 tpuasm 登记新的 libtpu 构建（新 release、nightly 或新 Python ABI）：下载 wheel，从已登记后端出发定位新地址，写入 TC/BCS 编解码后端和 TC 来源后端，并完成离线、TPU、旧版本回归和分发验证。升级 JAX 后来源捕获失效时，以及删除已登记的旧构建时，也用本流程。
---

# 为 tpuasm 添加 libtpu 后端

一个 libtpu 构建（由 GNU build-id 唯一确定）在一种 Python ABI 下，每个支持的目标需要一份编解码文件，另需一份来源文件：

| 文件 | 登记位置 | 作用 |
|---|---|---|
| `src/tpuasm/native_backends/libtpu_<stem>_tpu_v4_tc.cc` | `backends.LIBTPU_RELEASES` | v4 TC 编解码与逐槽校验 |
| `src/tpuasm/native_backends/libtpu_<stem>_tpu_v4_bcs.cc` | 同上 | v4 BCS 编解码 |
| `src/tpuasm/native_backends/libtpu_<stem>_tpu_v6e_tc.cc` | 同上 | v6e TC 编解码与逐槽校验，可选 |
| `src/tpuasm/source_backends/libtpu_<stem>.cc` | `tc_source_backend.SOURCE_BACKENDS` | TC 编译来源捕获，v4 与 v6e 共用 |

v6e 后端是可选的：`LIBTPU_RELEASES` 的每一行列出该版本支持的目标，没有登记 v6e 的版本对 `tpu-v6e-tc` 报告没有后端。登记 v6e 时还要按第 7 节重新生成字段表。

硬件程序格式在 `targets.py` 中按代际和执行单元定义，与 libtpu 版本无关，本流程不修改它。各常量的含义见[原生后端设计](../../../docs/design/native_backend.md)和[来源映射设计](../../../docs/design/tc_source_mapping.md#版本相关的部分)。

## 约定

下文用到以下变量：

- `PY`：目标 Python 解释器，其 ABI 必须与要登记的 wheel 相同。
- `WORK`：仓库外的临时目录，存放 wheel、符号表、临时 venv 和验证产物。
- `NEW_VERSION`、`NEW_LIB`：要登记的 libtpu 版本号及其 `libtpu.so`。
- `REF_VERSION`、`REF_LIB`：已登记版本中与新版本最接近的一个，及其 `libtpu.so`。

所有包操作都用 `"$PY" -m pip`，临时环境用 `"$PY" -m venv`，不使用 uv 的任何命令。不要为本流程新建测试框架；一次性的检查脚本放在 `$WORK`。

## 1. 准备环境和两份 libtpu

下载新旧两个 wheel。正式版直接从 PyPI 下载；nightly 在 [libtpu nightly wheel 索引](https://storage.googleapis.com/libtpu-wheels/index.html)找到 URL 后，把 URL 传给 `pip download`。参考版本只需解包，不必安装：

```sh
"$PY" -m pip download --no-deps "libtpu==$NEW_VERSION" -d "$WORK/new"
"$PY" -m pip download --no-deps "libtpu==$REF_VERSION" -d "$WORK/ref"
"$PY" -m zipfile -e "$WORK"/ref/libtpu-*.whl "$WORK/ref/x"
REF_LIB="$WORK/ref/x/libtpu/libtpu.so"
readelf -nW "$REF_LIB" | grep 'Build ID'
```

如果目标环境本身就要切换到新 libtpu（例如同时把 JAX 升级到 HEAD），先装 JAX，再用 `--no-deps` 覆盖 libtpu。顺序不能反：`jax[tpu]` 固定了 libtpu 版本，后装 JAX 会把 libtpu 换回去。

```sh
"$PY" -m pip install "$JAX_SOURCE[tpu]"
"$PY" -m pip install --no-deps "$WORK"/new/libtpu-*.whl
```

如果目标环境必须保留原来的 libtpu，就建临时 venv。从一个 venv 创建的新 venv 不继承前者的包，即使指定 `--system-site-packages` 也不行；需要 JAX 时，用 `.pth` 把共享环境追加到 `sys.path` 末尾，这样临时 venv 里的 libtpu 仍然排在前面：

```sh
"$PY" -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install --no-deps "$WORK"/new/libtpu-*.whl
echo "import site; site.addsitedir('<共享环境的 site-packages>')" > "$WORK"/venv/lib/python3.*/site-packages/zz_shared.pth
```

之后所有命令都使用实际装有新 libtpu 的解释器，下文仍写作 `$PY`。打印 `importlib.metadata.version('libtpu')`、`distribution('libtpu').locate_file('libtpu/libtpu.so')` 和 JAX 版本，确认导入的确实是目标文件，并记下 `NEW_LIB` 与 build-id。

## 2. 生成候选地址

在仓库根目录运行：

```sh
PYTHONPATH=src "$PY" tools/locate_libtpu_backend.py "$REF_LIB" "$REF_VERSION" "$NEW_LIB" > "$WORK/locate.txt"
```

脚本读取参考版本已登记的全部 VA：编解码常量、来源后端函数、hook 调用点和 signatures。它把每个 VA 换成“参考库中的符号 + 偏移”，再到新库里找同名（mangled）符号，得出候选 VA。脚本只负责提出候选，每一行都要按下表处理后才能采用：

| 输出 | 处理 |
|---|---|
| `bytes equal` 且候选唯一 | 直接采用。 |
| `bytes differ`，位于函数入口 | 用 `objdump` 对照新旧反汇编。若差异只在 rel32 调用目标、RIP 相对位移或栈帧大小，采用新地址；否则按函数体已改变处理，重新核对该函数被依赖的行为。 |
| `same-name candidates ...` | 同名的内部链接函数在不同编译单元中各有一份。反汇编相关调用方（例如 hook 所在的函数），看它实际调用的是哪一份。 |
| `no symbol` / `missing in target` | 符号被改名或剥离。从已确认的调用方、数据引用和重定位恢复，方法见 `$reverse-engineer-libtpu`。未定位完成前，不要运行猜测出来的地址。 |
| `kEmptyAnnotations`（数据，不比较字节） | 在 formatter 的调用方中找到 `lea <AnnotationMetadata_globals_>(%rip)` 及随后加上的偏移；再用 `readelf -rW` 确认该地址的重定位指向 `AnnotationMetadata` 的 vtable + 0x10。 |
| `call site kept at the same offset` | 调用点仍在函数内同一偏移，且目标仍是同一个被调函数，采用。 |
| `call moved` | 函数体被重新编排。脚本会列出新函数中调用同一被调函数的所有位置，逐一对照参数准备和控制流，把旧调用点对应到新调用点。 |
| signature `bytes differ` | 先判断旧字节在守护什么（见下一段），然后在新库中找到起同样作用的指令，取相同长度的新字节。 |

signatures 分三类。第一类是函数入口，与上面“函数入口”一行的处理相同。第二类是调用点及其参数准备，它的位置跟随对应的 call 一起确定。第三类是访问对象偏移的代码片段，例如 emitter 当前注释的偏移、module 中 SourceMap 的偏移、SourceInfo 中 ordinals 字段的偏移；这一类要在新库里找到访问同一字段的指令，并确认偏移常量没有变化。只要有一个偏移变了，就说明对象布局变了，处理方法见第 3 节。

对照函数时，先把地址和符号偏移归一化，再做 diff，只剩真正的代码差异：

```sh
norm() { objdump -d --no-show-raw-insn --start-address="$2" --stop-address="$3" "$1" | c++filt | sed -E 's/^ *[0-9a-f]+:\s*//; s/[0-9a-f]{6,} <([^>+]*)(\+0x[0-9a-f]+)?>/<\1>/; s/-?0x[0-9a-f]+\(%rip\)/RIP/'; }
norm "$REF_LIB" 0x<旧起点> 0x<旧终点> > "$WORK/old.s"
norm "$NEW_LIB" 0x<新起点> 0x<新终点> > "$WORK/new.s"
diff "$WORK/old.s" "$WORK/new.s"
```

函数的起点和大小用 `nm -nS "$NEW_LIB" | grep <mangled 名>` 查。符号值是 ELF VA。换算文件偏移时，要用包含该地址的 `PT_LOAD` 计算 `p_offset + VA - p_vaddr`，数据段尤其不能把 VA 直接当作文件偏移。

hook 会对调用点的寄存器状态做假设，调用点移动之后要逐一重新核对：

- **替换 hook**：`rdi` 是旧值，`rsi` 是新值；region 替换返回 `(rax, rdx)`。
- **合并 hook**：trampoline 前缀从某个被调用方保存的寄存器中取出第二个候选的指针。要确认该寄存器在调用点仍然保存着这个指针，并且两个候选之间的间距与 `tc_source_native.cc` 的假设一致。
- **load 合并 hook**（`source_combine_hook`，位于 `LoadStoreOptimizerImpl::SimplifyVectorCombineWithSublanesPerStrideInternal` 中两处设置 `combined load` / `combine strided load` 注释的调用）：trampoline 前缀从调用方栈帧读出两个原 load 的指针，放入 `rdx`、`rcx`。要确认这两个栈槽在调用点仍保存原 load，并把前缀中的 `rbp` 偏移随新版本更新；读取这两个栈槽的指令登记为第二类 signature。
- **store 注释 hook**：析构对象的第 0 个字段是 emitter。

## 3. 核对脚本证明不了的 ABI 和布局

符号和字节相同，只说明函数入口一样，不能说明调用方式和对象布局一样。复用公共桥接之前，逐项确认：

- **TC 和 BCS 的 program 对象**：从带 `proto2::Arena*` 参数的构造函数，以及拷贝构造中 repeated 字段的访问，读出对象大小、bundle repeated 字段（tagged pointer）的偏移和 bundle 数的偏移。`kDestroyProgram` 必须是非 deleting 的析构函数（D1/D2），不能是 D0。
- **`DecodeProgram` / `EncodeProgram`**：必须是接收 program 对象或 `Span` 的包装函数，不是带 decoder `this` 参数的虚函数；`StatusOr` 的成功标志和 payload 偏移；承载程序映像的 host DMA buffer 的指针、长度字段和释放方式。
- **formatter**：从真实调用方确认参数顺序、换行枚举取值、两个 metadata 参数，以及 `this` 是否带有实例状态。
- **bundle presence mask**：确认其偏移，以及各位与物理槽的对应关系。`kBundleSlotMasks` 按 `targets.py` 中 `slots` 的顺序排列；最终由逐槽拼回校验来证明。
- **libc++ 字符串**：长短两种表示的布局和释放方式；不能当成宿主 g++ 的 `std::string` 使用。
- **来源 hook 读取的对象偏移**：指令、region、module、HLO module、SourceMap、SourceInfo 和 emitter 中被读取的字段。它们写死在公共的 `tc_source_native.cc` 中，由第三类 signature 守护。

任何一项与现有版本不同时，都在新版本文件中实现同样的 C ABI（编解码为 `tpuasm_verify` / `tpuasm_program_proto` / `tpuasm_free`，来源为 `source_*` 导出函数），或者把相关偏移改为按版本定义的常量，不要直接改公共文件，以免破坏已有版本。

## 4. 写入并登记

1. 复制参考版本的各份 `.cc`，文件名中的 `<stem>` 取版本号并把 `.`、`+` 换成下划线，去掉 `nightly` 之类的后缀（参照已有文件名）。替换其中的常量，并在文件头注释中写明 libtpu 版本、GNU build-id 和 wheel 标签。
2. 在 `backends.py` 的 `LIBTPU_RELEASES` 数据表中加一行 `(version, stem, build_id, targets)`。版本号必须与 `importlib.metadata.version('libtpu')` 完全一致，`targets` 只列出已写好后端文件的目标。新的 Python ABI 需要各自登记和验证。
3. 在 `tc_source_backend.py` 中新增 `LIBTPU_<STEM>`：包含 `calls`（原始 call 字节、hook 名、前缀）和 `signatures`（新字节），并加入 `SOURCE_BACKENDS`，键为编解码后端的 `build_identifier`（`identifier` 去掉目标后缀，例如 `libtpu-0.0.49-cpython-314t-linux-x86_64`）。
4. 在[版本兼容性](../../../docs/compatibility.md)的“未发布”一节加入新构建（Python、libtpu 版本、build-id 和各目标的支持情况），并在 `CHANGELOG.md` 的“未发布”中写明。若同时升级了 JAX，一并更新该节的 JAX 版本与 commit，以及[来源映射设计](../../../docs/design/tc_source_mapping.md)中函数源码摘要对应的 JAX commit。已发布版本的小节记录的是发布时的事实，不修改。
5. 新增的文件类型若不在 `pyproject.toml` 的 package-data 中，要补上。
6. 在 `.github/workflows/tests.yml` 的 `offline` 矩阵中加入新版本，nightly 写 wheel 的 URL。若新版本取代 `0.0.49` 成为生成清单和字段表的版本，把其余 job 使用的版本和 `offline` 中按版本判断的条件一并改掉。若同时升级了 JAX 或 jaxlib，更新 `.github/actions/setup/action.yml` 中的 JAX commit 和 jaxlib 版本。

地址更新完后，再以新版本自身为参考跑一遍：`tools/locate_libtpu_backend.py "$NEW_LIB" "$NEW_VERSION" "$NEW_LIB"`。每一行都应是 `bytes equal` 或 `call site kept`；同名候选行则要求登记的地址出现在所列候选中。这一步能查出抄错的地址和字节。

## 5. 验证

按以下顺序执行，前一步失败就先修复，不要跳过。

1. **后端选择**：`PYTHONPATH=src "$PY" -c 'from tpuasm.backends import select_backend; print([select_backend(t)[0].identifier for t in ("tpu-v4-tc", "tpu-v4-bcs")])'` 应输出新版本的两个标识；登记了 v6e 时把 `"tpu-v6e-tc"` 也加入。不要通过改写 `RuntimeEnvironment.current()` 来冒充目标解释器。
2. **离线编解码**：

   ```sh
   PYTHONPATH=src "$PY" tests/reproduce_tpu_v4_tc.py
   PYTHONPATH=src "$PY" tests/reproduce_tpu_v4_bcs.py
   ```

   这两步检查 bundle 数、decode→encode 逐字节一致、逐槽拼回、编码歧义对和 BCS 的 semantic 互操作。若失败，依次排查函数入口、第 3 节的 ABI、输入格式。登记了 v6e 时，先按第 7 节重新生成字段表，再运行 `PYTHONPATH=src "$PY" tests/reproduce_tpu_v6e_tc.py`。
3. **TPU 来源捕获**（需要匹配的 TPU）：

   ```sh
   PYTHONPATH=src "$PY" tests/reproduce_tpu_v4_tc_source_mapping.py --output-dir "$WORK/source-check"
   PYTHONPATH=src "$PY" tests/reproduce_tpu_v4_tc_source_mapping.py --trace-only --output-dir "$WORK/trace-check"
   ```

   每个案例都必须满足 `failures == 0`、`isa_equal`、`numerical_equal`，最后打印 `full outputs`。若 `isa_equal` 成立但来源断言失败（缺少 primitive、scope 重复或归属错误），应先怀疑 JAX lowering 发生了变化，而不是 libtpu，见第 6 节。
4. **示例**：运行 `examples/pallas/run_all.sh tpu-v4-tc`，它会检查数值并重新生成 `examples/pallas/tpu_v4_tc/` 中的清单；登记了 v6e 时再运行 `examples/pallas/run_all.sh --aot tpu-v6e-tc`，重新生成 `tpu_v6e_tc/` 中的清单。去掉 `#` 注释后与 git 中的旧版本比较，分清哪些示例只是来源注释变了，哪些机器码变了；机器码变化属于编译器行为变化，需要在提交说明中列出。其中 `insert_program.py` 检查变长写回、独立计数器读数和循环成本，登记新版本时需确认装载尺寸与 metadata 迁移仍然成立。`replace_program.py` 在设备上先执行原程序、再执行写回的修改版：若它得到原结果，说明新版本 runtime 识别已装载程序的身份字段变了，按[回灌与执行](../../../docs/design/executable_replacement.md#程序身份)的方法重新确定需要改写的字段。
5. **拒绝路径**：在 `$WORK` 写一个临时脚本，检查空映像和未按块对齐的映像，并用 monkeypatch 模拟 ABI 不符、build-id 不符和版本未登记。只有拒绝路径允许 monkeypatch。
6. **旧版本回归**：为每个仍登记的版本建临时 venv（`pip install --no-deps` 该版本的 wheel，并加上 `.pth`），重跑第 2 步。这次若修改了 `tc_source_lowering.py` 或公共 `.cc` 文件，还要在旧版本上重跑第 3 步。
7. **分发**：

   ```sh
   "$PY" -m pip wheel --no-deps -w "$WORK/dist" .
   "$WORK/venv/bin/python" -m pip install --no-deps --force-reinstall "$WORK"/dist/tpuasm-*.whl
   ```

   在仓库之外、不设置 `PYTHONPATH` 的目录中完成以下检查：`python -m tpuasm <executable> --input-format executable --output-dir ...` 的输出与源码运行逐字节一致；`tpuasm <image> --target tpu-v4-tc --input-format image --output x.tpuasm` 后，再以 `--input-format listing --output-format image` 汇编回去，结果逐字节相同；把 `examples/pallas/` 目录复制到仓库外，设置 `TPUASM_EXAMPLES_TARGET` 后运行其中一个脚本，确认安装包中的来源后端和函数源码摘要校验能正常工作。
8. **收尾**：运行 `mypy src tests` 和 `MYPYPATH=src mypy tools`，以及 `git diff --check`；删除 `pip wheel` 在仓库中生成的 `build/` 和 `src/*.egg-info`。

示例可以在任何机器上用 `examples/pallas/run_all.sh --aot` 按参考拓扑离线编译两个目标并导出清单，这一步同时验证来源捕获和精确导出，但不检查数值；v4 的离线清单与在 TPU v4 上编译的逐字节相同。有对应的 TPU 时运行 `run_all.sh tpu-v4-tc` 或 `run_all.sh tpu-v6e-tc`，会检查数值。

没有 TPU 时只能完成第 1、2、5、6、7 步中的离线部分，以及第 4 步中用 `run_all.sh --aot` 离线编译示例。此时要写明数值和第 3 步的来源断言未经验证，不能把离线编解码与离线编译通过说成目标版本在设备上已验证。

## 6. JAX 变化的影响

来源捕获依赖 JAX 的两个函数：`jaxpr_subcomp` 和 `_lower_jaxpr_to_for_loop`。`tc_source_lowering.py` 会读取已安装 JAX 中这两个函数的原始源码字节，计算 SHA-256 并与 `_SOURCE_SHA256` 比较。摘要覆盖 `inspect.getsourcelines()` 确定的函数行范围，包括空白、注释和换行，不做格式归一化；每项摘要旁边的固定 commit 链接指向已验证的上游源码。

- **比较不通过**：对照固定 commit 链接阅读 JAX 在这两个函数上的 diff，确认每处文本替换在新源码中仍然恰好匹配一次，语义也没有变。验证适配逻辑后更新 `_SOURCE_SHA256` 及其源码链接，重跑第 5 节第 3 步。
- **比较通过但来源断言失败**：这两个函数调用的其他代码可能变了。用 `git log <旧 commit>..HEAD -- jax/_src/pallas/mosaic/lowering.py jaxlib/mlir/_mlir_libs/` 查找可疑提交，再把补丁前后生成的 MLIR location 或来源 JSON 与旧环境的结果对比。例如，`inlined_func_call` 会把被克隆 op 的 location 改写成调用方的类型和名称。修复时只改 location 的生成路径，并用复现脚本确认 `isa_equal`，然后同步更新设计文档。

## 7. 重新生成 v6e 字段表

v6e 的字段表 `src/tpuasm/tpu_v6e_tc_isa_data.py` 由 `tools/generate_tpu_v6e_tc_isa.py` 从已安装的 libtpu 生成，不手工修改。登记了 v6e 编解码后端之后运行：

```sh
PYTHONPATH=src "$PY" tools/generate_tpu_v6e_tc_isa.py
git diff --stat src/tpuasm/tpu_v6e_tc_isa_data.py
```

工具需要几分钟，会用 2 万个随机指令包核对字段表，不一致时失败。生成结果与旧版本不同时，逐项确认差异：新增或删除的形式、字段位置、互斥槽、`REJECTED` 和 `FORMATTER_ABORTS` 的变化，以及 formatter 助记符或操作数顺序的变化。字段表在版本之间不同时，同一份数据无法同时服务新旧版本，这时要先把字段表改为按版本选择，再登记新版本。之后重新生成指令索引（`PYTHONPATH=src "$PY" tools/generate_isa_reference.py`），并运行第 5 节第 2 步的 v6e 检查。各步骤的原理见 [v6e 设计文档](../../../docs/design/tpu_v6e_tc.md#字段表的生成)。

## 8. 删除已登记的构建

tpuasm 不为旧 libtpu 保持兼容，用户需要旧构建时安装仍支持它的已发布版本。因此删除前先确认该构建出现在[版本兼容性](../../../docs/compatibility.md)某个已发布版本的小节中；只在“未发布”中出现过的构建，删除后就没有可安装的版本，需先发布或经用户确认。生成清单和 v6e 字段表所用的版本（目前是 0.0.49）不能直接删除，要先按第 5、7 节把它们切换到新版本。

删除一个构建时，改动以下位置，文件名中的 `<stem>` 与登记时相同：

1. `src/tpuasm/backends.py`：从 `LIBTPU_RELEASES` 中删除该行。
2. `src/tpuasm/native_backends/libtpu_<stem>_*.cc`：删除该构建的全部编解码文件。
3. `src/tpuasm/tc_source_backend.py`：删除 `LIBTPU_<STEM>` 及其在 `SOURCE_BACKENDS` 中的条目；删除 `src/tpuasm/source_backends/libtpu_<stem>.cc`。
4. `.github/workflows/tests.yml`：从 `offline` 矩阵中删除该版本。
5. `docs/compatibility.md`：从“未发布”一节删除该行，并更新该节中与各构建 hook 差异有关的说明；已发布版本的小节不动。`CHANGELOG.md` 的“未发布”中加一个“移除”小节，写明删除的构建和仍支持它的最后一个 tpuasm 版本。
6. 用 `git grep -nF '<版本号>'` 和 `git grep -n '<stem>'` 检查其余引用。描述“当前支持”的句子要改；作为历史证据的句子（例如某结论在哪个版本上核对过、版本间清单差异的对照）仍然成立，保留原文。若公共 `.cc`、`tc_source_native.cc` 或 `tc_source_lowering.py` 中有只为该构建存在的分支或偏移常量，一并删除。

删除后验证：第 5 节第 1 步的后端选择对剩余版本仍然成立，对被删除的版本报告没有后端；为剩余的每个版本重跑第 5 节第 2 步；运行第 5 节第 8 步的 `mypy` 与 `git diff --check`，并按 `.github/workflows/docs.yml` 的步骤构建文档，确认没有失效链接。

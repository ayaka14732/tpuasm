# 原生编解码后端

TPU v4 TC 的程序映像按块编排：每个 512 字节块内含 10 个 51 字节的 bundle 和若干块内分隔字节；TPU v6e TC 的每个块是 8 个连续的 64 字节 bundle，没有分隔字节。各物理槽的编码器和解码器都位于 libtpu 内部，没有公开 API。tpuasm 在 Python 中维护自己的位模型，用来求解和检查冲突，但程序映像的最终字节由 libtpu 自己的 encoder 生成，读取时也由 libtpu 的 decoder 解释。这样做有两个好处：块内编排无需在 Python 中重新实现；Python 位模型中的错误会在编码后重新解码的核对中暴露出来。代价是每个原生后端只能配合一个特定的 libtpu 二进制使用。

## 后端的选择

`backends.select_backend(target)` 依次检查以下条件，任何一项不满足都在调用原生代码前报错：

1. 硬件目标已登记；
2. 已安装的 libtpu 发行版版本已在 `LIBTPU_RELEASES` 中登记；
3. 当前运行环境的 Python 实现、版本、`abiflags`、操作系统和 CPU 架构与后端的登记一致；
4. `libtpu.so` 的 ELF note 中的 GNU build-id 与后端的登记一致。

libtpu 为不同的 Python ABI 分别发布 wheel，其中的函数地址各不相同，所以运行环境是后端身份的一部分。版本号字符串只用来缩小候选范围，最终以 build-id 判定是否为同一个二进制。每个 `NativeBackend` 由 release、运行环境、build-id、版本源文件和硬件目标组成。每个 release 为 v4 TC 和 BCS 各登记一个后端，libtpu 0.0.49 另外登记 v6e TC 和 v6e TEC 后端；release 没有某个目标的后端时，选择后端直接报错。

## 编译与加载

`printer._load_native` 在首次使用某个后端时，用 g++ 把它的版本源文件编译成临时共享库，放在 `/tmp` 下，并在进程内按 (后端, 目标) 缓存。块大小、每块 bundle 数、槽数等硬件常量由 `HardwareTarget` 生成为 `target.h`，通过 `-include` 注入。因此，版本源文件只记录 libtpu 的 ABI 信息，不重复硬件格式。

版本源文件在 `tpuasm_backend` 命名空间中定义常量，然后 `#include "../native.cc"`。BCS 与 v6e TEC 的源文件另外定义 `TPUASM_CODEC_ONLY`，编译时去掉与 formatter 有关的代码。常量按作用分为以下几组：

| 常量 | 作用 |
|---|---|
| `kGetPjrtApi` | 某个导出符号的 ELF VA。运行时用 `dladdr` 求得加载基址后，检查该符号的实际地址与预期是否一致，以此确认后续的 VA 换算正确。 |
| `kDecodeProgram`、`kEncodeProgram` | 程序映像与 program 对象互相转换的包装函数。 |
| `kConstructProgram`、`kDestroyProgram` | 就地构造 program 对象；非 deleting 析构，不释放桥接层自己的缓冲区。 |
| `kProgramByteSize`、`kProtoSerializeToArray`、`kProtoParseFromArray` | program 对象与 protobuf 字节互相转换。 |
| `kProgramBundleCountOffset` | 解码后 program 对象中 bundle 数的位置。 |
| `kFunctionPrologue` | 调用函数前比对的入口字节。 |
| `kFormatBundle`、`kEmptyAnnotations`、`kProgramBundleStorageOffset`、`kBundleMaskOffset`、`kBundleSlotMasks` | 仅 TC 使用，服务于逐槽 formatter 校验（见下文）。 |
| `kScalarOneofPointerOffset`、`kScalarOneofCaseOffset`、`kScalarBundleMaskOffset`、`kScalarBundleCase`、`kBundleSharedMask`、`kScalarSlotCases`、`kScalarSlotMasks` | 仅 v6e TC 使用（定义 `TPUASM_SCALAR_ONEOF`）：两个标量槽位于标量子 bundle，与 DMA 同属一个 oneof；立即数与标量操作数消息在每个 bundle 中都存在。 |
| `kTripleCtor`、`kCreateInstPrinter`、`kPrintInst`、`kConsumeBundle` 等 | 仅 v6e TEC 源文件登记，桥接层不使用。离线生成 TEC 字段表时，[tec_llvm.cc](../../tools/tec_llvm.cc) 用它们驱动 LLVM TPU printer 与 TEC emitter（见 [TPU v6e TEC 目标](tpu_v6e_tec.md#字段表的生成)）。 |

地址一律是 ELF VA，使用前加上运行时基址。具体数值写在各版本源文件中，本文不重复。

Python 进程通过 ctypes 持有一个指向 libtpu 的句柄，使其一直保持加载。桥接函数每次调用都会 `dlopen` 并 `dlclose`，有了这个句柄，`dlclose` 就不会把 libtpu 卸载后再重新初始化。

## C ABI

三个函数的调用约定对所有版本相同。若新版本的 libtpu ABI 与 `native.cc` 的假设不符，应在新的版本源文件中另行实现同样的 C ABI，不修改公共文件：

- `tpuasm_verify(libtpu_path, image, size, error, capacity)`：只做校验，不返回数据。
- `tpuasm_program_proto(libtpu_path, input, size, encode, &output, &output_size, error, capacity)`：
  - encode 模式：构造 program 对象，用 `ParseFromArray` 读入 ISA program protobuf，编码后返回程序映像；
  - decode 模式：解码程序映像，检查 bundle 数，检查重新编码后逐字节一致，最后序列化为 protobuf 返回。
- `tpuasm_format(libtpu_path, image, size, &output, &output_size, error, capacity)`：仅 TC 后端导出，返回每个 bundle 一行的 formatter 文本，没有 formatter 输出的 bundle 写作 `<no formatter output>`。只有离线生成 v6e 字段表的工具使用它；formatter 遇到部分保留编码会终止进程，所以调用方在子进程中运行。
- `tpuasm_free(pointer)`：释放以上函数返回的缓冲区。

每次调用都会重新核对加载基址和函数入口字节。失败时返回非零状态并写入诊断信息，由 Python 转换为 `RuntimeError`。libtpu 内部的对象（`StatusOr`、libc++ 字符串、承载程序映像的 host 缓冲区）都按已核对的偏移读取，不当作宿主 g++ 的 `std::string` 等类型使用。

## 原生校验能证明什么

| 检查 | 证明的内容 |
|---|---|
| 解码出的 bundle 数 = 程序映像字节数 / 块大小 × 每块 bundle 数 | decoder 接受了完整的程序映像，没有提前截断。 |
| `encode(decode(image)) == image` | 对这份输入，解码得到的对象没有丢失机器信息，Python 从解码对象还原出的字段足以决定程序映像。 |
| 逐槽 formatter 输出能拼回整个 bundle 的输出（仅 TC） | `kBundleSlotMasks` 中 presence 位与物理槽的对应关系正确。 |

逐槽检查的做法是：暂时把解码后 bundle 对象的 presence mask 改成只含某一个槽的位，调用 libtpu formatter，然后恢复原掩码；各槽输出拼接的结果必须与整个 bundle 的 formatter 输出完全相同。presence mask 属于 libtpu 的内存表示，这项检查是槽名映射唯一的交叉证据。formatter 的文本在这里只用于比较，不会出现在任何输出中。

v6e TC 的 bundle 对象中，两个标量槽不是 presence 位，而是 oneof case 1 指向的标量子 bundle 里的两个 presence 位；DMA 是同一 oneof 的 case 4。只保留一个槽时，立即数和标量操作数两条消息的 presence 位保持不变，否则 formatter 读不到操作数。FormatterGl 对少数 descriptor 形式没有输出（例如 `vector_f32_remap`、`sync_*_yieldable`），这类槽不参与拼接比较，其余槽照常比较；对部分保留编码 formatter 会直接终止进程，v6e 的 Python 解码在调用原生校验之前拒绝这些编码（见 [TPU v6e TC 目标](tpu_v6e_tc.md#formatter-的限制)）。

这些检查不证明 formatter 的文本与机器字节一一对应。从 libtpu formatter 的文本出发，确实无法唯一恢复程序映像，因为可见文本不能决定全部机器位。它们也不验证程序在设备上的执行效果。

BCS 只做 codec 校验。libtpu 中没有与当前 Pufferfish BCS program 对象匹配的 formatter，旧代际的 formatter 不能用于这个对象。BCS 的 bundle 在程序映像中连续排列、没有分隔字节，所以 Python 直接从映像字节读取机器字，并与原生结果逐字节比较，槽的对应关系也由这项比较覆盖。

v6e TEC 同样只做 codec 校验：libtpu 没有 TEC bundle 的 descriptor formatter。TEC 的程序映像是连续的 64 字节 bundle，Python 同样直接读取机器字并与原生结果逐字节比较。

## 增加 libtpu 版本

步骤见 skill [add-libtpu-backend](../../.agents/skills/add-libtpu-backend/SKILL.md)。编解码后端只负责程序映像的编解码；某个版本已有编解码后端，并不意味着它支持编译来源捕获。来源捕获需要另外适配，见[来源映射](tc_source_mapping.md#版本相关的部分)。

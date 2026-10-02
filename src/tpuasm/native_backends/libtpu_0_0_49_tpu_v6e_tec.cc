// libtpu 0.0.49, GNU build-id 97e27df7268da25ab03e455e30dd86b0,
// from the cp314-cp314t manylinux_2_31_x86_64 wheel.
#include <cstddef>
#include <cstdint>

#define TPUASM_BACKEND_CONFIGURED 1
// libtpu has no descriptor formatter for TEC bundles; the bridge checks the
// codec round trip only.
#define TPUASM_CODEC_ONLY 1

namespace tpuasm_backend {
constexpr std::uintptr_t kGetPjrtApi = 0x0c43ba30;
constexpr std::uintptr_t kDecodeProgram = 0x1ad1d2f0;
constexpr std::uintptr_t kEncodeProgram = 0x1ad1c4f0;
constexpr std::uintptr_t kDestroyProgram = 0x1bd56520;
constexpr std::uintptr_t kConstructProgram = 0x1bd56450;
constexpr std::uintptr_t kProgramByteSize = 0x1bd566c0;
constexpr std::uintptr_t kProtoParseFromArray = 0x1d3abc00;
constexpr std::uintptr_t kProtoSerializeToArray = 0x1d3ac668;
constexpr std::size_t kProgramBundleCountOffset = 0x20;
constexpr unsigned char kFunctionPrologue[] = { 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57 };
// LLVM TPU MC printer and Ghostlite TEC emitter. The native bridge does not
// use them; tools/generate_tpu_v6e_tec_isa.py reads the compiler's syntax with
// them (see tools/tec_llvm.cc).
constexpr std::uintptr_t kTripleCtor = 0x109e7530;
constexpr std::uintptr_t kTargetOptionsCtor = 0x188fa5f0;
constexpr std::uintptr_t kAsmInfoAllocator = 0x170ddc00;
constexpr std::uintptr_t kCreateInstrInfo = 0x170dd740;
constexpr std::uintptr_t kCreateRegisterInfo = 0x170dd7a0;
constexpr std::uintptr_t kCreateSubtargetInfo = 0x170dd8f0;
constexpr std::uintptr_t kCreateInstPrinter = 0x170dd670;
constexpr std::uintptr_t kPrintInst = 0x170d6e70;
constexpr std::uintptr_t kRegisterName = 0x170cd6d0;
constexpr std::uintptr_t kStringOstreamCtor = 0x0f1effb0;
constexpr std::uintptr_t kConsumeBundle = 0x16e16bd0;
constexpr std::uintptr_t kConstructBundle = 0x1bd56d50;
constexpr std::uintptr_t kBundleByteSize = 0x1bd575e0;
constexpr std::uintptr_t kImmExprVtable = 0x1dfdd1e0;
}  // namespace tpuasm_backend

#include "../native.cc"

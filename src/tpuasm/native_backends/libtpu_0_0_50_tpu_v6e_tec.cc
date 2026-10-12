// libtpu 0.0.50, GNU build-id 4035cb37c50260442f2b5d54701a25fa,
// from the cp314-cp314t manylinux_2_31_x86_64 wheel.
#include <cstddef>
#include <cstdint>

#define TPUASM_BACKEND_CONFIGURED 1
// libtpu has no descriptor formatter for TEC bundles; the bridge checks the
// codec round trip only.
#define TPUASM_CODEC_ONLY 1

namespace tpuasm_backend {
constexpr std::uintptr_t kGetPjrtApi = 0x0b4f9b90;
constexpr std::uintptr_t kDecodeProgram = 0x191a1c20;
constexpr std::uintptr_t kEncodeProgram = 0x191a0e20;
constexpr std::uintptr_t kDestroyProgram = 0x19de3b30;
constexpr std::uintptr_t kConstructProgram = 0x19de3a70;
constexpr std::uintptr_t kProgramByteSize = 0x19de3900;
constexpr std::uintptr_t kProtoParseFromArray = 0x1b195120;
constexpr std::uintptr_t kProtoSerializeToArray = 0x1b195b08;
constexpr std::size_t kProgramBundleCountOffset = 0x20;
constexpr unsigned char kFunctionPrologue[] = { 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57 };
// LLVM TPU MC printer and Ghostlite TEC emitter. The native bridge does not
// use them; tools/generate_tpu_v6e_tec_isa.py reads the compiler's syntax with
// them (see tools/tec_llvm.cc).
constexpr std::uintptr_t kTripleCtor = 0x0f8ad1a0;
constexpr std::uintptr_t kTargetOptionsCtor = 0x171db4f0;
constexpr std::uintptr_t kAsmInfoAllocator = 0x15a35e90;
constexpr std::uintptr_t kCreateInstrInfo = 0x15a359d0;
constexpr std::uintptr_t kCreateRegisterInfo = 0x15a35a30;
constexpr std::uintptr_t kCreateSubtargetInfo = 0x15a35b80;
constexpr std::uintptr_t kCreateInstPrinter = 0x15a35900;
constexpr std::uintptr_t kPrintInst = 0x15a2f200;
constexpr std::uintptr_t kRegisterName = 0x15a25a00;
constexpr std::uintptr_t kStringOstreamCtor = 0x0e061420;
constexpr std::uintptr_t kConsumeBundle = 0x15761130;
constexpr std::uintptr_t kConstructBundle = 0x19de4b90;
constexpr std::uintptr_t kBundleByteSize = 0x19de4610;
constexpr std::uintptr_t kImmExprVtable = 0x1bd86b90;
}  // namespace tpuasm_backend

#include "../native.cc"

// libtpu 0.0.49, GNU build-id 97e27df7268da25ab03e455e30dd86b0,
// from the cp314-cp314t manylinux_2_31_x86_64 wheel.
#include <cstddef>
#include <cstdint>

#define TPUASM_BACKEND_CONFIGURED 1
#define TPUASM_CODEC_ONLY 1

namespace tpuasm_backend {
constexpr std::uintptr_t kGetPjrtApi = 0x0c43ba30;
constexpr std::uintptr_t kDecodeProgram = 0x1ad226c0;
constexpr std::uintptr_t kEncodeProgram = 0x1ad22160;
constexpr std::uintptr_t kDestroyProgram = 0x1bcddf10;
constexpr std::uintptr_t kConstructProgram = 0x1bcdde40;
constexpr std::uintptr_t kProgramByteSize = 0x1bcde0b0;
constexpr std::uintptr_t kProtoParseFromArray = 0x1d3abc00;
constexpr std::uintptr_t kProtoSerializeToArray = 0x1d3ac668;
constexpr std::size_t kProgramBundleCountOffset = 0x20;
constexpr unsigned char kFunctionPrologue[] = { 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57 };
}  // namespace tpuasm_backend

// BCS has no matching Pufferfish native formatter; verify its codec directly.
#include "../native.cc"

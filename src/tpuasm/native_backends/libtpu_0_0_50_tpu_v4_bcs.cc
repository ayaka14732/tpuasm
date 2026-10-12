// libtpu 0.0.50, GNU build-id 4035cb37c50260442f2b5d54701a25fa,
// from the cp314-cp314t manylinux_2_31_x86_64 wheel.
#include <cstddef>
#include <cstdint>

#define TPUASM_BACKEND_CONFIGURED 1
#define TPUASM_CODEC_ONLY 1

namespace tpuasm_backend {
constexpr std::uintptr_t kGetPjrtApi = 0x0b4f9b90;
constexpr std::uintptr_t kDecodeProgram = 0x191a7010;
constexpr std::uintptr_t kEncodeProgram = 0x191a6ab0;
constexpr std::uintptr_t kDestroyProgram = 0x19d8c9e0;
constexpr std::uintptr_t kConstructProgram = 0x19d8c920;
constexpr std::uintptr_t kProgramByteSize = 0x19d8c7b0;
constexpr std::uintptr_t kProtoParseFromArray = 0x1b195120;
constexpr std::uintptr_t kProtoSerializeToArray = 0x1b195b08;
constexpr std::size_t kProgramBundleCountOffset = 0x20;
constexpr unsigned char kFunctionPrologue[] = { 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57 };
}  // namespace tpuasm_backend

// BCS has no matching Pufferfish native formatter; verify its codec directly.
#include "../native.cc"

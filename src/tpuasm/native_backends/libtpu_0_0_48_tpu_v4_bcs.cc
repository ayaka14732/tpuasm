// libtpu 0.0.48, GNU build-id 3310a7c8c137cd515c7a2ba1ce2ea38c,
// from the cp314-cp314t manylinux_2_31_x86_64 wheel.
#include <cstddef>
#include <cstdint>

#define TPUASM_BACKEND_CONFIGURED 1
#define TPUASM_CODEC_ONLY 1

namespace tpuasm_backend {
constexpr std::uintptr_t kGetPjrtApi = 0x0c708cf0;
constexpr std::uintptr_t kDecodeProgram = 0x1af1e0c0;
constexpr std::uintptr_t kEncodeProgram = 0x1af1db60;
constexpr std::uintptr_t kDestroyProgram = 0x1bed95e0;
constexpr std::uintptr_t kConstructProgram = 0x1bed9510;
constexpr std::uintptr_t kProgramByteSize = 0x1bed9780;
constexpr std::uintptr_t kProtoParseFromArray = 0x1d58c850;
constexpr std::uintptr_t kProtoSerializeToArray = 0x1d58d238;
constexpr std::size_t kProgramBundleCountOffset = 0x20;
constexpr unsigned char kFunctionPrologue[] = { 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57 };
}  // namespace tpuasm_backend

// BCS has no matching Pufferfish native formatter; verify its codec directly.
#include "../native.cc"

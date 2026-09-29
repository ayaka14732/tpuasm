// libtpu 0.0.48.dev20260912+nightly, GNU build-id
// 825044f87748f172b7db935904b3f754, on CPython 3.14t Linux x86-64.
#include <cstddef>
#include <cstdint>

#define TPUASM_BACKEND_CONFIGURED 1
#define TPUASM_CODEC_ONLY 1

namespace tpuasm_backend {
constexpr std::uintptr_t kGetPjrtApi = 0x0c733cf0;
constexpr std::uintptr_t kDecodeProgram = 0x1af9f500;
constexpr std::uintptr_t kEncodeProgram = 0x1af9efa0;
constexpr std::uintptr_t kDestroyProgram = 0x1bf5aa90;
constexpr std::uintptr_t kConstructProgram = 0x1bf5a9c0;
constexpr std::uintptr_t kProgramByteSize = 0x1bf5ac30;
constexpr std::uintptr_t kProtoParseFromArray = 0x1d618b50;
constexpr std::uintptr_t kProtoSerializeToArray = 0x1d6195b8;
constexpr std::size_t kProgramBundleCountOffset = 0x20;
constexpr unsigned char kFunctionPrologue[] = { 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57 };
}  // namespace tpuasm_backend

// BCS has no matching Pufferfish native formatter; verify its codec directly.
#include "../native.cc"

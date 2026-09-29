// libtpu 0.0.48.dev20260912+nightly, GNU build-id
// 825044f87748f172b7db935904b3f754, on CPython 3.14t Linux x86-64.
#include <cstddef>
#include <cstdint>

#define TPUASM_BACKEND_CONFIGURED 1

namespace tpuasm_backend {
constexpr std::uintptr_t kGetPjrtApi = 0x0c733cf0;
constexpr std::uintptr_t kDecodeProgram = 0x1af9f1e0;
constexpr std::uintptr_t kEncodeProgram = 0x1af9edf0;
constexpr std::uintptr_t kDestroyProgram = 0x1bef3670;
constexpr std::uintptr_t kConstructProgram = 0x1bef35a0;
constexpr std::uintptr_t kProgramByteSize = 0x1bef3810;
constexpr std::uintptr_t kProtoParseFromArray = 0x1d618b50;
constexpr std::uintptr_t kProtoSerializeToArray = 0x1d6195b8;
constexpr std::uintptr_t kFormatBundle = 0x19d40f40;
constexpr std::uintptr_t kEmptyAnnotations = 0x1e6ec708;
constexpr std::size_t kProgramBundleStorageOffset = 0x18;
constexpr std::size_t kProgramBundleCountOffset = 0x20;
constexpr std::size_t kBundleMaskOffset = 0x10;
constexpr unsigned char kFunctionPrologue[] = { 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57 };
// Decoded protobuf presence bits, indexed by the hardware target's slot order.
constexpr std::uint32_t kBundleSlotMasks[] = {
    0x001U, 0x002U, 0x004U, 0x008U, 0x010U, 0x020U, 0x040U, 0x080U, 0x100U, 0x200U, 0x400U, 0x800U
};
}  // namespace tpuasm_backend

// This backend reuses the common bridge. A future backend may instead provide
// the same C ABI with a completely different libtpu ABI adapter.
#include "../native.cc"

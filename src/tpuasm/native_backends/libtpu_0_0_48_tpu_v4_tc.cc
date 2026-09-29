// libtpu 0.0.48, GNU build-id 3310a7c8c137cd515c7a2ba1ce2ea38c,
// from the cp314-cp314t manylinux_2_31_x86_64 wheel.
#include <cstddef>
#include <cstdint>

#define TPUASM_BACKEND_CONFIGURED 1

namespace tpuasm_backend {
constexpr std::uintptr_t kGetPjrtApi = 0x0c708cf0;
constexpr std::uintptr_t kDecodeProgram = 0x1af1dda0;
constexpr std::uintptr_t kEncodeProgram = 0x1af1d9b0;
constexpr std::uintptr_t kDestroyProgram = 0x1be721c0;
constexpr std::uintptr_t kConstructProgram = 0x1be720f0;
constexpr std::uintptr_t kProgramByteSize = 0x1be72360;
constexpr std::uintptr_t kProtoParseFromArray = 0x1d58c850;
constexpr std::uintptr_t kProtoSerializeToArray = 0x1d58d238;
constexpr std::uintptr_t kFormatBundle = 0x19ccccc0;
// AnnotationMetadata_globals_ + 0x30: the empty protobuf instance.
constexpr std::uintptr_t kEmptyAnnotations = 0x1e65ec68;
constexpr std::size_t kProgramBundleStorageOffset = 0x18;
constexpr std::size_t kProgramBundleCountOffset = 0x20;
constexpr std::size_t kBundleMaskOffset = 0x10;
constexpr unsigned char kFunctionPrologue[] = { 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57 };
// Decoded protobuf presence bits, indexed by the hardware target's slot order.
constexpr std::uint32_t kBundleSlotMasks[] = {
    0x001U, 0x002U, 0x004U, 0x008U, 0x010U, 0x020U, 0x040U, 0x080U, 0x100U, 0x200U, 0x400U, 0x800U
};
}  // namespace tpuasm_backend

#include "../native.cc"

// libtpu 0.0.50, GNU build-id 4035cb37c50260442f2b5d54701a25fa,
// from the cp314-cp314t manylinux_2_31_x86_64 wheel.
#include <cstddef>
#include <cstdint>

#define TPUASM_BACKEND_CONFIGURED 1

namespace tpuasm_backend {
constexpr std::uintptr_t kGetPjrtApi = 0x0b4f9b90;
constexpr std::uintptr_t kDecodeProgram = 0x191a6cf0;
constexpr std::uintptr_t kEncodeProgram = 0x191a6900;
constexpr std::uintptr_t kDestroyProgram = 0x19d36320;
constexpr std::uintptr_t kConstructProgram = 0x19d36260;
constexpr std::uintptr_t kProgramByteSize = 0x19d360f0;
constexpr std::uintptr_t kProtoParseFromArray = 0x1b195120;
constexpr std::uintptr_t kProtoSerializeToArray = 0x1b195b08;
constexpr std::uintptr_t kFormatBundle = 0x17fec020;
// AnnotationMetadata_globals_ + 0x50: the empty protobuf instance. The globals now start with the ClassData.
constexpr std::uintptr_t kEmptyAnnotations = 0x1c1dea70;
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

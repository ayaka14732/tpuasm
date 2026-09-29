// libtpu 0.0.49, GNU build-id 97e27df7268da25ab03e455e30dd86b0,
// from the cp314-cp314t manylinux_2_31_x86_64 wheel.
#include <cstddef>
#include <cstdint>

#define TPUASM_BACKEND_CONFIGURED 1
#define TPUASM_SCALAR_ONEOF 1
// Some descriptor forms have no FormatterGl output; their slots skip the cross-check.
#define TPUASM_FORMATTER_GAPS 1

namespace tpuasm_backend {
constexpr std::uintptr_t kGetPjrtApi = 0x0c43ba30;
constexpr std::uintptr_t kDecodeProgram = 0x1ad1cb50;
constexpr std::uintptr_t kEncodeProgram = 0x1ad1c100;
constexpr std::uintptr_t kDestroyProgram = 0x1bd373e0;
constexpr std::uintptr_t kConstructProgram = 0x1bd37310;
constexpr std::uintptr_t kProgramByteSize = 0x1bd37580;
constexpr std::uintptr_t kProtoParseFromArray = 0x1d3abc00;
constexpr std::uintptr_t kProtoSerializeToArray = 0x1d3ac668;
constexpr std::uintptr_t kFormatBundle = 0x19bb0580;
// AnnotationMetadata_globals_ + 0x30: the empty protobuf instance.
constexpr std::uintptr_t kEmptyAnnotations = 0x1e485b00;
constexpr std::size_t kProgramBundleStorageOffset = 0x18;
constexpr std::size_t kProgramBundleCountOffset = 0x20;
constexpr std::size_t kBundleMaskOffset = 0x10;
// Scalar oneof of the decoded bundle: case 1 is the scalar sub-bundle, case 4
// the DMA; the sub-bundle has its own presence bits.
constexpr std::size_t kScalarOneofPointerOffset = 0x88;
constexpr std::size_t kScalarOneofCaseOffset = 0x90;
constexpr std::size_t kScalarBundleMaskOffset = 0x10;
constexpr std::uint32_t kScalarBundleCase = 1;
// Immediates and vector scalar operands are present in every decoded bundle.
constexpr std::uint32_t kBundleSharedMask = 0x3;
constexpr unsigned char kFunctionPrologue[] = { 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57 };
// Per physical slot, in the hardware target's slot order: decoded presence
// bits, scalar oneof case and scalar sub-bundle presence bits.
constexpr std::uint32_t kBundleSlotMasks[] = {
    0x0000U, 0x0000U, 0x0000U, 0x0004U, 0x0008U, 0x0010U, 0x0020U, 0x0040U, 0x0080U, 0x0100U, 0x0200U, 0x0400U, 0x0800U, 0x1000U, 0x2000U
};
constexpr std::uint32_t kScalarSlotCases[] = { 1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
constexpr std::uint32_t kScalarSlotMasks[] = { 0x1U, 0x2U, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
}  // namespace tpuasm_backend

#include "../native.cc"

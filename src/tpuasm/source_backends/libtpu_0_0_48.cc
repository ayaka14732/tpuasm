// CPython 3.14t Linux x86-64; build-id 3310a7c8c137cd515c7a2ba1ce2ea38c.
#include <cstddef>
#include <cstdint>
namespace tpuasm_source_backend {
constexpr uintptr_t kScopedAnnotator = 0x15867690;
constexpr uintptr_t kEmit = 0x1578ff60;
constexpr uintptr_t kAnnotation = 0xf0184c0;
constexpr uintptr_t kSetAnnotation = 0xe853520;
constexpr uintptr_t kSerialize = 0x1d58d238;
constexpr uintptr_t kGetHloModule = 0x1aca10a0;
constexpr uintptr_t kOperands = 0xefd50b0;
constexpr uintptr_t kAppendOrdinal = 0xdb205b0;
constexpr uintptr_t kReplace = 0x15c16190;
constexpr uintptr_t kRegion = 0x15c18cd0;
constexpr uintptr_t kSetAnnotationText = 0xea10740;
constexpr uintptr_t kCandidateInstruction = 0xefdf850;
constexpr uintptr_t kGetFlag = 0x1d7334a0;
constexpr uintptr_t kSetFlag = 0x1d7036f8;
// Where a protobuf message keeps its dispatch table and the ByteSizeLong slot in it; HloModule::unique_id.
constexpr std::ptrdiff_t kMessageDispatchOffset = 0x0;
constexpr std::ptrdiff_t kByteSizeSlot = 0x18;
constexpr std::ptrdiff_t kHloModuleIdOffset = 0xaa4;
}
#include "../tc_source_native.cc"

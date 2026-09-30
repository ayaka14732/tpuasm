// CPython 3.14t Linux x86-64; build-id 97e27df7268da25ab03e455e30dd86b0.
#include <cstdint>
namespace tpuasm_source_backend {
constexpr uintptr_t kScopedAnnotator = 0x1543dcd0;
constexpr uintptr_t kEmit = 0x15365880;
constexpr uintptr_t kAnnotation = 0xed807b0;
constexpr uintptr_t kSetAnnotation = 0xe5b88d0;
constexpr uintptr_t kSourceMapSize = 0x197f1b20;
constexpr uintptr_t kSerialize = 0x1d3ac668;
constexpr uintptr_t kGetHloModule = 0x1aaa49b0;
constexpr uintptr_t kOperands = 0xed3d1c0;
constexpr uintptr_t kAppendOrdinal = 0xd8859a0;
constexpr uintptr_t kReplace = 0x157d2820;
constexpr uintptr_t kRegion = 0x157d5410;
constexpr uintptr_t kSetAnnotationText = 0xe770460;
constexpr uintptr_t kCandidateInstruction = 0xed47a40;
constexpr uintptr_t kAppendInstruction = 0x19752860;
constexpr uintptr_t kSetAnnotationInternal = 0x1972fda0;
constexpr uintptr_t kReplaceUsesAndDelete = 0x18c7b150;
constexpr uintptr_t kFusedLocGet = 0x19f037c0;
constexpr uintptr_t kAttributeContext = 0x19ea1b30;
constexpr uintptr_t kOperationVisitor = 0x16f5eb30;
constexpr uintptr_t kFusedLocLocations = 0x19f02d20;
constexpr uintptr_t kFusedLocTypeIdSlot = 0x1e52fc60;
constexpr uintptr_t kVmatprepSubr = 0x197bb7c0;
constexpr uintptr_t kVmatprepMubrMsk = 0x197bbad0;
constexpr uintptr_t kGhostliteMatprep = 0x155c0be0;
constexpr uintptr_t kGhostliteAnnotatorConstruct = 0x155ca1d0;
constexpr uintptr_t kGhostliteAnnotatorDestroy = 0x1555a740;
constexpr uintptr_t kGetFlag = 0x1d552270;
constexpr uintptr_t kSetFlag = 0x1d5227b8;
}
#include "../tc_source_native.cc"

namespace {
void preserve_direct_source(void* original, void* result) {
    if (!original) return;
    propagate_sources(original, result, false);
    auto annotation = reinterpret_cast<const void*(*)(void*)>(base + kAnnotation);
    std::string old = native_string(annotation(original));
    std::string current = native_string(annotation(result));
    if (!old.empty() && current.find(old) == std::string::npos) {
        if (!current.empty()) current += " :: ";
        current += old;
        reinterpret_cast<void(*)(void*, View)>(base + kSetAnnotation)(result, {current.data(), current.size()});
    }
}
}

// LoadSubr and PackToSingleQuadrant synthesize prep instructions for the exact
// latch/matmul they are rewriting. The original is alive until region substitution.
extern "C" void* source_subr_prep_hook(void* builder, void* operand, uint8_t mode, int mxu, void* original) {
    void* result = reinterpret_cast<void*(*)(void*, void*, uint8_t, int)>(base + kVmatprepSubr)(builder, operand, mode, mxu);
    try { preserve_direct_source(original, result); }
    catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
    return result;
}

extern "C" void* source_mubr_prep_hook(void* builder, void* mask, void* operand, uint8_t format, int mxu, void* original) {
    void* result = reinterpret_cast<void*(*)(void*, void*, void*, uint8_t, int)>(base + kVmatprepMubrMsk)(builder, mask, operand, format, mxu);
    try { preserve_direct_source(original, result); }
    catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
    return result;
}

// CodeGenerator has already installed this LLO instruction's annotation. The
// Ghostlite prep emitter lacks the ScopedAnnotator used by latch/matmul. Snapshot
// occupied slots around that exact emission, using the native guard's ABI.
// Other emitter implementations retain their original virtual dispatch.
extern "C" void* source_matprep_hook(void* emitter, uint16_t opcode, uint64_t mask, uint32_t reg, uint8_t format, int mxu) {
    auto emit = read<void*(*)(void*, uint16_t, uint64_t, uint32_t, uint8_t, int)>(read<void*>(emitter, 0), 0x408);
    if (reinterpret_cast<uintptr_t>(emit) != base + kGhostliteMatprep) return emit(emitter, opcode, mask, reg, format, mxu);
    struct Guard {
        alignas(8) unsigned char bytes[0x30];
        explicit Guard(void* emitter) {
            reinterpret_cast<void(*)(void*, void*, void*)>(base + kGhostliteAnnotatorConstruct)(bytes, emitter, read<void*>(emitter, 0x1b8));
        }
        ~Guard() { reinterpret_cast<void(*)(void*)>(base + kGhostliteAnnotatorDestroy)(bytes); }
    } guard(emitter);
    return emit(emitter, opcode, mask, reg, format, mxu);
}

// DecomposeDmaDone appends its wait and sync decrement before destroying the
// original dma.done. The call-site trampoline passes that original in rcx.
extern "C" void* source_dma_append_hook(void* region, void* instruction, uint32_t element_type, void* original) {
    void* result = reinterpret_cast<void*(*)(void*, void*, uint32_t)>(base + kAppendInstruction)(region, instruction, element_type);
    try {
        preserve_direct_source(original, result);
    } catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
    return result;
}

// AnnotateNewInstructions replaces the annotation of every instruction lowered
// from one MLIR op with the op's root location, discarding text the region
// builder attached (e.g. "smod.u32 w/div 2", "Set PCR instruction"). Keep that
// text in the order LloInstruction::append_annotation uses: "loc(...) :: old".
extern "C" void source_location_annotation_hook(void* inst, void* location) {
    std::string combined;
    try {
        std::string old = native_string(reinterpret_cast<const void*(*)(void*)>(base + kAnnotation)(inst));
        std::string text = native_string(location);
        if (!old.empty() && old != text) combined = text + " :: " + old;
    } catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
    if (combined.empty()) {
        // The caller owns and destroys the by-value string argument.
        reinterpret_cast<void(*)(void*, void*)>(base + kSetAnnotationInternal)(inst, location);
        return;
    }
    reinterpret_cast<void(*)(void*, View)>(base + kSetAnnotation)(inst, {combined.data(), combined.size()});
}

// MLIR CSEDriver::replaceUsesAndDelete keeps the dominating op and drops the
// erased op's location (it only copies it over an UnknownLoc), so an expression
// written twice, e.g. `index * 8` in two helpers, keeps the first source only.
// Both ops compute the same value; record both locations on the survivor with
// FusedLoc::get, which also flattens and deduplicates. mlir::Operation keeps
// its Location at +0x18, as replaceUsesAndDelete itself reads and writes.
extern "C" void source_cse_hook(void* driver, void* known, void* op, void* existing, bool dominance) {
    void* erased = read<void*>(op, 0x18);
    reinterpret_cast<void(*)(void*, void*, void*, void*, bool)>(base + kReplaceUsesAndDelete)(driver, known, op, existing, dominance);
    try {
        void* kept = read<void*>(existing, 0x18);
        if (kept == erased) return;
        void* locations[2] = {kept, erased};
        void* context = reinterpret_cast<void*(*)(const void*)>(base + kAttributeContext)(&kept);
        void* fused = reinterpret_cast<void*(*)(void* const*, std::size_t, void*, void*)>(base + kFusedLocGet)(locations, 2, nullptr, context);
        std::memcpy(static_cast<char*>(existing) + 0x18, &fused, sizeof fused);
        remapped.fetch_add(1, std::memory_order_relaxed);
    } catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
}

// LloSourceMapSerializer::OperationVisitor parses one SourceInfo from an op's
// location; for a FusedLoc (e.g. from source_cse_hook) only the first part
// survives. Visit the op once per fused part so every part gets the op's
// ordinals. The op's location is restored before returning.
struct LocationRange { void* const* data; std::size_t size; };
extern "C" void source_fused_location_hook(void* serializer, void* op) {
    auto visit = reinterpret_cast<void(*)(void*, void*)>(base + kOperationVisitor);
    void* location = read<void*>(op, 0x18);
    LocationRange parts{nullptr, 0};
    try {
        // AbstractAttribute::typeID at +0x90, compared with the relocated GOT slot
        // that readOptionalAttribute<FusedLoc> uses for isa<FusedLoc>.
        uintptr_t fused_type = read<uintptr_t>(reinterpret_cast<void*>(base + kFusedLocTypeIdSlot), 0);
        if (read<uintptr_t>(read<void*>(location, 0), 0x90) == fused_type)
            parts = reinterpret_cast<LocationRange(*)(void* const*)>(base + kFusedLocLocations)(&location);
    } catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
    if (parts.size < 2) return visit(serializer, op);
    std::vector<void*> copy(parts.data, parts.data + parts.size);
    for (void* part : copy) {
        std::memcpy(static_cast<char*>(op) + 0x18, &part, sizeof part);
        visit(serializer, op);
    }
    std::memcpy(static_cast<char*>(op) + 0x18, &location, sizeof location);
}

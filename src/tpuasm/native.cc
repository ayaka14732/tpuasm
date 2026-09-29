// Common host-only TPU ISA decoder bridge. A version-specific translation unit
// supplies libtpu offsets and decoded-object layout before including it.
// The loader generates target.h from the shared hardware definition.
#ifndef TPUASM_BACKEND_CONFIGURED
#error "compile native.cc with a tpuasm native backend configuration"
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace {
using namespace tpuasm_backend;
namespace target = tpuasm_target;

#ifndef TPUASM_CODEC_ONLY
static_assert(sizeof(kBundleSlotMasks) / sizeof(kBundleSlotMasks[0]) == target::kSlotCount);

constexpr std::uint32_t decoded_slot_mask() {
    std::uint32_t mask = 0;
    for (const auto bits : kBundleSlotMasks) mask |= bits;
    return mask;
}

// Which physical slots a decoded bundle holds. A target whose scalar slots sit
// in a protobuf oneof (a scalar sub-bundle or a DMA) also records the oneof
// case and the sub-bundle presence bits; other targets use presence bits only.
struct Presence {
    std::uint32_t mask = 0;
    std::uint32_t scalar_case = 0;
    std::uint32_t scalar_mask = 0;
};

#ifdef TPUASM_SCALAR_ONEOF
static_assert(sizeof(kScalarSlotCases) / sizeof(kScalarSlotCases[0]) == target::kSlotCount);
static_assert(sizeof(kScalarSlotMasks) / sizeof(kScalarSlotMasks[0]) == target::kSlotCount);

std::uint32_t* scalar_mask_word(unsigned char* bundle) {
    auto* scalar = *reinterpret_cast<unsigned char**>(bundle + kScalarOneofPointerOffset);
    return reinterpret_cast<std::uint32_t*>(scalar + kScalarBundleMaskOffset);
}
#endif

Presence read_presence(unsigned char* bundle, int pc) {
    Presence presence;
    presence.mask = *reinterpret_cast<const std::uint32_t*>(bundle + kBundleMaskOffset);
    std::uint32_t known = decoded_slot_mask();
#ifdef TPUASM_SCALAR_ONEOF
    known |= kBundleSharedMask;
    presence.scalar_case = *reinterpret_cast<const std::uint32_t*>(bundle + kScalarOneofCaseOffset);
    bool known_case = presence.scalar_case == 0;
    std::uint32_t known_scalar = 0;
    for (std::size_t slot = 0; slot != target::kSlotCount; ++slot) {
        known_case |= presence.scalar_case == kScalarSlotCases[slot];
        known_scalar |= kScalarSlotMasks[slot];
    }
    if (presence.scalar_case == kScalarBundleCase) presence.scalar_mask = *scalar_mask_word(bundle);
    if (!known_case || (presence.scalar_mask & ~known_scalar) != 0) {
        throw std::runtime_error("unexpected decoded scalar oneof at PC " + std::to_string(pc));
    }
#endif
    if ((presence.mask & ~known) != 0) throw std::runtime_error("unexpected decoded bundle presence mask at PC " + std::to_string(pc));
    return presence;
}

void write_presence(unsigned char* bundle, const Presence& presence, const Presence& original) {
    *reinterpret_cast<std::uint32_t*>(bundle + kBundleMaskOffset) = presence.mask;
#ifdef TPUASM_SCALAR_ONEOF
    *reinterpret_cast<std::uint32_t*>(bundle + kScalarOneofCaseOffset) = presence.scalar_case;
    // The sub-bundle pointer is only valid while the original case selects it.
    if (original.scalar_case == kScalarBundleCase) *scalar_mask_word(bundle) = presence.scalar_mask;
#else
    (void)original;
#endif
}

bool slot_present(const Presence& presence, std::size_t slot) {
#ifdef TPUASM_SCALAR_ONEOF
    if (kScalarSlotCases[slot] != 0) {
        return presence.scalar_case == kScalarSlotCases[slot] && (kScalarSlotMasks[slot] == 0 || (presence.scalar_mask & kScalarSlotMasks[slot]) != 0);
    }
#endif
    return (presence.mask & kBundleSlotMasks[slot]) != 0;
}

// Keep only one physical slot. Shared operand messages stay present so the
// formatter still reads the immediates and scalar operands of that slot.
// Add one physical slot of the original bundle to a presence view. Shared
// operand messages stay present so the formatter still reads the immediates
// and scalar operands of the kept slots.
Presence with_slot(Presence view, const Presence& original, std::size_t slot) {
    view.mask |= kBundleSlotMasks[slot];
#ifdef TPUASM_SCALAR_ONEOF
    view.mask |= original.mask & kBundleSharedMask;
    if (kScalarSlotCases[slot] != 0) {
        view.scalar_case = kScalarSlotCases[slot];
        if (kScalarSlotCases[slot] == kScalarBundleCase) view.scalar_mask |= kScalarSlotMasks[slot];
    }
#else
    (void)original;
#endif
    return view;
}

#endif

using Decode = void (*)(void*, const void*, std::size_t);
using Encode = void (*)(void*, const void*);
using Destroy = void (*)(void*);
using Format = void (*)(void*, void*, const void*, std::int64_t, int, const void*, const void*);
using Construct = void (*)(void*, void*);
using ByteSize = std::size_t (*)(const void*);
using Serialize = bool (*)(const void*, void*, int);
using Parse = bool (*)(void*, const void*, int);

void diagnostic(char* output, std::size_t capacity, const std::string& message) {
    if (output != nullptr && capacity != 0) {
        std::snprintf(output, capacity, "%s", message.c_str());
    }
}

struct Library {
    void* handle;
    explicit Library(const char* path) : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {}
    ~Library() {
        if (handle != nullptr) dlclose(handle);
    }
};

struct Program {
    void* value = nullptr;
    Destroy destroy = nullptr;
    ~Program() {
        if (value != nullptr) destroy(value);
    }
};

unsigned char* checked_base(const Library& library) {
    if (library.handle == nullptr) {
        const char* reason = dlerror();
        throw std::runtime_error(reason == nullptr ? "cannot load libtpu" : reason);
    }
    void* exported = dlsym(library.handle, "GetPjrtApi");
    Dl_info info = {};
    if (exported == nullptr || dladdr(exported, &info) == 0 || info.dli_fbase == nullptr) {
        throw std::runtime_error("cannot locate loaded libtpu base");
    }
    auto* base = static_cast<unsigned char*>(info.dli_fbase);
    if (static_cast<unsigned char*>(exported) - base != kGetPjrtApi) {
        throw std::runtime_error("libtpu exported symbol offset differs");
    }
    return base;
}

#ifndef TPUASM_CODEC_ONLY
// libtpu uses libc++'s 24-byte std::__u::string, including its short-string form.
std::string copy_libtpu_string(const unsigned char* object) {
    const bool long_form = static_cast<signed char>(object[23]) < 0;
    const std::size_t size = long_form ? *reinterpret_cast<const std::size_t*>(object + 8) : object[23];
    const char* data = long_form ? *reinterpret_cast<char* const*>(object) : reinterpret_cast<const char*>(object);
    return std::string(data, size);
}

void release_libtpu_string(unsigned char* object) {
    if (static_cast<signed char>(object[23]) < 0) std::free(*reinterpret_cast<void**>(object));
}

std::string format_bundle(Format format, unsigned char* base, const void* bundle, std::int64_t pc) {
    alignas(32) unsigned char returned[128] = {};
    unsigned char formatter = 0;  // FormatterPf and FormatterGl have no instance state in these builds.
    format(returned, &formatter, bundle, pc, 1, base + kEmptyAnnotations, nullptr);
    if (*reinterpret_cast<const std::uintptr_t*>(returned) != 1) {
        throw std::runtime_error("native ISA formatter failed at PC " + std::to_string(pc));
    }
    std::string result = copy_libtpu_string(returned + sizeof(std::uintptr_t));
    release_libtpu_string(returned + sizeof(std::uintptr_t));
    return result;
}

void** bundle_array(void* program) {
    auto* storage = static_cast<unsigned char*>(program) + kProgramBundleStorageOffset;
    const std::uintptr_t tagged = *reinterpret_cast<const std::uintptr_t*>(storage);
    return reinterpret_cast<void**>((tagged & 1) ? tagged + 7 : reinterpret_cast<std::uintptr_t>(storage));
}
#endif
}  // namespace

extern "C" int tpuasm_verify(
    const char* libtpu_path,
    const void* image,
    std::size_t image_size,
    char* error,
    std::size_t error_capacity) {
    if (libtpu_path == nullptr || image == nullptr || image_size == 0 || image_size % target::kImageBlockSize != 0) {
        diagnostic(error, error_capacity, "expected a nonempty block-aligned TPU program image");
        return 1;
    }
    try {
        Library library(libtpu_path);
        auto* base = checked_base(library);
        for (const std::uintptr_t address : { kDecodeProgram, kEncodeProgram }) {
            if (std::memcmp(base + address, kFunctionPrologue, sizeof(kFunctionPrologue)) != 0) {
                throw std::runtime_error("libtpu function signature differs at " + std::to_string(address));
            }
        }
        const auto decode = reinterpret_cast<Decode>(base + kDecodeProgram);
        const auto encode = reinterpret_cast<Encode>(base + kEncodeProgram);
#ifndef TPUASM_CODEC_ONLY
        if (std::memcmp(base + kFormatBundle, kFunctionPrologue, sizeof(kFunctionPrologue)) != 0) {
            throw std::runtime_error("libtpu formatter signature differs");
        }
        const auto format = reinterpret_cast<Format>(base + kFormatBundle);
#endif
        alignas(32) unsigned char decoded[512] = {};
        Program program;
        program.destroy = reinterpret_cast<Destroy>(base + kDestroyProgram);
        decode(decoded, image, image_size);
        if (*reinterpret_cast<const std::uintptr_t*>(decoded) != 1) {
            throw std::runtime_error("libtpu rejected the TPU program image");
        }
        program.value = decoded + sizeof(std::uintptr_t);
        const int count = *reinterpret_cast<const int*>(static_cast<unsigned char*>(program.value) + kProgramBundleCountOffset);
        if (count <= 0 || static_cast<std::size_t>(count) != image_size / target::kImageBlockSize * target::kBundlesPerBlock) {
            throw std::runtime_error("decoded bundle count differs from the program image byte count");
        }

        // Verify the decoded program reencodes to the input bytes before
        // formatting. This does not make the formatter's text reversible.
        alignas(32) unsigned char encoded[128] = {};
        encode(encoded, program.value);
        if (*reinterpret_cast<const std::uintptr_t*>(encoded) != 1) {
            throw std::runtime_error("libtpu could not reencode the decoded program");
        }
        void* encoded_bytes = *reinterpret_cast<void**>(encoded + 8);
        const std::size_t encoded_size = *reinterpret_cast<const std::size_t*>(encoded + 16);
        const bool exact = encoded_bytes != nullptr && encoded_size == image_size &&
                           std::memcmp(encoded_bytes, image, image_size) == 0;
        std::free(encoded_bytes);
        if (!exact) throw std::runtime_error("decoded program does not reencode to the original machine bytes");

#ifndef TPUASM_CODEC_ONLY
        void** bundles = bundle_array(program.value);
        for (int pc = 0; pc < count; ++pc) {
            auto* bundle = static_cast<unsigned char*>(bundles[pc]);
            if (bundle == nullptr) throw std::runtime_error("null decoded bundle at PC " + std::to_string(pc));
            const Presence presence = read_presence(bundle, pc);
            Presence formatted;
            std::string rebuilt = "{";
            bool first = true;
            for (std::size_t slot = 0; slot != target::kSlotCount; ++slot) {
                if (!slot_present(presence, slot)) continue;
                write_presence(bundle, with_slot(Presence(), presence, slot), presence);
                std::string part;
                try {
                    part = format_bundle(format, base, bundle, pc);
                } catch (const std::runtime_error&) {
                    write_presence(bundle, presence, presence);
#ifdef TPUASM_FORMATTER_GAPS
                    // This form has no formatter; the remaining slots are still cross-checked.
                    continue;
#else
                    throw;
#endif
                } catch (...) {
                    write_presence(bundle, presence, presence);
                    throw;
                }
                write_presence(bundle, presence, presence);
                formatted = with_slot(formatted, presence, slot);
                if (part.size() < 2 || part.front() != '{' || part.back() != '}') {
                    throw std::runtime_error("unexpected native slot format at PC " + std::to_string(pc));
                }
                const std::string instruction = part.substr(1, part.size() - 2);
                if (instruction.empty()) continue;
                if (!first) rebuilt += "; ";
                rebuilt += instruction;
                first = false;
            }
            rebuilt += '}';
            write_presence(bundle, formatted, presence);
            std::string whole;
            try {
                whole = format_bundle(format, base, bundle, pc);
            } catch (...) {
                write_presence(bundle, presence, presence);
                throw;
            }
            write_presence(bundle, presence, presence);
            if (rebuilt != whole) {
                throw std::runtime_error("individual physical slots do not reconstruct bundle at PC " + std::to_string(pc));
            }
        }
#endif
        diagnostic(error, error_capacity, "verified and reencoded " + std::to_string(count) + " bundles exactly");
        return 0;
    } catch (const std::exception& exception) {
        diagnostic(error, error_capacity, exception.what());
        return 2;
    }
}

extern "C" int tpuasm_program_proto(
    const char* libtpu_path,
    const void* input,
    std::size_t input_size,
    int encode_mode,
    void** output,
    std::size_t* output_size,
    char* error,
    std::size_t error_capacity) {
    if (libtpu_path == nullptr || input == nullptr || input_size == 0 ||
        output == nullptr || output_size == nullptr ||
        input_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        diagnostic(error, error_capacity, "invalid program input");
        return 1;
    }
    *output = nullptr;
    *output_size = 0;
    try {
        Library library(libtpu_path);
        auto* base = checked_base(library);
        for (const std::uintptr_t address : { kDecodeProgram, kEncodeProgram }) {
            if (std::memcmp(base + address, kFunctionPrologue, sizeof(kFunctionPrologue)) != 0) {
                throw std::runtime_error("libtpu codec signature differs at " + std::to_string(address));
            }
        }
        const auto decode = reinterpret_cast<Decode>(base + kDecodeProgram);
        const auto encode = reinterpret_cast<Encode>(base + kEncodeProgram);
        Program program;
        program.destroy = reinterpret_cast<Destroy>(base + kDestroyProgram);
        alignas(32) unsigned char decoded[512] = {};
        alignas(32) unsigned char constructed[512] = {};
        if (encode_mode) {
            const auto construct = reinterpret_cast<Construct>(base + kConstructProgram);
            const auto parse = reinterpret_cast<Parse>(base + kProtoParseFromArray);
            construct(constructed, nullptr);
            program.value = constructed;
            if (!parse(program.value, input, static_cast<int>(input_size))) {
                throw std::runtime_error("libtpu rejected the ISA program protobuf");
            }
        } else {
            if (input_size % target::kImageBlockSize != 0) {
                throw std::runtime_error("TPU program image is not block-aligned");
            }
            decode(decoded, input, input_size);
            if (*reinterpret_cast<const std::uintptr_t*>(decoded) != 1) {
                throw std::runtime_error("libtpu rejected the TPU program image");
            }
            program.value = decoded + sizeof(std::uintptr_t);
            const int count = *reinterpret_cast<const int*>(static_cast<unsigned char*>(program.value) + kProgramBundleCountOffset);
            if (count <= 0 || static_cast<std::size_t>(count) != input_size / target::kImageBlockSize * target::kBundlesPerBlock) {
                throw std::runtime_error("decoded bundle count differs from the program image byte count");
            }
        }
        alignas(32) unsigned char encoded[128] = {};
        encode(encoded, program.value);
        if (*reinterpret_cast<const std::uintptr_t*>(encoded) != 1) {
            throw std::runtime_error("libtpu could not encode the ISA program");
        }
        void* encoded_bytes = *reinterpret_cast<void**>(encoded + 8);
        const std::size_t encoded_size = *reinterpret_cast<const std::size_t*>(encoded + 16);
        if (encoded_bytes == nullptr || encoded_size == 0 || encoded_size % target::kImageBlockSize != 0) {
            std::free(encoded_bytes);
            throw std::runtime_error("libtpu produced an invalid program image byte count");
        }
        if (encode_mode) {
            *output = encoded_bytes;
            *output_size = encoded_size;
        } else {
            const bool exact = encoded_size == input_size && std::memcmp(encoded_bytes, input, input_size) == 0;
            std::free(encoded_bytes);
            if (!exact) throw std::runtime_error("decoded program does not reencode to the original machine bytes");
            const auto byte_size = reinterpret_cast<ByteSize>(base + kProgramByteSize);
            const auto serialize = reinterpret_cast<Serialize>(base + kProtoSerializeToArray);
            const std::size_t size = byte_size(program.value);
            if (size == 0 || size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("ISA program protobuf has invalid size");
            }
            void* serialized = std::malloc(size);
            if (serialized == nullptr) throw std::bad_alloc();
            if (!serialize(program.value, serialized, static_cast<int>(size))) {
                std::free(serialized);
                throw std::runtime_error("cannot serialize the ISA program protobuf");
            }
            *output = serialized;
            *output_size = size;
        }
        diagnostic(error, error_capacity, "OK");
        return 0;
    } catch (const std::exception& exception) {
        diagnostic(error, error_capacity, exception.what());
        return 2;
    }
}

#ifndef TPUASM_CODEC_ONLY
// Formatter text of every bundle, one line each. Only offline ISA table
// generators read it; listings never contain formatter text. The formatter
// aborts the process on some reserved encodings, so callers isolate it.
extern "C" int tpuasm_format(
    const char* libtpu_path,
    const void* image,
    std::size_t image_size,
    void** output,
    std::size_t* output_size,
    char* error,
    std::size_t error_capacity) {
    if (libtpu_path == nullptr || image == nullptr || image_size == 0 || image_size % target::kImageBlockSize != 0 || output == nullptr || output_size == nullptr) {
        diagnostic(error, error_capacity, "expected a nonempty block-aligned TPU program image");
        return 1;
    }
    *output = nullptr;
    *output_size = 0;
    try {
        Library library(libtpu_path);
        auto* base = checked_base(library);
        for (const std::uintptr_t address : { kDecodeProgram, kFormatBundle }) {
            if (std::memcmp(base + address, kFunctionPrologue, sizeof(kFunctionPrologue)) != 0) {
                throw std::runtime_error("libtpu function signature differs at " + std::to_string(address));
            }
        }
        const auto decode = reinterpret_cast<Decode>(base + kDecodeProgram);
        const auto format = reinterpret_cast<Format>(base + kFormatBundle);
        alignas(32) unsigned char decoded[512] = {};
        Program program;
        program.destroy = reinterpret_cast<Destroy>(base + kDestroyProgram);
        decode(decoded, image, image_size);
        if (*reinterpret_cast<const std::uintptr_t*>(decoded) != 1) {
            throw std::runtime_error("libtpu rejected the TPU program image");
        }
        program.value = decoded + sizeof(std::uintptr_t);
        const int count = *reinterpret_cast<const int*>(static_cast<unsigned char*>(program.value) + kProgramBundleCountOffset);
        void** bundles = bundle_array(program.value);
        std::string text;
        for (int pc = 0; pc < count; ++pc) {
            // Some descriptor forms have no formatter; report them per bundle.
            try {
                text += format_bundle(format, base, bundles[pc], pc);
            } catch (const std::runtime_error&) {
                text += "<no formatter output>";
            }
            text += '\n';
        }
        void* buffer = std::malloc(text.size());
        if (buffer == nullptr) throw std::bad_alloc();
        std::memcpy(buffer, text.data(), text.size());
        *output = buffer;
        *output_size = text.size();
        diagnostic(error, error_capacity, "OK");
        return 0;
    } catch (const std::exception& exception) {
        diagnostic(error, error_capacity, exception.what());
        return 2;
    }
}
#endif

extern "C" void tpuasm_free(void* pointer) {
    std::free(pointer);
}

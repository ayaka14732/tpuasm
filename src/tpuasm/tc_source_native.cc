// Version-locked native source preservation; no host C++ objects cross libtpu ABI.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>

namespace {
using namespace tpuasm_source_backend;
struct View { const char* data; std::size_t size; };
struct NativeString { alignas(8) unsigned char bytes[24]{}; };
static uintptr_t base;
static std::atomic<uint64_t> calls{0}, added{0}, failures{0}, rewrites{0}, remapped{0};
template<class T> T read(const void* p, ptrdiff_t offset) {
    T value;
    std::memcpy(&value, static_cast<const char*>(p) + offset, sizeof(T));
    return value;
}
std::string native_string(const void* value) {
    int8_t small = read<int8_t>(value, 23);
    return small >= 0 ? std::string(static_cast<const char*>(value), small)
                      : std::string(read<const char*>(value, 0), read<std::size_t>(value, 8));
}
void release(NativeString& s) {
    if (read<int8_t>(&s, 23) < 0) std::free(read<void*>(&s, 0));
}
uint64_t varint(std::string_view s, std::size_t& p) {
    uint64_t value = 0;
    for (int shift = 0; shift < 70 && p < s.size(); shift += 7) {
        unsigned char b = s[p++];
        value |= uint64_t(b & 127) << shift;
        if (b < 128) return value;
    }
    throw std::runtime_error("invalid source protobuf");
}
void varint(std::string& s, uint64_t v) {
    while (v >= 128) { s += char((v & 127) | 128); v >>= 7; }
    s += char(v);
}
struct Field { int number; int wire; uint64_t integer; std::string_view bytes; };
std::vector<Field> fields(std::string_view s) {
    std::vector<Field> out;
    std::size_t p = 0;
    while (p < s.size()) {
        auto tag = varint(s, p);
        Field f{int(tag >> 3), int(tag & 7), 0, {}};
        if (f.wire == 0) f.integer = varint(s, p);
        else if (f.wire == 2) {
            auto n = varint(s, p);
            if (n > s.size() - p) throw std::runtime_error("invalid source field length");
            f.bytes = s.substr(p, n); p += n;
        } else throw std::runtime_error("unexpected source wire type");
        out.push_back(f);
    }
    return out;
}
void field(std::string& s, int n, uint64_t v) { varint(s, uint64_t(n) << 3); varint(s, v); }
void field(std::string& s, int n, std::string_view v) {
    varint(s, (uint64_t(n) << 3) | 2); varint(s, v.size()); s.append(v);
}
std::string source_record(void* inst) {
    void* region = read<void*>(inst, 0);
    if (!region) return {};
    void* module = read<void*>(region, 0x38);
    if (!module) return {};
    void* map = read<void*>(module, 0x340);
    if (!map) return {};
    auto size = reinterpret_cast<std::size_t(*)(void*)>(base + kSourceMapSize)(map);
    std::string raw(size, '\0');
    if (!reinterpret_cast<bool(*)(void*, void*, int)>(base + kSerialize)(map, raw.data(), raw.size()))
        throw std::runtime_error("source map serialization failed");
    int ordinal = read<int>(inst, -0xc);
    std::string subset;
    for (const auto& f : fields(raw)) {
        if (f.number == 2) field(subset, 2, f.bytes);
        if (f.number != 1) continue;
        bool match = false;
        for (const auto& info : fields(f.bytes)) {
            if (info.number != 4) continue;
            if (info.wire == 0) match |= int(info.integer) == ordinal;
            else {
                std::size_t p = 0;
                while (p < info.bytes.size()) match |= int(varint(info.bytes, p)) == ordinal;
            }
        }
        if (match) field(subset, 1, f.bytes);
    }
    std::string result;
    field(result, 1, uint64_t(ordinal));
    field(result, 2, subset);
    // LloSerializer::LloModuleToProto reads the same root HLO association.
    void* root = read<void*>(module, 0x250);
    void* hlo = root ? read<void*>(root, 0x28) : nullptr;
    auto name = native_string(static_cast<char*>(module) + 0x328);
    if (name.empty() && hlo) name = native_string(static_cast<char*>(hlo) + 0x78);
    field(result, 3, name);
    if (hlo) {
        void* hlo_module = reinterpret_cast<void*(*)(void*)>(base + kGetHloModule)(hlo);
        if (hlo_module) {
            field(result, 4, native_string(static_cast<char*>(hlo_module) + 8));
            field(result, 5, uint64_t(read<int>(hlo_module, 0xaa4)));
        }
    }
    return result;
}

std::vector<void*> source_locations(void* source_map, const std::unordered_set<int>& ordinals) {
    std::vector<void*> locations;
    if (!source_map || ordinals.empty()) return locations;
    // LloSourceMap.locations is a native RepeatedPtrField<SourceInfo>.
    int count = read<int>(source_map, 0x20);
    uintptr_t tagged = read<uintptr_t>(source_map, 0x18);
    for (int i = 0; i < count; ++i) {
        void* loc = tagged & 1 ? read<void*>(reinterpret_cast<void*>(tagged - 1), 8 + 8 * i) : read<void*>(source_map, 0x18);
        // SourceInfo.ordinals is a native RepeatedField<int>, inline or heap.
        void* field = static_cast<char*>(loc) + 0x38;
        void* data = read<uint8_t>(field, 0) & 1 ? read<void*>(field, 8) : field;
        for (int j = 0; j < read<int>(field, 4); ++j) {
            if (ordinals.count(read<int>(data, 8 + 4 * j))) {
                locations.push_back(loc);
                break;
            }
        }
    }
    return locations;
}

bool append_sources(void* value, const std::vector<void*>& locations) {
    if (locations.empty()) return false;
    int ordinal = read<int>(value, -0xc);
    auto append = reinterpret_cast<void(*)(void*, const void*, int)>(base + kAppendOrdinal);
    bool changed = false;
    for (void* loc : locations) {
        void* field = static_cast<char*>(loc) + 0x38;
        void* data = read<uint8_t>(field, 0) & 1 ? read<void*>(field, 8) : field;
        bool present = false;
        for (int j = 0; j < read<int>(field, 4); ++j) present |= read<int>(data, 8 + 4 * j) == ordinal;
        if (present) continue;
        // Native append handles inline storage, arena ownership, and growth.
        append(field, loc, ordinal);
        remapped.fetch_add(1, std::memory_order_relaxed);
        changed = true;
    }
    if (changed) rewrites.fetch_add(1, std::memory_order_relaxed);
    return changed;
}

std::vector<void*> operands(void* value) {
    std::vector<void*> result;
    int count = read<int>(value, -0x10) - (read<uint16_t>(value, 0x1c) & 3);
    auto operand = reinterpret_cast<void*(*)(void*, int)>(base + kOperands);
    for (int i = 0; i < count; ++i) result.push_back(operand(value, i));
    return result;
}

void annotate_source_record(void* inst) {
    std::string record = source_record(inst);
    if (record.empty()) return;
    const void* old = reinterpret_cast<const void*(*)(void*)>(base + kAnnotation)(inst);
    std::string annotation = native_string(old);
    if (annotation.find("[[tpuasm:v1:") != std::string::npos) return;
    annotation += " [[tpuasm:v1:";
    const char hex[] = "0123456789abcdef";
    for (unsigned char c : record) {
        annotation += hex[c >> 4];
        annotation += hex[c & 15];
    }
    annotation += "]]";
    // Native setter owns a copy for the entire LLO instruction lifetime,
    // including delayed emission and bundle finalization.
    reinterpret_cast<void(*)(void*, View)>(base + kSetAnnotation)(inst, {annotation.data(), annotation.size()});
    added.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

extern "C" uintptr_t source_emit_hook(void* generator, void* inst) noexcept {
    calls.fetch_add(1, std::memory_order_relaxed);
    try {
        annotate_source_record(inst);
    } catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
    return reinterpret_cast<uintptr_t(*)(void*, void*)>(base + kEmit)(generator, inst);
}

// StoreCommon's inner annotator sees the store slot already allocated by its
// caller. Its empty slot delta still consumes the current annotation, leaving
// the outer annotator with no text. Preserve the view until the outer guard
// records the actual newly occupied slot. Ownership remains with the LLO value.
extern "C" void source_store_annotation_hook(void* guard) {
    void* emitter = read<void*>(guard, 0);
    View annotation = read<View>(emitter, 0x188);
    reinterpret_cast<void(*)(void*)>(base + kScopedAnnotator)(guard);
    std::memcpy(static_cast<char*>(emitter) + 0x188, &annotation, sizeof(annotation));
}

static void propagate_sources(void* old_value, void* new_value, bool expression = true) {
    void* region = read<void*>(old_value, 0);
    if (!region) return;
    void* module = read<void*>(region, 0x38);
    if (!module) return;
    void* new_region = read<void*>(new_value, 0);
    if (new_region && read<void*>(new_region, 0x38) != module) return;
    void* source_map = read<void*>(module, 0x340);
    std::vector<void*> locations;
    if (source_map) {
        auto graph_operands = [expression](void* value) {
            return expression ? operands(value) : std::vector<void*>{};
        };
        std::unordered_set<void*> old_graph;
        std::vector<void*> pending{old_value};
        while (!pending.empty()) {
            void* value = pending.back();
            pending.pop_back();
            if (!old_graph.insert(value).second) continue;
            auto children = graph_operands(value);
            pending.insert(pending.end(), children.begin(), children.end());
        }
        // Eliminating an operation in favor of an existing value must not attach
        // its name to that value's other uses.
        if (old_graph.count(new_value)) return;
        std::unordered_set<void*> boundary, visited;
        pending = {new_value};
        while (!pending.empty()) {
            void* value = pending.back();
            pending.pop_back();
            if (!visited.insert(value).second) continue;
            if (old_graph.count(value)) {
                boundary.insert(value);
                continue;
            }
            auto children = graph_operands(value);
            pending.insert(pending.end(), children.begin(), children.end());
        }
        visited.clear();
        std::unordered_set<int> origins;
        pending = {old_value};
        while (!pending.empty()) {
            void* value = pending.back();
            pending.pop_back();
            if (boundary.count(value) || !visited.insert(value).second) continue;
            origins.insert(read<int>(value, -0xc));
            auto children = graph_operands(value);
            pending.insert(pending.end(), children.begin(), children.end());
        }
        locations = source_locations(source_map, origins);
    }
    append_sources(new_value, locations);
}

extern "C" void* source_replace_hook(void* old_value, void* new_value) {
    try { propagate_sources(old_value, new_value); }
    catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
    return reinterpret_cast<void*(*)(void*, void*)>(base + kReplace)(old_value, new_value);
}

extern "C" void source_coalesce_hook(void* new_value, const char* annotation, void* second_candidate) {
    reinterpret_cast<void(*)(void*, const char*)>(base + kSetAnnotationText)(new_value, annotation);
    try {
        auto instruction = reinterpret_cast<void*(*)(void*)>(base + kCandidateInstruction);
        propagate_sources(instruction(static_cast<char*>(second_candidate) - 0x28), new_value, false);
        propagate_sources(instruction(second_candidate), new_value, false);
    }
    catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
}

// The load-store optimizer replaces two sublane-half loads with one combined
// load. The trampoline passes both original loads from the caller's frame.
extern "C" void source_combine_hook(void* new_value, const char* annotation, void* first, void* second) {
    reinterpret_cast<void(*)(void*, const char*)>(base + kSetAnnotationText)(new_value, annotation);
    try {
        propagate_sources(first, new_value, false);
        propagate_sources(second, new_value, false);
    }
    catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
}

struct RegionResult { void* region; void* value; };
extern "C" RegionResult source_region_hook(void* old_value) {
    auto result = reinterpret_cast<RegionResult(*)(void*)>(base + kRegion)(old_value);
    if (result.value) {
        try { propagate_sources(old_value, result.value); }
        catch (...) { failures.fetch_add(1, std::memory_order_relaxed); }
    }
    return result;
}


extern "C" void source_configure(uintptr_t address) { base = address; }
extern "C" uint64_t source_counter(int index) {
    switch (index) {
        case 0: return calls.load(); case 1: return added.load(); case 2: return failures.load();
        case 3: return rewrites.load(); case 4: return remapped.load(); default: return 0;
    }
}
extern "C" int source_flag(const char* name, const char* value, char* output, std::size_t capacity) {
    View key{name, std::strlen(name)};
    NativeString old;
    if (!reinterpret_cast<bool(*)(View, NativeString*)>(base + kGetFlag)(key, &old)) return 1;
    auto previous = native_string(&old);
    release(old);
    if (previous.size() + 1 > capacity) return 2;
    std::memcpy(output, previous.c_str(), previous.size() + 1);
    if (value) {
        NativeString result;
        reinterpret_cast<void(*)(NativeString*, View, View)>(base + kSetFlag)(&result, key, {value, std::strlen(value)});
        bool ok = !native_string(&result).empty();
        release(result);
        if (!ok) return 3;
    }
    return 0;
}

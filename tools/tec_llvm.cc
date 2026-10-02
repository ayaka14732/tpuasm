// Host-only bridge to libtpu's LLVM TPU MC printer and Ghostlite SparseCore TEC
// emitter, used by generate_tpu_v6e_tec_isa.py. The generator includes the
// libtpu function addresses of the matching TEC native backend before this file.
//
// libtpu compiles SparseCore kernels to LLVM MCInsts and converts each bundle of
// MCInsts to a SparseCoreTecBundle protobuf in ConsumeBundle. Feeding the same
// MCInst to TPUInstPrinter and to ConsumeBundle pairs the compiler's printed
// syntax with the fields the emitter writes.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
using namespace tpuasm_backend;

// LLVM object layouts in the libtpu 0.0.49 LLVM snapshot.
// MCInstrInfo: last MCInstrDesc, name index table, name data, opcode count.
constexpr std::size_t kInstrInfoLastDesc = 0x0;
constexpr std::size_t kInstrInfoNameIndices = 0x8;
constexpr std::size_t kInstrInfoNameData = 0x10;
constexpr std::size_t kInstrInfoOpcodeCount = 0x28;
// MCInstrDesc entries are 32 bytes, stored in reverse opcode order. Operand
// counts sit at 4, the operand info offset (in 6-byte MCOperandInfo entries
// counted from the end of the table) at 12.
constexpr std::size_t kInstrDescSize = 32;
constexpr std::size_t kInstrDescOperandCount = 4;
constexpr std::size_t kInstrDescOperandOffset = 12;
constexpr std::size_t kOperandInfoSize = 6;
// MCRegisterInfo: register classes and their count.
constexpr std::size_t kRegisterInfoClasses = 0x20;
constexpr std::size_t kRegisterInfoClassCount = 0x28;
// Register classes are 64-byte entries with a self-relative register list and
// the register count at 16.
constexpr std::size_t kRegisterClassSize = 64;
constexpr std::size_t kRegisterClassCount = 16;
// MCInst: opcode, slot flags, VS lane flags (3 bits per operand), operand
// vector with 6 inline 16-byte MCOperands.
constexpr std::uint32_t kBundleOpcode = 0x16;
// MCOperand kinds.
constexpr std::uint8_t kExprOperand = 5;
constexpr std::uint8_t kInstOperand = 6;
// MCExpr kinds: constant and target (TPUMCImmExpr).
constexpr std::uint8_t kConstantExpr = 1;
constexpr std::uint8_t kTargetExpr = 5;

unsigned char* base = nullptr;
void* triple = nullptr;
void* subtarget = nullptr;
void* instr_info = nullptr;
void* register_info = nullptr;
void* printer = nullptr;

template <typename T>
T at(std::uintptr_t address) { return reinterpret_cast<T>(base + address); }

template <typename T>
T load(const void* object, std::size_t offset) {
    T value;
    std::memcpy(&value, static_cast<const unsigned char*>(object) + offset, sizeof(T));
    return value;
}

// libtpu uses libc++'s 24-byte std::__u::string, including its short-string form.
std::string libtpu_string(const unsigned char* object) {
    const bool long_form = static_cast<signed char>(object[23]) < 0;
    const std::size_t size = long_form ? load<std::size_t>(object, 8) : object[23];
    const char* data = long_form ? load<const char*>(object, 0) : reinterpret_cast<const char*>(object);
    return std::string(data, size);
}

struct Operand {
    std::uint8_t kind;
    std::uint8_t padding[7];
    std::uint64_t value;
};

struct Inst {
    std::uint32_t opcode;
    std::uint32_t slot_flags;
    std::uint64_t lane_flags;
    Operand* data;
    std::uint32_t size;
    std::uint32_t capacity;
    Operand inline_storage[6];
};
static_assert(sizeof(Inst) == 128);

struct Constant {
    std::uint8_t kind;
    std::uint8_t padding[15];
    std::int64_t value;
};

// TPUMCImmExpr: an MCTargetExpr holding the constant, the shared immediate
// slot, the selector encoding and the immediate type chosen by the backend.
struct ImmExpr {
    void* vtable;
    std::uint64_t header;
    std::uint64_t location;
    std::uint32_t imm_kind;
    std::uint32_t padding0;
    const Constant* value;
    std::uint8_t slot;
    std::uint8_t encoding;
    std::uint16_t padding1;
    std::uint32_t type;
    void* context;
};
static_assert(sizeof(ImmExpr) == 0x38);

struct Bundle {
    std::vector<Inst> insts;
    std::vector<Operand> operands;
    std::vector<Constant> constants;
    std::vector<ImmExpr> expressions;
    std::vector<Operand> members;
    Inst bundle {};
};

// A record is an instruction count, then per instruction: opcode | slot flags
// << 32, lane flags, operand count and (kind, value, extra) triples. Kind 5
// builds a TPUMCImmExpr with extra = slot | encoding << 8 | type << 16.
std::size_t parse(const std::uint64_t* words, std::size_t pos, Bundle& out) {
    const std::size_t count = words[pos++];
    std::size_t operand_total = 0, expression_total = 0;
    for (std::size_t scan = pos, i = 0; i != count; ++i) {
        scan += 2;
        const std::size_t operands = words[scan++];
        for (std::size_t j = 0; j != operands; ++j, scan += 3) expression_total += words[scan] == kExprOperand;
        operand_total += operands;
    }
    out.insts.assign(count, Inst {});
    out.operands.assign(operand_total, Operand {});
    out.constants.assign(expression_total, Constant {});
    out.expressions.assign(expression_total, ImmExpr {});
    std::size_t used = 0, expression = 0;
    for (Inst& inst : out.insts) {
        inst.opcode = static_cast<std::uint32_t>(words[pos] & 0xffffffff);
        inst.slot_flags = static_cast<std::uint32_t>(words[pos] >> 32);
        inst.lane_flags = words[pos + 1];
        const std::size_t operands = words[pos + 2];
        pos += 3;
        inst.data = out.operands.data() + used;
        inst.size = inst.capacity = static_cast<std::uint32_t>(operands);
        for (std::size_t j = 0; j != operands; ++j, pos += 3) {
            Operand& operand = out.operands[used++];
            operand.kind = static_cast<std::uint8_t>(words[pos]);
            operand.value = words[pos + 1];
            if (operand.kind != kExprOperand) continue;
            Constant& constant = out.constants[expression];
            constant.kind = kConstantExpr;
            constant.value = static_cast<std::int64_t>(words[pos + 1]);
            ImmExpr& immediate = out.expressions[expression++];
            immediate.vtable = base + kImmExprVtable + 0x10;
            immediate.header = kTargetExpr;
            immediate.value = &constant;
            immediate.slot = static_cast<std::uint8_t>(words[pos + 2]);
            immediate.encoding = static_cast<std::uint8_t>(words[pos + 2] >> 8);
            immediate.type = static_cast<std::uint32_t>(words[pos + 2] >> 16 & 0xff);
            // The MCOperand points at the MCExpr header behind the vtable.
            operand.value = reinterpret_cast<std::uint64_t>(&immediate.header);
        }
    }
    out.members.assign(count, Operand {});
    for (std::size_t i = 0; i != count; ++i) {
        out.members[i].kind = kInstOperand;
        out.members[i].value = reinterpret_cast<std::uint64_t>(&out.insts[i]);
    }
    out.bundle.opcode = kBundleOpcode;
    out.bundle.data = out.members.data();
    out.bundle.size = out.bundle.capacity = static_cast<std::uint32_t>(count);
    return pos;
}

std::string print(const Bundle& bundle) {
    alignas(16) unsigned char text[24] = {};
    alignas(16) unsigned char stream[0x40] = {};
    at<void (*)(void*, void*)>(kStringOstreamCtor)(stream, text);
    at<void (*)(void*, const void*, std::uint64_t, const char*, std::size_t, void*, void*)>(kPrintInst)(printer, &bundle.bundle, 0, "", 0, subtarget, stream);
    return libtpu_string(text);
}

// Serialized SparseCoreTecBundle, or the emitter's status message.
bool consume(const Bundle& bundle, std::string& out) {
    alignas(64) unsigned char proto[256] = {};
    at<void (*)(void*, void*)>(kConstructBundle)(proto, nullptr);
    const std::uintptr_t status = at<std::uintptr_t (*)(void*, const void*, void*, bool)>(kConsumeBundle)(printer, &bundle.bundle, proto, true);
    if (status != 1) {
        // A non-inlined absl::Status points at a StatusRep whose message follows the code.
        out = (status & 1) == 0 ? libtpu_string(reinterpret_cast<const unsigned char*>(status) + 8) : "status " + std::to_string(status);
        return false;
    }
    const std::size_t size = at<std::size_t (*)(const void*)>(kBundleByteSize)(proto);
    out.assign(size, '\0');
    at<bool (*)(const void*, void*, int)>(kProtoSerializeToArray)(proto, out.data(), static_cast<int>(size));
    return true;
}

void write_all(int fd, const void* data, std::size_t size) {
    const char* pointer = static_cast<const char*>(data);
    while (size) {
        const ssize_t written = write(fd, pointer, size);
        if (written <= 0) _exit(3);
        pointer += written;
        size -= static_cast<std::size_t>(written);
    }
}

void write_blob(int fd, std::uint8_t tag, const std::string& data) {
    const std::uint32_t size = static_cast<std::uint32_t>(data.size());
    write_all(fd, &tag, 1);
    write_all(fd, &size, 4);
    write_all(fd, data.data(), data.size());
}

const unsigned char* instr_desc(unsigned opcode) {
    return load<const unsigned char*>(instr_info, kInstrInfoLastDesc) - kInstrDescSize * opcode;
}
}  // namespace

extern "C" int tec_llvm_open(const char* path) {
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) return 1;
    Dl_info info = {};
    void* exported = dlsym(handle, "GetPjrtApi");
    if (exported == nullptr || dladdr(exported, &info) == 0) return 1;
    base = static_cast<unsigned char*>(info.dli_fbase);
    if (static_cast<unsigned char*>(exported) - base != static_cast<std::ptrdiff_t>(kGetPjrtApi)) return 1;
    // The TPU target ignores the triple's text; the constructors only keep it.
    triple = std::calloc(1, 256);
    at<void (*)(void*, const char*)>(kTripleCtor)(triple, "googletpu");
    void* options = std::calloc(1, 4096);
    at<void (*)(void*)>(kTargetOptionsCtor)(options);
    instr_info = at<void* (*)()>(kCreateInstrInfo)();
    register_info = at<void* (*)(void*)>(kCreateRegisterInfo)(triple);
    void* asm_info = at<void* (*)(void*, void*, void*)>(kAsmInfoAllocator)(register_info, triple, options);
    // The TPU target's processor table names the v6e TEC subtarget sparsecore-tec-v6e.
    static const char cpu[] = "sparsecore-tec-v6e";
    subtarget = at<void* (*)(void*, const char*, std::size_t, const char*, std::size_t)>(kCreateSubtargetInfo)(triple, cpu, sizeof(cpu) - 1, "", 0);
    // Syntax variant 0 is TPUInstPrinter, the SparseCore printer used by the bundle dumps.
    printer = at<void* (*)(void*, unsigned, void*, void*, void*)>(kCreateInstPrinter)(triple, 0, asm_info, instr_info, register_info);
    return 0;
}

extern "C" unsigned tec_llvm_opcode_count() { return load<std::uint32_t>(instr_info, kInstrInfoOpcodeCount); }

extern "C" const char* tec_llvm_opcode_name(unsigned opcode) {
    const auto* indices = load<const std::uint32_t*>(instr_info, kInstrInfoNameIndices);
    return load<const char*>(instr_info, kInstrInfoNameData) + indices[opcode];
}

// Writes (register class, flags, operand type) per operand; returns the count.
extern "C" unsigned tec_llvm_operands(unsigned opcode, std::int32_t* output, unsigned capacity) {
    const unsigned char* desc = instr_desc(opcode);
    const unsigned count = load<std::uint16_t>(desc, kInstrDescOperandCount);
    const unsigned char* table_end = load<const unsigned char*>(instr_info, kInstrInfoLastDesc) + kInstrDescSize;
    const unsigned char* info = table_end + kOperandInfoSize * load<std::uint16_t>(desc, kInstrDescOperandOffset);
    for (unsigned i = 0; i != count && i < capacity; ++i) {
        const unsigned char* entry = info + kOperandInfoSize * i;
        output[3 * i] = load<std::int16_t>(entry, 0);
        output[3 * i + 1] = entry[2];
        output[3 * i + 2] = entry[3];
    }
    return count;
}

extern "C" unsigned tec_llvm_class_count() { return load<std::uint32_t>(register_info, kRegisterInfoClassCount); }

extern "C" unsigned tec_llvm_class_registers(unsigned index, std::uint16_t* output, unsigned capacity) {
    const unsigned char* entry = load<const unsigned char*>(register_info, kRegisterInfoClasses) + kRegisterClassSize * index;
    const auto* registers = reinterpret_cast<const std::uint16_t*>(entry + load<std::int32_t>(entry, 0));
    const unsigned count = load<std::uint16_t>(entry, kRegisterClassCount);
    for (unsigned i = 0; i != count && i < capacity; ++i) output[i] = registers[i];
    return count;
}

extern "C" const char* tec_llvm_register_name(unsigned reg) { return at<const char* (*)(unsigned)>(kRegisterName)(reg); }

// Prints and consumes each record in a forked child that streams a text blob
// (tag 0) and a result blob (tag 1 protobuf, tag 2 message) per record. An
// emitter CHECK failure kills the child; the parent returns what arrived and
// the child's stderr goes to log_path.
extern "C" int tec_llvm_batch(const std::uint64_t* words, std::size_t count, const char* log_path, unsigned char** output, std::size_t* output_size) {
    int fds[2];
    if (pipe(fds) != 0) return -1;
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        const int log = open(log_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        dup2(log, 2);
        std::size_t pos = 0;
        while (pos < count) {
            Bundle bundle;
            pos = parse(words, pos, bundle);
            write_blob(fds[1], 0, print(bundle));
            std::string result;
            const bool accepted = consume(bundle, result);
            write_blob(fds[1], accepted ? 1 : 2, result);
        }
        close(fds[1]);
        _exit(0);
    }
    close(fds[1]);
    std::vector<unsigned char> data;
    unsigned char chunk[1 << 16];
    for (ssize_t n; (n = read(fds[0], chunk, sizeof(chunk))) > 0;) data.insert(data.end(), chunk, chunk + n);
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    *output = static_cast<unsigned char*>(std::malloc(data.size() + 1));
    std::memcpy(*output, data.data(), data.size());
    *output_size = data.size();
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

extern "C" void tec_llvm_free(void* pointer) { std::free(pointer); }

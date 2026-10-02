"""libtpu 中 LLVM TPU MC printer 与 Ghostlite TEC emitter 的 Python 接口，供 generate_tpu_v6e_tec_isa.py 使用。

原生部分见 tec_llvm.cc。函数地址取自对应 TEC 原生后端文件中的常量；emitter 遇到不支持的操作数时可能以 CHECK 失败终止进程，所以每批指令在 fork 出的子进程中处理，子进程终止后从下一条指令继续。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

from tpuasm.backends import NativeBackend

WORKERS = min(64, os.cpu_count() or 1)
# MCOperand kinds used in records.
REGISTER, IMMEDIATE, EXPRESSION = 1, 2, 5

@dataclass(frozen=True)
class McInst:
    """一条 MCInst：slot_flags 是 SparseCoreMCSlot 位，lane_flags 每个操作数 3 位，给出标量寄存器经由的 VS 槽。

    ``operands`` 的每项为 (种类, 值, 附加)。EXPRESSION 表示 TPUMCImmExpr，附加为 slot | encoding << 8 | type << 16：共享立即数槽、selector 编码和立即数类型。
    """
    opcode: int
    slot_flags: int
    operands: tuple[tuple[int, int, int], ...]
    lane_flags: int = 0

@dataclass(frozen=True)
class Result:
    text: str
    bundle: bytes | None
    message: str

class Llvm:
    def __init__(self, backend: NativeBackend, libtpu: Path) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix='tpuasm-tec-llvm-', dir='/tmp')
        directory = Path(self._directory.name)
        source = Path(__file__).with_name('tec_llvm.cc')
        backend_source = (Path(__file__).resolve().parents[1] / 'src' / 'tpuasm' / backend.native_source).read_text()
        constants = re.findall(r'constexpr std::(?:uintptr_t|size_t) k\w+ = 0x[0-9a-f]+;', backend_source)
        header = directory / 'constants.h'
        header.write_text('#include <cstddef>\n#include <cstdint>\nnamespace tpuasm_backend {\n' + '\n'.join(constants) + '\n}\n')
        output = directory / 'tec_llvm.so'
        subprocess.run(['g++', '-std=c++17', '-O2', '-fPIC', '-shared', '-Wall', '-Wextra', '-include', str(header), str(source), '-o', str(output), '-ldl'], check=True)
        self._log = directory / 'child.log'
        native = ctypes.CDLL(str(output))
        native.tec_llvm_opcode_name.restype = ctypes.c_char_p
        native.tec_llvm_register_name.restype = ctypes.c_char_p
        native.tec_llvm_batch.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
        native.tec_llvm_free.argtypes = [ctypes.c_void_p]
        if native.tec_llvm_open(str(libtpu).encode()):
            raise RuntimeError(f'cannot open {libtpu} for the LLVM TPU printer')
        self._native = native
        self.names = [native.tec_llvm_opcode_name(opcode).decode() for opcode in range(native.tec_llvm_opcode_count())]
        buffer = (ctypes.c_uint16 * 4096)()
        self.class_registers = []
        for index in range(native.tec_llvm_class_count()):
            count = native.tec_llvm_class_registers(index, buffer, 4096)
            self.class_registers.append(tuple(buffer[:count]))
        top = max(register for registers in self.class_registers for register in registers)
        self.register_names = {register: native.tec_llvm_register_name(register).decode() for register in range(1, top + 1)}
        self.registers = {name: register for register, name in self.register_names.items()}

    def operands(self, opcode: int) -> list[tuple[int, int, int]]:
        """每个操作数的 (寄存器类或 -1, MCOI 标志, 操作数类型)；标志 2 表示谓词操作数。"""
        buffer = (ctypes.c_int32 * 192)()
        count = self._native.tec_llvm_operands(opcode, buffer, 64)
        return [(buffer[3 * index], buffer[3 * index + 1], buffer[3 * index + 2]) for index in range(count)]

    def run(self, bundles: list[list[McInst]]) -> list[Result]:
        """逐个 bundle 打印并交给 emitter；子进程终止的 bundle 以 CHECK 消息为结果。

        emitter 的 CHECK 失败会先符号化整个调用栈，每次约 0.2 秒，所以把 bundle 分成若干段并行处理。
        """
        size = max(64, -(-len(bundles) // WORKERS))
        chunks = [bundles[start:start + size] for start in range(0, len(bundles), size)]
        with ThreadPoolExecutor(WORKERS) as pool:
            parts = list(pool.map(self._run_chunk, chunks, range(len(chunks))))
        return [result for part in parts for result in part]

    def _run_chunk(self, bundles: list[list[McInst]], worker: int) -> list[Result]:
        log = self._log.with_name(f'child{worker}.log')
        offsets = []
        words: list[int] = []
        for insts in bundles:
            offsets.append(len(words))
            words.append(len(insts))
            for inst in insts:
                words.extend((inst.opcode | inst.slot_flags << 32, inst.lane_flags, len(inst.operands)))
                for kind, value, extra in inst.operands:
                    words.extend((kind, value & (1 << 64) - 1, extra))
        array = (ctypes.c_uint64 * max(len(words), 1))(*words)
        results: list[Result] = []
        start = 0
        while start < len(bundles):
            output, length = ctypes.c_void_p(), ctypes.c_size_t()
            pointer = ctypes.cast(ctypes.byref(array, 8 * offsets[start]), ctypes.c_void_p)
            self._native.tec_llvm_batch(pointer, len(words) - offsets[start], str(log).encode(), ctypes.byref(output), ctypes.byref(length))
            data = ctypes.string_at(output, length.value)
            self._native.tec_llvm_free(output)
            blobs = []
            position = 0
            while position + 5 <= len(data):
                tag, count = data[position], struct.unpack_from('<I', data, position + 1)[0]
                if position + 5 + count > len(data):
                    break
                blobs.append((tag, data[position + 5:position + 5 + count]))
                position += 5 + count
            for index in range(len(blobs) // 2):
                text = blobs[2 * index][1].decode()
                tag, payload = blobs[2 * index + 1]
                results.append(Result(text, payload, '') if tag == 1 else Result(text, None, payload.decode(errors='replace')))
            start += len(blobs) // 2
            if start < len(bundles):
                text = blobs[-1][1].decode() if len(blobs) % 2 else ''
                results.append(Result(text, None, 'CHECK ' + self._check_message(log)))
                start += 1
        return results

    def _check_message(self, log: Path) -> str:
        lines = log.read_text(errors='replace').splitlines() if log.exists() else []
        for line in lines:
            if 'Check failed' in line or 'RET_CHECK' in line:
                return line.split('] ', 1)[-1].strip()
        return lines[0].strip() if lines else 'terminated'

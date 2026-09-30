"""进程内、版本限定的静态编译来源保留。"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import mmap
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
from typing import Iterator
import warnings

from .tc_source_backend import SOURCE_BACKENDS
from .tc_source_lowering import lowering_sources
from .backends import select_backend
from .targets import TPU_V4_TC

_LOCK = threading.RLock()
_state: CompilerSourceMapping | None = None
_retained: list[CompilerSourceMapping] = []
_FLAGS = ('xla_jf_emit_annotations', 'xla_mosaic_enable_llo_source_annotations', 'xla_xprof_register_llo_debug_info')

def _maps() -> list[tuple[int, int, str, int, str]]:
    rows = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        start, end = (int(value, 16) for value in parts[0].split('-'))
        rows.append((start, end, parts[1], int(parts[2], 16), parts[5] if len(parts) == 6 else ''))
    return rows

class CompilerSourceMapping:
    """一次上下文的原生计数；计数代表发射捕获，不保证所有来源均可恢复。"""

    def __init__(self) -> None:
        # Every registered libtpu build has a TPU v4 TC backend; it locates the library.
        backend, path = select_backend(TPU_V4_TC.identifier)
        if backend.build_identifier not in SOURCE_BACKENDS:
            raise RuntimeError(f'compiler source mapping is not supported by {backend.build_identifier}')
        if mmap.PAGESIZE != 4096:
            raise RuntimeError('compiler source mapping requires 4096-byte pages')
        self.backend = SOURCE_BACKENDS[backend.build_identifier]
        self.library = ctypes.CDLL(str(path))
        with path.open('rb') as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as elf:
            phoff = struct.unpack_from('<Q', elf, 32)[0]
            size, count = struct.unpack_from('<HH', elf, 54)
            segments = [struct.unpack_from('<IIQQQQQQ', elf, phoff + index * size) for index in range(count)]
            segments = [segment for segment in segments if segment[0] == 1]
            for address, raw in self.backend.signatures.items():
                segment = next(s for s in segments if s[3] <= address and address + len(raw) <= s[3] + s[5] and s[1] & 1)
                offset = segment[2] + address - segment[3]
                if elf[offset:offset + len(raw)] != raw:
                    raise RuntimeError(f'compiler ELF signature differs at {address:#x}')
        rows = _maps()
        first = segments[0]
        bases = {start - (first[3] & -4096) for start, _, _, offset, name in rows if name == str(path) and offset == (first[2] & -4096)}
        if len(bases) != 1:
            raise RuntimeError('expected one loaded libtpu image')
        self.base = bases.pop()
        for address, raw in self.backend.signatures.items():
            if not any(start <= self.base + address and self.base + address + len(raw) <= end and perms == 'r-xp' for start, end, perms, _, _ in rows):
                raise RuntimeError('expected private RX libtpu text mappings')
            if ctypes.string_at(self.base + address, len(raw)) != raw:
                raise RuntimeError(f'loaded compiler signature differs at {address:#x}')
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        with tempfile.TemporaryDirectory(prefix='tpuasm-source-', dir='/tmp') as directory:
            output = Path(directory) / 'native.so'
            source = Path(__file__).parent / self.backend.native_source
            subprocess.run(['g++', '-std=c++17', '-O2', '-fPIC', '-shared', '-Wall', '-Wextra', str(source), '-o', str(output)], check=True)
            self.native = ctypes.CDLL(str(output))
        self.native.source_configure.argtypes = [ctypes.c_size_t]
        self.native.source_configure.restype = None
        self.native.source_configure(self.base)
        self.native.source_counter.argtypes = [ctypes.c_int]
        self.native.source_counter.restype = ctypes.c_uint64
        self.native.source_flag.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.gateway = self._allocate_gateway()
        try:
            self.replacements = {}
            for index, (site, (raw, symbol, prefix)) in enumerate(self.backend.calls.items()):
                destination = self.gateway + index * 32
                target = ctypes.cast(getattr(self.native, symbol), ctypes.c_void_p).value
                code = prefix + bytes.fromhex('ff2500000000') + struct.pack('<Q', target)
                ctypes.memmove(destination, code, len(code))
                if ctypes.string_at(destination, len(code)) != code:
                    raise RuntimeError('compiler trampoline readback failed')
                # A tail call (jmp rel32) stays a jmp so the hook returns to the original caller.
                opcode = b'\xe9' if raw[0] == 0xe9 else b'\xe8'
                self.replacements[site] = opcode + struct.pack('<i', destination - (self.base + site + 5)) + b'\x90' * (len(raw) - 5)
            self._protect(self.gateway, 5)
        except BaseException:
            self._unmap()
            raise

    def counters(self) -> dict[str, int]:
        return {name: self.native.source_counter(i) for i, name in enumerate(('emitted', 'annotated', 'failures', 'rewrites', 'propagated'))}

    def _flag(self, name: str, value: str | None = None) -> str:
        output = ctypes.create_string_buffer(128)
        if self.native.source_flag(name.encode(), value.encode() if value is not None else None, output, len(output)):
            raise RuntimeError(f'cannot read/set compiler flag {name}')
        return output.value.decode()

    def _allocate_gateway(self) -> int:
        sites = [self.base + site + 5 for site in self.backend.calls]
        candidates = []
        rows = _maps()
        for left, right in zip(rows, rows[1:]):
            low = (max(left[1], *(site - (1 << 31) for site in sites)) + 4095) & -4096
            high = min(right[0] - 4096, *(site + (1 << 31) - 4096 for site in sites)) & -4096
            if low <= high:
                candidates.append(min(max(sites[0] & -4096, low), high))
        for address in sorted(set(candidates), key=lambda p: abs(p - sites[0])):
            pointer = self.libc.mmap(address, 4096, 3, 0x100022, -1, 0)
            if pointer == ctypes.c_void_p(-1).value:
                continue
            if pointer == address:
                return pointer
            self.libc.munmap(pointer, 4096)
        raise RuntimeError('no free rel32 gateway page; enter compiler_source_mapping before TPU initialization')

    def _protect(self, page: int, permissions: int) -> None:
        if self.libc.mprotect(page, 4096, permissions):
            raise OSError(ctypes.get_errno(), 'mprotect failed')

    def _write(self, site: int, raw: bytes) -> None:
        address = self.base + site
        pages = range(address & -4096, (address + len(raw) + 4095) & -4096, 4096)
        changed = []
        try:
            for page in pages:
                self._protect(page, 7)
                changed.append(page)
            ctypes.memmove(address, raw, len(raw))
            if ctypes.string_at(address, len(raw)) != raw:
                raise RuntimeError('compiler call-site readback failed')
        finally:
            for page in reversed(changed):
                self._protect(page, 5)

    def _unmap(self) -> None:
        if self.libc.munmap(self.gateway, 4096):
            raise OSError(ctypes.get_errno(), 'munmap failed')

    def _original(self) -> bool:
        return all(ctypes.string_at(self.base + site, len(raw)) == raw for site, (raw, _, _) in self.backend.calls.items())

    def _install(self) -> None:
        if not self._original():
            raise RuntimeError('compiler call site changed before installation')
        attempted = []
        try:
            for site, replacement in self.replacements.items():
                attempted.append(site)
                self._write(site, replacement)
        except BaseException:
            for site in reversed(attempted):
                self._write(site, self.backend.calls[site][0])
            raise

    def _restore(self) -> None:
        for site, raw in self.replacements.items():
            if ctypes.string_at(self.base + site, len(raw)) != raw:
                raise RuntimeError('compiler call site changed while source mapping was active')
        for site in reversed(self.backend.calls):
            self._write(site, self.backend.calls[site][0])

@contextmanager
def compiler_source_mapping() -> Iterator[CompilerSourceMapping]:
    """在 lowering/compile 期间保留静态来源，支持嵌套与异常恢复。

    这是进程级补丁。锁只协调本接口调用者；上下文期间不得有绕过本接口的并发编译。缓存产物不会重新生成来源；可用 jax.clear_caches() 并禁用持久编译缓存重新编译。
    """
    global _state
    with _LOCK:
        if _retained:
            raise RuntimeError('a previous compiler patch restoration failed; restart the process')
        if _state is not None:
            yield _state
            return
        state = CompilerSourceMapping()
        old_env = os.environ.get('LIBTPU_INIT_ARGS')
        old_flags = {}
        installed = False
        try:
            for name in _FLAGS:
                old_flags[name] = state._flag(name, 'true')
            os.environ['LIBTPU_INIT_ARGS'] = (old_env or '') + ''.join(f' --{name}=true' for name in _FLAGS)
            state._install()
            installed = True
            _state = state
            with lowering_sources():
                yield state
        finally:
            _state = None
            try:
                if installed:
                    state._restore()
                for name, value in old_flags.items():
                    state._flag(name, value)
                if not state._original():
                    raise RuntimeError('compiler patch rollback did not restore original bytes')
                state._unmap()
            except BaseException:
                _retained.append(state)
                raise
            finally:
                if old_env is None:
                    os.environ.pop('LIBTPU_INIT_ARGS', None)
                else:
                    os.environ['LIBTPU_INIT_ARGS'] = old_env
        counts = state.counters()
        if counts['failures']:
            raise RuntimeError(f'native source preservation failed: {counts}')
        if not counts['annotated']:
            warnings.warn('no Pallas source records captured; compilation may have used a cache or contained no source map', RuntimeWarning, stacklevel=2)

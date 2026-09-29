"""把已登记 libtpu 后端的每个 ELF VA 换成“符号 + 偏移”，再到另一个 libtpu 中查同名符号，输出候选地址供人工核对。

用法：PYTHONPATH=src python tools/locate_libtpu_backend.py REFERENCE_LIBTPU REFERENCE_VERSION TARGET_LIBTPU
REFERENCE_LIBTPU 必须是 REFERENCE_VERSION 已登记后端对应的同一个 libtpu.so。不加载 libtpu，不需要 JAX 或 TPU。
"""
from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
from pathlib import Path
import re
import struct
import subprocess

from tpuasm.backends import LIBTPU_RELEASES, _build_id
from tpuasm.tc_source_backend import SOURCE_BACKENDS

PACKAGE = Path(__file__).resolve().parents[1] / 'src' / 'tpuasm'

@dataclass(frozen=True)
class Symbol:
    address: int
    size: int
    name: str

class Library:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = path.read_bytes()
        phoff = struct.unpack_from('<Q', self.data, 32)[0]
        size, count = struct.unpack_from('<HH', self.data, 54)
        headers = [struct.unpack_from('<IIQQQQQQ', self.data, phoff + index * size) for index in range(count)]
        # (VA, file size, file offset) of each PT_LOAD; VA and file offset differ outside the first segment.
        self.loads = [(header[3], header[5], header[2]) for header in headers if header[0] == 1]
        output = subprocess.run(['nm', '--defined-only', '-nS', str(path)], check=True, capture_output=True, text=True).stdout
        self.symbols = []
        for line in output.splitlines():
            parts = line.split(maxsplit=3)
            if len(parts) == 4 and int(parts[1], 16):
                self.symbols.append(Symbol(int(parts[0], 16), int(parts[1], 16), parts[3]))
        self.symbols.sort(key=lambda symbol: symbol.address)
        self.starts = [symbol.address for symbol in self.symbols]
        self.by_name: dict[str, list[Symbol]] = {}
        for symbol in self.symbols:
            self.by_name.setdefault(symbol.name, []).append(symbol)

    def read(self, address: int, size: int) -> bytes:
        for start, length, offset in self.loads:
            if start <= address and address + size <= start + length:
                return self.data[offset + address - start:offset + address - start + size]
        raise ValueError(f'{address:#x} is outside file-backed PT_LOAD segments')

    def containing(self, address: int) -> list[Symbol]:
        """包含该地址的最内层符号；别名（如 C1/C2、D1/D2）一并返回。"""
        index = bisect.bisect_right(self.starts, address)
        found = []
        for symbol in reversed(self.symbols[max(0, index - 64):index]):
            if symbol.address <= address < symbol.address + symbol.size:
                found.append(symbol)
        if not found:
            return []
        innermost = max(symbol.address for symbol in found)
        return [symbol for symbol in found if symbol.address == innermost]

    def call_target(self, site: int) -> int | None:
        raw = self.read(site, 5)
        return site + 5 + struct.unpack('<i', raw[1:])[0] if raw[0] == 0xe8 else None

    def calls_in(self, function: Symbol, target: int) -> list[int]:
        body = self.read(function.address, function.size)
        return [
            function.address + index
            for index in range(len(body) - 4)
            if body[index] == 0xe8 and function.address + index + 5 + struct.unpack_from('<i', body, index + 1)[0] == target
        ]

def demangle(names: set[str]) -> dict[str, str]:
    ordered = sorted(names)
    output = subprocess.run(['c++filt'], input='\n'.join(ordered), check=True, capture_output=True, text=True).stdout.splitlines()
    return dict(zip(ordered, output))

def constants(path: Path) -> dict[str, int]:
    return {match[1]: int(match[2], 16) for match in re.finditer(r'constexpr (?:std::)?uintptr_t (k\w+) = (0x[0-9a-f]+);', path.read_text())}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('reference_libtpu', type=Path)
    parser.add_argument('reference_version')
    parser.add_argument('target_libtpu', type=Path)
    args = parser.parse_args()
    release, = (item for item in LIBTPU_RELEASES if item.version == args.reference_version)
    old, new = Library(args.reference_libtpu), Library(args.target_libtpu)
    if {backend.build_id for backend in release.backends} != {_build_id(old.path)}:
        raise SystemExit('reference libtpu build-id differs from the registered backend')
    rows: list[tuple[str, str, int, list[Symbol], int | None, str]] = []

    def locate(group: str, label: str, address: int, size: int) -> list[tuple[int, Symbol]]:
        """返回新库候选 (VA, 符号)；size 为 0 时不比较字节。"""
        owners = old.containing(address)
        # Aliases such as C1/C2 or D1/D2 share one address.
        candidates = list({symbol.address + address - owner.address: symbol for owner in owners for symbol in new.by_name.get(owner.name, [])}.items())
        status = 'no symbol' if not owners else 'missing in target' if not candidates else ''
        if candidates and size:
            same = [old.read(address, size) == new.read(candidate, size) for candidate, _ in candidates]
            status = 'bytes equal' if all(same) else 'bytes differ' if not any(same) else 'bytes mixed'
            if len(candidates) == 1 and not same[0]:
                status += f'\n{"":12}old {old.read(address, size).hex()}\n{"":12}new {new.read(candidates[0][0], size).hex()}'
        if len(candidates) > 1:
            status += f'; same-name candidates {", ".join(hex(candidate) for candidate, _ in candidates)}'
        rows.append((group, label, address, owners, candidates[0][0] if len(candidates) == 1 else None, status))
        return candidates

    for backend in release.backends:
        for name, address in constants(PACKAGE / backend.native_source).items():
            locate(backend.target.identifier, name, address, 0 if name == 'kEmptyAnnotations' else 16)
    source = next((SOURCE_BACKENDS[backend.build_identifier] for backend in release.backends if backend.build_identifier in SOURCE_BACKENDS), None)
    if source is not None:
        for name, address in constants(PACKAGE / source.native_source).items():
            locate('source', name, address, 16)
        for site, (raw, hook, _) in source.calls.items():
            callee = old.call_target(site)
            assert callee is not None and old.read(site, 5) == raw
            callee_names = {symbol.name for symbol in old.containing(callee)}
            owners = old.containing(site)
            candidates = list({symbol.address + site - owner.address: symbol for owner in owners for symbol in new.by_name.get(owner.name, [])}.items())
            found: set[int] = set()
            for _, function in candidates:
                for target in {symbol.address for name in callee_names for symbol in new.by_name.get(name, [])}:
                    found.update(new.calls_in(function, target))
            same = [candidate for candidate, _ in candidates if candidate in found]
            status = 'call site kept at the same offset' if same else f'call moved; calls to the same callee in target function: {", ".join(hex(item) for item in sorted(found)) or "none"}'
            rows.append(('call', hook, site, owners, same[0] if same else None, status))
        for address, raw in source.signatures.items():
            locate('signature', f'{len(raw)} bytes', address, len(raw))
    names = demangle({symbol.name for row in rows for symbol in row[3]})
    for group, label, address, owners, candidate, status in rows:
        owner = f'{names[owners[0].name]}+{address - owners[0].address:#x}' if owners else '?'
        located = f'{candidate:#x}' if candidate is not None else '?'
        print(f'{group:10} {label:24} {address:#010x} -> {located:>10}  {status}\n{"":12}{owner[:200]}')

if __name__ == '__main__':
    main()

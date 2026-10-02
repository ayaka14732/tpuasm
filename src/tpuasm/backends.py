"""按运行环境选择与特定 libtpu 二进制匹配的原生后端。"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
import mmap
from pathlib import Path
import platform
import struct
import sys

from .targets import HardwareTarget, TPU_V4_TC, TPU_V4_BCS, TPU_V6E_TC, TPU_V6E_TEC, hardware_target

@dataclass(frozen=True)
class RuntimeEnvironment:
    python_implementation: str
    python_version: tuple[int, int]
    python_abiflags: str
    system: str
    machine: str

    @classmethod
    def current(cls) -> RuntimeEnvironment:
        return cls(sys.implementation.name, sys.version_info[:2], getattr(sys, 'abiflags', ''), platform.system(), platform.machine())

    def label(self) -> str:
        major, minor = self.python_version
        return f'{self.python_implementation} {major}.{minor}{self.python_abiflags} on {self.system} {self.machine}'

@dataclass(frozen=True)
class NativeBackend:
    identifier: str
    python_implementation: str
    python_version: tuple[int, int]
    python_abiflags: str
    system: str
    machine: str
    build_id: str
    native_source: str
    target: HardwareTarget

    def accepts(self, environment: RuntimeEnvironment) -> bool:
        return (
            environment.python_implementation == self.python_implementation
            and environment.python_version == self.python_version
            and environment.python_abiflags == self.python_abiflags
            and environment.system == self.system
            and environment.machine == self.machine
        )

    @property
    def build_identifier(self) -> str:
        """不含硬件目标的 libtpu 构建标识；同一构建的各目标后端共用。"""
        return self.identifier.removesuffix('-' + self.target.identifier)

    def environment_label(self) -> str:
        major, minor = self.python_version
        return f'{self.python_implementation} {major}.{minor}{self.python_abiflags} on {self.system} {self.machine}'

@dataclass(frozen=True)
class LibtpuRelease:
    version: str
    backends: tuple[NativeBackend, ...]

LIBTPU_RELEASES = tuple(
    LibtpuRelease(
        version=version,
        backends=tuple(
            NativeBackend(
                identifier=f'libtpu-{version}-cpython-314t-linux-x86_64-{target.identifier}',
                python_implementation='cpython',
                python_version=(3, 14),
                python_abiflags='t',
                system='Linux',
                machine='x86_64',
                build_id=build_id,
                native_source=f'native_backends/libtpu_{stem}_{target.identifier.replace("-", "_")}.cc',
                target=target,
            )
            for target in targets
        ),
    )
    for version, stem, build_id, targets in (
        ('0.0.48', '0_0_48', '3310a7c8c137cd515c7a2ba1ce2ea38c', (TPU_V4_TC, TPU_V4_BCS)),
        ('0.0.49', '0_0_49', '97e27df7268da25ab03e455e30dd86b0', (TPU_V4_TC, TPU_V4_BCS, TPU_V6E_TC, TPU_V6E_TEC)),
    )
)

def _build_id(path: Path) -> str:
    with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if data[:4] != b'\x7fELF' or data[4] != 2 or data[5] != 1:
            raise RuntimeError('libtpu is not an ELF64 little-endian library')
        phoff = struct.unpack_from('<Q', data, 32)[0]
        phsize, phcount = struct.unpack_from('<HH', data, 54)
        for index in range(phcount):
            kind, _, offset, _, _, size, _, _ = struct.unpack_from('<IIQQQQQQ', data, phoff + index * phsize)
            if kind != 4:
                continue
            pos = offset
            while pos < offset + size:
                name_size, desc_size, tag = struct.unpack_from('<III', data, pos)
                pos += 12
                name = data[pos:pos + name_size]
                pos += (name_size + 3) & ~3
                descriptor = data[pos:pos + desc_size]
                pos += (desc_size + 3) & ~3
                if name == b'GNU\0' and tag == 3:
                    return descriptor.hex()
    raise RuntimeError('libtpu has no GNU build-id')

def _release(version: str) -> LibtpuRelease:
    for release in LIBTPU_RELEASES:
        if release.version == version:
            return release
    known = ', '.join(item.version for item in LIBTPU_RELEASES)
    raise RuntimeError(f'unsupported libtpu version {version!r}; registered versions: {known}')

def select_backend(target: str) -> tuple[NativeBackend, Path]:
    """根据已安装 libtpu 与当前 Python/平台选择并验证后端。"""
    hardware = hardware_target(target)
    try:
        package = distribution('libtpu')
    except PackageNotFoundError as exception:
        raise RuntimeError('libtpu is not installed; tpuasm selects a decoder backend at runtime') from exception
    release = _release(package.version)
    if not release.backends:
        raise RuntimeError(f'libtpu {release.version} is registered, but its decoder backend is not implemented yet')
    environment = RuntimeEnvironment.current()
    for_target = tuple(backend for backend in release.backends if backend.target == hardware)
    if not for_target:
        raise RuntimeError(f'libtpu {release.version} has no {target} decoder backend')
    candidates = tuple(backend for backend in for_target if backend.accepts(environment))
    if not candidates:
        supported = ', '.join(backend.environment_label() for backend in for_target)
        raise RuntimeError(f'libtpu {release.version} has no {target} decoder backend for {environment.label()}; supported environments: {supported}')
    path = Path(str(package.locate_file('libtpu/libtpu.so'))).resolve()
    build_id = _build_id(path)
    for backend in candidates:
        if backend.build_id == build_id:
            return backend, path
    expected = ', '.join(backend.build_id for backend in candidates)
    raise RuntimeError(f'installed libtpu {release.version} has build-id {build_id}; matching backend build-id: {expected}')

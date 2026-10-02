"""把修改后的 TensorCore 或 SparseCore TEC 程序映像写回 serialized executable，并按原调用约定装载执行。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import io
from typing import Any, cast

import jax
from jax.experimental import serialize_executable
from jax.stages import Compiled
from jaxlib.xla_client import LoadedExecutable

from ._protobuf import encode_varint, field_spans, replace_fields
from .printer import _verify_image
from .program_container import executable_records, resolve_executable_target
from .targets import TPU_V4_BCS, TPU_V6E_TEC
from .tc_source_mapping import _Program, _one, _programs, _values
from .tc_relocation import BundleInsertion, insert_program

def _span(data: bytes, start: int, limit: int, number: int, what: str) -> tuple[int, int]:
    found = [(first, last) for field, wire, first, last in field_spans(data, start, limit) if field == number and wire == 2]
    if len(found) != 1:
        raise ValueError(f'expected exactly one {what}, found {len(found)}')
    return found[0]

def _identity(old: bytes, content: bytes) -> bytes:
    # The compiler derives these values from its internal program representation; tpuasm only needs new, deterministic values of the same size.
    if len(old) != 32:
        raise ValueError(f'expected a 32-byte program identity, found {len(old)} bytes')
    return hashlib.sha256(old + content).digest()

def replace_executable_programs(serialized: bytes, images: Mapping[tuple[int, int], bytes], *, target: str | None = None) -> bytes:
    """用新的程序映像替换 serialized executable 中的 TC 或 TEC 程序映像，返回新的 executable 字节。

    本函数处理等长替换：新映像必须与原映像的字节数相同。需要增加 bundle 时，使用 :func:`insert_executable_bundles`，显式给出插入点以迁移分支、装载参数和元数据。

    内容有变化的记录会得到新的程序身份：segment set 的 hash 和 core program 的 fingerprint（及其在 sequencer 与 CompilerMetadata 中的副本）改为由原值和新内容导出的 SHA-256。runtime 以这两项识别已装载的程序；不改变它们时，同一进程中已装载过原程序的 runtime 会继续执行原程序。与原映像相同的替换不改变任何字节。

    Args:
        serialized: ``bytes(compiled.runtime_executable().serialize())`` 得到的字节。
        images: 以 :func:`executable_programs` 给出的 ``(record, index)`` 为键的新程序映像，例如 :func:`assemble_listing` 的结果。
        target: ``'tpu-v4-tc'``、``'tpu-v6e-tc'`` 或 ``'tpu-v6e-tec'``；省略时从容器推断，规则同 :func:`executable_programs`，TEC 必须显式指定。BCS 程序在容器中保存为 semantic protobuf，不支持替换。

    Returns:
        新的 serialized executable 字节，长度与输入相同。可用 :func:`load_executable` 装载执行。

    Raises:
        ValueError: 目标是 BCS、键不存在、映像长度不同，或容器缺少程序身份字段。
        RuntimeError: 原生后端不可用，或新映像未通过原生校验。

    Examples:
        修改一条指令后写回::

            from tpuasm import assemble_listing, executable_programs, format_assembly, replace_executable_programs

            serialized = bytes(compiled.runtime_executable().serialize())
            (record, index, image), = executable_programs(serialized)
            source = format_assembly(image, target='tpu-v4-tc').replace('v0, 1.0', 'v0, 0.5')
            patched = replace_executable_programs(serialized, {(record, index): assemble_listing(source)})
    """
    hardware = resolve_executable_target(serialized, target)
    if hardware == TPU_V4_BCS:
        raise ValueError('tpu-v4-bcs programs are stored as semantic protobuf; replacing them is not supported')
    if hardware == TPU_V6E_TEC:
        from .tpu_v6e_tec_program import executable_tec_programs
        found = executable_tec_programs(serialized)
    else:
        found = _programs(serialized)
    programs = {(program.record, program.image_index): program for program in found}
    for key, image in images.items():
        if key not in programs:
            raise ValueError(f'the executable has no program image {key[0]}:{key[1]}')
        program = programs[key]
        if len(image) != len(program.image):
            raise ValueError(f'program image {key[0]}:{key[1]} has {len(program.image)} bytes, the replacement has {len(image)}; only same-size replacement is supported')
        _verify_image(image, target=hardware.identifier)
    return _write_programs(serialized, programs, {key: (image, None) for key, image in images.items() if image != programs[key].image})

def _write_programs(serialized: bytes, programs: Mapping[tuple[int, int], _Program], updates: Mapping[tuple[int, int], tuple[bytes, bytes | None]]) -> bytes:
    if not updates:
        return serialized
    records = executable_records(serialized)
    for record in sorted({key[0] for key in updates}):
        core = records[record]
        sets = _values(core, 8)
        hashes = []
        metadata = None
        metadata_record = None
        for set_index, segment_set in enumerate(sets):
            changes = sorted(
                (programs[key].image_offset, programs[key], image, revised)
                for key, (image, revised) in updates.items()
                if key[0] == record and programs[key].segment_set_index == set_index
            )
            if not changes:
                continue
            data = _one(segment_set, 3, b'')
            segments = _values(segment_set, 2)
            for segment_index, segment in enumerate(segments):
                data_range = _one(segment, 3, b'')
                if not data_range:
                    continue
                offset, size = _one(data_range, 1), _one(data_range, 2)
                old_size = size
                shift = 0
                for at, program, image, revised in changes:
                    delta = len(image) - len(program.image)
                    if segment_index == program.segment_index:
                        # A TEC image is only part of its segment, so the size changes by the difference.
                        size += delta
                    elif delta and offset < at + len(program.image) and at < offset + size:
                        raise ValueError('resized code segment overlaps another initialized segment')
                    if offset >= at + len(program.image):
                        shift += delta
                range_changes: dict[int, Sequence[int | bytes]] = {}
                if shift:
                    range_changes[1] = [offset + shift]
                if size != old_size:
                    range_changes[2] = [size]
                if range_changes:
                    segments[segment_index] = replace_fields(segment, {3: [replace_fields(data_range, range_changes)]})
            cursor = 0
            pieces: list[bytes] = []
            for at, program, image, revised in changes:
                if at < cursor:
                    raise ValueError('replacement code segments overlap')
                pieces.extend((data[cursor:at], image))
                cursor = at + len(program.image)
                if revised is not None:
                    metadata, metadata_record = revised, program.metadata_record
            pieces.append(data[cursor:])
            data = b''.join(pieces)
            hashes.append(_identity(_one(segment_set, 4, b''), data))
            sets[set_index] = replace_fields(segment_set, {2: segments, 3: [data], 4: [hashes[-1]]})
        first, last = _span(core, 0, len(core), 3, 'core program fingerprint')
        fingerprint = core[first:last]
        tensor_core = _one(core, 5, b'')
        sequencer = _one(tensor_core, 1, b'')
        if _one(sequencer, 4, b'') != fingerprint:
            raise ValueError(f'record {record}: fingerprint copies disagree')
        if metadata_record is None:
            metadata_record = next(program.metadata_record for program in programs.values() if program.record == record)
        if metadata_record is not None and _one(records[metadata_record], 26, b'') != fingerprint:
            raise ValueError(f'record {record}: fingerprint copies disagree')
        replacement = _identity(fingerprint, b''.join(hashes) + (metadata or b''))
        sequencer = replace_fields(sequencer, {4: [replacement]})
        records[record] = replace_fields(core, {3: [replacement], 5: [replace_fields(tensor_core, {1: [sequencer]})], 8: sets})
        if metadata_record is not None:
            records[metadata_record] = replace_fields(metadata if metadata is not None else records[metadata_record], {26: [replacement]})
    return b''.join(encode_varint(len(record)) + record for record in records)

def insert_executable_bundles(serialized: bytes, insertions: Mapping[tuple[int, int], Sequence[BundleInsertion]], *, target: str | None = None) -> bytes:
    """在 TensorCore executable 中插入独立 bundle，返回已迁移的 executable。

    插入位置是输入映像的 bundle 编号。自动处理直接分支、装载块数、块对齐、代码 segment、protobuf 长度、overlay、符号范围、注释和程序身份。新增 bundle 不继承原源码归属。原 bundle 除需要迁移的分支与装载指令外保持原编码。

    当前支持一个前缀 overlay 和一个主程序 overlay，在主程序内部插入；拒绝多个代码映像与尾部 continuation 内部插入，也拒绝在原 sbr/scall（包括间接形式）的延迟窗口内插入：v4 为分支后的 1 个 bundle，v6e 为 4 个。片段中每条 sbr/scall 的延迟窗口必须完整落在片段内。调用者负责寄存器与内存资源、片段自身的流水线及延迟槽内容，以及通过寄存器或内存保存的间接跳转地址；本函数不分配资源或重新调度。

    Args:
        serialized: 原 serialized executable 字节。
        insertions: 以 ``(record, index)`` 为键的 :class:`BundleInsertion` 序列。同一映像中的全部位置使用原编号。
        target: 可选的 TensorCore 目标，省略时从容器推断。

    Returns:
        可交给 :func:`load_executable` 的 executable 字节；空编辑保持原字节。

    Raises:
        ValueError: 插入点、汇编片段、元数据或 overlay 布局不支持。
        RuntimeError: 原生编解码或校验失败。
    """
    hardware = resolve_executable_target(serialized, target)
    if hardware in (TPU_V4_BCS, TPU_V6E_TEC):
        raise ValueError('bundle insertion supports TensorCore targets only')
    programs = {(p.record, p.image_index): p for p in _programs(serialized)}
    updates = {}
    for key, edits in insertions.items():
        if key not in programs:
            raise ValueError(f'the executable has no program image {key[0]}:{key[1]}')
        if edits:
            updates[key] = insert_program(programs[key], list(edits), hardware)
    return _write_programs(serialized, programs, updates)

class _Unpickler(serialize_executable._JaxPjrtUnpickler):
    """JAX 的 executable unpickler，只把其中的 serialized executable 换成给定字节。"""

    def __init__(self, file: io.BytesIO, backend: Any, devices: Sequence[Any], serialized: bytes) -> None:
        super().__init__(file, backend, devices)
        self.serialized = serialized
        self.count = 0

    def persistent_load(self, pid: Any) -> Any:
        if pid[0] == 'exec':
            self.count += 1
            pid = ('exec', self.serialized)
        return super().persistent_load(pid)

def load_executable(serialized: bytes, template: Compiled, *, devices: Sequence[jax.Device] | None = None) -> Compiled:
    """按 template 的输入输出结构与分片装载 serialized executable，返回可调用的 Compiled。

    template 提供 executable 以外的全部内容：参数 pytree、aval、分片和输出结构。装载过程与 ``jax.experimental.serialize_executable.deserialize_and_load`` 相同，只是把其中的 executable 换成 serialized。serialized 必须与 template 的调用约定一致，例如由 template 自身的 executable 经 :func:`replace_executable_programs` 得到。

    Args:
        serialized: 要装载的 serialized executable 字节。
        template: 已编译的 JAX 对象。可以是离线编译的结果，此时需要用 devices 指定执行设备。
        devices: 执行设备，顺序与 template 的设备分配一致；省略时使用 ``template.runtime_executable().local_devices()``。

    Returns:
        执行 serialized 中程序的 Compiled 对象。

    Raises:
        ValueError: template 不能序列化，或其中不是恰好一个 executable。
        jax.errors.JaxRuntimeError: runtime 拒绝装载 serialized。

    Examples:
        装载替换了程序映像的 executable 并执行::

            from tpuasm import load_executable

            patched_compiled = load_executable(patched, compiled)
            result = patched_compiled(x)
    """
    if devices is None:
        devices = cast(LoadedExecutable, template.runtime_executable()).local_devices()
    payload, in_tree, out_tree = serialize_executable.serialize(template)
    unpickler = _Unpickler(io.BytesIO(payload), devices[0].client, devices, serialized)
    unloaded, args_info, no_kwargs = unpickler.load()
    if unpickler.count != 1:
        raise ValueError(f'expected exactly one executable in the template, found {unpickler.count}')
    # The remaining steps follow jax.experimental.serialize_executable.deserialize_and_load.
    return Compiled(unloaded.load(), [], in_tree.unflatten(args_info), out_tree, no_kwargs=no_kwargs)

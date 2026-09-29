"""从 TPU executable 的实际 ISA 字节打印带物理槽的 bundle。"""
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import subprocess
import tempfile
from typing import cast

from jax.stages import Compiled
from jaxlib.xla_client import LoadedExecutable

from .assembler import assemble_listing, check_source_mapping_supported, format_assembly, supports_source_mapping
from .backends import NativeBackend, select_backend
from .targets import TARGETS, TPU_V4_BCS
from .program_container import resolve_executable_target, core_program_target

_native_libraries: dict[tuple[str, str], ctypes.CDLL] = {}
_native_directories: dict[tuple[str, str], tempfile.TemporaryDirectory] = {}
_libtpu_libraries: dict[str, ctypes.CDLL] = {}

def _retain_libtpu(path: Path) -> None:
    # A session performs many codec calls; keep libtpu's process resources alive.
    key = str(path)
    if key not in _libtpu_libraries:
        _libtpu_libraries[key] = ctypes.CDLL(key)

def _load_native(backend: NativeBackend) -> ctypes.CDLL:
    target = backend.target
    key = (backend.identifier, target.identifier)
    if key not in _native_libraries:
        directory = tempfile.TemporaryDirectory(prefix='tpuasm-', dir='/tmp')
        output = Path(directory.name) / 'native.so'
        source = Path(__file__).parent / backend.native_source
        # Share hardware constants with C++ instead of copying them per libtpu version.
        header = Path(directory.name) / 'target.h'
        header.write_text(
            '#include <cstddef>\n'
            'namespace tpuasm_target {\n'
            f'constexpr std::size_t kImageBlockSize = {target.image_block_size};\n'
            f'constexpr std::size_t kBundlesPerBlock = {target.bundles_per_block};\n'
            f'constexpr std::size_t kSlotCount = {len(target.slots)};\n'
            '}\n'
        )
        command = ['g++', '-std=c++17', '-O2', '-fPIC', '-shared', '-Wall', '-Wextra', '-include', str(header), str(source), '-o', str(output), '-ldl']
        subprocess.run(command, check=True)
        native = ctypes.CDLL(str(output))
        native.tpuasm_verify.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        native.tpuasm_verify.restype = ctypes.c_int
        native.tpuasm_program_proto.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        native.tpuasm_program_proto.restype = ctypes.c_int
        native.tpuasm_free.argtypes = [ctypes.c_void_p]
        _native_directories[key] = directory
        _native_libraries[key] = native
    return _native_libraries[key]

def _program_proto(data: bytes, *, encode: bool, target: str) -> bytes:
    backend, libtpu = select_backend(target)
    _retain_libtpu(libtpu)
    native = _load_native(backend)
    source = ctypes.create_string_buffer(data)
    output = ctypes.c_void_p()
    output_size = ctypes.c_size_t()
    diagnostic = ctypes.create_string_buffer(4096)
    status = native.tpuasm_program_proto(
        str(libtpu).encode(),
        source,
        len(data),
        int(encode),
        ctypes.byref(output),
        ctypes.byref(output_size),
        diagnostic,
        len(diagnostic),
    )
    if status:
        raise RuntimeError(diagnostic.value.decode('utf-8', 'replace'))
    try:
        return ctypes.string_at(output, output_size.value)
    finally:
        native.tpuasm_free(output)

def executable_programs(serialized: bytes, *, target: str | None = None) -> list[tuple[int, int, bytes]]:
    """按容器顺序提取 serialized executable 中指定目标的程序映像。

    TC 路径只解析容器，不选择原生后端；BCS 路径提取 semantic body 后调用匹配的原生 codec 生成并验证机器映像。不读写文件，数据段不会被当作程序映像。当前容器字段布局的来源和扩展边界见仓库的 ``docs/design/architecture.md``。

    Args:
        serialized: ``bytes(compiled.runtime_executable().serialize())`` 得到的字节。
        target: ``'tpu-v4-tc'``、``'tpu-v4-bcs'`` 或 ``'tpu-v6e-tc'``。省略时从容器的 program oneof / ABI 推断；目标不唯一或缺少证据时要求显式指定。executable 不一定包含 BCS 程序。

    Returns:
        按容器顺序排列的 ``(record, index, image)`` 列表。record 是从零开始的记录号，index 是该记录内从零开始的程序映像索引，image 是程序映像字节。

    Raises:
        ValueError: 目标/容器格式无效、没有对应程序、代码范围超出 initialized data，或遇到不支持的压缩代码。
        RuntimeError: BCS 原生后端不可用或 codec 校验失败。

    Examples:
        从已保存的 executable 提取程序映像::

            from pathlib import Path
            from tpuasm import executable_programs

            serialized = Path('/tmp/my-executable.bin').read_bytes()
            for record, index, image in executable_programs(serialized):
                print(record, index, len(image))
    """
    if resolve_executable_target(serialized, target) == TPU_V4_BCS:
        from .tpu_v4_bcs_program import executable_bcs_programs
        return executable_bcs_programs(serialized)
    from .tc_source_mapping import _programs
    return [(program.record, program.image_index, program.image) for program in _programs(serialized)]

def _verify_image(image: bytes, *, target: str) -> None:
    backend, libtpu = select_backend(target)
    _retain_libtpu(libtpu)
    native = _load_native(backend)
    source = ctypes.create_string_buffer(image)
    diagnostic = ctypes.create_string_buffer(4096)
    status = native.tpuasm_verify(
        str(libtpu).encode(),
        source,
        len(image),
        diagnostic,
        len(diagnostic),
    )
    if status:
        raise RuntimeError(diagnostic.value.decode('utf-8', 'replace'))

def dump_executable(serialized: bytes, output_dir: Path, *, encoding: str = 'exact', sources: bool = True, source_map_json: bool = False, target: str | None = None) -> list[Path]:
    """将 serialized executable 中指定目标的程序映像导出为 .tpuasm 文件。

    所有程序映像完成格式化后才开始创建目录和写入文件。自动创建输出目录及父目录，覆盖同名文件，保留其他文件；文件系统写入失败仍可能留下部分文件。

    Args:
        serialized: ``bytes(compiled.runtime_executable().serialize())`` 得到的字节。
        output_dir: 输出目录，使用 pathlib.Path。
        encoding: ``'exact'`` （默认）通过命名编码约束及重汇编校验保留原字节；``'canonical'`` 按确定规则重新分配共享资源，不保证字节相同。具体保证见 :func:`format_assembly`。
        target: 可省略并从容器推断；缺少或存在多个目标时必须指定。可选 ``'tpu-v4-tc'``、``'tpu-v4-bcs'``、``'tpu-v6e-tc'``；BCS 无编译来源注释。
        sources: 默认 True，将已保存的 TC 来源显示为注释；False 关闭注释。
        source_map_json: 默认 False；True 另外写入每份程序映像对应的``program-<target>-<record>-<index>.sources.json``，不受 sources 开关影响。BCS 不支持该选项，指定 True 会报错。

    Returns:
        与 :func:`executable_programs` 顺序相同的 pathlib.Path 列表，文件名为``program-<target>-<record>-<index>.tpuasm``；target 包含 TPU 代际与执行单元。返回值只含 .tpuasm 路径，不含 JSON 路径。

    Raises:
        ValueError: 容器或来源元数据无效，或程序映像无法按所选 encoding 导出。
        RuntimeError: 原生后端不可用或原生校验失败。
        OSError: 输出目录无法创建或文件写入失败。

    Examples:
        从已保存的 executable 导出所有程序映像::

            from pathlib import Path
            from tpuasm import dump_executable

            serialized = Path('/tmp/my-executable.bin').read_bytes()
            paths = dump_executable(serialized, Path('/tmp/tpuasm-output'))
    """
    from .tc_source_mapping import executable_source_maps
    hardware = resolve_executable_target(serialized, target)
    target = hardware.identifier
    if source_map_json:
        check_source_mapping_supported(hardware)
    programs = executable_programs(serialized, target=target)
    maps = executable_source_maps(serialized) if supports_source_mapping(hardware) and (sources or source_map_json) else [None] * len(programs)
    formatted = []
    for (record, index, image), mapping in zip(programs, maps):
        text = format_assembly(image, encoding=encoding, target=target, source_map=mapping if sources else None)
        formatted.append((record, index, text, mapping))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for record, index, text, mapping in formatted:
        prefix = 'program-' + hardware.identifier
        path = output_dir / f'{prefix}-{record}-{index}.tpuasm'
        path.write_text(text)
        paths.append(path)
        if source_map_json and mapping is not None:
            path.with_suffix('.sources.json').write_text(mapping.to_json())
    return paths

def dump_compiled(compiled: Compiled, output_dir: Path, *, encoding: str = 'exact', sources: bool = True, source_map_json: bool = False, target: str | None = None) -> list[Path]:
    """将已编译的 JAX 对象中指定目标的程序映像导出为 .tpuasm 文件。

    调用 ``bytes(compiled.runtime_executable().serialize())``，然后交给
    :func:`dump_executable`；本函数不替调用者编译 kernel。

    Args:
        compiled: 已编译的 JAX 对象，例如 ``jax.jit(kernel).lower(*example_inputs).compile()`` 的结果。
        output_dir: 输出目录，使用 pathlib.Path；自动创建目录及父目录并覆盖同名文件。
        encoding: ``'exact'`` （默认）保留原机器字节；``'canonical'`` 重新分配共享资源，不保证字节相同。具体保证见 :func:`format_assembly`。
        target: 可省略并从容器推断；缺少或存在多个目标时必须指定。可选 ``'tpu-v4-tc'``、``'tpu-v4-bcs'``、``'tpu-v6e-tc'``；BCS 无编译来源注释。
        sources: 默认 True，将已保存的 TC 来源显示为注释；False 关闭注释。
        source_map_json: 默认 False；True 另外写入 TC .sources.json，不受 sources 开关影响；BCS 明确拒绝 True。

    Returns:
        与 :func:`dump_executable` 相同的 pathlib.Path 列表，按容器顺序仅包含 .tpuasm 路径；文件名对两目标均包含完整 target。

    Raises:
        ValueError: 容器或来源元数据无效，或程序映像无法按所选 encoding 导出。
        RuntimeError: 原生后端不可用或原生校验失败。
        OSError: 输出目录无法创建或文件写入失败；可能已写入部分文件。
    """
    executable = cast(LoadedExecutable, compiled.runtime_executable())
    return dump_executable(bytes(executable.serialize()), output_dir, encoding=encoding, sources=sources, source_map_json=source_map_json, target=target)

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Assemble and disassemble TPU programs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  tpuasm executable.bin --input-format executable --output-dir asm\n'
            '  tpuasm program.bin --input-format image --target tpu-v4-tc --output program.tpuasm\n'
            '  tpuasm program.tpuasm --input-format listing --output program.bin\n'
            '  tpuasm executable.bin --input-format executable --output-format executable --replace 1:0 program-tpu-v4-tc-1-0.tpuasm --output patched.bin'
        ),
    )
    parser.add_argument('input', type=Path, metavar='FILE', help='serialized executable, machine image, .tpuasm listing or BCS protobuf')
    parser.add_argument(
        '--input-format',
        choices=('executable', 'image', 'listing', 'semantic-proto', 'core-program'),
        required=True,
        metavar='FORMAT',
        help='executable, image, listing, semantic-proto or core-program (last two: BCS)',
    )
    parser.add_argument(
        '--output-format',
        choices=('image', 'listing', 'semantic-proto', 'executable'),
        metavar='FORMAT',
        help='image, listing, semantic-proto (BCS) or executable (with --replace/--insert); default: image for listing input, otherwise listing',
    )
    parser.add_argument(
        '--target',
        choices=tuple(TARGETS),
        metavar='TARGET',
        help='tpu-v4-tc, tpu-v4-bcs or tpu-v6e-tc; required for image and semantic-proto, otherwise inferred when unambiguous',
    )
    parser.add_argument('--encoding', choices=('exact', 'canonical'), metavar='MODE', help='binary-to-listing export: exact preserves bytes (default); canonical for editing')
    parser.add_argument('--output-dir', type=Path, metavar='DIR', help='required for executable input with listing output; one .tpuasm per program')
    parser.add_argument('--output', type=Path, metavar='FILE', help='required for single-program input and executable output')
    parser.add_argument('--no-sources', action='store_true', help='omit executable source annotations')
    parser.add_argument('--source-map-json', action='store_true', help='export source maps (TC executable only)')
    parser.add_argument(
        '--replace',
        nargs=2,
        action='append',
        metavar=('RECORD:INDEX', 'LISTING'),
        help='executable output: replace TC program image RECORD:INDEX (as in program-<target>-<record>-<index>.tpuasm) with the assembled LISTING; same size only; repeatable',
    )
    parser.add_argument('--insert', nargs=2, action='append', metavar=('RECORD:INDEX:PC', 'LISTING'), help='executable output: insert a .target assembly fragment before the original image bundle PC (decimal or 0x hex); repeatable')
    parser.add_argument('--insert-branch-target', choices=('inserted', 'original'), default='inserted', help='direct branches targeting an insertion execute the inserted fragment (default) or skip to the original bundle')
    args = parser.parse_args()
    target = args.target
    output_format = args.output_format or ('image' if args.input_format == 'listing' else 'listing')
    if args.encoding is not None and (args.input_format == 'listing' or output_format != 'listing'):
        parser.error('--encoding requires binary input and listing output')
    if (args.no_sources or args.source_map_json) and args.input_format != 'executable':
        parser.error('source options require executable input')
    if (args.replace is not None or args.insert is not None) != (output_format == 'executable'):
        parser.error('--replace/--insert require executable output and vice versa')
    if args.insert_branch_target != 'inserted' and args.insert is None:
        parser.error('--insert-branch-target requires --insert')
    if output_format == 'executable':
        if args.input_format != 'executable' or args.output is None or args.output_dir is not None or args.no_sources or args.source_map_json:
            parser.error('executable output requires executable input, --replace/--insert and --output')
        from .assembly_syntax import parse_assembly
        from .executable_replacement import BundleInsertion, insert_executable_bundles, replace_executable_programs
        serialized = args.input.read_bytes()
        hardware = resolve_executable_target(serialized, target)
        images = {}
        for key, path in args.replace or ():
            record, separator, index = key.partition(':')
            if not separator or not record.isdigit() or not index.isdigit():
                parser.error(f'--replace expects RECORD:INDEX, got {key!r}')
            source = Path(path).read_text(encoding='utf-8')
            if parse_assembly(source, filename=path).hardware != hardware:
                parser.error(f'{path} does not declare .target {hardware.identifier}')
            images[int(record), int(index)] = assemble_listing(source, filename=path)
        serialized = replace_executable_programs(serialized, images, target=hardware.identifier)
        insertions: dict[tuple[int, int], list[BundleInsertion]] = {}
        for key, path in args.insert or ():
            parts = key.split(':')
            try:
                if len(parts) != 3:
                    raise ValueError('expected three coordinates')
                record_number, image_index = int(parts[0]), int(parts[1])
                pc = int(parts[2], 16 if parts[2].startswith('0x') else 10)
            except ValueError:
                parser.error(f'--insert expects RECORD:INDEX:PC, got {key!r}')
            edit = BundleInsertion(pc, Path(path).read_text(encoding='utf-8'), args.insert_branch_target)
            insertions.setdefault((record_number, image_index), []).append(edit)
        if insertions:
            serialized = insert_executable_bundles(serialized, insertions, target=hardware.identifier)
        args.output.write_bytes(serialized)
        print(args.output)
        return
    if args.input_format == 'executable':
        if args.output_dir is None or args.output is not None or output_format != 'listing':
            parser.error('executable input requires --output-dir and listing output')
        if target == TPU_V4_BCS.identifier and args.source_map_json:
            parser.error('tpu-v4-bcs compiler source mapping is not supported')
        for path in dump_executable(args.input.read_bytes(), args.output_dir, encoding=args.encoding or 'exact', sources=not args.no_sources, source_map_json=args.source_map_json, target=target):
            print(path)
        return
    if args.output is None or args.output_dir is not None:
        parser.error('single-program input requires --output')
    from .tpu_v4_bcs_program import encode_tpu_v4_bcs_program, decode_tpu_v4_bcs_program, extract_tpu_v4_bcs_program
    if args.input_format == 'listing':
        from .assembly_syntax import parse_assembly
        source = args.input.read_text(encoding='utf-8')
        declared = parse_assembly(source, filename=str(args.input)).hardware.identifier
        if args.target is not None and args.target != declared:
            parser.error('--target disagrees with the listing .target')
        target = declared
        if output_format == 'listing':
            parser.error('listing input requires image or semantic-proto output')
        image = assemble_listing(source, filename=str(args.input))
    elif args.input_format == 'image':
        if target is None:
            parser.error('raw image input requires --target')
        if output_format == 'image':
            parser.error('image input requires listing or semantic-proto output')
        image = args.input.read_bytes()
    else:
        proto = args.input.read_bytes()
        if args.input_format == 'core-program':
            inferred = core_program_target(proto)
            if target is None and inferred is not None:
                target = inferred.identifier
            elif inferred is not None and target != inferred.identifier:
                parser.error('--target disagrees with the core-program target')
        if target != TPU_V4_BCS.identifier:
            parser.error('BCS protobuf input needs target tpu-v4-bcs, explicitly or from core-program metadata')
        image = encode_tpu_v4_bcs_program(extract_tpu_v4_bcs_program(proto) if args.input_format == 'core-program' else proto)
    assert target is not None
    if output_format == 'listing':
        args.output.write_text(format_assembly(image, encoding=args.encoding or 'exact', target=target), encoding='utf-8')
    elif output_format == 'semantic-proto':
        if target != TPU_V4_BCS.identifier:
            parser.error('semantic-proto output requires BCS input')
        args.output.write_bytes(decode_tpu_v4_bcs_program(image))
    else:
        args.output.write_bytes(image)
    print(args.output)

if __name__ == '__main__':
    main()

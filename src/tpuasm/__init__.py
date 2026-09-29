"""TPU 指令包的汇编器与反汇编器。"""

from .printer import dump_compiled, dump_executable, executable_programs
from .assembler import assemble_listing, format_assembly
from .assembly_syntax import AssemblyBundle, AssemblyInstruction, AssemblyLocation, AssemblyProgram, EncodingConstraint, parse_assembly
from .targets import HardwareTarget
from .executable_replacement import BundleInsertion, insert_executable_bundles, load_executable, replace_executable_programs
from .tpu_v4_bcs_program import encode_tpu_v4_bcs_program, decode_tpu_v4_bcs_program, extract_tpu_v4_bcs_program
from .tc_compiler import CompilerSourceMapping, compiler_source_mapping
from .tc_source_mapping import (
    BundleAnnotation,
    FunctionSource,
    InstructionOrigin,
    Overlay,
    ProgramSourceMap,
    SlotSource,
    SourceFrame,
    SourceLocation,
    SourceRange,
    executable_source_maps,
    source_maps_json,
)

__all__ = [
    'assemble_listing',
    'encode_tpu_v4_bcs_program',
    'decode_tpu_v4_bcs_program',
    'extract_tpu_v4_bcs_program',
    'dump_compiled',
    'dump_executable',
    'executable_programs',
    'replace_executable_programs',
    'BundleInsertion',
    'insert_executable_bundles',
    'load_executable',
    'format_assembly',
    'parse_assembly',
    'AssemblyProgram',
    'AssemblyBundle',
    'AssemblyInstruction',
    'EncodingConstraint',
    'AssemblyLocation',
    'HardwareTarget',
    'compiler_source_mapping',
    'CompilerSourceMapping',
    'executable_source_maps',
    'source_maps_json',
    'BundleAnnotation',
    'FunctionSource',
    'InstructionOrigin',
    'Overlay',
    'ProgramSourceMap',
    'SlotSource',
    'SourceFrame',
    'SourceLocation',
    'SourceRange',
]

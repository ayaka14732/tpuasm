"""TPU v4 TC 汇编、编码核对与可逆源码导出。"""
from __future__ import annotations

from .assembly_export import decoded_labels, export_program, validate_encoding
from .assembly_model import Bits, Form as AnyForm
from .tpu_v4_tc_codec import EMPTY_WORD, decode_program, decoded_signature, encode_program
from .tpu_v4_tc_constraints import GLOBAL_FIELDS, bind_constraint, export_constraints
from .tpu_v4_tc_isa import SIGNATURES_BY_MNEMONIC
from .tpu_v4_tc_model import TcForm as Form
from .tc_solver import BundleSolver, TcIsa
from .assembly_syntax import AssemblyBundle, AssemblyInstruction, AssemblyProgram, EncodingConstraint, AssemblyLocation
from .printer import _verify_image
from .targets import TPU_V4_TC
from .tc_source_mapping import ProgramSourceMap

def _bind(pin: EncodingConstraint, form: AnyForm | None) -> Bits:
    assert form is None or isinstance(form, Form)
    return bind_constraint(pin, form)

def _writable(form: AnyForm) -> int:
    assert isinstance(form, Form)
    return form.writable_mask

ISA = TcIsa(TPU_V4_TC, 51, EMPTY_WORD, 0, GLOBAL_FIELDS, _bind, SIGNATURES_BY_MNEMONIC, _writable, lambda form: ())

def _verify(solved: list[tuple[int, tuple[AnyForm, ...]]], image: bytes) -> None:
    decoded = decode_program(image)
    if len(decoded) != len(solved):
        raise RuntimeError('encoded bundle count differs from source')
    for pc, ((word, forms), (observed, actual)) in enumerate(zip(solved, decoded)):
        if word != observed or {form.slot for form in forms} != {form.slot for form, pred in actual}:
            raise ValueError(f'bundle {pc:#x}: backend encoding differs from solved hardware fields')
    # The native verifier independently checks byte roundtrip and every occupied slot.
    _verify_image(image, target=TPU_V4_TC.identifier)

def assemble_program(program: AssemblyProgram) -> bytes:
    solved = [BundleSolver(ISA, bundle, program, pc).solve() for pc, bundle in enumerate(program.bundles)]
    image = encode_program(solved)
    _verify(solved, image)
    return image

def _decoded_source(image: bytes) -> tuple[AssemblyProgram, list[tuple[int, tuple[Form, ...]]]]:
    decoded = decode_program(image)
    bundles = []
    solved = []
    for pc, (word, forms) in enumerate(decoded):
        instructions = []
        location = AssemblyLocation('<decoded>', pc + 2, 1)
        for form, predicate in forms:
            signature, operands = decoded_signature(form, word)
            if predicate == 31:
                raise ValueError(f'bundle {pc:#x}: invisible NEVER instruction cannot be exported')
            instructions.append(AssemblyInstruction(form.slot, signature.mnemonic, operands, predicate, location))
        bundles.append(AssemblyBundle(tuple(instructions), (), location))
        solved.append((word, tuple(form for form, _ in forms)))
    return AssemblyProgram(tuple(bundles), decoded_labels(bundles), TPU_V4_TC), solved

def format_program(image: bytes, *, encoding: str, source_map: ProgramSourceMap | None = None) -> str:
    validate_encoding(encoding)
    # Keep the established native checks before exporting any source.
    _verify_image(image, target=TPU_V4_TC.identifier)
    program, solved = _decoded_source(image)

    def solve(bundle: AssemblyBundle, pc: int, pins: tuple[EncodingConstraint, ...]) -> int:
        return BundleSolver(ISA, bundle, program, pc).solve(pins)[0]

    return export_program(image, program, solved, encoding=encoding, solve=solve, list_constraints=export_constraints, source_map=source_map)

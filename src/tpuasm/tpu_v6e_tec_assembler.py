"""TPU v6e TEC 汇编、编码核对与可逆源码导出。"""
from __future__ import annotations

from .assembly_export import decoded_labels, export_program, validate_encoding
from .assembly_model import Bits, Form
from .assembly_syntax import AssemblyBundle, AssemblyInstruction, AssemblyLocation, AssemblyProgram, EncodingConstraint
from .printer import _verify_image
from .targets import TPU_V6E_TEC
from .tc_solver import BundleSolver, TcIsa
from .tpu_v6e_tec_codec import WORD_BYTES, decode_program, decoded_signature, encode_program
from .tpu_v6e_tec_constraints import GLOBAL_FIELDS, bind_constraint, export_constraints
from .tpu_v6e_tec_isa import SIGNATURES_BY_MNEMONIC
from .tpu_v6e_tec_isa_data import EMPTY_WORD
from .tpu_v6e_tec_model import SHARED_MASK, TecForm

def _bind(pin: EncodingConstraint, form: Form | None) -> Bits:
    assert form is None or isinstance(form, TecForm)
    return bind_constraint(pin, form)

def _writable(form: Form) -> int:
    assert isinstance(form, TecForm)
    return form.writable_mask

def _excludes(form: Form) -> tuple[str, ...]:
    assert isinstance(form, TecForm)
    return form.excludes

ISA = TcIsa(TPU_V6E_TEC, WORD_BYTES, EMPTY_WORD, SHARED_MASK, GLOBAL_FIELDS, _bind, SIGNATURES_BY_MNEMONIC, _writable, _excludes, prefer_low_lanes=True)

def assemble_program(program: AssemblyProgram) -> bytes:
    solved = [BundleSolver(ISA, bundle, program, pc).solve() for pc, bundle in enumerate(program.bundles)]
    image = encode_program(solved)
    # A predicate the encoder turns into NEVER drops the slot; reject such silent changes.
    for pc, ((_, forms), (_, actual)) in enumerate(zip(solved, decode_program(image))):
        if {form.slot for form in forms} != {form.slot for form, _ in actual}:
            raise program.bundles[pc].location.error('encoding changes the occupied slot set', pc)
    _verify_image(image, target=TPU_V6E_TEC.identifier)
    return image

def _decoded_source(image: bytes) -> tuple[AssemblyProgram, list[tuple[int, tuple[Form, ...]]]]:
    decoded = decode_program(image)
    bundles = []
    solved: list[tuple[int, tuple[Form, ...]]] = []
    for pc, (word, forms) in enumerate(decoded):
        instructions = []
        location = AssemblyLocation('<decoded>', pc + 2, 1)
        for form, predicate in forms:
            signature, operands = decoded_signature(form, word)
            instructions.append(AssemblyInstruction(form.slot, signature.mnemonic, operands, predicate, location))
        bundles.append(AssemblyBundle(tuple(instructions), (), location))
        solved.append((word, tuple(form for form, _ in forms)))
    return AssemblyProgram(tuple(bundles), decoded_labels(bundles), TPU_V6E_TEC), solved

def format_program(image: bytes, *, encoding: str) -> str:
    validate_encoding(encoding)
    program, solved = _decoded_source(image)
    _verify_image(image, target=TPU_V6E_TEC.identifier)

    def solve(bundle: AssemblyBundle, pc: int, pins: tuple[EncodingConstraint, ...]) -> int:
        return BundleSolver(ISA, bundle, program, pc).solve(pins)[0]

    def list_constraints(forms: tuple[Form, ...], word: int) -> dict[str, str]:
        return export_constraints(tuple(form for form in forms if isinstance(form, TecForm)), word)

    return export_program(image, program, solved, encoding=encoding, solve=solve, list_constraints=list_constraints)

"""BCS 的 bundle 资源求解、原生编码交叉检查和可逆源码导出。"""
from __future__ import annotations

from itertools import product

from ._protobuf import fields
from ._protobuf import message
from .assembly_export import decoded_labels, export_program, validate_encoding
from .assembly_model import Bits, combine
from .assembly_operands import decode_operand, encode_operand
from .assembly_operands import ordered_operands
from .assembly_syntax import AssemblyBundle, AssemblyInstruction, AssemblyProgram, EncodingConstraint, AssemblyLocation, integer
from .tpu_v4_bcs_isa import BcsForm, EMPTY_WORD, FORMS_BY_BRANCH, IMMEDIATES, SIGNATURES_BY_BRANCH, SIGNATURES_BY_MNEMONIC
from .printer import _program_proto, _verify_image
from .targets import TPU_V4_BCS

TARGET = TPU_V4_BCS.identifier

def _instruction_bits(instruction: AssemblyInstruction, program: AssemblyProgram, pc: int) -> tuple[BcsForm, list[Bits]]:
    signature = SIGNATURES_BY_MNEMONIC.get((instruction.slot, instruction.mnemonic))
    if signature is None:
        raise instruction.location.error(f'unsupported BCS slot/mnemonic {instruction.mnemonic!r}', pc, instruction.slot)
    form = signature.form
    assert isinstance(form, BcsForm)
    operands = instruction.operands
    if instruction.mnemonic in ('sbr.rel', 'sbr.abs', 'scall.rel', 'scall.abs'):
        index = 1 if instruction.mnemonic.startswith('scall') else 0
        if len(operands) > index:
            value = program.target(operands[index], pc, instruction.mnemonic.endswith('.rel'), instruction.location)
            operands = operands[:index] + (str(value),) + operands[index + 1:]
    try:
        ordered = ordered_operands(signature, operands)
        if len(ordered) != len(signature.operands):
            raise ValueError(f'expected {len(signature.operands)} operands, got {len(ordered)}')
        candidates = [form.fixed(instruction.predicate)]
        for expression, text in zip(signature.operands, ordered):
            candidates = combine(candidates, encode_operand(expression, text, form))
        if not candidates:
            raise ValueError('operands do not match their types/ranges or require conflicting hardware fields')
        return form, candidates
    except ValueError as error:
        raise instruction.location.error(str(error), pc, instruction.slot) from error

def _constraint_fields(forms: tuple[BcsForm, ...]) -> dict[str, tuple[BcsForm | None, str]]:
    registry: dict[str, tuple[BcsForm | None, str]] = {name: (None, name) for name in IMMEDIATES}
    for form in forms:
        registry.update((f'{form.slot}.{name}', (form, name)) for name in form.fields if name not in IMMEDIATES)
    return registry

def _solve(bundle: AssemblyBundle, program: AssemblyProgram, pc: int, pins: tuple[EncodingConstraint, ...] | None = None) -> tuple[int, tuple[BcsForm, ...]]:
    domains = [_instruction_bits(instruction, program, pc) for instruction in sorted(bundle.instructions, key=lambda inst: inst.slot)]
    forms = tuple(form for form, candidates in domains)
    registry = _constraint_fields(forms)
    bound = Bits()
    for pin in bundle.constraints if pins is None else pins:
        if pin.name not in registry:
            raise pin.location.error(f'unknown BCS encoding constraint {pin.name!r}', pc)
        form, name = registry[pin.name]
        field = IMMEDIATES[name] if form is None else form.fields[name]
        try:
            merged = bound.merge(field.bind(integer(pin.value)))
        except ValueError as error:
            raise pin.location.error(str(error), pc) from error
        if merged is None:
            raise pin.location.error('conflicting BCS encoding constraints', pc)
        bound = merged
    best: tuple[int, bytes] | None = None
    word = 0
    for choices in product(*(candidates for form, candidates in domains)):
        merged = bound
        for candidate in choices:
            combined = merged.merge(candidate)
            if combined is None:
                break
            merged = combined
        else:
            candidate_word = (EMPTY_WORD & ~merged.mask) | merged.value
            key = (merged.immediates.bit_count(), candidate_word.to_bytes(32, 'little'))
            if best is None or key < best:
                best, word = key, candidate_word
    if best is None:
        raise bundle.location.error('BCS slots and encoding constraints require conflicting hardware fields', pc)
    return word, forms

def _encode(solved: list[tuple[int, tuple[BcsForm, ...]]]) -> bytes:
    bundles = []
    for word, forms in solved:
        slots = []
        for form in forms:
            payload = message([(field.number, field.read(word)) for field in form.fields.values()])
            slot = message([(1, form.predicate_field.read(word)), (form.branch, payload)])
            slots.append((TPU_V4_BCS.slots.index(form.slot) + 1, slot))
        bundles.append((1, message(slots)))
    image = _program_proto(message(bundles), encode=True, target=TARGET)
    expected = b''.join(word.to_bytes(32, 'little') for word, forms in solved)
    if image != expected:
        raise ValueError('BCS native encoder changed the solved hardware fields or bundle count')
    _verify_image(image, target=TARGET)
    return image

def assemble_program(program: AssemblyProgram) -> bytes:
    solved = [_solve(bundle, program, pc) for pc, bundle in enumerate(program.bundles)]
    image = _encode(solved)
    decoded, _ = _decode(image)
    # Some machine encodings suppress an occupied slot (DMA owns overlapping
    # S1 bits, and NEVER removes an instruction). Reject such silent changes.
    for pc, (requested, actual) in enumerate(zip(program.bundles, decoded.bundles)):
        if {inst.slot for inst in requested.instructions} != {inst.slot for inst in actual.instructions}:
            raise requested.location.error('BCS encoding changes the occupied slot set', pc)
    return image

def _decode(image: bytes) -> tuple[AssemblyProgram, list[tuple[int, tuple[BcsForm, ...]]]]:
    serialized = _program_proto(image, encode=False, target=TARGET)
    bundles = []
    solved = []
    program_fields = fields(serialized)
    for pc, (number, wire, bundle) in enumerate(program_fields):
        if number != 1 or wire != 2 or not isinstance(bundle, bytes):
            raise ValueError('unsupported BCS program protobuf structure')
        word = int.from_bytes(image[pc * 32:(pc + 1) * 32], 'little')
        instructions = []
        forms = []
        location = AssemblyLocation('<decoded>', pc + 2, 1)
        for slot_number, slot_wire, slot_bytes in fields(bundle):
            if slot_number not in (1, 2) or slot_wire != 2 or not isinstance(slot_bytes, bytes):
                raise ValueError(f'unsupported BCS slot at bundle {pc:#x}')
            slot = TPU_V4_BCS.slots[slot_number - 1]
            operation = [(key, data) for key, kind, data in fields(slot_bytes) if key >= 5 and kind == 2]
            if len(operation) != 1:
                raise ValueError(f'unsupported BCS instruction structure at bundle {pc:#x}')
            branch, payload = operation[0]
            form = FORMS_BY_BRANCH.get((slot, branch))
            if form is None or not isinstance(payload, bytes):
                raise ValueError(f'unsupported BCS instruction {slot}.{branch} at bundle {pc:#x}')
            for key, kind, value in fields(payload):
                registered = next((field for field in form.fields.values() if field.number == key), None)
                if kind != 0 or not isinstance(value, int) or registered is None:
                    raise ValueError(f'unsupported BCS operand field at bundle {pc:#x}')
                if (value & ((1 << registered.width) - 1)) != registered.read(word):
                    raise ValueError(f'BCS decoded operand differs from hardware field at bundle {pc:#x}')
            signature = SIGNATURES_BY_BRANCH[(slot, branch)]
            operands = tuple(decode_operand(expression, form, word) for expression in signature.operands)
            if signature.keywords:
                operands = tuple(f'{name}={value}' for name, value in zip(signature.keywords, operands))
            predicate = form.predicate_field.read(word)
            if predicate == 31:
                raise ValueError(f'bundle {pc:#x}: invisible NEVER instruction cannot be exported')
            instructions.append(AssemblyInstruction(slot, signature.mnemonic, operands, predicate, location))
            forms.append(form)
        bundles.append(AssemblyBundle(tuple(instructions), (), location))
        solved.append((word, tuple(forms)))
    return AssemblyProgram(tuple(bundles), decoded_labels(bundles), TPU_V4_BCS), solved

def format_program(image: bytes, *, encoding: str) -> str:
    validate_encoding(encoding)
    _verify_image(image, target=TARGET)
    program, solved = _decode(image)

    def solve(bundle: AssemblyBundle, pc: int, pins: tuple[EncodingConstraint, ...]) -> int:
        return _solve(bundle, program, pc, pins)[0]

    def list_constraints(forms: tuple[BcsForm, ...], word: int) -> dict[str, str]:
        registry = _constraint_fields(forms)
        return {
            name: str((IMMEDIATES[field] if form is None else form.fields[field]).read(word))
            for name, (form, field) in sorted(registry.items())
        }

    def validate_canonical(rebuilt: bytes) -> None:
        restored, _ = _decode(rebuilt)
        # Compare semantic operands, independent of source locations and labels.
        original = [[(inst.slot, inst.mnemonic, inst.operands, inst.predicate) for inst in bundle.instructions] for bundle in program.bundles]
        actual = [[(inst.slot, inst.mnemonic, inst.operands, inst.predicate) for inst in bundle.instructions] for bundle in restored.bundles]
        if original != actual:
            raise ValueError('BCS canonical encoding changes visible operands')

    return export_program(image, program, solved, encoding=encoding, solve=solve, list_constraints=list_constraints, validate_canonical=validate_canonical)

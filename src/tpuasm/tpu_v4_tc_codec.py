"""通过内部结构化后端交换已求解的硬件字段；不公开 protobuf 语法。"""
from __future__ import annotations

from ._protobuf import fields, message
from .assembly_model import Bits, Form as AnyForm, Signature
from .tpu_v4_tc_model import FORMS_BY_BRANCH, TcForm as Form
from .assembly_operands import decode_operand
from .tpu_v4_tc_isa import SIGNATURES_BY_BRANCH
from .tpu_v4_tc_isa_data import OPCODES
from .printer import _program_proto
from .targets import TPU_V4_TC

EMPTY_WORD = sum(31 << (start + width) for start, width in OPCODES.values())

def decode_program(image: bytes) -> list[tuple[int, tuple[tuple[Form, int], ...]]]:
    serialized = _program_proto(image, encode=False, target=TPU_V4_TC.identifier)
    program: list[tuple[int, tuple[tuple[Form, int], ...]]] = []
    for number, wire, bundle in fields(serialized):
        if number != 1 or wire != 2 or not isinstance(bundle, bytes):
            raise ValueError('unsupported TC program structure')
        bits = Bits()
        forms = []
        for slot_number, slot_wire, slot_bytes in fields(bundle):
            if slot_wire != 2 or not isinstance(slot_bytes, bytes) or not 1 <= slot_number <= len(TPU_V4_TC.slots):
                raise ValueError('unsupported TC bundle field')
            slot = TPU_V4_TC.slots[slot_number - 1]
            slot_fields = fields(slot_bytes)
            pred = next(value for key, kind, value in slot_fields if key == 1)
            if not isinstance(pred, int):
                raise ValueError('invalid predicate field')
            operations = [(key, value) for key, kind, value in slot_fields if key >= 5 and kind == 2]
            if len(operations) != 1 or not isinstance(operations[0][1], bytes):
                raise ValueError('unsupported instruction structure')
            branch, payload = operations[0]
            assert isinstance(payload, bytes)
            form = FORMS_BY_BRANCH.get((slot, branch))
            if form is None:
                raise ValueError(f'unsupported instruction form in {slot}: {branch}')
            decoded = {key: value for key, kind, value in fields(payload) if kind == 0 and isinstance(value, int)}
            bound = form.fixed(pred)
            for field in form.fields.values():
                if field.number in decoded:
                    value = decoded[field.number] & ((1 << field.width) - 1)
                    merged = bound.merge(field.bind(value))
                    if merged is None:
                        raise ValueError(f'inconsistent decoded fields in {slot}')
                    bound = merged
            merged = bits.merge(bound)
            if merged is None:
                raise ValueError(f'inconsistent decoded shared fields at bundle {len(program):#x}')
            bits = merged
            forms.append((form, pred))
        program.append(((EMPTY_WORD & ~bits.mask) | bits.value, tuple(forms)))
    return program

def encode_program(bundles: list[tuple[int, tuple[AnyForm, ...]]]) -> bytes:
    serialized = []
    for word, forms in bundles:
        slots = []
        for form in sorted(forms, key=lambda item: TPU_V4_TC.slots.index(item.slot)):
            assert isinstance(form, Form)
            payload = message([(field.number, field.read(word)) for field in form.fields.values()])
            slot = message([(1, form.predicate_field.read(word)), (form.branch, payload)])
            slots.append((TPU_V4_TC.slots.index(form.slot) + 1, slot))
        serialized.append((1, message(slots)))
    return _program_proto(message(serialized), encode=True, target=TPU_V4_TC.identifier)

def decoded_signature(form: Form, word: int) -> tuple[Signature, tuple[str, ...]]:
    for signature in SIGNATURES_BY_BRANCH[(form.slot, form.branch)]:
        if any(form.fields[name].read(word) != value for name, value in signature.fixed_fields):
            continue
        try:
            operands = tuple(decode_operand(operand, form, word) for operand in signature.operands)
        except ValueError:
            continue
        if signature.keywords:
            count = len(signature.keywords)
            operands = operands[:-count] + tuple(f'{key}={value}' for key, value in zip(signature.keywords, operands[-count:]))
        return signature, operands
    raise ValueError(f'no supported signature for {form.slot} {form.name}')

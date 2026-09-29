"""通过原生后端交换 TPU v6e TC 的已求解硬件字段；不公开 protobuf 语法。

v6e 的 bundle 在程序映像中按 64 字节连续排列，没有块内分隔字节，所以机器字直接取自映像字节。原生解码得到的 protobuf 只用来识别每个槽的形式，并逐字段核对与机器字一致。
"""
from __future__ import annotations

from ._protobuf import fields, message
from .assembly_model import Form, Signature
from .assembly_operands import decode_operand
from .printer import _program_proto
from .targets import TPU_V6E_TC
from .tpu_v6e_tc_isa import SIGNATURES_BY_BRANCH
from .tpu_v6e_tc_isa_data import FORMATTER_ABORTS, SHARED_FIELDS, SLOTS
from .tpu_v6e_tc_model import FORMS_BY_BRANCH, SHARED, GlForm

TARGET = TPU_V6E_TC.identifier
WORD_BYTES = 64
_PATHS = {path: slot for slot, (path, _, _) in SLOTS.items()}
_SHARED_MESSAGES = {number for _, number, _, _, _ in SHARED_FIELDS}

def _words(image: bytes) -> list[int]:
    return [int.from_bytes(image[offset:offset + WORD_BYTES], 'little') for offset in range(0, len(image), WORD_BYTES)]

def _slot_form(slot: str, payload: bytes, word: int, pc: int) -> GlForm:
    operations = [(number, value) for number, wire, value in fields(payload) if wire == 2 and (slot, number) in FORMS_BY_BRANCH]
    if len(operations) != 1:
        raise ValueError(f'unsupported instruction structure in {slot} at bundle {pc:#x}')
    branch, body = operations[0]
    form = FORMS_BY_BRANCH[(slot, branch)]
    assert isinstance(body, bytes)
    decoded = {number: value for number, wire, value in fields(body) if isinstance(value, int)}
    for name in form.layout:
        field = form.fields[name]
        if decoded.get(field.number, 0) & ((1 << field.width) - 1) != field.read(word):
            raise ValueError(f'decoded {slot} operand differs from the machine word at bundle {pc:#x}')
        if field.read(word) in FORMATTER_ABORTS.get(form.enums.get(name, ''), ()):
            raise ValueError(f'reserved {slot} {name} encoding at bundle {pc:#x}')
    head = {number: value for number, wire, value in fields(payload) if isinstance(value, int)}
    if head.get(1, 0) != form.predicate_field.read(word) or (form.inversion_field is not None and head.get(2, 0) != form.inversion_field.read(word)):
        raise ValueError(f'decoded {slot} predicate differs from the machine word at bundle {pc:#x}')
    return form

def decode_program(image: bytes) -> list[tuple[int, tuple[tuple[GlForm, int], ...]]]:
    """返回每个 bundle 的机器字和 (形式, 汇编谓词)；原生解码结果必须与机器字逐字段一致。"""
    serialized = _program_proto(image, encode=False, target=TARGET)
    words = _words(image)
    bundles = [value for number, wire, value in fields(serialized) if number == 1 and wire == 2]
    if len(bundles) != len(words):
        raise ValueError('decoded bundle count differs from the program image')
    program = []
    for pc, (word, bundle) in enumerate(zip(words, bundles)):
        assert isinstance(bundle, bytes)
        forms = []
        for number, wire, value in fields(bundle):
            if wire != 2 or not isinstance(value, bytes):
                raise ValueError(f'unsupported TC bundle field at bundle {pc:#x}')
            if number in _SHARED_MESSAGES:
                decoded = {key: item for key, kind, item in fields(value) if isinstance(item, int)}
                for name, message_number, field_number, _, _ in SHARED_FIELDS:
                    if message_number == number and decoded.get(field_number, 0) != SHARED[name].read(word):
                        raise ValueError(f'decoded {name} differs from the machine word at bundle {pc:#x}')
                continue
            if (number,) in _PATHS:
                forms.append(_slot_form(_PATHS[(number,)], value, word, pc))
                continue
            for inner, inner_wire, payload in fields(value):
                if (number, inner) not in _PATHS or inner_wire != 2 or not isinstance(payload, bytes):
                    raise ValueError(f'unsupported TC bundle field at bundle {pc:#x}')
                forms.append(_slot_form(_PATHS[(number, inner)], payload, word, pc))
        forms.sort(key=lambda form: TPU_V6E_TC.slots.index(form.slot))
        program.append((word, tuple((form, form.predicate(word)) for form in forms)))
    return program

def encode_program(bundles: list[tuple[int, tuple[Form, ...]]]) -> bytes:
    """按机器字写出每个槽的全部字段；原生编码结果必须与机器字逐字节相同。"""
    serialized = []
    for word, forms in bundles:
        items: list[tuple[int, int | bytes]] = []
        scalar: list[tuple[int, int | bytes]] = []
        for form in sorted(forms, key=lambda item: TPU_V6E_TC.slots.index(item.slot)):
            assert isinstance(form, GlForm)
            head: list[tuple[int, int | bytes]] = [(1, form.predicate_field.read(word))]
            if form.inversion_field is not None:
                head.append((2, form.inversion_field.read(word)))
            payload = message([(field.number, field.read(word)) for field in form.own_fields])
            body = message(head + [(form.branch, payload)])
            path = SLOTS[form.slot][0]
            if len(path) == 2:
                scalar.append((path[1], body))
            else:
                items.append((path[0], body))
        if scalar:
            items.append((1, message(scalar)))
        for number in sorted(_SHARED_MESSAGES):
            items.append((number, message([(field_number, SHARED[name].read(word)) for name, message_number, field_number, _, _ in SHARED_FIELDS if message_number == number])))
        serialized.append((1, message(sorted(items, key=lambda item: item[0]))))
    image = _program_proto(message(serialized), encode=True, target=TARGET)
    expected = b''.join(word.to_bytes(WORD_BYTES, 'little') for word, _ in bundles)
    if image != expected:
        raise ValueError('native v6e encoder changed the solved hardware fields or bundle count')
    return image

def decoded_signature(form: GlForm, word: int) -> tuple[Signature, tuple[str, ...]]:
    for signature in SIGNATURES_BY_BRANCH[(form.slot, form.branch)]:
        if any(form.fields[name].read(word) != value for name, value in signature.fixed_fields):
            continue
        try:
            operands = tuple(decode_operand(operand, form, word) for operand in signature.operands)
        except ValueError:
            continue
        if signature.keywords:
            count = len(signature.keywords)
            defaults = dict(signature.defaults)
            named = tuple(f'{key}={value}' for key, value in zip(signature.keywords, operands[-count:]) if defaults.get(key) != value)
            operands = operands[:-count] + named
        return signature, operands
    raise ValueError(f'no supported signature for {form.slot} {form.name}')

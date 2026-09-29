"""命名硬件约束：共享资源与每个已登记指令形式的物理字段。"""
from __future__ import annotations

import re

from .assembly_model import Bits, Field
from .tpu_v4_tc_model import TcForm as Form
from .assembly_syntax import EncodingConstraint, integer
from .tpu_v4_tc_isa import READ_PORTS, SCALAR_CONSTANTS, VECTOR_CONSTANTS, WRITE_PORTS

GLOBAL_FIELDS = {
    **{f'imm{i}': Field(0, start, 16) for i, start in enumerate((338, 320, 304, 288, 272, 256))},
    **{f'vs{i}': Field(0, start, 5) for i, start in enumerate((251, 246, 241))},
    **{name: Field(0, start, 5) for name, start in (
        ('port.va0.x', 225), ('port.va0.y', 203), ('port.va1.x', 182), ('port.va1.y', 172), ('port.vst.src', 152),
        ('port.va0.dst', 198), ('port.va1.dst', 167), ('port.vld.dst', 129), ('port.aux.dst', 17),
    )},
}
_SOURCE_PORTS = ('port.va0.x', 'port.va0.y', 'port.va1.x', 'port.va1.y', 'port.vst.src')
_DEST_PORTS = ('port.va0.dst', 'port.va1.dst', 'port.vld.dst', 'port.aux.dst')
_ALIASES = {'sy': 'y', 'y_src': 'y', 'read_port': 'read', 'base_address': 'base', 'sublane_mask': 'sm', 'stride': 'ss', 'dest_sync_flag': 'dst_flag', 'sync_flag_number': 'flag', 'operand': 'value'}

def resource_name(form: Form, name: str) -> str:
    field = form.fields[name]
    for public, global_field in GLOBAL_FIELDS.items():
        if field.start == global_field.start and field.width == global_field.width:
            return public
    if name == 'dest' and form.slot.startswith('vr'):
        return form.slot + '.write'
    return form.slot + '.' + _ALIASES.get(name, name)

def selector_values(form: Form, name: str) -> dict[int, str]:
    if name in ('sy', 'reg_value', 'address', 'smem_address', 'length', 'stride_descriptor') or (name == 'dest_sync_flag' and form.fields[name].width == 6):
        values = {i: f's{i}' for i in range(32)}
        for prefix, start in (('lo', 32), ('ones_hi', 36), ('hi', 40)):
            values.update({start + i: f'{prefix}(imm{i})' for i in range(4)})
        values.update({44: 'pair(imm1, imm0)', 45: 'pair(imm3, imm2)'})
        values.update({i: f'const({value})' for i, value in SCALAR_CONSTANTS.items()})
        return values
    if name == 'y_src':
        values = {0: f'port.{form.slot}.y', **{i: f'const({value})' for i, value in VECTOR_CONSTANTS.items()}}
        for prefix, start in (('lo', 8), ('ones_hi', 14), ('hi', 20)):
            values.update({start + i: f'{prefix}(imm{i})' for i in range(6)})
        values.update({26 + i: f'pair(imm{2 * i + 1}, imm{2 * i})' for i in range(3)})
        values.update({29 + i: f'vs{i}' for i in range(3)})
        return values
    if name == 'read_port':
        return dict(enumerate(_SOURCE_PORTS))
    if name == 'dest' and form.slot.startswith('vr'):
        return dict(enumerate(_DEST_PORTS))
    if name == 'offset' and form.slot in ('vld', 'vst', 'cld'):
        return {i: f'imm{i + 2}' for i in range(4)}
    if name == 'base_address':
        return {0: 'const(0)', **{i + 1: f'vs{i}' for i in range(3)}}
    if name == 'shuffle':
        return {**{i + 1: f'vs{i}' for i in range(3)}, **{i + 4: f'pair(imm{2 * i + 1}, imm{2 * i})' for i in range(3)}}
    msc_names = ('sublane_mask', 'stride', 'matrix_width', 'rotate_count', 'operand', 'operand0', 'operand1', 'sync_flag_number', 'interrupt_number')
    if name in msc_names or (name == 'delay_count' and form.name == 'Delay') or (name == 'dest' and form.name.startswith('AtomicRemote')):
        zero = 128 if name == 'matrix_width' else (1 if name in ('stride', 'rotate_count', 'delay_count') else 0)
        return {0: f'const({zero})', **{i + 1: f'vs{i}' for i in range(3)}, **{i + 4: f'imm{i + 2}' for i in range(4)}}
    return {}

def registry(form: Form) -> dict[str, tuple[Field, dict[int, str]]]:
    result = {resource_name(form, name): (field, selector_values(form, name)) for name, field in form.fields.items()}
    result[form.slot + '.opcode'] = (form.opcode_field, {})
    for name, start, width, value in form.extra:
        result[form.slot + '.' + name] = (Field(0, start, width), {})
    return result

def _value(text: str, values: dict[int, str], name: str) -> int:
    if values:
        normalized = re.sub(r'\s+', '', text)
        for value, label in values.items():
            if normalized == re.sub(r'\s+', '', label):
                return value
        match = re.fullmatch(r'const\((.+)\)', normalized)
        if match:
            number = integer(match[1]) & 0xffffffff
            for value, label in values.items():
                if label == f'const({number})':
                    return value
        raise ValueError(f'invalid source {text!r} for {name}')
    if name.startswith('vs'):
        if not re.fullmatch(r's[0-9]+', text):
            raise ValueError(f'{name} requires an sN register')
        return int(text[1:])
    return integer(text)

def bind_constraint(pin: EncodingConstraint, form: Form | None = None) -> Bits:
    if pin.name in GLOBAL_FIELDS:
        return GLOBAL_FIELDS[pin.name].bind(_value(pin.value, {}, pin.name))
    resources = registry(form) if form is not None else {}
    if pin.name not in resources:
        raise ValueError(f'unknown or unavailable encoding constraint {pin.name!r}')
    field, values = resources[pin.name]
    return field.bind(_value(pin.value, values, pin.name))

def export_constraints(forms: tuple[Form, ...], word: int) -> dict[str, str]:
    values: dict[str, str] = {}
    for form in forms:
        for name, (field, selectors) in registry(form).items():
            value = field.read(word)
            if selectors and value not in selectors:
                raise ValueError(f'no named encoding for {name}={value}')
            display = selectors[value] if selectors else (f's{value}' if name.startswith('vs') else str(value))
            if name in values and values[name] != display:
                raise ValueError(f'conflicting definitions of {name}')
            values[name] = display
    return dict(sorted(values.items()))

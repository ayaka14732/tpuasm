"""TPU v6e 的命名硬件约束：共享立即数与标量操作数、跨槽读端口，以及每个已登记形式的字段。"""
from __future__ import annotations

import re

from .assembly_model import Bits, Field
from .assembly_syntax import EncodingConstraint, integer
from .tpu_v6e_tc_isa import value_names
from .tpu_v6e_tc_isa_data import FORMATTER_ABORTS
from .tpu_v6e_tc_model import FORMS, SHARED, GlForm

# 本槽字段也被其他槽的形式读写时，以本槽为名公开为全局端口，例如向量存储经 VALU0 的 x 读端口取寄存器。
_PORT_ALIASES = {'x': 'x', 'y_vreg': 'y', 'y_src': 'ysrc', 'dest': 'dst', 'source_vreg': 'src', 'vmsk': 'vmsk'}

def _ports() -> dict[str, tuple[Field, str]]:
    """跨槽共享的字段：全局名称、字段与枚举名（寄存器号字段为空）。"""
    owners: dict[tuple[int, int], set[str]] = {}
    for form in FORMS:
        for name in form.layout:
            field = form.fields[name]
            owners.setdefault((field.start, field.width), set()).add(form.slot)
    ports: dict[str, tuple[Field, str]] = {}
    for form in FORMS:
        if not re.fullmatch(r'va[0-3]|vst', form.slot):
            continue
        for name, alias in _PORT_ALIASES.items():
            if name not in form.layout:
                continue
            field = form.fields[name]
            slots = owners[(field.start, field.width)]
            if len(slots - {'s0', 's1', 'dma'}) > 1 or (form.slot.startswith('va') and 'dma' in slots):
                ports.setdefault(f'port.{form.slot}.{alias}', (Field(0, field.start, field.width), form.enums.get(name, '')))
    return ports

_PORTS = _ports()
GLOBAL_FIELDS = {**{name: Field(0, field.start, field.width) for name, field in SHARED.items()}, **{name: field for name, (field, _) in _PORTS.items()}}
_GLOBAL_VALUES = {name: value_names(enum) for name, (_, enum) in _PORTS.items() if enum}
_GLOBAL_RANGES = {(field.start, field.width): name for name, field in GLOBAL_FIELDS.items()}

def resource_name(form: GlForm, name: str) -> str:
    field = form.fields[name]
    return _GLOBAL_RANGES.get((field.start, field.width), form.slot + '.' + name)

def selector_values(form: GlForm, name: str) -> dict[int, str]:
    enum = form.enums.get(name)
    if enum is None:
        return {}
    aborts = FORMATTER_ABORTS.get(enum, ())
    return {value: label for value, label in value_names(enum).items() if value not in aborts}

def registry(form: GlForm) -> dict[str, tuple[Field, dict[int, str]]]:
    return {resource_name(form, name): (form.fields[name], selector_values(form, name)) for name in form.layout}

def _value(text: str, values: dict[int, str], name: str) -> int:
    if values:
        normalized = re.sub(r'\s+', '', text)
        for value, label in values.items():
            if normalized == label:
                return value
        raise ValueError(f'invalid value {text!r} for {name}; expected one of {", ".join(values.values())}')
    if re.fullmatch(r'vs[0-3]', name):
        if not re.fullmatch(r's[0-9]+', text):
            raise ValueError(f'{name} requires an sN register')
        return int(text[1:])
    return integer(text)

def bind_constraint(pin: EncodingConstraint, form: GlForm | None = None) -> Bits:
    if pin.name in GLOBAL_FIELDS:
        return GLOBAL_FIELDS[pin.name].bind(_value(pin.value, _GLOBAL_VALUES.get(pin.name, {}), pin.name))
    if form is not None and pin.name == form.slot + '.form':
        if pin.value != form.name:
            raise ValueError(f'{pin.name} selects {pin.value}, not {form.name}')
        return Bits()
    resources = registry(form) if form is not None else {}
    if pin.name not in resources:
        raise ValueError(f'unknown or unavailable encoding constraint {pin.name!r}')
    field, values = resources[pin.name]
    return field.bind(_value(pin.value, values, pin.name))

def export_constraints(forms: tuple[GlForm, ...], word: int) -> dict[str, str]:
    values: dict[str, str] = {name: f's{field.read(word)}' if name.startswith('vs') else str(field.read(word)) for name, field in SHARED.items()}
    for form in forms:
        values[form.slot + '.form'] = form.name
        for name, (field, selectors) in registry(form).items():
            value = field.read(word)
            if selectors and value not in selectors:
                raise ValueError(f'no named encoding for {name}={value}')
            display = selectors[value] if selectors else (f's{value}' if name.startswith('vs') else str(value))
            if name in values and values[name] != display:
                raise ValueError(f'conflicting definitions of {name}')
            values[name] = display
    return dict(sorted(values.items()))

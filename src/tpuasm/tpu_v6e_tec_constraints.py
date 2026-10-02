"""TPU v6e TEC 的命名硬件约束：共享立即数与标量操作数，以及每个已登记形式的字段。"""
from __future__ import annotations

import os
import re

from .assembly_model import Bits, Field
from .assembly_syntax import EncodingConstraint, integer
from .tpu_v6e_tec_isa_data import ENUMS
from .tpu_v6e_tec_model import SHARED, TecForm

GLOBAL_FIELDS = {name: Field(0, field.start, field.width) for name, field in SHARED.items()}

def value_names(kind: str) -> dict[int, str]:
    """descriptor 枚举值去掉公共前缀后的小写名称，用作命名编码约束的取值。"""
    names = ENUMS[kind]
    prefix = os.path.commonprefix(list(names.values()))
    prefix = prefix[:prefix.rfind('_') + 1]
    return {value: name[len(prefix):].lower() for value, name in names.items()}

def registry(form: TecForm) -> dict[str, tuple[Field, dict[int, str]]]:
    return {f'{form.slot}.{name}': (form.fields[name], value_names(form.enums[name]) if name in form.enums else {}) for name in form.layout}

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

def bind_constraint(pin: EncodingConstraint, form: TecForm | None = None) -> Bits:
    if pin.name in GLOBAL_FIELDS:
        return GLOBAL_FIELDS[pin.name].bind(_value(pin.value, {}, pin.name))
    if form is not None and pin.name == form.slot + '.form':
        if pin.value != form.name:
            raise ValueError(f'{pin.name} selects {pin.value}, not {form.name}')
        return Bits()
    resources = registry(form) if form is not None else {}
    if pin.name not in resources:
        raise ValueError(f'unknown or unavailable encoding constraint {pin.name!r}')
    field, values = resources[pin.name]
    return field.bind(_value(pin.value, values, pin.name))

def export_constraints(forms: tuple[TecForm, ...], word: int) -> dict[str, str]:
    values: dict[str, str] = {name: f's{field.read(word)}' if re.fullmatch(r'vs[0-3]', name) else str(field.read(word)) for name, field in SHARED.items()}
    for form in forms:
        values[form.slot + '.form'] = form.name
        for name, (field, selectors) in registry(form).items():
            value = field.read(word)
            if selectors and value not in selectors:
                raise ValueError(f'no named encoding for {name}={value}')
            values[name] = selectors[value] if selectors else str(value)
    return dict(sorted(values.items()))

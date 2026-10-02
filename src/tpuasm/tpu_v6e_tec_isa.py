"""TPU v6e TEC 的助记符签名与共享操作数选择器。

助记符、操作数顺序和地址写法取自 libtpu 编译 SparseCore kernel 所用的 LLVM TPU printer（见 tpu_v6e_tec_isa_data 的 INSTRUCTIONS）；每个操作数对应的字段由 TEC emitter 探测得到。选择器的数值含义沿用 v6e TC 的规则，所需的共享立即数由求解器分配。
"""
from __future__ import annotations

import re
from typing import Any

from .assembly_expressions import choice, direct, literal, numeric, packed, pattern, register, table
from .assembly_model import Expression, Signature
from .tpu_v6e_tec_constraints import value_names
from .tpu_v6e_tec_isa_data import INSTRUCTIONS
from .tpu_v6e_tec_model import FORMS, FORMS_BY_NAME, SHARED, TecForm

LANE = SHARED['imm0'].width
_WORDS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8}

def constant(name: str) -> int | None:
    """选择器的内置常量，名称取自 descriptor。"""
    name = re.sub(r'^int_', '', name)
    if name in _WORDS:
        return _WORDS[name]
    if name == 'negative_one':
        return 0xffffffff
    if match := re.fullmatch(r'hex_([0-9a-f]+)', name):
        return int(match[1], 16)
    if match := re.fullmatch(r'(0x[0-9a-f]+)', name):
        return int(match[1], 16)
    return {'float_one': 0x3f800000, 'float_negative_one': 0xbf800000, 'float_half': 0x3f000000}.get(name)

def source(name: str, kind: str) -> Expression | None:
    """一个选择器取值对应的数值；共享立即数 20 位，拼接只取各字段低 16 位，与 v6e TC 相同。"""
    if match := re.fullmatch(r'(?:zero_)?imm_?([0-5])', name):
        return numeric(kind, 32, ((f'imm{match[1]}', 0),))
    if match := re.fullmatch(r'ones_imm([0-5])', name):
        return numeric(kind, 32, ((f'imm{match[1]}', 0),), (0xffffffff << LANE) & 0xffffffff)
    if match := re.fullmatch(r'imm([0-5])_zero', name):
        return numeric(kind, 32, ((f'imm{match[1]}', 32 - LANE),))
    if match := re.fullmatch(r'imm([0-5])_imm([0-5])', name):
        return numeric(kind, 32, ((f'imm{match[1]}', 16), (f'imm{match[2]}', 0)), part_width=16)
    value = constant(name)
    if value is not None:
        return numeric(kind, 32, base=value)
    return None

def expression(form: TecForm, descriptor: tuple[Any, ...]) -> Expression:
    kind = descriptor[0]
    if kind == 'literal':
        return literal(descriptor[1])
    if kind == 'register':
        _, field, prefix, count = descriptor
        if count < 1 << form.fields[field].width:
            # The field also encodes selector constants; only the register numbers have this syntax.
            return choice(field, [(number, literal(f'{prefix}{number}')) for number in range(count)])
        return register(field, prefix)
    if kind == 'lane':
        return choice(descriptor[1], [(selector, register(f'vs{lane}', 's')) for selector, lane in descriptor[2]])
    if kind == 'selector':
        _, field, values, display = descriptor
        names = value_names(form.enums[field])
        alternatives = []
        for value in values:
            operand = source(names[value], display)
            if operand is not None:
                alternatives.append((value, operand))
        return choice(field, alternatives)
    if kind == 'immediate':
        _, field, display = descriptor
        return numeric(display, LANE, ((field, 0),))
    if kind == 'number':
        _, field, display = descriptor
        return direct(field, form.fields[field].width, display)
    if kind == 'packed':
        return packed(descriptor[2], 32, descriptor[1])
    if kind == 'table':
        return table(descriptor[1])
    if kind == 'pattern':
        return pattern(descriptor[1], tuple(expression(form, part) for part in descriptor[2]))
    raise ValueError(f'unknown TEC operand descriptor {kind!r}')

def signature(entry: tuple[Any, ...]) -> Signature:
    slot, name, _, mnemonic, fixed, operands = entry
    form = FORMS_BY_NAME[(slot, name)]
    return Signature(form, mnemonic, tuple(expression(form, descriptor) for descriptor in operands), fixed)

def fallback(form: TecForm) -> Signature:
    """没有 printer 写法的形式：助记符取 descriptor 中的形式名，各字段按字段顺序写作具名参数；selector 取值能写成数值时写数值，否则写取值名。"""
    operands: list[Expression] = []
    for name in form.layout:
        if name in form.enums:
            alternatives = []
            for value, label in value_names(form.enums[name]).items():
                if value >> form.fields[name].width:
                    continue
                operand = source(label, 'hex')
                alternatives.append((value, operand if operand is not None else literal(label)))
            operands.append(choice(name, alternatives))
        else:
            operands.append(direct(name, form.fields[name].width, 'hex'))
    return Signature(form, form.name, tuple(operands), keywords=form.layout)

# Every form also has the named-field syntax for encodings that no printer syntax covers.
FALLBACKS = tuple(fallback(form) for form in FORMS)
SIGNATURES = tuple(signature(entry) for entry in INSTRUCTIONS) + FALLBACKS
SIGNATURES_BY_MNEMONIC: dict[tuple[str, str], list[Signature]] = {}
SIGNATURES_BY_BRANCH: dict[tuple[str, int], list[Signature]] = {}
# Decoding tries the signatures that fix more fields first, e.g. vld before vld.msk, and the named-field syntax last.
for _signature in sorted(SIGNATURES, key=lambda item: (item in FALLBACKS, -len(item.fixed_fields))):
    SIGNATURES_BY_MNEMONIC.setdefault((_signature.form.slot, _signature.mnemonic), []).append(_signature)
    SIGNATURES_BY_BRANCH.setdefault((_signature.form.slot, _signature.form.branch), []).append(_signature)

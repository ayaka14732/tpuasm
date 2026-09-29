"""将有类型的汇编操作数转换成可合并的字段约束，并逆向打印。"""
from __future__ import annotations

import itertools
import re

from .assembly_model import Bits, Expression, Field, Form, Signature, combine
from .assembly_syntax import integer, split_operands
from .assembly_values import display_number, number

def _literal_equal(left: str, right: str) -> bool:
    return re.sub(r'\s+', '', left) == re.sub(r'\s+', '', right)

def address_parts(text: str) -> tuple[str, str, str | None, dict[str, str]]:
    match = re.fullmatch(r'\[\s*([a-z][a-z0-9_]*)\s*:\s*(.+)\]', text)
    if match is None:
        raise ValueError('expected [space:address]')
    items = split_operands(match[2])
    modifiers = {}
    for item in items[1:]:
        pair = item.split('=', 1)
        if len(pair) != 2 or pair[0].strip() not in ('sm', 'ss'):
            raise ValueError(f'unknown address modifier {item!r}')
        key, value = (x.strip() for x in pair)
        if key in modifiers:
            raise ValueError(f'duplicate address modifier {key!r}')
        modifiers[key] = value
    term = r'(?:sy\.[a-z0-9_]+|[sv][0-9]+|[+-]?(?:0x[0-9a-fA-F]+|[0-9]+))'
    expression = re.fullmatch(rf'\s*({term})\s*(?:([+-])\s*({term}))?\s*', items[0])
    if expression is None:
        raise ValueError(f'invalid address expression {items[0]!r}')
    base, sign, offset = expression.groups()
    if sign == '-':
        if offset.startswith(('s', 'v')):
            raise ValueError('register subtraction is not an address operand')
        offset = str(-integer(offset))
    return match[1], base, offset, modifiers

def _part_width(field: Field, limit: int | None) -> int:
    return field.width if limit is None else min(field.width, limit)

def _number_bits(expression: Expression, text: str, form: Form) -> list[Bits]:
    """数值拆入若干立即数字段；各字段按位或拼接。

    字段中超出操作数位宽或 part_width 的位不影响数值，保持未约束。v6e vtrace 的现有 formatter 模型仍有重叠位：为 1 的位可以来自任一字段或两者，每种分配各是一个候选。
    """
    _, kind, width, parts, base, bias = expression
    value = number(text, kind, width) - bias
    if value < 0:
        return []
    mask = 0
    for name, shift, limit in parts:
        mask |= ((1 << _part_width(form.fields[name], limit)) - 1) << shift
    if value & ~mask != base:
        return []
    owners: dict[int, list[tuple[int, int]]] = {}
    for index, (name, shift, limit) in enumerate(parts):
        for bit in range(_part_width(form.fields[name], limit)):
            if bit + shift < width:
                owners.setdefault(bit + shift, []).append((index, bit))
    # Each assignment holds the (mask, value) bound in every part's field.
    assignments = [[(0, 0)] * len(parts)]
    for position, bit_owners in sorted(owners.items()):
        if len(bit_owners) == 1 or not value >> position & 1:
            choices = [tuple(value >> position & 1 for _ in bit_owners)]
        else:
            choices = [combination for combination in itertools.product((0, 1), repeat=len(bit_owners)) if any(combination)]
        expanded = []
        for assignment in assignments:
            for combination in choices:
                updated = list(assignment)
                for (index, bit), set_bit in zip(bit_owners, combination):
                    part_mask, part_value = updated[index]
                    updated[index] = (part_mask | 1 << bit, part_value | set_bit << bit)
                expanded.append(updated)
        assignments = expanded
    result = []
    for assignment in assignments:
        bits: Bits | None = Bits()
        for (name, _, _), (part_mask, part_value) in zip(parts, assignment):
            field = form.fields[name]
            used = form.bind(name, 0, consumed=True)
            bits = bits.merge(Bits(part_mask << field.start, part_value << field.start, used.immediates, used.scalars)) if bits is not None else None
        if bits is not None:
            result.append(bits)
    return list(dict.fromkeys(result))

def _dma_bits(expression: Expression, text: str, form: Form) -> list[Bits]:
    dest = expression[1]
    space, base, offset, modifiers = address_parts(text)
    if offset is not None or modifiers:
        return []
    bits = encode_operand(('register', 'dest_address' if dest else 'source_address', 's'), base, form)
    fields = []
    memory_field = 'destination_memory_id' if dest else 'source_memory_id'
    core_field = 'destination_core_id' if dest else 'source_core_id'
    if dest and re.fullmatch(r'oq[0-9]+', space):
        fields = [('dma_type_0', 1), ('outfeed_queue_id', int(space[2:]))]
    elif not dest and space in ('memseti', 'memsetd'):
        fields = [('src_opcode', 2 if space == 'memseti' else 3)]
    elif space in ('hbm', 'cmem'):
        fields = [(core_field, 1), (memory_field, 0 if space == 'hbm' else 2)]
    elif space in ('vmem', 'smem', 'imem'):
        fields = [(core_field, 0), (memory_field, ('vmem', 'smem', 'imem').index(space))]
    else:
        return []
    if dest and not space.startswith('oq'):
        fields.append(('dma_type_0', 0))
    if not dest and space not in ('memseti', 'memsetd'):
        fields.append(('src_opcode', 0))
    for name, value in fields:
        bits = combine(bits, [form.bind(name, value)])
    return bits

def encode_operand(expression: Expression, text: str, form: Form) -> list[Bits]:
    kind = expression[0]
    if kind == 'literal':
        return [Bits()] if _literal_equal(text, expression[1]) else []
    if kind == 'register':
        _, field, prefix = expression
        match = re.fullmatch(prefix + r'([0-9]+)', text)
        return [form.bind(field, int(match[1]), consumed=True)] if match else []
    if kind == 'number':
        return _number_bits(expression, text, form)
    if kind == 'choice':
        _, field, alternatives = expression
        result = []
        for selector, operand in alternatives:
            try:
                matches = encode_operand(operand, text, form)
            except ValueError:
                continue
            result.extend(combine([form.bind(field, selector)], matches))
        return list(dict.fromkeys(result))
    if kind == 'table':
        normalized = re.sub(r'\s+', '', text)
        result = []
        for label, assignments in expression[1]:
            if re.sub(r'\s+', '', label) == normalized:
                bits = [Bits()]
                for name, value in assignments:
                    bits = combine(bits, [form.bind(name, value)])
                result.extend(bits)
        return result
    if kind == 'memory':
        _, space, base_expr, offset_expr, modifiers = expression
        actual_space, base, offset, extras = address_parts(text)
        if actual_space != space or not set(extras) <= {key for key, _, _ in modifiers}:
            return []
        if offset_expr is None:
            if offset is not None:
                return []
            bits = encode_operand(base_expr, base, form)
        else:
            if form.name == 'ScalarLoadSmemOffset' and offset is None:
                return []
            if offset is None and not base.startswith('s'):
                base, offset = '0', base
            bits = combine(encode_operand(base_expr, base, form), encode_operand(offset_expr, offset or '0', form))
        for key, field, modifier in modifiers:
            modifier_bits = encode_operand(modifier, extras[key], form) if key in extras else [form.bind(field, 0)]
            bits = combine(bits, modifier_bits)
        return bits
    if kind == 'dma_address':
        return _dma_bits(expression, text, form)
    if kind == 'trace':
        if re.fullmatch(r's[0-9]+', text):
            result = [Bits()]
            for field in ('operand0', 'operand1'):
                choices = []
                for i in range(3):
                    choices.extend(combine([form.bind(field, i + 1)], encode_operand(('register', f'vs{i}', 's'), text, form)))
                result = combine(result, choices)
            return result
        value = number(text, 'hex')
        bits = [Bits()]
        for field, part in (('operand0', value & 0xffff), ('operand1', value >> 16)):
            immediate_choices: list[tuple[int, tuple[tuple[str, int], ...]]] = [(0, ())] if part == 0 else []
            immediate_choices.extend((i + 4, ((f'imm{i + 2}', part),)) for i in range(4))
            candidates = []
            for selector, assignments in immediate_choices:
                bound = form.bind(field, selector)
                for name, val in assignments:
                    merged = bound.merge(form.bind(name, val, consumed=True))
                    assert merged is not None
                    bound = merged
                candidates.append(bound)
            bits = combine(bits, candidates)
        return bits
    raise ValueError(f'unknown operand signature {kind!r}')

def _is_zero(text: str) -> bool:
    try:
        return integer(text) == 0
    except ValueError:
        return False

def decode_operand(expression: Expression, form: Form, word: int) -> str:
    kind = expression[0]
    if kind == 'literal':
        return expression[1]
    if kind == 'register':
        return expression[2] + str(form.fields[expression[1]].read(word))
    if kind == 'number':
        _, number_kind, width, parts, base, bias = expression
        value = base
        for name, shift, limit in parts:
            part_field = form.fields[name]
            value |= (part_field.read(word) & ((1 << _part_width(part_field, limit)) - 1)) << shift
        return display_number(value + bias, number_kind, width)
    if kind == 'choice':
        selector = form.fields[expression[1]].read(word)
        for value, operand in expression[2]:
            if value == selector:
                return decode_operand(operand, form, word)
        raise ValueError(f'unsupported selector {expression[1]}={selector}')
    if kind == 'table':
        for label, assignments in expression[1]:
            if all(form.fields[name].read(word) == value for name, value in assignments):
                return label
        raise ValueError('unsupported encoding for a fixed operand')
    if kind == 'memory':
        _, space, base_expr, offset_expr, modifiers = expression
        base = decode_operand(base_expr, form, word)
        offset = decode_operand(offset_expr, form, word) if offset_expr is not None else '0'
        if _is_zero(base):
            address = offset if offset_expr is not None else base
        elif _is_zero(offset) and form.name != 'ScalarLoadSmemOffset':
            address = base
        else:
            address = base + (' + ' if not offset.startswith('-') else ' - ') + offset.lstrip('-')
        for key, field, modifier in modifiers:
            if form.fields[field].read(word):
                address += f', {key}=' + decode_operand(modifier, form, word)
        return f'[{space}:{address}]'
    if kind == 'dma_address':
        dest = expression[1]
        fields = form.fields
        if dest and fields['dma_type_0'].read(word):
            space = 'oq' + str(fields['outfeed_queue_id'].read(word))
        elif not dest and fields['src_opcode'].read(word) in (2, 3):
            space = 'memseti' if fields['src_opcode'].read(word) == 2 else 'memsetd'
        else:
            core = fields['destination_core_id' if dest else 'source_core_id'].read(word)
            memory_id = fields['destination_memory_id' if dest else 'source_memory_id'].read(word)
            if core == 1 and memory_id in (0, 2):
                space = 'hbm' if memory_id == 0 else 'cmem'
            elif core == 0 and memory_id < 3:
                space = ('vmem', 'smem', 'imem')[memory_id]
            else:
                raise ValueError(f'unsupported DMA address core={core}, memory={memory_id}')
        address_register = fields['dest_address' if dest else 'source_address'].read(word)
        return f'[{space}:s{address_register}]'
    if kind == 'trace':
        low_selector = form.fields['operand0'].read(word)
        high_selector = form.fields['operand1'].read(word)
        if 1 <= low_selector <= 3 and 1 <= high_selector <= 3:
            low = form.fields[f'vs{low_selector - 1}'].read(word)
            high = form.fields[f'vs{high_selector - 1}'].read(word)
            if low == high:
                return f's{low}'
            raise ValueError('vtrace halves must read the same scalar register')
        value = 0
        for field, shift in (('operand0', 0), ('operand1', 16)):
            selector = form.fields[field].read(word)
            if selector == 0:
                part = 0
            elif 4 <= selector <= 7:
                part = form.fields[f'imm{selector - 2}'].read(word)
            else:
                raise ValueError('vtrace requires immediate sources')
            value |= part << shift
        return hex(value)
    raise ValueError(f'unknown operand signature {kind!r}')

def ordered_operands(signature: Signature, operands: tuple[str, ...]) -> tuple[str, ...]:
    if not signature.keywords:
        return operands
    count = len(signature.operands) - len(signature.keywords)
    positional = operands[:count]
    keywords = {}
    for item in operands[count:]:
        pair = item.split('=', 1)
        if len(pair) != 2:
            raise ValueError('expected a named operand')
        name, value = (part.strip() for part in pair)
        if name in keywords:
            raise ValueError(f'duplicate named operand {name!r}')
        keywords[name] = value
    for name, value in signature.defaults:
        keywords.setdefault(name, value)
    if set(keywords) != set(signature.keywords):
        raise ValueError(f'expected named operands {", ".join(signature.keywords)}')
    return positional + tuple(keywords[name] for name in signature.keywords)

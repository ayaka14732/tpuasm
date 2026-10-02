"""TPU v6e（Ghostlite）各执行单元共用的 ISA descriptor 读取与 encoder 字段探测，供 v6e TC 与 TEC 字段表生成工具使用。

执行单元之间只有 bundle 消息、槽路径、共享字段和互斥槽不同，由 ``GlSpec`` 描述；探测流程见 v6e TC 设计文档的“字段表的生成”。
"""
from __future__ import annotations

from dataclasses import dataclass
import mmap
from pathlib import Path
import random
import re
import sys

from tpuasm._protobuf import fields, message, read_varint
from tpuasm.printer import _program_proto
from tpuasm.targets import hardware_target

PACKAGE = 'asic_sw.deepsea.gxc.glc.isa'
WORD_BITS = 512
ALWAYS, NEVER = 14, 15

@dataclass(frozen=True)
class GlSpec:
    """一个执行单元的 bundle 结构。

    ``slots`` 是物理槽在 bundle 中的字段路径，两级路径位于标量子 bundle；``shared`` 的每项为 (名称, bundle 字段号, 消息内字段号)；``exclusive`` 是同属一个 protobuf oneof、不能同时出现的槽对。
    """
    target: str
    bundle: str
    scalar: str
    slots: dict[str, tuple[int, ...]]
    shared: tuple[tuple[str, int, int], ...]
    exclusive: frozenset[frozenset[str]]

# ---------------------------------------------------------------- descriptors

@dataclass(frozen=True)
class FieldInfo:
    name: str
    number: int
    kind: int
    type_name: str
    oneof: int | None

@dataclass(frozen=True)
class MessageInfo:
    fields: tuple[FieldInfo, ...]

def _parse(data: bytes, pos: int, end: int, valid: range) -> list[tuple[int, int | bytes]]:
    """读取到第一个不合法字段为止；嵌入的 descriptor 后面紧接着其他数据。"""
    result: list[tuple[int, int | bytes]] = []
    while pos < end:
        try:
            tag, next_pos = read_varint(data, pos)
            number, wire = tag >> 3, tag & 7
            if number not in valid or wire not in (0, 2):
                break
            value: int | bytes
            if wire == 0:
                value, next_pos = read_varint(data, next_pos)
            else:
                size, next_pos = read_varint(data, next_pos)
                if next_pos + size > end:
                    break
                value = bytes(data[next_pos:next_pos + size])
                next_pos += size
        except (ValueError, IndexError):
            break
        result.append((number, value))
        pos = next_pos
    return result

def _message_fields(data: bytes) -> list[tuple[int, int | bytes]]:
    return _parse(data, 0, len(data), range(1, 64))

def _text(value: int | bytes) -> str:
    assert isinstance(value, bytes)
    return value.decode()

def _enum(data: bytes, prefix: str, enums: dict[str, dict[int, str]]) -> None:
    name = ''
    values: dict[int, str] = {}
    for number, value in _message_fields(data):
        if number == 1:
            name = _text(value)
        elif number == 2 and isinstance(value, bytes):
            item = dict(_message_fields(value))
            number_value = item.get(2, 0)
            assert isinstance(number_value, int)
            # Aliased enum values keep the first declared name.
            values.setdefault(number_value, _text(item[1]))
    enums[f'{prefix}.{name}'] = values

def _message(data: bytes, prefix: str, messages: dict[str, MessageInfo], enums: dict[str, dict[int, str]]) -> None:
    name = ''
    items = []
    nested = []
    for number, value in _message_fields(data):
        if number == 1:
            name = _text(value)
        elif number == 2 and isinstance(value, bytes):
            item = dict(_message_fields(value))
            type_name = _text(item[6]) if 6 in item else ''
            oneof = item.get(9)
            assert oneof is None or isinstance(oneof, int)
            items.append(FieldInfo(_text(item[1]), int(item[3]), int(item[5]), type_name, oneof))
        elif number in (3, 4) and isinstance(value, bytes):
            nested.append((number, value))
    full = f'{prefix}.{name}'
    messages[full] = MessageInfo(tuple(items))
    for number, value in nested:
        if number == 3:
            _message(value, full, messages, enums)
        else:
            _enum(value, full, enums)

def descriptors(path: Path) -> tuple[dict[str, MessageInfo], dict[str, dict[int, str]]]:
    messages: dict[str, MessageInfo] = {}
    enums: dict[str, dict[int, str]] = {}
    with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        names = sorted({match[0] for match in re.finditer(rb'platforms/asic_sw/lib/deepsea/(?:gxc|common/isa)/[a-z_/0-9]+\.proto', data)})
        for name in names:
            for match in re.finditer(re.escape(name), data):
                start = match.start() - 2
                # A FileDescriptorProto starts with field 1 (name) and a one-byte length.
                if start < 0 or data[start] != 0x0a or data[start + 1] != len(name):
                    continue
                chunk = bytes(data[start:start + (1 << 24)])
                items = _parse(chunk, 0, len(chunk), range(1, 15))
                package = next(_text(value) for number, value in items if number == 2)
                for number, value in items:
                    if number == 4 and isinstance(value, bytes):
                        _message(value, '.' + package, messages, enums)
                    elif number == 5 and isinstance(value, bytes):
                        _enum(value, '.' + package, enums)
    return messages, enums

# ---------------------------------------------------------------- encoder

def enum_name(type_name: str) -> str:
    """去掉包名：`.asic_sw.deepsea.gxc.glc.isa.VectorY` 写作 `VectorY`，嵌套枚举保留外层消息名。"""
    parts = type_name.lstrip('.').split('.')
    package = max(index for index, part in enumerate(parts) if part[0].islower())
    return '.'.join(parts[package + 1:])

def encode(spec: GlSpec, bundles: list[bytes]) -> list[int | None]:
    """编码若干独立的 bundle，返回机器字；被拒绝的 bundle 由二分定位后记为 None。"""
    if not bundles:
        return []
    padded = bundles + [b''] * (-len(bundles) % hardware_target(spec.target).bundles_per_block)
    try:
        image = _program_proto(message([(1, bundle) for bundle in padded]), encode=True, target=spec.target)
    except RuntimeError:
        if len(bundles) == 1:
            return [None]
        middle = len(bundles) // 2
        return encode(spec, bundles[:middle]) + encode(spec, bundles[middle:])
    return [int.from_bytes(image[64 * index:64 * index + 64], 'little') for index in range(len(bundles))]

def bits(value: int) -> list[int]:
    return [index for index in range(WORD_BITS) if value >> index & 1]

def runs(mask: int) -> list[tuple[int, int]]:
    result = []
    index = 0
    while index < WORD_BITS:
        if mask >> index & 1:
            start = index
            while index < WORD_BITS and mask >> index & 1:
                index += 1
            result.append((start, index - start))
        else:
            index += 1
    return result

class Isa:
    def __init__(self, spec: GlSpec, messages: dict[str, MessageInfo], enums: dict[str, dict[int, str]]) -> None:
        self.spec = spec
        self.messages = messages
        self.enums = enums
        bundle = messages[f'.{PACKAGE}.{spec.bundle}']
        scalar = messages[f'.{PACKAGE}.{spec.scalar}']
        self.slot_messages = {}
        for slot, path in spec.slots.items():
            field = next(item for item in bundle.fields if item.number == path[0])
            if len(path) == 2:
                field = next(item for item in scalar.fields if item.number == path[1])
            self.slot_messages[slot] = messages[field.type_name]
        self.shared_kinds = {}
        for name, number, index in spec.shared:
            owner = next(item for item in bundle.fields if item.number == number)
            self.shared_kinds[name] = next(item.kind for item in messages[owner.type_name].fields if item.number == index)

    def slot_field(self, slot: str, name: str) -> FieldInfo | None:
        return next((item for item in self.slot_messages[slot].fields if item.name == name), None)

    def forms(self, slot: str) -> list[FieldInfo]:
        return [item for item in self.slot_messages[slot].fields if item.oneof is not None and item.kind == 11]

    def operands(self, form: FieldInfo) -> list[FieldInfo]:
        return [item for item in self.messages[form.type_name].fields if item.kind != 11]

    def enum_values(self, field: FieldInfo) -> list[int]:
        """保留值和无效值会使 formatter 终止进程，不作为可选编码。"""
        names = self.enums[field.type_name]
        return sorted(value for value, name in names.items() if value >= 0 and 'RESERVED' not in name and 'INVALID' not in name)

    def slot_bundle(self, slot: str, form: FieldInfo, values: dict[str, int], predicate: int = ALWAYS, inverted: int = 0) -> list[tuple[int, int | bytes]]:
        head: list[tuple[int, int | bytes]] = []
        predication = self.slot_field(slot, 'predication')
        inversion = self.slot_field(slot, 'predication_inversion')
        if predication is not None:
            head.append((predication.number, predicate))
        if inversion is not None:
            head.append((inversion.number, inverted))
        payload = message([(item.number, values.get(item.name, 0)) for item in self.operands(form)])
        body = message(head + [(form.number, payload)])
        path = self.spec.slots[slot]
        if len(path) == 2:
            return [(path[0], message([(path[1], body)]))]
        return [(path[0], body)]

def bundle(spec: GlSpec, parts: list[tuple[int, int | bytes]], shared: dict[str, int] | None = None) -> bytes:
    """合并各槽的字段；两个标量槽写入同一个标量子 bundle。``shared`` 给出非零的共享字段。"""
    merged: dict[int, list[tuple[int, int | bytes]]] = {}
    plain = []
    for number, value in parts:
        if number == 1:
            assert isinstance(value, bytes)
            merged.setdefault(1, []).extend((key, item) for key, _, item in fields(value))
        else:
            plain.append((number, value))
    items = plain + [(number, message(values)) for number, values in merged.items()]
    if shared:
        groups: dict[int, list[tuple[int, int | bytes]]] = {}
        for name, number, index in spec.shared:
            groups.setdefault(number, []).append((index, shared.get(name, 0)))
        items.extend((number, message(values)) for number, values in groups.items())
    return message(sorted(items, key=lambda item: item[0]))

@dataclass
class FormData:
    slot: str
    form: FieldInfo
    base: int
    layout: dict[str, tuple[int, int, str]]
    fixed_mask: int = 0
    fixed_value: int = 0
    excludes: tuple[str, ...] = ()
    # Bits above a field that the encoder writes as 0; they join the fixed bits.
    cleared: int = 0

    @property
    def field_mask(self) -> int:
        mask = 0
        for start, width, _ in self.layout.values():
            mask |= ((1 << width) - 1) << start
        return mask

def probe(isa: Isa) -> tuple[int, dict[str, tuple[int, int]], dict[str, tuple[int, int | None]], list[FormData], dict[str, list[str]]]:
    spec = isa.spec
    empty, = encode(spec, [b''])
    assert empty is not None
    shared = {}
    for name, number, index in spec.shared:
        # A bool field has one bit; the encoder stores any nonzero value as 1.
        words = encode(spec, [message([(number, message([(index, 1 << bit)]))]) for bit in range(1 if isa.shared_kinds[name] == 8 else 32)])
        changes = [bits(word ^ empty) for word in words if word is not None]
        changes = [item for item in changes if item]
        assert all(len(item) == 1 for item in changes), name
        start = changes[0][0]
        assert [item[0] for item in changes] == list(range(start, start + len(changes)))
        shared[name] = (start, len(changes))
    forms: list[FormData] = []
    rejected: dict[str, list[str]] = {}
    for slot in spec.slots:
        candidates = isa.forms(slot)
        bases = encode(spec, [bundle(spec, isa.slot_bundle(slot, form, {})) for form in candidates])
        for form, base in zip(candidates, bases):
            if base is None:
                rejected.setdefault(slot, []).append(form.name)
                continue
            requests: list[tuple[FieldInfo, int]] = []
            for operand in isa.operands(form):
                if operand.kind == 14:
                    requests.extend((operand, value) for value in isa.enum_values(operand))
                else:
                    requests.extend((operand, 1 << bit) for bit in range(1 if operand.kind == 8 else 32))
            words = encode(spec, [bundle(spec, isa.slot_bundle(slot, form, {operand.name: value})) for operand, value in requests])
            layout: dict[str, tuple[int, int, str]] = {}
            changed: dict[str, int] = {}
            for (operand, value), word in zip(requests, words):
                if word is None:
                    # The TEC encoder rejects integers wider than the field instead of truncating them.
                    assert operand.kind != 14, (slot, form.name, operand.name, value)
                    continue
                difference = word ^ base
                if operand.kind == 14:
                    changed[operand.name] = changed.get(operand.name, 0) | difference
                elif difference:
                    assert len(bits(difference)) == 1, (slot, form.name, operand.name)
                    changed.setdefault(operand.name, 0)
                    changed[operand.name] |= difference
            for operand in isa.operands(form):
                positions = bits(changed.get(operand.name, 0))
                assert positions, (slot, form.name, operand.name)
                start, width = positions[0], positions[-1] - positions[0] + 1
                enum = enum_name(operand.type_name) if operand.kind == 14 else ''
                layout[operand.name] = (start, width, enum)
            # Enum values are copied verbatim: every value is value << start.
            for (operand, value), word in zip(requests, words):
                start, width, _ = layout[operand.name]
                assert word is None or word ^ base == (value << start if operand.kind == 14 else word ^ base)
            forms.append(FormData(slot, form, base, layout))
    predicates = {}
    for slot in spec.slots:
        data = next(item for item in forms if item.slot == slot)
        words = encode(spec, [bundle(spec, isa.slot_bundle(slot, data.form, {}, predicate=value)) for value in range(NEVER)])
        mask = 0
        for word in words:
            assert word is not None
            mask |= word ^ data.base
        start = bits(mask)[0]
        assert bits(mask) == list(range(start, start + 4)), slot
        # The predicate value is copied verbatim; NEVER removes the whole slot.
        assert all(word is not None and word ^ data.base == (value ^ ALWAYS) << start for value, word in enumerate(words)), slot
        inversion = None
        if isa.slot_field(slot, 'predication_inversion') is not None:
            plain, inverted = encode(spec, [bundle(spec, isa.slot_bundle(slot, data.form, {}, predicate=3, inverted=value)) for value in (0, 1)])
            assert plain is not None and inverted is not None
            inversion, = bits(plain ^ inverted)
        predicates[slot] = (start, inversion)
    return empty, shared, predicates, forms, rejected

def probe_widths(isa: Isa, predicates: dict[str, tuple[int, int | None]], forms: list[FormData]) -> None:
    """找出 encoder 以 0 写入的字段高位。

    逐 bit 探测只能看到写成 1 的位。encoder 可能把较窄的字段写入更宽的操作数位段，高位写 0，例如 TEC 的向量掩码写入 VALU 6 位的 x 字段。对每个字段，另选一个可同时出现、覆盖其上方位段的其他槽字段，把后者置为全 1 后与本形式组合编码；被清零的连续高位记入 ``cleared``，由 ``assign_fixed`` 并入固定位。字段本身的位宽仍是可写入的取值范围。
    """
    spec = isa.spec
    requests = []
    for data in forms:
        for operand in isa.operands(data.form):
            start, width, _ = data.layout[operand.name]
            top = start + width
            best = None
            for partner in forms:
                if partner.slot == data.slot or frozenset((data.slot, partner.slot)) in spec.exclusive or data.slot in partner.excludes:
                    continue
                for other in isa.operands(partner.form):
                    other_start, other_width, _ = partner.layout[other.name]
                    if other_start < top < other_start + other_width and (best is None or other_start + other_width > best[3]):
                        best = (partner, other, other_width, other_start + other_width)
            if best is None:
                continue
            partner, other, other_width, _ = best
            partner_values = {**base_values(isa, partner), other.name: isa.enum_values(other)[-1] if other.kind == 14 else (1 << other_width) - 1}
            own_values = {**base_values(isa, data), operand.name: 0}
            parts = isa.slot_bundle(partner.slot, partner.form, partner_values)
            requests.append((data, operand.name, top, partner.layout[other.name], bundle(spec, parts), bundle(spec, parts + isa.slot_bundle(data.slot, data.form, own_values))))
    words = encode(spec, [item for request in requests for item in request[4:]])
    for index, (data, name, top, (other_start, other_width, _), _, _) in enumerate(requests):
        alone, combined = words[2 * index], words[2 * index + 1]
        if alone is None or combined is None:
            continue
        cleared = alone & ~combined
        extension = 0
        while top + extension < other_start + other_width and cleared >> (top + extension) & 1:
            extension += 1
        data.cleared |= ((1 << extension) - 1) << top

def predicate_mask(predicates: dict[str, tuple[int, int | None]], slot: str) -> int:
    start, inversion = predicates[slot]
    return (15 << start) | (1 << inversion if inversion is not None else 0)

def assign_fixed(spec: GlSpec, empty: int, predicates: dict[str, tuple[int, int | None]], forms: list[FormData]) -> None:
    """形式的固定位：本槽独占的位，加上本槽某些形式在共享位上写入的操作码。"""
    region = {slot: predicate_mask(predicates, slot) for slot in spec.slots}
    for data in forms:
        region[data.slot] |= (data.base ^ empty) | data.field_mask
    own = {}
    for slot in spec.slots:
        others = 0
        for other in spec.slots:
            if other != slot and frozenset((slot, other)) not in spec.exclusive:
                others |= region[other]
        own[slot] = region[slot] & ~others
    opcode_bits = dict.fromkeys(spec.slots, 0)
    for data in forms:
        opcode_bits[data.slot] |= (data.base ^ empty) & ~data.field_mask & ~predicate_mask(predicates, data.slot) & ~own[data.slot]
    for data in forms:
        data.fixed_mask = (own[data.slot] | opcode_bits[data.slot] | data.cleared) & ~data.field_mask & ~predicate_mask(predicates, data.slot)
        data.fixed_value = data.base & data.fixed_mask
        assert (data.base ^ empty) & ~data.fixed_mask & ~data.field_mask & ~predicate_mask(predicates, data.slot) == 0, (data.slot, data.form.name)

def base_values(isa: Isa, data: FormData) -> dict[str, int]:
    return {operand.name: isa.enum_values(operand)[0] if operand.kind == 14 else 0 for operand in isa.operands(data.form)}

def model_bits(isa: Isa, data: FormData, values: dict[str, int], predicates: dict[str, tuple[int, int | None]]) -> tuple[int, int] | None:
    """按字段表预测一个形式写入的 (mask, value)；字段与固定位冲突时返回 None。"""
    mask, value = data.fixed_mask, data.fixed_value
    for name, field_value in values.items():
        start, width, _ = data.layout[name]
        field_mask = ((1 << width) - 1) << start
        if (value ^ (field_value << start)) & mask & field_mask:
            return None
        mask |= field_mask
        value = (value & ~field_mask) | (field_value << start)
    start, _ = predicates[data.slot]
    mask |= predicate_mask(predicates, data.slot)
    value = (value & ~predicate_mask(predicates, data.slot)) | (ALWAYS << start)
    return mask, value

def probe_exclusions(isa: Isa, predicates: dict[str, tuple[int, int | None]], forms: list[FormData]) -> None:
    """找出与另一个槽同时出现时 encoder 总是拒绝的形式，例如占用相邻向量 ALU 的 64 位乘法。"""
    spec = isa.spec
    by_slot: dict[str, list[FormData]] = {}
    for data in forms:
        by_slot.setdefault(data.slot, []).append(data)
    requests = []
    for data in forms:
        own = model_bits(isa, data, base_values(isa, data), predicates)
        assert own is not None
        for other in spec.slots:
            if other == data.slot or frozenset((data.slot, other)) in spec.exclusive:
                continue
            tried = 0
            for partner in by_slot[other]:
                bits_other = model_bits(isa, partner, base_values(isa, partner), predicates)
                assert bits_other is not None
                if (own[1] ^ bits_other[1]) & own[0] & bits_other[0]:
                    continue
                parts = isa.slot_bundle(data.slot, data.form, base_values(isa, data)) + isa.slot_bundle(other, partner.form, base_values(isa, partner))
                requests.append((data, other, bundle(spec, parts)))
                tried += 1
                if tried == 3:
                    break
    words = encode(spec, [item for _, _, item in requests])
    accepted: dict[tuple[int, str], bool] = {}
    for (data, other, _), word in zip(requests, words):
        key = (id(data), other)
        accepted[key] = accepted.get(key, False) or word is not None
    for data in forms:
        data.excludes = tuple(other for other in spec.slots if accepted.get((id(data), other)) is False)

def validate(isa: Isa, empty: int, shared: dict[str, tuple[int, int]], predicates: dict[str, tuple[int, int | None]], forms: list[FormData], count: int) -> None:
    """随机组合各槽形式、字段和共享操作数，核对预测的机器字与 encoder 输出完全相同。"""
    spec = isa.spec
    rng = random.Random(0)
    by_slot: dict[str, list[FormData]] = {}
    for data in forms:
        by_slot.setdefault(data.slot, []).append(data)
    cases: list[tuple[bytes, int]] = []
    details = []
    while len(cases) < count:
        present: list[str] = []
        for slot in spec.slots:
            if rng.random() < 0.4 and not any(frozenset((slot, other)) in spec.exclusive for other in present):
                present.append(slot)
        mask = value = 0
        shared_values = {name: rng.getrandbits(width) if rng.random() < 0.5 else 0 for name, (_, width) in shared.items()}
        for name, part in shared_values.items():
            start, width = shared[name]
            mask |= ((1 << width) - 1) << start
            value |= part << start
        parts: list[tuple[int, int | bytes]] = []
        chosen: list[tuple[str, str, dict[str, int], int, int]] = []
        consistent = True
        for slot in present:
            data = rng.choice(by_slot[slot])
            if any(other in present for other in data.excludes):
                consistent = False
            slot_mask, slot_value = data.fixed_mask, data.fixed_value
            values = {}
            for operand in isa.operands(data.form):
                start, width, _ = data.layout[operand.name]
                field_value = rng.choice(isa.enum_values(operand)) if operand.kind == 14 else rng.getrandbits(width)
                values[operand.name] = field_value
                field_mask = ((1 << width) - 1) << start
                if (slot_value ^ (field_value << start)) & slot_mask & field_mask:
                    consistent = False
                slot_mask |= field_mask
                slot_value = (slot_value & ~field_mask) | (field_value << start)
            predicate = rng.choice([*range(ALWAYS), *[ALWAYS] * 10])
            start, inversion = predicates[slot]
            inverted = rng.randint(0, 1) if inversion is not None and predicate != ALWAYS else 0
            slot_mask |= predicate_mask(predicates, slot)
            slot_value = (slot_value & ~predicate_mask(predicates, slot)) | (predicate << start) | (inverted << inversion if inversion is not None else 0)
            if (value ^ slot_value) & mask & slot_mask:
                consistent = False
            mask |= slot_mask
            value = (value & ~slot_mask) | slot_value
            parts.extend(isa.slot_bundle(slot, data.form, values, predicate, inverted))
            chosen.append((slot, data.form.name, values, predicate, inverted))
        if consistent:
            cases.append((bundle(spec, parts, shared_values), (empty & ~mask) | value))
            details.append(chosen)
    words = encode(spec, [item for item, _ in cases])
    mismatches = sum(word != expected for word, (_, expected) in zip(words, cases))
    for word, (_, expected), chosen in zip(words, cases, details):
        if word != expected:
            print('MISMATCH', bits(word ^ expected) if word is not None else 'rejected', chosen, file=sys.stderr)
    if mismatches:
        raise SystemExit(f'{mismatches} of {count} random bundles differ from the predicted machine words')
    print(f'{count} random bundles match the predicted machine words', file=sys.stderr)

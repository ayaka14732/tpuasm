"""从已安装 libtpu 的 ISA descriptor、encoder 和 formatter 生成 TPU v6e TC 的字段表。

用法：PYTHONPATH=src python tools/generate_tpu_v6e_tc_isa.py [--output PATH]
需要已登记 tpu-v6e-tc 后端的 libtpu，不需要 TPU。默认覆盖 src/tpuasm/tpu_v6e_tc_isa_data.py。

步骤：从 libtpu 内嵌的 FileDescriptorProto 读取 bundle、槽、形式和枚举；逐字段、逐 bit 改变 encoder 输入，求出字段位置和每个形式的固定位；用随机指令包核对预测的机器字；最后在子进程中调用 formatter，记录助记符、操作数顺序和随字段变化的文本。formatter 遇到部分保留编码会终止进程，因此每批格式化都在子进程中运行，失败时二分定位。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import distribution
import itertools
import json
import math
import mmap
from pathlib import Path
import random
import re
import subprocess
import sys

from tpuasm._protobuf import fields, message, read_varint
from tpuasm.backends import select_backend
from tpuasm.printer import _load_native, _program_proto, _retain_libtpu
from tpuasm.targets import TPU_V6E_TC

TARGET = TPU_V6E_TC.identifier
PACKAGE = 'asic_sw.deepsea.gxc.glc.isa'
WORD_BITS = 512
# 物理槽在 TensorCoreBundle 中的字段路径；s0/s1 位于标量子 bundle，与 DMA 同属一个 oneof。
SLOT_PATHS = {
    's0': (1, 1),
    's1': (1, 2),
    'dma': (4,),
    'va0': (5,),
    'va1': (6,),
    'va2': (7,),
    'va3': (8,),
    'vst': (9,),
    'vld0': (10,),
    'vld1': (11,),
    'misc': (12,),
    'vx0': (13,),
    'vx1': (14,),
    'vr0': (15,),
    'vr1': (16,),
}
SHARED_MESSAGES = ((2, 'imm', 6), (3, 'vs', 4))
EXCLUSIVE = {frozenset(('s0', 'dma')), frozenset(('s1', 'dma'))}
ALWAYS, NEVER = 14, 15

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

def encode(bundles: list[bytes]) -> list[int | None]:
    """编码若干独立的 bundle，返回机器字；被拒绝的 bundle 由二分定位后记为 None。"""
    if not bundles:
        return []
    padded = bundles + [b''] * (-len(bundles) % TPU_V6E_TC.bundles_per_block)
    try:
        image = _program_proto(message([(1, bundle) for bundle in padded]), encode=True, target=TARGET)
    except RuntimeError:
        if len(bundles) == 1:
            return [None]
        middle = len(bundles) // 2
        return encode(bundles[:middle]) + encode(bundles[middle:])
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
    def __init__(self, messages: dict[str, MessageInfo], enums: dict[str, dict[int, str]]) -> None:
        self.messages = messages
        self.enums = enums
        bundle = messages[f'.{PACKAGE}.TensorCoreBundle']
        scalar = messages[f'.{PACKAGE}.ScalarSubBundle']
        self.slot_messages = {}
        for slot, path in SLOT_PATHS.items():
            field = next(item for item in bundle.fields if item.number == path[0])
            if len(path) == 2:
                field = next(item for item in scalar.fields if item.number == path[1])
            self.slot_messages[slot] = messages[field.type_name]

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
        path = SLOT_PATHS[slot]
        if len(path) == 2:
            return [(path[0], message([(path[1], body)]))]
        return [(path[0], body)]

def bundle(parts: list[tuple[int, int | bytes]], immediates: tuple[int, ...] = (), scalars: tuple[int, ...] = ()) -> bytes:
    """合并各槽的字段；两个标量槽写入同一个标量子 bundle。"""
    merged: dict[int, list[tuple[int, int | bytes]]] = {}
    plain = []
    for number, value in parts:
        if number == 1:
            assert isinstance(value, bytes)
            merged.setdefault(1, []).extend((key, item) for key, _, item in fields(value))
        else:
            plain.append((number, value))
    items = plain + [(number, message(values)) for number, values in merged.items()]
    if immediates:
        items.append((2, message([(index + 1, value) for index, value in enumerate(immediates)])))
    if scalars:
        items.append((3, message([(index + 1, value) for index, value in enumerate(scalars)])))
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

    @property
    def field_mask(self) -> int:
        mask = 0
        for start, width, _ in self.layout.values():
            mask |= ((1 << width) - 1) << start
        return mask

def probe(isa: Isa) -> tuple[int, dict[str, tuple[int, int]], dict[str, tuple[int, int | None]], list[FormData], dict[str, list[str]]]:
    empty, = encode([b''])
    assert empty is not None
    shared = {}
    for number, prefix, count in SHARED_MESSAGES:
        for index in range(count):
            words = encode([message([(number, message([(index + 1, 1 << bit)]))]) for bit in range(32)])
            changes = [bits(word ^ empty) for word in words if word is not None]
            changes = [item for item in changes if item]
            assert all(len(item) == 1 for item in changes), (prefix, index)
            start = changes[0][0]
            assert [item[0] for item in changes] == list(range(start, start + len(changes)))
            shared[f'{prefix}{index}'] = (start, len(changes))
    forms: list[FormData] = []
    rejected: dict[str, list[str]] = {}
    for slot in SLOT_PATHS:
        candidates = isa.forms(slot)
        bases = encode([bundle(isa.slot_bundle(slot, form, {})) for form in candidates])
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
            words = encode([bundle(isa.slot_bundle(slot, form, {operand.name: value})) for operand, value in requests])
            layout: dict[str, tuple[int, int, str]] = {}
            changed: dict[str, int] = {}
            for (operand, value), word in zip(requests, words):
                assert word is not None, (slot, form.name, operand.name, value)
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
                assert word is not None and word ^ base == (value << start if operand.kind == 14 else word ^ base)
            forms.append(FormData(slot, form, base, layout))
    predicates = {}
    for slot in SLOT_PATHS:
        data = next(item for item in forms if item.slot == slot)
        words = encode([bundle(isa.slot_bundle(slot, data.form, {}, predicate=value)) for value in range(NEVER)])
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
            plain, inverted = encode([bundle(isa.slot_bundle(slot, data.form, {}, predicate=3, inverted=value)) for value in (0, 1)])
            assert plain is not None and inverted is not None
            inversion, = bits(plain ^ inverted)
        predicates[slot] = (start, inversion)
    return empty, shared, predicates, forms, rejected

def predicate_mask(predicates: dict[str, tuple[int, int | None]], slot: str) -> int:
    start, inversion = predicates[slot]
    return (15 << start) | (1 << inversion if inversion is not None else 0)

def assign_fixed(empty: int, predicates: dict[str, tuple[int, int | None]], forms: list[FormData]) -> None:
    """形式的固定位：本槽独占的位，加上本槽某些形式在共享位上写入的操作码。"""
    region = {slot: predicate_mask(predicates, slot) for slot in SLOT_PATHS}
    for data in forms:
        region[data.slot] |= (data.base ^ empty) | data.field_mask
    own = {}
    for slot in SLOT_PATHS:
        others = 0
        for other in SLOT_PATHS:
            if other != slot and frozenset((slot, other)) not in EXCLUSIVE:
                others |= region[other]
        own[slot] = region[slot] & ~others
    opcode_bits = dict.fromkeys(SLOT_PATHS, 0)
    for data in forms:
        opcode_bits[data.slot] |= (data.base ^ empty) & ~data.field_mask & ~predicate_mask(predicates, data.slot) & ~own[data.slot]
    for data in forms:
        data.fixed_mask = (own[data.slot] | opcode_bits[data.slot]) & ~data.field_mask & ~predicate_mask(predicates, data.slot)
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
    by_slot: dict[str, list[FormData]] = {}
    for data in forms:
        by_slot.setdefault(data.slot, []).append(data)
    requests = []
    for data in forms:
        own = model_bits(isa, data, base_values(isa, data), predicates)
        assert own is not None
        for other in SLOT_PATHS:
            if other == data.slot or frozenset((data.slot, other)) in EXCLUSIVE:
                continue
            tried = 0
            for partner in by_slot[other]:
                bits_other = model_bits(isa, partner, base_values(isa, partner), predicates)
                assert bits_other is not None
                if (own[1] ^ bits_other[1]) & own[0] & bits_other[0]:
                    continue
                parts = isa.slot_bundle(data.slot, data.form, base_values(isa, data)) + isa.slot_bundle(other, partner.form, base_values(isa, partner))
                requests.append((data, other, bundle(parts)))
                tried += 1
                if tried == 3:
                    break
    words = encode([item for _, _, item in requests])
    accepted: dict[tuple[int, str], bool] = {}
    for (data, other, _), word in zip(requests, words):
        key = (id(data), other)
        accepted[key] = accepted.get(key, False) or word is not None
    for data in forms:
        data.excludes = tuple(other for other in SLOT_PATHS if accepted.get((id(data), other)) is False)

def validate(isa: Isa, empty: int, shared: dict[str, tuple[int, int]], predicates: dict[str, tuple[int, int | None]], forms: list[FormData], count: int) -> None:
    """随机组合各槽形式、字段和共享操作数，核对预测的机器字与 encoder 输出完全相同。"""
    rng = random.Random(0)
    by_slot: dict[str, list[FormData]] = {}
    for data in forms:
        by_slot.setdefault(data.slot, []).append(data)
    cases: list[tuple[bytes, int]] = []
    details = []
    while len(cases) < count:
        present = [slot for slot in SLOT_PATHS if rng.random() < 0.4]
        if 'dma' in present and ('s0' in present or 's1' in present):
            present.remove('dma')
        mask = value = 0
        immediates = tuple(rng.getrandbits(shared[f'imm{index}'][1]) if rng.random() < 0.5 else 0 for index in range(6))
        scalars = tuple(rng.getrandbits(shared[f'vs{index}'][1]) if rng.random() < 0.5 else 0 for index in range(4))
        for name, part in [*zip((f'imm{index}' for index in range(6)), immediates), *zip((f'vs{index}' for index in range(4)), scalars)]:
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
            cases.append((bundle(parts, immediates, scalars), (empty & ~mask) | value))
            details.append(chosen)
    words = encode([item for item, _ in cases])
    mismatches = sum(word != expected for word, (_, expected) in zip(words, cases))
    for word, (_, expected), chosen in zip(words, cases, details):
        if word != expected:
            print('MISMATCH', bits(word ^ expected) if word is not None else 'rejected', chosen, file=sys.stderr)
    if mismatches:
        raise SystemExit(f'{mismatches} of {count} random bundles differ from the predicted machine words')
    print(f'{count} random bundles match the predicted machine words', file=sys.stderr)

# ---------------------------------------------------------------- formatter

FORMAT_WORKER = '''
import ctypes, json, sys
libtpu = ctypes.CDLL(sys.argv[2])
native = ctypes.CDLL(sys.argv[1])
native.tpuasm_format.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_char_p, ctypes.c_size_t]
native.tpuasm_free.argtypes = [ctypes.c_void_p]
image = sys.stdin.buffer.read()
output, size, error = ctypes.c_void_p(), ctypes.c_size_t(), ctypes.create_string_buffer(4096)
if native.tpuasm_format(sys.argv[2].encode(), ctypes.create_string_buffer(image), len(image), ctypes.byref(output), ctypes.byref(size), error, len(error)):
    raise SystemExit(error.value.decode())
print(json.dumps(ctypes.string_at(output, size.value).decode().splitlines()))
native.tpuasm_free(output)
'''

@dataclass(frozen=True)
class Formatter:
    """在子进程中调用已编译的原生桥接；formatter 终止进程只影响该子进程。"""
    native: str
    libtpu: str
    empty: int

    def _run(self, words: list[int]) -> list[str] | None:
        padded = words + [self.empty] * (-len(words) % TPU_V6E_TC.bundles_per_block)
        image = b''.join(word.to_bytes(64, 'little') for word in padded)
        result = subprocess.run([sys.executable, '-c', FORMAT_WORKER, self.native, self.libtpu], input=image, capture_output=True)
        if result.returncode:
            return None
        return json.loads(result.stdout)[:len(words)]

    def aborts(self, words: list[int]) -> list[bool]:
        """逐个机器字判断 formatter 是否终止进程。"""
        if not words:
            return []
        if self._run(words) is not None:
            return [False] * len(words)
        if len(words) == 1:
            return [True]
        middle = len(words) // 2
        return self.aborts(words[:middle]) + self.aborts(words[middle:])

    def __call__(self, words: list[int]) -> list[str | None]:
        """formatter 终止进程或没有输出的机器字记为 None。"""
        if not words:
            return []
        texts = self._run(words)
        if texts is not None:
            return [None if text == '<no formatter output>' else text for text in texts]
        if len(words) == 1:
            return [None]
        middle = len(words) // 2
        return self(words[:middle]) + self(words[middle:])

def _split(text: str, separator: str) -> list[str]:
    depth = 0
    parts = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in '[(<':
            depth += 1
        elif char in '])>':
            depth -= 1
        elif depth == 0 and text.startswith(separator, index):
            parts.append(text[start:index].strip())
            index += len(separator)
            start = index
            continue
        index += 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]

def parse_format(text: str) -> tuple[str, tuple[str, ...]]:
    """把单槽 formatter 文本拆成助记符与按显示顺序排列的操作数（目的在前）。"""
    body = text.strip()
    assert body.startswith('{') and body.endswith('}'), text
    body = body[1:-1].strip()
    halves = _split(body, ' = ')
    left, right = (halves[0], halves[1]) if len(halves) == 2 else ('', halves[0])
    mnemonic, _, tail = right.partition(' ')
    destinations = [] if left in ('', '_') else _split(left, ', ')
    return mnemonic, tuple(destinations + _split(tail, ', '))

def small(operand: FieldInfo, width: int, isa: Isa) -> list[int] | None:
    """可枚举全部取值的小字段；寄存器号等宽字段返回 None。"""
    if operand.kind == 14:
        values = isa.enum_values(operand)
        return values if len(values) <= 16 else None
    return list(range(1 << width)) if width <= 2 else None

def format_base(isa: Isa, data: FormData) -> dict[str, int]:
    """格式化时的基础取值：宽寄存器字段各不相同，改变选择器时显示的寄存器随之变化。"""
    values = {}
    for index, operand in enumerate(isa.operands(data.form)):
        width = data.layout[operand.name][1]
        if operand.kind == 14:
            values[operand.name] = isa.enum_values(operand)[0]
        else:
            values[operand.name] = index % ((1 << width) - 1) + 1 if width > 2 else 0
    return values

def describe(isa: Isa, format_words: Formatter, forms: list[FormData], shared: dict[str, tuple[int, int]]) -> dict[tuple[str, str], dict[str, object]]:
    """用 formatter 求出助记符、目的与源的顺序、只影响助记符的字段及随小字段变化的记号。

    每个形式先以全部基础取值格式化，再逐个改变字段，比较哪些记号随之变化。一个字段只改变
    助记符时，枚举它的全部取值；只由一个小字段决定的记号记录每个取值的文本。
    """
    requests: list[tuple[int, int, str, int]] = []
    words: list[int] = []
    bases = []
    probes: dict[tuple[int, str], list[int]] = {}
    for position, data in enumerate(forms):
        base_values = format_base(isa, data)
        bases.append(base_values)

        def word(values: dict[str, int]) -> int:
            result = data.base
            for name, value in values.items():
                start, width, _ = data.layout[name]
                result = (result & ~(((1 << width) - 1) << start)) | (value << start)
            return result

        words.append(word(base_values))
        requests.append((position, len(words) - 1, '', 0))
        # The same operands with nonzero shared immediates and scalar lanes.
        lanes = word(base_values)
        for name, (start, width) in shared.items():
            lanes = (lanes & ~(((1 << width) - 1) << start)) | ((0x11111 * (int(name[-1]) + 1) if name.startswith('imm') else 20 + int(name[-1])) << start)
        words.append(lanes)
        requests.append((position, len(words) - 1, '*', 0))
        for operand in isa.operands(data.form):
            start, width, _ = data.layout[operand.name]
            options = small(operand, width, isa)
            if options is None:
                options = [value for value in (5, 3) if value < 1 << width] if operand.kind != 14 else isa.enum_values(operand)[1:3]
            probes[(position, operand.name)] = options
            for value in options:
                if value == base_values[operand.name]:
                    continue
                words.append(word({**base_values, operand.name: value}))
                requests.append((position, len(words) - 1, operand.name, value))
    texts = format_words(words)
    result: dict[tuple[str, str], dict[str, object]] = {}
    grouped: dict[int, list[tuple[str, int, str | None]]] = {}
    for position, index, name, value in requests:
        grouped.setdefault(position, []).append((name, value, texts[index]))
    for position, data in enumerate(forms):
        entries = grouped[position]
        base_text = entries[0][2]
        record: dict[str, object] = {'mnemonic': None, 'tokens': (), 'mnemonic_fields': (), 'owners': {}, 'implicit': ()}
        result[(data.slot, data.form.name)] = record
        if base_text is None:
            continue
        mnemonic, tokens = parse_format(base_text)
        record['mnemonic'] = mnemonic
        record['tokens'] = tokens
        owners: dict[int, list[str]] = {}
        mnemonic_fields = []
        implicit = []
        for name, value, text in entries[1:]:
            if text is None:
                continue
            other_mnemonic, other_tokens = parse_format(text)
            if name == '*':
                implicit = [index for index, (left, right) in enumerate(zip(tokens, other_tokens)) if left != right] if len(other_tokens) == len(tokens) else list(range(len(tokens)))
                continue
            if other_mnemonic != mnemonic and name not in mnemonic_fields:
                mnemonic_fields.append(name)
            if len(other_tokens) != len(tokens):
                continue
            for index, (left, right) in enumerate(zip(tokens, other_tokens)):
                if left != right and name not in owners.setdefault(index, []):
                    owners[index].append(name)
        record['implicit'] = tuple(implicit)
        record['mnemonic_fields'] = tuple(mnemonic_fields)
        record['owners'] = {index: tuple(names) for index, names in owners.items()}
    return result

def formatter_aborts(isa: Isa, format_words: Formatter, forms: list[FormData]) -> dict[str, tuple[int, ...]]:
    """每个枚举中使 formatter 终止进程的取值。

    先以每个枚举的第一个字段核对全部取值，再以其余形式、字段核对剩下的取值；同一形式在各槽中的副本只核对一次。
    """
    fields: dict[str, list[tuple[FormData, FieldInfo]]] = {}
    seen = set()
    for data in forms:
        if data.form.name in seen and data.slot not in ('s0', 'dma'):
            continue
        seen.add(data.form.name)
        for operand in isa.operands(data.form):
            if operand.kind == 14:
                fields.setdefault(data.layout[operand.name][2], []).append((data, operand))

    def probe(pairs: list[tuple[FormData, FieldInfo, int]]) -> list[tuple[str, int]]:
        words = []
        for data, operand, value in pairs:
            word = data.base
            for name, field_value in {**format_base(isa, data), operand.name: value}.items():
                start, width, _ = data.layout[name]
                word = (word & ~(((1 << width) - 1) << start)) | (field_value << start)
            words.append(word)
        return [(data.layout[operand.name][2], value) for (data, operand, value), aborted in zip(pairs, format_words.aborts(words)) if aborted]

    result: dict[str, set[int]] = {}
    first = [(users[0][0], users[0][1], value) for users in fields.values() for value in sorted(isa.enums[users[0][1].type_name]) if value >= 0]
    for enum, value in probe(first):
        result.setdefault(enum, set()).add(value)
    rest = [(data, operand, value) for enum, users in fields.items() for data, operand in users[1:] for value in sorted(isa.enums[operand.type_name]) if value >= 0 and value not in result.get(enum, set())]
    for enum, value in probe(rest):
        result.setdefault(enum, set()).add(value)
    return {enum: tuple(sorted(values)) for enum, values in sorted(result.items())}

def token_tables(isa: Isa, format_words: Formatter, forms: list[FormData], described: dict[tuple[str, str], dict[str, object]]) -> dict[tuple[str, str], dict[int, tuple[tuple[str, ...], tuple[tuple[tuple[int, ...], str], ...]]]]:
    """只由小字段决定、又不改变助记符的记号：枚举这些字段的全部组合，记录每个组合的文本。"""
    requests = []
    words = []
    for data in forms:
        key = (data.slot, data.form.name)
        record = described[key]
        owners = record['owners']
        mnemonic_fields = record['mnemonic_fields']
        assert isinstance(owners, dict) and isinstance(mnemonic_fields, tuple)
        operands = {item.name: item for item in isa.operands(data.form)}
        for index, names in owners.items():
            options = [small(operands[name], data.layout[name][1], isa) for name in names]
            if any(item is None for item in options) or set(names) & set(mnemonic_fields):
                continue
            # Unit and register-group selectors; selector enums with many values are operands, not fixed text.
            if math.prod(len(item) for item in options if item is not None) > 8:
                continue
            for combination in itertools.product(*[item for item in options if item is not None]):
                values = {**format_base(isa, data), **dict(zip(names, combination))}
                word = data.base
                for name, value in values.items():
                    start, width, _ = data.layout[name]
                    word = (word & ~(((1 << width) - 1) << start)) | (value << start)
                words.append(word)
                requests.append((key, index, names, combination))
    texts = format_words(words)
    result: dict[tuple[str, str], dict[int, list[tuple[tuple[int, ...], str]]]] = {}
    fields_of: dict[tuple[tuple[str, str], int], tuple[str, ...]] = {}
    for (key, index, names, combination), text in zip(requests, texts):
        fields_of[(key, index)] = names
        if text is None:
            continue
        _, tokens = parse_format(text)
        result.setdefault(key, {}).setdefault(index, []).append((combination, tokens[index]))
    return {key: {index: (fields_of[(key, index)], tuple(entries)) for index, entries in tables.items()} for key, tables in result.items()}

def mnemonic_variants(isa: Isa, format_words: Formatter, forms: list[FormData], described: dict[tuple[str, str], dict[str, object]]) -> dict[tuple[str, str], tuple[tuple[tuple[int, ...], str], ...]]:
    """枚举只影响助记符的字段的全部组合，记录每个组合的 formatter 助记符。"""
    requests = []
    words = []
    for data in forms:
        names = described[(data.slot, data.form.name)]['mnemonic_fields']
        assert isinstance(names, tuple)
        options = []
        for name in names:
            operand = next(item for item in isa.operands(data.form) if item.name == name)
            values = small(operand, data.layout[name][1], isa)
            if values is not None:
                options.append(values)
        # A wide selector that switches the mnemonic (register or immediate move) is left to the signature rules.
        if not names or len(options) != len(names):
            continue
        for combination in itertools.product(*options):
            value = data.base
            for operand in isa.operands(data.form):
                start, width, _ = data.layout[operand.name]
                field_value = combination[names.index(operand.name)] if operand.name in names else format_base(isa, data)[operand.name]
                value = (value & ~(((1 << width) - 1) << start)) | (field_value << start)
            words.append(value)
            requests.append(((data.slot, data.form.name), combination))
    texts = format_words(words)
    result: dict[tuple[str, str], list[tuple[tuple[int, ...], str]]] = {}
    for (key, combination), text in zip(requests, texts):
        if text is not None:
            result.setdefault(key, []).append((combination, parse_format(text)[0]))
    return {key: tuple(value) for key, value in result.items()}

# ---------------------------------------------------------------- output

def render(empty: int, shared: dict[str, tuple[int, int]], predicates: dict[str, tuple[int, int | None]], isa: Isa, forms: list[FormData], rejected: dict[str, list[str]], described: dict[tuple[str, str], dict[str, object]], tables: dict[tuple[str, str], dict[int, tuple[tuple[str, ...], tuple[tuple[tuple[int, ...], str], ...]]]], variants: dict[tuple[str, str], tuple[tuple[tuple[int, ...], str], ...]], aborts: dict[str, tuple[int, ...]], version: str) -> str:
    enum_names = sorted({enum for data in forms for _, _, enum in data.layout.values() if enum})
    enum_types = {enum: next(item.type_name for data in forms for item in isa.operands(data.form) if data.layout[item.name][2] == enum) for enum in enum_names}
    layouts: list[tuple[tuple[str, int, int, int, str], ...]] = []
    lines = [
        f'"""TPU v6e TC 字段位置、指令形式与 formatter 记号；由 tools/generate_tpu_v6e_tc_isa.py 从 libtpu {version} 生成，不要手工修改。',
        '',
        '位编号相对一个 little-endian 64 字节 bundle。字段名、字段号和枚举来自 ISA descriptor，位位置和固定位来自 encoder，助记符和记号来自 formatter；不含程序样例或运行时地址。',
        '"""',
        'from __future__ import annotations',
        '',
        f'EMPTY_WORD = {empty:#x}',
        '# 物理槽：bundle 内的字段路径、谓词起始 bit、取反 bit（没有取反字段时为 None）。',
        'SLOTS: dict[str, tuple[tuple[int, ...], int, int | None]] = {',
    ]
    for slot, path in SLOT_PATHS.items():
        start, inversion = predicates[slot]
        lines.append(f'    {slot!r}: ({path!r}, {start}, {inversion!r}),')
    lines.extend(('}', '# 共享操作数：名称、所在 bundle 字段号、消息内字段号、起始 bit、位宽。', 'SHARED_FIELDS: tuple[tuple[str, int, int, int, int], ...] = ('))
    for number, prefix, count in SHARED_MESSAGES:
        for index in range(count):
            start, width = shared[f'{prefix}{index}']
            lines.append(f'    ({prefix + str(index)!r}, {number}, {index + 1}, {start}, {width}),')
    lines.extend((')', '# descriptor 中的枚举；同值别名只保留首个名称。', 'ENUMS: dict[str, dict[int, str]] = {'))
    for enum in enum_names:
        values = isa.enums[enum_types[enum]]
        lines.append(f'    {enum!r}: {{')
        lines.extend(f'        {value}: {name!r},' for value, name in sorted(values.items()))
        lines.append('    },')
    lines.extend(('}',))
    form_lines = []
    for data in forms:
        layout = tuple((operand.name, operand.number, *data.layout[operand.name]) for operand in isa.operands(data.form))
        if layout not in layouts:
            layouts.append(layout)
        record = described[(data.slot, data.form.name)]
        fixed = tuple((start, width, data.fixed_value >> start & ((1 << width) - 1)) for start, width in runs(data.fixed_mask))
        form_lines.append(f'    ({data.slot!r}, {data.form.number}, {data.form.name!r}, {layouts.index(layout)}, {fixed!r}, {data.excludes!r}, {record["mnemonic"]!r}, {record["tokens"]!r}),')
    lines.append('# 字段组：每项为 (字段名, 字段号, 起始 bit, 位宽, 枚举名)。')
    lines.append('FIELD_LAYOUTS: tuple[tuple[tuple[str, int, int, int, str], ...], ...] = (')
    for layout in layouts:
        lines.append(f'    {layout!r},')
    lines.append(')')
    lines.append('# 形式：槽、oneof 字段号、descriptor 名称、字段组编号、固定位 (起始 bit, 位宽, 取值)、encoder 不允许同时出现的其他槽、formatter 助记符、formatter 操作数记号（目的在前；字段取基础值）。formatter 没有输出时后两项为 None 和 ()。')
    lines.append('INSTRUCTION_FORMS: tuple[tuple[str, int, str, int, tuple[tuple[int, int, int], ...], tuple[str, ...], str | None, tuple[str, ...]], ...] = (')
    lines.extend(form_lines)
    lines.append(')')
    lines.append('# 每个记号随哪些字段变化：(槽, 形式) -> {记号序号: 字段名元组}。')
    lines.append('TOKEN_FIELDS: dict[tuple[str, str], dict[int, tuple[str, ...]]] = {')
    for data in forms:
        owners = described[(data.slot, data.form.name)]['owners']
        assert isinstance(owners, dict)
        if owners:
            lines.append(f'    ({data.slot!r}, {data.form.name!r}): {dict(sorted(owners.items()))!r},')
    lines.append('}')
    lines.append('# 字段取基础值时仍随共享立即数或标量操作数变化的记号：(槽, 形式) -> 记号序号元组。')
    lines.append('SHARED_TOKENS: dict[tuple[str, str], tuple[int, ...]] = {')
    for data in forms:
        implicit = described[(data.slot, data.form.name)]['implicit']
        if implicit:
            lines.append(f'    ({data.slot!r}, {data.form.name!r}): {implicit!r},')
    lines.append('}')
    lines.append('# 只由小字段决定的记号的全部文本：(槽, 形式) -> {记号序号: (字段名元组, ((取值元组, 文本), ...))}。')
    lines.append('TOKEN_TABLES: dict[tuple[str, str], dict[int, tuple[tuple[str, ...], tuple[tuple[tuple[int, ...], str], ...]]]] = {')
    for data in forms:
        key = (data.slot, data.form.name)
        if key in tables:
            lines.append(f'    {key!r}: {dict(sorted(tables[key].items()))!r},')
    lines.append('}')
    lines.append('# 改变 formatter 助记符的字段及其全部取值组合：(槽, 形式) -> (字段名元组, ((取值元组, 助记符), ...))。')
    lines.append('MNEMONIC_VARIANTS: dict[tuple[str, str], tuple[tuple[str, ...], tuple[tuple[tuple[int, ...], str], ...]]] = {')
    for data in forms:
        key = (data.slot, data.form.name)
        if key in variants:
            names = described[key]['mnemonic_fields']
            lines.append(f'    {key!r}: ({names!r}, {variants[key]!r}),')
    lines.append('}')
    lines.append('# 使 formatter 终止进程的枚举取值：汇编与导出都不使用它们，原生校验前先拒绝。')
    lines.append(f'FORMATTER_ABORTS: dict[str, tuple[int, ...]] = {aborts!r}')
    lines.append('# encoder 以任何操作数都拒绝的 (槽, 形式)：这些形式不能在该槽发射。')
    lines.append('REJECTED: dict[str, tuple[str, ...]] = {')
    for slot, names in rejected.items():
        lines.append(f'    {slot!r}: {tuple(names)!r},')
    lines.append('}')
    return '\n'.join(lines) + '\n'

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'src' / 'tpuasm' / 'tpu_v6e_tc_isa_data.py')
    parser.add_argument('--validate', type=int, default=20000, metavar='N', help='number of random bundles checked against the encoder')
    args = parser.parse_args()
    backend, path = select_backend(TARGET)
    _retain_libtpu(path)
    native = _load_native(backend)
    messages, enums = descriptors(path)
    isa = Isa(messages, enums)
    empty, shared, predicates, forms, rejected = probe(isa)
    assign_fixed(empty, predicates, forms)
    probe_exclusions(isa, predicates, forms)
    validate(isa, empty, shared, predicates, forms, args.validate)
    formatter = Formatter(native._name, str(path), empty)
    described = describe(isa, formatter, forms, shared)
    tables = token_tables(isa, formatter, forms, described)
    variants = mnemonic_variants(isa, formatter, forms, described)
    aborts = formatter_aborts(isa, formatter, forms)
    version = distribution('libtpu').version
    args.output.write_text(render(empty, shared, predicates, isa, forms, rejected, described, tables, variants, aborts, version), encoding='utf-8')
    print(args.output)

if __name__ == '__main__':
    main()

"""从已安装 libtpu 的 ISA descriptor、TEC encoder、LLVM TPU printer 与 TEC emitter 生成 TPU v6e TEC 的字段表与指令语法。

用法：PYTHONPATH=src python tools/generate_tpu_v6e_tec_isa.py [--output PATH]
需要已登记 tpu-v6e-tec 后端的 libtpu 与 g++，不需要 TPU。默认覆盖 src/tpuasm/tpu_v6e_tec_isa_data.py。

步骤：字段位置、固定位与互斥槽的探测与 v6e TC 相同（ghostlite_isa.py）。指令语法取自编译器：libtpu 把 SparseCore kernel 编译为 LLVM MCInst，再由 emitter 转成 SparseCoreTecBundle。工具为每个 LLVM opcode 与 SparseCoreMCSlot 构造 MCInst，交给 emitter 并用 TPUInstPrinter 打印；逐个改变寄存器、立即数和 selector 编码，比较 emitter 写入的字段与打印文本的变化，得到每条指令的助记符、操作数顺序以及每个操作数对应的字段。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from importlib.metadata import distribution
import itertools
from pathlib import Path
import re
import sys
from typing import Any

from ghostlite_isa import FormData, GlSpec, Isa, assign_fixed, descriptors, probe, probe_exclusions, probe_widths, runs, validate
from tec_llvm import EXPRESSION, IMMEDIATE, REGISTER, Llvm, McInst
from tpuasm._protobuf import fields
from tpuasm.backends import select_backend
from tpuasm.targets import TPU_V6E_TEC

SPEC = GlSpec(
    target=TPU_V6E_TEC.identifier,
    bundle='SparseCoreTecBundle',
    scalar='TecScalarSubBundle',
    # 物理槽在 SparseCoreTecBundle 中的字段路径；s0/s1 位于标量子 bundle，与 DMA、stream 同属一个 oneof。
    slots={
        's0': (1, 1),
        's1': (1, 2),
        'dma': (2,),
        'misc': (3,),
        'va0': (4,),
        'va1': (5,),
        'va2': (6,),
        'vld': (9,),
        'vst': (10,),
        'stream': (11,),
        'vr': (12,),
        'vx': (13,),
    },
    shared=tuple((f'imm{index}', 8, index + 1) for index in range(6)) + tuple(item for index in range(4) for item in ((f'vs{index}', 7, 2 * index + 1), (f'vs{index}_used', 7, 2 * index + 2))),
    exclusive=frozenset(frozenset(pair) for pair in (('s0', 'dma'), ('s1', 'dma'), ('s0', 'stream'), ('s1', 'stream'), ('dma', 'stream'))),
)

# ---------------------------------------------------------------- emitter

@dataclass(frozen=True)
class Decoded:
    """emitter 为一个槽写入的内容：形式与字段，以及 bundle 级共享字段。"""
    slot: str
    form: str
    values: dict[str, int]
    shared: dict[str, int]

class Emitter:
    """把 MCInst 交给 LLVM printer 与 TEC emitter，并把 emitter 的 protobuf 读成槽、形式和字段。"""

    def __init__(self, llvm: Llvm, isa: Isa) -> None:
        self.llvm = llvm
        self.isa = isa
        self.paths = {path: slot for slot, path in SPEC.slots.items()}
        self.shared_names = {(number, index): name for name, number, index in SPEC.shared}
        self.class_of_register: dict[int, int] = {}
        for index, registers in enumerate(llvm.class_registers):
            for register in registers:
                self.class_of_register.setdefault(register, index)

    def run(self, insts: list[McInst]) -> list[tuple[str, list[Decoded] | None, str]]:
        """每条 MCInst 单独成 bundle：(打印文本, 解码结果或 None, emitter 消息)。"""
        results = self.llvm.run([[inst] for inst in insts])
        return [(result.text, self.decode(result.bundle) if result.bundle is not None else None, result.message) for result in results]

    def decode(self, bundle: bytes) -> list[Decoded]:
        shared: dict[str, int] = {}
        slots: list[tuple[str, bytes]] = []
        for number, _, value in fields(bundle):
            assert isinstance(value, bytes)
            if number in (7, 8):
                for index, _, item in fields(value):
                    if (number, index) in self.shared_names:
                        assert isinstance(item, int)
                        shared[self.shared_names[(number, index)]] = item
            elif number == 1:
                slots.extend((self.paths[(1, index)], item) for index, _, item in fields(value) if isinstance(item, bytes))
            else:
                slots.append((self.paths[(number,)], value))
        result = []
        for slot, payload in slots:
            forms = {form.number: form for form in self.isa.forms(slot)}
            form_name = ''
            values: dict[str, int] = {}
            for number, _, value in fields(payload):
                if number in forms:
                    assert isinstance(value, bytes)
                    form = forms[number]
                    form_name = form.name
                    names = {operand.number: operand.name for operand in self.isa.operands(form)}
                    values = {names[key]: item for key, _, item in fields(value) if key in names and isinstance(item, int)}
            result.append(Decoded(slot, form_name, values, {name: value for name, value in shared.items() if value}))
        return result

# ---------------------------------------------------------------- baseline search

SIMPLE_REGISTER = re.compile(r'[a-z]+[0-9]+')

def usable_registers(llvm: Llvm, register_class: int) -> list[int]:
    """寄存器类中名称为字母加编号的寄存器；隐含队列寄存器（如 (erf)）另行处理。"""
    registers = llvm.class_registers[register_class]
    simple = [register for register in registers if SIMPLE_REGISTER.fullmatch(llvm.register_names[register]) and ',' not in llvm.register_names[register]]
    return simple or list(registers)

@dataclass
class Candidate:
    """一条 MCInst 的可变部分：操作数与每个操作数经由的 VS 槽。"""
    opcode: int
    slot_flags: int
    operands: list[tuple[int, int, int]]
    lanes: dict[int, int] = field(default_factory=dict)
    # Operands the search must not change, e.g. an enumerated shape operand.
    locked: frozenset[int] = frozenset()

    def inst(self) -> McInst:
        lane_flags = sum(lane << 3 * index for index, lane in self.lanes.items())
        return McInst(self.opcode, self.slot_flags, tuple(self.operands), lane_flags)

def mode_seeds(name: str) -> tuple[int, ...]:
    """stream 与 DMA 指令操作数 0 的模式字：两端的地址空间、方向和 circular buffer 等选择。

    从 0 出发的搜索到不了编译器使用的组合，所以以编译器写出的模式字为起点，它们取自编译 SparseCore kernel 时交给 emitter 的 MCInst。每个起点各自搜索可用的槽与其余操作数，形状展开再从起点逐段改变其中的位。几段位要同时改变、其余操作数也随之不同的组合各需要一个起点，例如 stream 的对端从带偏移寄存器的 HBM 换成 SPMEM。
    """
    if name.startswith('STREAM_'):
        # Bits 0..2 select the stream kind; the opcode name alone does not.
        mode = (1 if '_STRIDED_' in name else 2 if '_INDIRECT_' in name else 0) | (0x8 if '_SCATTER_' in name else 0) | (0x20 if '_cb_upd_' in name else 0x10 if '_cb_' in name else 0)
        # TileSPMEM with HBM (4-byte granules, offset register) or with SPMEM.
        return (0x640000c0 | mode, 0x68000000 | mode)
    if name.startswith('DMA_'):
        # TIMEM from HBM, and HBM from SPMEM.
        return (0x60000000, 0x03000000)
    return ()

def initial_candidate(llvm: Llvm, opcode: int, slot_flags: int, seed: int = 0) -> Candidate:
    operands = []
    for register_class, flags, _ in llvm.operands(opcode):
        if register_class >= 0 and flags & 2:
            operands.append((REGISTER, llvm.registers['always'], 0))
        elif register_class >= 0:
            registers = usable_registers(llvm, register_class)
            operands.append((REGISTER, registers[1] if len(registers) > 1 else registers[0], 0))
        else:
            operands.append((IMMEDIATE, 0, 0))
    seeds = mode_seeds(llvm.names[opcode])
    if not seeds:
        return Candidate(opcode, slot_flags, operands)
    operands[0] = (IMMEDIATE, seeds[seed], 0)
    return Candidate(opcode, slot_flags, operands, locked=frozenset({0}))

def constant_encoding(enums: dict[str, dict[int, str]], enum: str) -> int | None:
    """emitter 报告的 selector 枚举中表示 0 的取值。"""
    for full, values in enums.items():
        if full.rsplit('.', 1)[-1] != enum:
            continue
        for value, name in values.items():
            if name.endswith('_ZERO') and 'RESERVED' not in name:
                return value
    return None

def adjust(llvm: Llvm, enums: dict[str, dict[int, str]], candidate: Candidate, message: str, tried: set[tuple[int, str]]) -> bool:
    """按 emitter 的报错改一个操作数；没有适用的规则时返回 False。"""
    operands = candidate.operands
    info = llvm.operands(candidate.opcode)

    def first(predicate: Any) -> int | None:
        for index, operand in enumerate(operands):
            if not info[index][1] & 2 and index not in candidate.locked and predicate(index, operand) and (index, message) not in tried:
                tried.add((index, message))
                return index
        return None

    if 'Invalid base encoding for VS' in message or 'vs_encoding ==' in message:
        # A named lane (VS0..VS3) goes to a scalar register operand; VSN means no lane.
        match = re.search(r'VSEncoding::VS([0-3N])', message)
        if match and match[1] == 'N':
            index = first(lambda index, operand: index in candidate.lanes)
            if index is None:
                return False
            del candidate.lanes[index]
            return True
        lane = int(match[1]) + 1 if match else 1
        # Register-or-immediate operands (types 27..29) are the ones that usually go through a lane.
        index = first(lambda index, operand: info[index][2] in (27, 28, 29) and candidate.lanes.get(index) != lane)
        if index is None:
            index = first(lambda index, operand: operand[0] == REGISTER and llvm.register_names[operand[1]].startswith('s') and candidate.lanes.get(index) != lane)
        if index is None:
            return False
        if operands[index][0] != REGISTER:
            operands[index] = (REGISTER, llvm.registers['s1'], 0)
        candidate.lanes[index] = lane
        return True
    if match := re.match(r'Invalid value for ([A-Za-z_]+): ([0-9]+)', message):
        encoding = constant_encoding(enums, match[1])
        index = first(lambda index, operand: operand[0] == EXPRESSION and operand[2] >> 8 & 0xff == int(match[2]))
        if index is None or encoding is None:
            return False
        operands[index] = (EXPRESSION, 0, encoding << 8)
        return True
    if re.search(r'isExpr\(\)|expression or register|Invalid operand for Vector Y|must be Sreg or Immediate', message):
        # Selector operands (types 19 and 23) come first; the message does not name the operand.
        index = first(lambda index, operand: operand[0] == IMMEDIATE and info[index][2] in (19, 23))
        if index is None:
            index = first(lambda index, operand: operand[0] == IMMEDIATE)
        if index is None:
            return False
        operands[index] = (EXPRESSION, 0, 0)
        return True
    if match := re.search(r'regno (?:<=|>=) llvm::TPU::P(?:13|0) \(([0-9]+) vs', message):
        index = first(lambda index, operand: operand[0] == REGISTER and operand[1] == int(match[1]))
        if index is None:
            return False
        operands[index] = (REGISTER, llvm.registers['p1'], 0)
        return True
    if match := re.search(r'regno == llvm::TPU::([A-Z0-9_]+) \(([0-9]+) vs', message):
        target = llvm.registers.get(f'({match[1].lower()})')
        index = first(lambda index, operand: operand[0] == REGISTER and operand[1] == int(match[2]))
        if index is None or target is None:
            return False
        operands[index] = (REGISTER, target, 0)
        return True
    if re.search(r'Invalid [A-Z]+ instance', message):
        # An implicit queue register such as (xrf0) must be the first register of its class.
        index = first(lambda index, operand: operand[0] == REGISTER and llvm.register_names[operand[1]].startswith('(') and operand[1] != llvm.class_registers[info[index][0]][0])
        if index is None:
            return False
        operands[index] = (REGISTER, llvm.class_registers[info[index][0]][0], 0)
        return True
    if 'non-zero HBM indices' in message:
        # Operands that already go through a lane are strides or offsets, not the memory index.
        index = first(lambda index, operand: operand[0] == REGISTER and info[index][2] in (27, 28, 29) and index not in candidate.lanes)
        if index is not None:
            operands[index] = (IMMEDIATE, 0, 0)
            return True
    if match := re.search(r'Expected immediate for [A-Za-z]+ in operand ([0-9]+)', message):
        index = int(match[1])
        if operands[index][0] == IMMEDIATE:
            return False
        operands[index] = (IMMEDIATE, 0, 0)
        tried.add((index, 'register'))
        return True
    if re.search(r'Unsupported .* type|Does not support|IsIovaDma|isImm\(\)', message):
        # The emitter rejects an enumerated immediate such as a memory type pair; count it up.
        for index, operand in enumerate(operands):
            if operand[0] == IMMEDIATE and not info[index][1] & 2 and index not in candidate.locked and operand[1] < 255 and (index, 'count') not in tried:
                operands[index] = (IMMEDIATE, operand[1] + 1, 0)
                if operand[1] + 1 == 255:
                    tried.add((index, 'count'))
                return True
        return False
    if 'isReg()' in message:
        # Operand types 27..29 take a register or an immediate, 19 and 23 a selector; packed
        # immediates (type 24) such as stream configuration words never become registers.
        index = first(lambda index, operand: operand[0] != REGISTER and info[index][2] in (27, 28, 29) and (index, 'register') not in tried)
        if index is None:
            index = first(lambda index, operand: operand[0] != REGISTER and info[index][2] in (19, 23) and (index, 'register') not in tried)
        if index is None:
            return False
        operands[index] = (REGISTER, llvm.registers['s1'], 0)
        return True
    return False

# SparseCoreMCSlot names in emitter CHECK messages and their slot flag bits.
SLOT_BITS = {'S0': 1, 'S1': 2, 'SM': 4, 'V0': 16, 'V1': 32, 'V2': 64, 'VLD': 256, 'VST': 1024, 'VEX0': 4096, 'VRES': 16384}
SCALAR_FLAGS = (2, 4)
VECTOR_FLAGS = (16, 32, 64, 256, 1024, 4096, 16384)

def baselines(emitter: Emitter, enums: dict[str, dict[int, str]]) -> dict[tuple[int, int, int], tuple[Candidate, str, Decoded]]:
    """为每个 (opcode, 槽位, 模式字起点) 找一条 emitter 接受、且只占一个槽的 MCInst。

    每个 opcode 先以 S0 槽尝试。emitter 对其他标量 opcode 以 CHECK 失败终止并指出所需槽位，对向量 opcode 报告不支持；据此只尝试可能的槽位，减少子进程终止的次数。
    """
    llvm = emitter.llvm
    pending = {
        (opcode, 1, seed): initial_candidate(llvm, opcode, 1, seed)
        for opcode, name in enumerate(llvm.names)
        # Generic MIR opcodes and TensorCore/BarnaCore variants never reach the TEC emitter.
        if not name.startswith(('G_', 'tc_', 'bc'))
        for seed in range(max(1, len(mode_seeds(name))))
    }
    tried: dict[tuple[int, int, int], set[tuple[int, str]]] = {key: set() for key in pending}
    found: dict[tuple[int, int, int], tuple[Candidate, str, Decoded]] = {}
    seen = set(pending)

    def enqueue(opcode: int, flags: int, seed: int, template: Candidate) -> None:
        if (opcode, flags, seed) not in seen:
            seen.add((opcode, flags, seed))
            pending[(opcode, flags, seed)] = Candidate(opcode, flags, list(template.operands), dict(template.lanes), template.locked)
            tried[(opcode, flags, seed)] = set()

    for _ in range(300):
        keys = list(pending)
        if not keys:
            break
        for key, (text, decoded, message) in zip(keys, emitter.run([pending[key].inst() for key in keys])):
            candidate = pending.pop(key)
            opcode, flags, seed = key
            if decoded is not None:
                if len(decoded) == 1:
                    found[key] = (candidate, text, decoded[0])
                if flags == 1:
                    for other in SCALAR_FLAGS:
                        enqueue(opcode, other, seed, candidate)
                continue
            if match := re.search(r'SparseCoreMCSlot::SLOT_([A-Z0-9]+)', message):
                needed = SLOT_BITS.get(match[1])
                if needed is not None:
                    enqueue(opcode, needed, seed, candidate)
                    enqueue(opcode, flags | needed, seed, candidate)
                continue
            if message.startswith('Unsupported opcode') or message.startswith('Invalid slot'):
                if flags == 1:
                    for other in SCALAR_FLAGS + VECTOR_FLAGS:
                        enqueue(opcode, other, seed, candidate)
                continue
            if adjust(llvm, enums, candidate, message.removeprefix('CHECK '), tried[key]):
                pending[key] = candidate
            elif flags == 1:
                # The operands are wrong, not the slot: the other scalar slots may accept other operands.
                for other in SCALAR_FLAGS:
                    enqueue(opcode, other, seed, initial_candidate(llvm, opcode, other, seed))
    return found

# ---------------------------------------------------------------- printed syntax

def split_top(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current = ''
    for char in text:
        if char in '[(':
            depth += 1
        elif char in '])':
            depth -= 1
        if char == ',' and depth == 0:
            parts.append(current.strip())
            current = ''
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts

@dataclass(frozen=True)
class Printed:
    """TPUInstPrinter 的一条指令：`目的 = 助记符 @谓词 源`，目的与源按显示顺序合并为 tokens。"""
    mnemonic: str
    predicate: str
    destinations: int
    tokens: tuple[str, ...]

def parse_printed(text: str) -> Printed:
    body = text.strip()
    assert body.startswith('{') and body.endswith('}'), text
    body = body[1:-1].strip()
    depth = 0
    split = None
    for index, char in enumerate(body):
        if char in '[(':
            depth += 1
        elif char in '])':
            depth -= 1
        elif char == '=' and depth == 0:
            split = index
            break
    left, right = (body[:split], body[split + 1:]) if split is not None else ('', body)
    mnemonic, _, rest = right.strip().partition('\t') if '\t' in right.strip() else right.strip().partition(' ')
    rest = rest.strip()
    predicate = ''
    if match := re.match(r'(@!?p[0-9]+)\s*(.*)', rest):
        predicate, rest = match[1], match[2]
    destinations = [] if left.strip() in ('', '_') else split_top(left.strip())
    return Printed(mnemonic.strip(), predicate, len(destinations), tuple(destinations + split_top(rest)))

# ---------------------------------------------------------------- shape operands

@dataclass(frozen=True)
class Base:
    """一条可以被 emitter 接受的 MCInst：opcode、槽位、固定的形状操作数，以及打印文本和 emitter 结果。"""
    opcode: int
    slot_flags: int
    shape: tuple[tuple[int, int], ...]
    candidate: Candidate
    printed: Printed
    decoded: Decoded

def register_class(llvm: Llvm, opcode: int, index: int, register: int) -> int:
    """操作数的寄存器类；可取寄存器或立即数的操作数没有类，取寄存器自身所在的第一个类。"""
    declared = llvm.operands(opcode)[index][0]
    if declared >= 0:
        return declared
    return next(number for number, registers in enumerate(llvm.class_registers) if register in registers)

def is_predicate(llvm: Llvm, opcode: int, index: int) -> bool:
    return bool(llvm.operands(opcode)[index][1] & 2)

def replaced(candidate: Candidate, index: int, operand: tuple[int, int, int], lane: int | None = None) -> Candidate:
    operands = list(candidate.operands)
    operands[index] = operand
    lanes = dict(candidate.lanes)
    if lane is not None:
        lanes[index] = lane
    return Candidate(candidate.opcode, candidate.slot_flags, operands, lanes, candidate.locked)

def outline(llvm: Llvm, printed: Printed) -> tuple[Any, ...]:
    """打印文本去掉寄存器与数值后的样子：助记符、目的个数和各位置的固定文本，例如地址空间名。"""
    names = '|'.join(sorted((re.escape(name) for name in llvm.registers), key=len, reverse=True))
    pattern = re.compile(rf'(?<![A-Za-z0-9_])(?:{names}|-?[0-9]+)(?![A-Za-z0-9_])|\$-?(?:0x[0-9a-fA-F]+|[0-9]+)')
    return (printed.mnemonic, printed.destinations, tuple(pattern.sub('#', token) for token in printed.tokens))

def stem(shape: tuple[Any, ...]) -> tuple[Any, ...]:
    """去掉地址空间名后的编号，例如 ``[hbm4b0:#]`` 写作 ``[hbm4b:#]``。"""
    mnemonic, destinations, tokens = shape
    return (mnemonic, destinations, tuple(re.sub(r'(?<=[a-z])[0-9]+(?=:)', '', token) for token in tokens))

def shape_values(llvm: Llvm, base: Base, index: int, value: int, texts: dict[int, str]) -> list[int]:
    """形状操作数要试的取值。有模式字时，单独翻转后改变打印样子的位按相邻关系分段，每段取遍全部组合，其余位保持模式字；否则取 0..255。"""
    if index not in base.candidate.locked:
        return list(range(256))
    reference = outline(llvm, base.printed)
    changing = [bit for bit in range(32) if texts.get(value ^ 1 << bit) and outline(llvm, parse_printed(texts[value ^ 1 << bit])) != reference]
    values: list[int] = []
    for _, group in itertools.groupby(enumerate(changing), lambda item: item[1] - item[0]):
        bits = [bit for _, bit in group]
        low, width = bits[0], min(len(bits), 8)
        mask = (1 << width) - 1 << low
        values.extend(value & ~mask | number << low for number in range(1 << width))
    return list(dict.fromkeys(values))

def expand_shapes(emitter: Emitter, bases: list[Base], widths: dict[tuple[str, str], dict[str, int]]) -> list[Base]:
    """立即数操作数若改变打印样子（助记符、显示位置的个数或地址空间名，例如 stream 的模式字），按每个取值分出一条独立指令。

    只保留 emitter 直接接受、形式不变、且写入的字段值不超出字段位宽的取值。超出位宽的值会被 encoder 拒绝，例如 stream 的存储器编号被 emitter 加到 2 位的存储器类型上；改变形式的取值使助记符与编码不一致，例如 ``stream.strided`` 的 opcode 配上 indirect 的模式位。
    """
    llvm = emitter.llvm
    # Keys use the position of the base: one opcode and slot can have a base per mode word seed.
    flips: list[tuple[int, int, int]] = []
    for position, base in enumerate(bases):
        for index, (kind, value, _) in enumerate(base.candidate.operands):
            if kind == IMMEDIATE and index in base.candidate.locked:
                flips.extend((position, index, value ^ 1 << bit) for bit in range(32))
    flipped: dict[tuple[int, int], dict[int, str]] = {}
    for (position, index, number), result in zip(flips, llvm.run([[replaced(bases[position].candidate, index, (IMMEDIATE, number, 0)).inst()] for position, index, number in flips])):
        flipped.setdefault((position, index), {})[number] = result.text
    requests: list[tuple[int, int, int]] = []
    for position, base in enumerate(bases):
        for index, (kind, value, _) in enumerate(base.candidate.operands):
            if kind == IMMEDIATE and not is_predicate(llvm, base.opcode, index):
                requests.extend((position, index, number) for number in shape_values(llvm, base, index, value, flipped.get((position, index), {})))
    results = emitter.run([replaced(bases[position].candidate, index, (IMMEDIATE, number, 0)).inst() for position, index, number in requests])
    shapes: dict[tuple[int, int], list[Base]] = {}
    for (position, index, number), (text, decoded, _) in zip(requests, results):
        # An empty text means the printer itself rejected the value.
        if text and outline(llvm, parse_printed(text)) != outline(llvm, bases[position].printed):
            shapes.setdefault((position, index), [])
    shape_operands: dict[int, set[int]] = {}
    for position, index in shapes:
        shape_operands.setdefault(position, set()).add(index)
    for (position, index, number), (text, decoded, _) in zip(requests, results):
        base = bases[position]
        variants = shapes.get((position, index))
        if variants is None or not text or decoded is None or len(decoded) != 1 or (decoded[0].slot, decoded[0].form) != (base.decoded.slot, base.decoded.form):
            continue
        # A number after an address space name is a memory index; the emitter adds it to the 2-bit memory type, so only the starting value is meaningful.
        if number != base.candidate.operands[index][1] and stem(outline(llvm, parse_printed(text))) == stem(outline(llvm, base.printed)):
            continue
        limits = widths.get((decoded[0].slot, decoded[0].form))
        if limits is None or any(value >= 1 << limits.get(name, 64) for name, value in decoded[0].values.items()):
            continue
        candidate = replaced(base.candidate, index, (IMMEDIATE, number, 0))
        # Every shape operand stays at its value in the other operands' variants, e.g. the stream memory id with each mode word.
        locked = Candidate(candidate.opcode, candidate.slot_flags, candidate.operands, candidate.lanes, candidate.locked | shape_operands[position])
        variants.append(Base(base.opcode, base.slot_flags, base.shape + ((index, number),), locked, parse_printed(text), decoded[0]))
    expanded = [base for position, base in enumerate(bases) if position not in shape_operands]
    return expanded + [variant for variants in shapes.values() for variant in variants]

# ---------------------------------------------------------------- operand probing

def changes(before: Decoded, after: Decoded) -> set[str]:
    """两次 emitter 结果中不同的字段：形式字段用字段名，共享字段加前缀 ``shared.``。"""
    names = {name for name in set(before.values) | set(after.values) if before.values.get(name, 0) != after.values.get(name, 0)}
    names |= {'shared.' + name for name in set(before.shared) | set(after.shared) if before.shared.get(name, 0) != after.shared.get(name, 0)}
    if (before.form, before.slot) != (after.form, after.slot):
        names.add('form')
    return names

def register_number(name: str) -> tuple[str, int] | None:
    match = re.fullmatch(r'([a-z]+)([0-9]+)', name)
    return (match[1], int(match[2])) if match else None

@dataclass
class Observation:
    label: str
    value: int
    text: str
    decoded: Decoded | None

@dataclass
class Probed:
    base: Base
    observations: dict[int, list[Observation]] = field(default_factory=dict)

ENCODING_SCAN = range(64)
# The emitter checks the value against the selector, e.g. ones_immN needs the upper 12 bits set.
SELECTOR_VALUES = (0, 0xfff00001, 0x12345000)
IMMEDIATE_VALUES = (3, 5, 0x1234, 0xfffff) + tuple(1 << bit for bit in range(32))

def probe_operands(emitter: Emitter, bases: list[Base]) -> list[Probed]:
    """逐个改变每条指令的寄存器、selector 编码与立即数，记录 emitter 的结果与打印文本。"""
    llvm = emitter.llvm
    probed = [Probed(base) for base in bases]
    requests: list[tuple[Probed, int, str, int, Candidate]] = []
    for item in probed:
        candidate = item.base.candidate
        for index, (kind, value, extra) in enumerate(candidate.operands):
            if is_predicate(llvm, candidate.opcode, index) or index in candidate.locked:
                continue
            if kind == REGISTER:
                others = [register for register in usable_registers(llvm, register_class(llvm, candidate.opcode, index, value)) if register != value]
                for register in dict.fromkeys(others[2:3] + others[5:6] + others[-1:]):
                    requests.append((item, index, 'register', register, replaced(candidate, index, (REGISTER, register, 0))))
                if index in candidate.lanes:
                    for lane in range(1, 5):
                        requests.append((item, index, 'lane', lane, replaced(candidate, index, (kind, value, extra), lane)))
            elif kind == EXPRESSION:
                for encoding in ENCODING_SCAN:
                    for number in SELECTOR_VALUES:
                        requests.append((item, index, 'encoding', encoding, replaced(candidate, index, (EXPRESSION, number, encoding << 8))))
                for number in IMMEDIATE_VALUES:
                    requests.append((item, index, 'expression', number, replaced(candidate, index, (EXPRESSION, number, extra))))
            else:
                for number in IMMEDIATE_VALUES:
                    requests.append((item, index, 'immediate', number, replaced(candidate, index, (IMMEDIATE, number, 0))))
                # The emitter ignores a plain immediate where it expects a selector expression.
                for encoding in ENCODING_SCAN:
                    for number in SELECTOR_VALUES:
                        requests.append((item, index, 'encoding', encoding, replaced(candidate, index, (EXPRESSION, number, encoding << 8))))
    for (item, index, label, value, _), (text, decoded, _) in zip(requests, emitter.run([request[4].inst() for request in requests])):
        single = decoded[0] if decoded is not None and len(decoded) == 1 else None
        item.observations.setdefault(index, []).append(Observation(label, value, text, single))
    return probed

# ---------------------------------------------------------------- instruction records

# Tokens that only name what the mnemonic writes; the tpuasm syntax omits them, as for v6e TC.
IMPLIED = frozenset(('(pc)', '(tag)', '(tm)'))

@dataclass
class Record:
    """一条 LLVM 指令在一个槽中的语法：助记符、按显示顺序的操作数描述，以及操作数之外的字段取值。"""
    slot: str
    form: str
    llvm: str
    mnemonic: str
    operands: list[tuple[Any, ...]]
    values: dict[str, int]
    owned: set[str]
    # Fields written by operands that the syntax does not show; they stay free for encoding constraints.
    hidden: set[str]
    # Bits by which the shape operands differ from the nearest starting MCInst; aliases keep the closest syntax.
    distance: int
    notes: list[str] = field(default_factory=list)

def _accepted(base: Base, observations: list[Observation], label: str) -> list[Observation]:
    return [item for item in observations if item.label == label and item.decoded is not None and (item.decoded.slot, item.decoded.form) == (base.decoded.slot, base.decoded.form)]

def _changed(base: Base, observations: list[Observation]) -> set[str]:
    return {name for item in observations if item.decoded is not None for name in changes(base.decoded, item.decoded)}

def _selector(base: Base, observations: list[Observation]) -> tuple[tuple[Any, ...], set[str]] | None:
    """selector 编码扫描：取值恰好等于编码的唯一字段就是 selector。"""
    scan = _accepted(base, observations, 'encoding')
    candidates = None
    for item in scan:
        assert item.decoded is not None
        if item.value:
            matching = {name for name, number in item.decoded.values.items() if number == item.value}
            candidates = matching if candidates is None else candidates & matching
    if candidates and len(candidates) == 1:
        name, = candidates
        return ('selector', name, tuple(sorted({item.value for item in scan}))), {name, *(f'shared.imm{slot}' for slot in range(6))}
    return None

def classify(llvm: Llvm, base: Base, index: int, observations: list[Observation]) -> tuple[tuple[Any, ...], set[str]]:
    """一个操作数的描述与它写入的字段。"""
    kind, value, _ = base.candidate.operands[index]
    if kind == REGISTER:
        registers = _accepted(base, observations, 'register')
        changed = _changed(base, registers)
        lanes = _accepted(base, observations, 'lane')
        selectors = {name for name in _changed(base, lanes) if not name.startswith('shared.')}
        if len(selectors) == 1:
            name, = selectors
            return ('lane', name, tuple(sorted((item.decoded.values.get(name, 0), item.value - 1) for item in lanes if item.decoded is not None))), {name} | {f'shared.vs{lane}' for lane in range(4)} | {f'shared.vs{lane}_used' for lane in range(4)}
        if not changed:
            return ('implicit', llvm.register_names[value]), set()
        lane_fields = {name.removeprefix('shared.') for name in changed if re.fullmatch(r'shared\.vs[0-3]', name)}
        if len(changed) == 1 and lane_fields:
            # The register always goes through one fixed lane, e.g. a remote chip id.
            lane, = lane_fields
            return ('register', lane, 's', 32), changed | {f'shared.{lane}_used'}
        if len(changed) == 1 and not next(iter(changed)).startswith('shared.'):
            name, = changed
            numbers = [register_number(llvm.register_names[item.value]) for item in registers]
            if all(number is not None and item.decoded is not None and item.decoded.values.get(name, 0) == number[1] for number, item in zip(numbers, registers)):
                prefix = register_number(llvm.register_names[registers[0].value])
                assert prefix is not None
                count = len(usable_registers(llvm, register_class(llvm, base.opcode, index, value)))
                return ('register', name, prefix[0], count), changed
        return ('unknown', f'register fields {sorted(changed)}'), changed
    found = _selector(base, observations)
    if found is not None:
        return found
    label = 'expression' if kind == EXPRESSION else 'immediate'
    values = _accepted(base, observations, label)
    changed = _changed(base, values)
    if not changed:
        # No accepted value changes a field, e.g. an HBM index the emitter only accepts as 0.
        return ('dropped',), set()
    if len(changed) == 1:
        name, = changed
        if not name.startswith('shared.') and all(item.decoded is not None and item.decoded.values.get(name, 0) == item.value for item in values):
            return ('number', name), changed
        if name.startswith('shared.imm') and all(item.decoded is not None and item.decoded.shared.get(name.removeprefix('shared.'), 0) == item.value for item in values):
            return ('immediate', name.removeprefix('shared.')), changed
    packed = _packed(base, values)
    if packed is not None:
        return ('packed', packed), changed
    return ('unknown', f'{label} fields {sorted(changed)}'), changed

def _packed(base: Base, values: list[Observation]) -> tuple[tuple[str, int, int], ...] | None:
    """立即数的每个 bit 恰好写入某个字段的一个 bit 时，按值的连续位段写出 (字段, 值中起始 bit, 位宽)。"""
    single = {item.value.bit_length() - 1: item for item in values if item.value and item.value & (item.value - 1) == 0}
    mapping: dict[int, tuple[str, int]] = {}
    for bit, item in single.items():
        assert item.decoded is not None
        changed = changes(base.decoded, item.decoded)
        if not changed:
            continue
        if len(changed) != 1 or next(iter(changed)).startswith('shared.'):
            return None
        name, = changed
        difference = item.decoded.values.get(name, 0) ^ base.decoded.values.get(name, 0)
        if difference & (difference - 1):
            return None
        mapping[bit] = (name, difference.bit_length() - 1)
    if not mapping:
        return None
    parts: list[tuple[str, int, int]] = []
    for bit in sorted(mapping):
        name, position = mapping[bit]
        if parts and parts[-1][0] == name and parts[-1][1] + parts[-1][2] == bit and position == parts[-1][2]:
            parts[-1] = (name, parts[-1][1], parts[-1][2] + 1)
        elif position == 0:
            parts.append((name, bit, 1))
        else:
            return None
    # The baseline value must already be zero in the packed fields for the parts to describe the operand.
    if any(base.decoded.values.get(name, 0) for name, _, _ in parts):
        return None
    return tuple(parts)

def hidden_operands(probed: Probed) -> set[int]:
    """语法中不出现的操作数：改变取值时打印文本始终不变，它们写入的字段由编码约束表达；或者是 emitter 接受的取值都不改变字段的立即数，例如只能为 0 的 HBM 编号，打印时保持原值。"""
    base = probed.base
    return {
        index for index, observations in probed.observations.items()
        if all(parse_printed(item.text) == base.printed for item in observations if item.text) or (base.candidate.operands[index][0] != REGISTER and not _changed(base, observations))
    }

def labeled(llvm: Llvm, base: Base, hidden: set[int]) -> tuple[Candidate, dict[int, list[str]]]:
    """打印用的 MCInst：每个操作数取互不相同、容易在文本中找到的值；返回各操作数可能的显示形式。"""
    candidate = base.candidate
    operands = list(candidate.operands)
    renderings: dict[int, list[str]] = {}
    used: set[str] = set()
    for index, (kind, value, extra) in enumerate(candidate.operands):
        if is_predicate(llvm, candidate.opcode, index) or index in candidate.locked or index in hidden:
            continue
        if kind == REGISTER:
            name = llvm.register_names[value]
            options = usable_registers(llvm, register_class(llvm, candidate.opcode, index, value))
            if register_number(name) is not None:
                for register in options[4 + index:] + options:
                    if llvm.register_names[register] not in used:
                        value = register
                        break
            operands[index] = (REGISTER, value, 0)
            used.add(llvm.register_names[value])
            renderings[index] = [llvm.register_names[value]]
            number = register_number(llvm.register_names[value])
            if number is not None:
                # Some printers show a register number as an immediate, e.g. cbreg:$0x5.
                renderings[index] += [f'$0x{number[1]:X}', f'$0x{number[1]:x}']
        else:
            label = 0x1a20 + 0x111 * index
            operands[index] = (kind, label, extra & 0xffff)
            renderings[index] = [f'$0x{label:X}', f'${label}', f'$0x{label:x}']
    return Candidate(candidate.opcode, candidate.slot_flags, operands, dict(candidate.lanes), candidate.locked), renderings

def templates(printed: Printed, renderings: dict[int, list[str]]) -> list[tuple[str, tuple[tuple[int, str], ...]]]:
    """每个显示位置的模板与其中依次出现的 (操作数, 显示文本)；``{n}`` 为占位符。"""
    alternatives = sorted(((text, index) for index, texts in renderings.items() for text in texts), key=lambda item: -len(item[0]))
    pattern = re.compile('|'.join(r'(?<![A-Za-z0-9_$])' + re.escape(text) + r'(?![A-Za-z0-9_])' for text, _ in alternatives)) if alternatives else None
    owner = {text: index for text, index in alternatives}
    result = []
    for token in printed.tokens:
        found: list[tuple[int, str]] = []

        def substitute(match: re.Match[str]) -> str:
            found.append((owner[match[0]], match[0]))
            return '{' + str(len(found) - 1) + '}'

        result.append((pattern.sub(substitute, token) if pattern else token, tuple(found)))
    return result

def plain(text: str) -> str:
    """去掉单个隐含寄存器的括号和编号，例如 ``(drf_0)`` 写作 ``drf``，与 v6e TC 的写法一致。"""
    match = re.fullmatch(r'\(([a-z0-9]+?)(?:_?[0-9]+)?\)', text)
    return match[1] if match else text

def display(descriptor: tuple[Any, ...], mnemonic: str, shown: str) -> tuple[Any, ...]:
    """补上数值的显示方式：浮点运算的立即数写成浮点数，其余按 printer 的十六进制或十进制。"""
    kind = descriptor[0]
    if kind == 'register' and shown.startswith('$'):
        return ('number', descriptor[1], 'hex')
    if kind == 'selector':
        return descriptor + ('f32' if re.search(r'\.f32(?:\.|$)', mnemonic) else 'hex',)
    if kind == 'immediate':
        return descriptor + ('s32' if '.rel' in mnemonic else 'hex',)
    if kind == 'packed':
        return descriptor + ('hex',)
    if kind == 'number':
        return descriptor + ('hex' if shown.startswith('$0x') else 'u32',)
    return descriptor

def describe(llvm: Llvm, probed: Probed, hidden: set[int], printed_labels: str) -> Record:
    base = probed.base
    notes = []
    descriptors: dict[int, tuple[Any, ...]] = {}
    owned: set[str] = set()
    hidden_fields: set[str] = set()
    for index, observations in probed.observations.items():
        if index in hidden:
            descriptors[index] = ('dropped',)
            hidden_fields |= _changed(base, observations)
            continue
        descriptor, fields_changed = classify(llvm, base, index, observations)
        descriptors[index] = descriptor
        owned |= fields_changed
        if descriptor[0] == 'unknown':
            notes.append(f'operand {index}: {descriptor[1]}')
    _, renderings = labeled(llvm, base, hidden)
    printed = parse_printed(printed_labels)
    for index, descriptor in descriptors.items():
        if descriptor[0] == 'implicit':
            renderings[index] = [llvm.register_names[base.candidate.operands[index][1]]]
    operands: list[tuple[Any, ...]] = []
    shown: set[int] = set()
    for token, (template, found) in zip(printed.tokens, templates(printed, renderings)):
        shown.update(index for index, _ in found)
        parts = [display(descriptors[index], printed.mnemonic, text) for index, text in found]
        if not found:
            if token not in IMPLIED and token != '_':
                operands.append(('literal', plain(token)))
        elif template == '{0}':
            part = parts[0]
            if part[0] == 'implicit':
                index = found[0][0]
                operands.append(('literal', plain(llvm.register_names[llvm.class_registers[register_class(llvm, base.opcode, index, base.candidate.operands[index][1])][0]])))
            elif part[0] != 'dropped':
                operands.append(part)
        else:
            if any(part[0] in ('dropped', 'implicit') for part in parts):
                notes.append(f'token {token!r} mixes unencoded operands')
            operands.append(('pattern', template.replace('$', ''), tuple(parts)))
    for index, descriptor in descriptors.items():
        if index not in shown and descriptor[0] not in ('dropped', 'implicit'):
            notes.append(f'operand {index} {descriptor} is not printed')
    start = initial_candidate(llvm, base.opcode, base.slot_flags).operands
    seeds = mode_seeds(llvm.names[base.opcode])
    distance = sum(min(bin(value ^ seed).count('1') for seed in seeds) if index == 0 and seeds else bin(value ^ start[index][1]).count('1') for index, value in base.shape)
    return Record(base.decoded.slot, base.decoded.form, llvm.names[base.opcode], printed.mnemonic, operands, dict(base.decoded.values), owned, hidden_fields - owned, distance, notes)

def encoding_parts(descriptor: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    """操作数描述中决定编码的部分：去掉固定文本并拆开复合写法，例如地址空间名不同、或偏移寄存器写在另一端地址中，但字段相同的写法编码相同。"""
    if descriptor[0] == 'literal':
        return []
    if descriptor[0] == 'pattern':
        return [leaf for part in descriptor[2] for leaf in encoding_parts(part)]
    return [descriptor]

def preference(record: Record) -> tuple[Any, ...]:
    """同一编码有多种写法时的顺序：先取与编译器 MCInst 最接近的，再取数值按整数显示的，再取 32 位整数类型的助记符，例如 simm.s32 先于 simm.f32，vimm.s32 先于 vimm.bf16。"""
    return (record.distance, 'f32' in repr(record.operands), re.search(r'\.[su]32(?:\.|$)', record.mnemonic) is None, record.llvm)

def finalize(records: list[Record], forms: list[FormData]) -> list[tuple[Any, ...]]:
    """去掉有疑问和 encoder 不接受的记录，求每条记录需要固定的字段，并处理编码相同的写法。

    操作数不控制的字段，若本指令取非零值、同一形式中其他同样不控制它的指令取值不同，或它是枚举字段，就由 opcode 决定，作为固定字段（例如不带 ``.msk`` 的 ``vld`` 固定全 1 掩码，``vmul.f32 vD, vY, vX`` 固定 y 选择寄存器，``dma.local`` 固定两端的地址空间）；不显示的操作数写入的字段，以及其余字段（例如 selector 选中立即数时闲置的寄存器号），留给编码约束。

    编码相同的写法按 ``preference`` 排序，解码取第一个。不同 LLVM opcode 的助记符不同的写法都保留供汇编使用，例如 ``simm.s32`` 与 ``simm.f32``；形状展开得到的写法只在带来新编码时保留，助记符相同、只是固定文本不同的写法（例如模式字中不影响编码的地址空间名）会误导读者，也只保留第一个。
    """
    encodable = {(data.slot, data.form.name) for data in forms}
    kept = [record for record in records if not record.notes and (record.slot, record.form) in encodable]
    layouts = {(data.slot, data.form.name): [name for name in data.layout] for data in forms}
    selectors = {(data.slot, data.form.name): {name for name, (_, _, enum) in data.layout.items() if enum} for data in forms}
    by_form: dict[tuple[str, str], list[Record]] = {}
    for record in kept:
        by_form.setdefault((record.slot, record.form), []).append(record)
    entries = []
    spellings: dict[tuple[Any, ...], set[str]] = {}
    for key, group in sorted(by_form.items()):
        for record in sorted(group, key=preference):
            fixed = []
            for name in layouts[key]:
                if name in record.owned or name in record.hidden:
                    continue
                value = record.values.get(name, 0)
                others = {other.values.get(name, 0) for other in group if name not in other.owned}
                if value or len(others) > 1 or name in selectors[key]:
                    fixed.append((name, value))
            # Operand order does not matter: the printer may list the same fields in another order, e.g. a stream whose mode word swaps the two ends.
            encoding = (record.slot, record.form, tuple(fixed), tuple(sorted(repr(leaf) for operand in record.operands for leaf in encoding_parts(operand))))
            # A shape variant only adds a new encoding; other spellings of one encoding come from different LLVM opcodes.
            if record.mnemonic in spellings.setdefault(encoding, set()) or (spellings[encoding] and record.distance):
                continue
            spellings[encoding].add(record.mnemonic)
            entries.append((record.slot, record.form, record.llvm, record.mnemonic, tuple(fixed), tuple(record.operands)))
    order = list(SPEC.slots)
    # Within a form the entries keep the preference order, which decoding follows.
    return sorted(entries, key=lambda entry: (order.index(entry[0]), entry[1]))

# ---------------------------------------------------------------- output

def render(empty: int, shared: dict[str, tuple[int, int]], predicates: dict[str, tuple[int, int | None]], isa: Isa, forms: list[FormData], rejected: dict[str, list[str]], instructions: list[tuple[Any, ...]], version: str) -> str:
    enum_names = sorted({enum for data in forms for _, _, enum in data.layout.values() if enum})
    enum_types = {enum: next(item.type_name for data in forms for item in isa.operands(data.form) if data.layout[item.name][2] == enum) for enum in enum_names}
    layouts: list[tuple[tuple[str, int, int, int, str], ...]] = []
    lines = [
        f'"""TPU v6e TEC 字段位置、指令形式与指令语法；由 tools/generate_tpu_v6e_tec_isa.py 从 libtpu {version} 生成，不要手工修改。',
        '',
        '位编号相对一个 little-endian 64 字节 bundle。字段名、字段号和枚举来自 ISA descriptor，位位置和固定位来自 encoder；助记符和操作数的显示形式来自 LLVM TPU printer，操作数对应的字段来自 TEC emitter。不含程序样例或运行时地址。',
        '"""',
        'from __future__ import annotations',
        '',
        f'EMPTY_WORD = {empty:#x}',
        '# 物理槽：bundle 内的字段路径、谓词起始 bit、取反 bit。',
        'SLOTS: dict[str, tuple[tuple[int, ...], int, int]] = {',
    ]
    for slot, path in SPEC.slots.items():
        start, inversion = predicates[slot]
        lines.append(f'    {slot!r}: ({path!r}, {start}, {inversion!r}),')
    lines.extend(('}', '# 共享操作数：名称、所在 bundle 字段号、消息内字段号、起始 bit、位宽。', 'SHARED_FIELDS: tuple[tuple[str, int, int, int, int], ...] = ('))
    for name, number, index in SPEC.shared:
        start, width = shared[name]
        lines.append(f'    ({name!r}, {number}, {index}, {start}, {width}),')
    lines.extend((')', '# descriptor 中的枚举；同值别名只保留首个名称。', 'ENUMS: dict[str, dict[int, str]] = {'))
    for enum in enum_names:
        values = isa.enums[enum_types[enum]]
        lines.append(f'    {enum!r}: {{')
        lines.extend(f'        {value}: {name!r},' for value, name in sorted(values.items()))
        lines.append('    },')
    lines.append('}')
    form_lines = []
    for data in forms:
        layout = tuple((operand.name, operand.number, *data.layout[operand.name]) for operand in isa.operands(data.form))
        if layout not in layouts:
            layouts.append(layout)
        fixed = tuple((start, width, data.fixed_value >> start & ((1 << width) - 1)) for start, width in runs(data.fixed_mask))
        form_lines.append(f'    ({data.slot!r}, {data.form.number}, {data.form.name!r}, {layouts.index(layout)}, {fixed!r}, {data.excludes!r}),')
    lines.append('# 字段组：每项为 (字段名, 字段号, 起始 bit, 位宽, 枚举名)。')
    lines.append('FIELD_LAYOUTS: tuple[tuple[tuple[str, int, int, int, str], ...], ...] = (')
    lines.extend(f'    {layout!r},' for layout in layouts)
    lines.append(')')
    lines.append('# 形式：槽、oneof 字段号、descriptor 名称、字段组编号、固定位 (起始 bit, 位宽, 取值)、encoder 不允许同时出现的其他槽。')
    lines.append('INSTRUCTION_FORMS: tuple[tuple[str, int, str, int, tuple[tuple[int, int, int], ...], tuple[str, ...]], ...] = (')
    lines.extend(form_lines)
    lines.append(')')
    lines.append('# 指令：槽、形式、LLVM opcode、printer 助记符、opcode 决定的字段取值、按显示顺序的操作数描述（目的在前）。')
    lines.append('INSTRUCTIONS: tuple[tuple[str, str, str, str, tuple[tuple[str, int], ...], tuple[tuple[object, ...], ...]], ...] = (')
    lines.extend(f'    {entry!r},' for entry in instructions)
    lines.append(')')
    lines.append('# encoder 以任何操作数都拒绝的 (槽, 形式)：这些形式不能在该槽发射。')
    lines.append('REJECTED: dict[str, tuple[str, ...]] = {')
    for slot, names in rejected.items():
        lines.append(f'    {slot!r}: {tuple(names)!r},')
    lines.append('}')
    return '\n'.join(lines) + '\n'

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'src' / 'tpuasm' / 'tpu_v6e_tec_isa_data.py')
    parser.add_argument('--validate', type=int, default=20000, metavar='N', help='number of random bundles checked against the encoder')
    args = parser.parse_args()
    backend, path = select_backend(SPEC.target)
    messages, enums = descriptors(path)
    isa = Isa(SPEC, messages, enums)
    empty, shared, predicates, forms, rejected = probe(isa)
    assign_fixed(SPEC, empty, predicates, forms)
    probe_exclusions(isa, predicates, forms)
    probe_widths(isa, predicates, forms)
    assign_fixed(SPEC, empty, predicates, forms)
    validate(isa, empty, shared, predicates, forms, args.validate)
    emitter = Emitter(Llvm(backend, path), isa)
    found = baselines(emitter, enums)
    bases = [Base(opcode, flags, (), candidate, parse_printed(text), decoded) for (opcode, flags, _), (candidate, text, decoded) in sorted(found.items())]
    widths = {(data.slot, data.form.name): {name: width for name, (_, width, _) in data.layout.items()} for data in forms}
    bases = expand_shapes(emitter, bases, widths)
    probed = probe_operands(emitter, bases)
    hidden = [hidden_operands(item) for item in probed]
    labels = [result.text for result in emitter.llvm.run([[labeled(emitter.llvm, item.base, keep)[0].inst()] for item, keep in zip(probed, hidden)])]
    records = [describe(emitter.llvm, item, keep, text) for item, keep, text in zip(probed, hidden, labels)]
    for record in records:
        if record.notes:
            print('skipped', record.llvm, record.slot, record.form, '; '.join(record.notes), file=sys.stderr)
    instructions = finalize(records, forms)
    covered = {(entry[0], entry[1]) for entry in instructions}
    missing = [(data.slot, data.form.name) for data in forms if (data.slot, data.form.name) not in covered]
    print(f'{len(instructions)} instructions cover {len(covered)} of {len(forms)} slot forms; without printer syntax: {missing}', file=sys.stderr)
    args.output.write_text(render(empty, shared, predicates, isa, forms, rejected, instructions, distribution('libtpu').version), encoding='utf-8')
    print(args.output)

if __name__ == '__main__':
    main()

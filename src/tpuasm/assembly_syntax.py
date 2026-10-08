"""汇编语法：显式 bundle、标签、操作数与命名编码约束。"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .targets import HardwareTarget, hardware_target

_IDENTIFIER = r'[A-Za-z_][A-Za-z_0-9]*'
_NAME = r'[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*'
_INTEGER = re.compile(r'([+-]?)(0x[0-9a-fA-F]+|[0-9]+)\Z')
_BRANCHES = frozenset(('sbr.rel', 'sbr.abs', 'scall.rel', 'scall.abs'))

@dataclass(frozen=True)
class AssemblyLocation:
    """汇编源码中的位置，用于诊断。

    Attributes:
        filename: 解析时给出的文件名。
        line: 行号，从 1 开始。
        column: 列号，从 1 开始。
    """
    filename: str
    line: int
    column: int

    def error(self, message: str, pc: int | None = None, slot: str | None = None) -> ValueError:
        """构造带文件名、行列及可选 bundle 编号和物理槽的 ValueError，由调用者抛出。"""
        """构造带文件名、行列及可选 bundle 编号和物理槽的 ValueError，由调用者抛出。"""
        context = f': bundle {pc:#x}' if pc is not None else ''
        context += f': {slot}' if slot is not None else ''
        return ValueError(f'{self.filename}:{self.line}:{self.column}{context}: {message}')

@dataclass(frozen=True)
class AssemblyInstruction:
    """一个物理槽中的指令。

    Attributes:
        slot: 物理槽名称，例如 ``'s0'``、``'va0'``。
        mnemonic: 助记符，例如 ``'vxor.8x128.u32'``。
        operands: 操作数，保留源码文本，例如 ``('v1', '0x13579bdf', 'v0')``。
        predicate: 谓词：15 表示无条件，0–14 表示 ``@pN``，16–30 表示 ``@!pN``。
        location: 指令在源码中的位置。
    """
    slot: str
    mnemonic: str
    operands: tuple[str, ...]
    predicate: int
    location: AssemblyLocation

@dataclass(frozen=True)
class EncodingConstraint:
    """``.encoding { name = value }`` 中的一条命名编码约束，exact 导出用它固定原机器编码。

    Attributes:
        name: 约束名称，例如 ``'s0.y'``。
        value: 约束值的源码文本，例如 ``'pair(imm1, imm0)'``。
        location: 约束在源码中的位置。
    """
    name: str
    value: str
    location: AssemblyLocation

@dataclass(frozen=True)
class AssemblyBundle:
    """一个 bundle；空 bundle 的 instructions 为空。

    Attributes:
        instructions: 按源码顺序排列的指令，每个物理槽至多一条。
        constraints: 该 bundle 的命名编码约束。
        location: bundle 在源码中的位置。
    """
    instructions: tuple[AssemblyInstruction, ...]
    constraints: tuple[EncodingConstraint, ...]
    location: AssemblyLocation

@dataclass(frozen=True)
class AssemblyProgram:
    """:func:`parse_assembly` 的结果。

    Attributes:
        bundles: 按 bundle 编号排列，``.empty`` 和 ``.align`` 已展开为空 bundle。完整清单的下标与程序映像中的 bundle 编号一致，可直接用作 :class:`BundleInsertion` 的插入点。
        labels: 标签到 bundle 编号的映射。
        hardware: ``.target`` 声明的硬件目标。
    """
    bundles: tuple[AssemblyBundle, ...]
    labels: dict[str, int]
    hardware: HardwareTarget

    def target(self, operand: str, pc: int, relative: bool, location: AssemblyLocation) -> int:
        """解析位于 bundle ``pc`` 的分支目标操作数：标签换算为 bundle 编号，relative 为 True 时返回相对 ``pc`` 的位移；其他操作数按整数解析。"""
        """解析位于 bundle ``pc`` 的分支目标操作数：标签换算为 bundle 编号，relative 为 True 时返回相对 ``pc`` 的位移；其他操作数按整数解析。"""
        if re.fullmatch(_IDENTIFIER, operand):
            if operand not in self.labels:
                raise location.error(f'undefined label {operand!r}', pc)
            return self.labels[operand] - (pc if relative else 0)
        try:
            return integer(operand)
        except ValueError as error:
            raise location.error(str(error), pc) from error

def integer(text: str) -> int:
    match = _INTEGER.fullmatch(text)
    if match is None:
        raise ValueError(f'expected an integer, got {text!r}')
    sign, digits = match.groups()
    return (-1 if sign == '-' else 1) * int(digits, 16 if digits.startswith('0x') else 10)

def branch_target(mnemonic: str, operands: tuple[str, ...], pc: int) -> tuple[int, int] | None:
    if mnemonic not in _BRANCHES:
        return None
    index = 1 if mnemonic.startswith('scall') else 0
    target = integer(operands[index]) + (pc if mnemonic.endswith('.rel') else 0)
    return index, target

def split_operands(text: str) -> tuple[str, ...]:
    if not text.strip():
        return ()
    stack = []
    parts = []
    start = 0
    for index, char in enumerate(text):
        if char in '[(':
            stack.append(char)
        elif char in '])':
            if not stack or stack.pop() != {']': '[', ')': '('}[char]:
                raise ValueError('mismatched operand brackets')
        elif char == ',' and not stack:
            parts.append(text[start:index].strip())
            start = index + 1
    if stack:
        raise ValueError('unclosed operand')
    parts.append(text[start:].strip())
    if any(not part for part in parts):
        raise ValueError('empty operand or trailing comma')
    return tuple(parts)

def _statements(text: str, filename: str) -> list[tuple[str, AssemblyLocation]]:
    result = []
    stack = []
    buffer: list[str] = []
    line = column = 1
    location = AssemblyLocation(filename, line, column)
    comment = False
    brace_depth = 0
    needs_separator = False
    for char in text.replace('\r\n', '\n'):
        if char == '#':
            comment = True
        if not comment or char == '\n':
            if needs_separator and not char.isspace() and char != ';':
                raise AssemblyLocation(filename, line, column).error('expected newline or ; after bundle')
            if char in ';\n':
                needs_separator = False
            if char in '[(':
                stack.append(char)
            elif char in '])':
                if not stack or stack.pop() != {']': '[', ')': '('}[char]:
                    raise AssemblyLocation(filename, line, column).error('mismatched operand brackets')
            if not stack and char in '{};\n':
                if char == '{':
                    if buffer and ''.join(buffer).strip() != '.encoding':
                        raise location.error('expected newline or ; before bundle')
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    needs_separator = brace_depth == 0
                if buffer:
                    result.append((''.join(buffer).strip(), location))
                    buffer = []
                if char in '{}':
                    result.append((char, AssemblyLocation(filename, line, column)))
            elif buffer or not char.isspace():
                if not buffer:
                    location = AssemblyLocation(filename, line, column)
                buffer.append(char)
        if char == '\n':
            line += 1
            column = 1
            comment = False
        else:
            column += 1
    if stack:
        raise location.error('unclosed operand')
    if buffer:
        result.append((''.join(buffer).strip(), location))
    return result

def parse_assembly(text: str, *, filename: str = '<assembly>', fragment: bool = False) -> AssemblyProgram:
    """将 .tpuasm 源码解析为按 bundle 编号排列的语法树，用于在清单中定位指令；不做编码，也不读写文件。

    只检查语法、物理槽名称和 bundle 数，不检查助记符、操作数类型和范围；这些由 :func:`assemble_listing` 负责。

    Args:
        text: .tpuasm 源码，首行必须声明 ``.target``，例如 :func:`format_assembly` 的输出。
        filename: 错误诊断中的文件名，默认 ``'<assembly>'``；不会打开对应路径。
        fragment: 默认 False，要求 bundle 总数为目标块容量的正整数倍；True 用于解析插入片段，不要求凑齐整块，也可以一个 bundle 都没有。

    Returns:
        :class:`AssemblyProgram`，``bundles[pc]`` 是编号为 ``pc`` 的 bundle。

    Raises:
        ValueError: 语法错误、未知物理槽或 bundle 数不合要求；消息包含文件名、行列和 bundle 编号。

    Examples:
        找到带唯一立即数的标记指令所在的 bundle::

            program = parse_assembly(format_assembly(image, target='tpu-v4-tc'))
            pc, = [pc for pc, bundle in enumerate(program.bundles) if any('0x13579bdf' in instruction.operands for instruction in bundle.instructions)]
    """
    statements = _statements(text, filename)
    if not statements:
        raise ValueError(f'{filename}: assembly contains no bundles')
    declaration = statements[0][0].split()
    if len(declaration) != 2 or declaration[0] != '.target':
        raise statements[0][1].error('first statement must be .target TARGET')
    try:
        hardware = hardware_target(declaration[1])
    except ValueError as error:
        raise statements[0][1].error(str(error)) from error
    bundles: list[AssemblyBundle] = []
    labels: dict[str, int] = {}
    instructions: list[AssemblyInstruction] | None = None
    constraints: list[EncodingConstraint] = []
    encoding = False
    pending_encoding = False
    saw_encoding = False
    bundle_location = statements[0][1]
    for statement, location in statements[1:]:
        pc = len(bundles)
        if pending_encoding:
            if statement != '{':
                raise location.error('expected { after .encoding', pc)
            encoding, pending_encoding = True, False
            continue
        if encoding:
            if statement == '}':
                encoding = False
                continue
            match = re.fullmatch(rf'({_NAME})\s*=\s*(.+)', statement)
            if match is None:
                raise location.error('expected a named encoding constraint', pc)
            name, value = match.groups()
            if any(pin.name == name for pin in constraints):
                raise location.error(f'duplicate encoding constraint {name!r}', pc)
            constraints.append(EncodingConstraint(name, value.strip(), location))
            continue
        if statement == '{':
            if instructions is not None:
                raise location.error('nested bundle', pc)
            instructions, constraints = [], []
            saw_encoding = False
            bundle_location = location
        elif statement == '}':
            if instructions is None:
                raise location.error('unexpected }', pc)
            bundles.append(AssemblyBundle(tuple(instructions), tuple(constraints), bundle_location))
            instructions = None
        elif statement == '.encoding':
            if instructions is None or saw_encoding:
                raise location.error('.encoding must occur once at the end of a bundle', pc)
            pending_encoding = saw_encoding = True
        elif instructions is not None:
            if saw_encoding:
                raise location.error('instruction cannot follow .encoding', pc)
            match = re.fullmatch(r'([a-z][a-z0-9]*):\s*(?:@(!?)p([0-9]+)\s+)?([a-z][a-z0-9_.]*)(?:\s+(.+))?', statement)
            if match is None:
                raise location.error('expected slot: [@pN] mnemonic operands', pc)
            slot, inverted, pred, mnemonic, tail = match.groups()
            if slot not in hardware.slots:
                raise location.error(f'unknown physical slot {slot!r}', pc)
            if any(inst.slot == slot for inst in instructions):
                raise location.error('duplicate slot', pc, slot)
            if pred is not None and not 0 <= int(pred) <= 14:
                raise location.error('predicate register must be p0..p14', pc, slot)
            predicate = 15 if pred is None else int(pred) + (16 if inverted else 0)
            try:
                operands = split_operands(tail or '')
            except ValueError as error:
                raise location.error(str(error), pc, slot) from error
            instructions.append(AssemblyInstruction(slot, mnemonic, operands, predicate, location))
        elif statement.startswith(('.empty', '.align')):
            parts = statement.split()
            if len(parts) != 2 or parts[0] not in ('.empty', '.align'):
                raise location.error('expected .empty N or .align N', pc)
            try:
                count = integer(parts[1])
            except ValueError as error:
                raise location.error(str(error), pc) from error
            if count <= 0:
                raise location.error('directive count must be positive', pc)
            count = count if parts[0] == '.empty' else -pc % count
            bundles.extend(AssemblyBundle((), (), location) for _ in range(count))
        else:
            match = re.fullmatch(rf'({_IDENTIFIER}):', statement)
            if match is None:
                raise location.error('expected a bundle, label, .empty or .align', pc)
            name = match[1]
            if name in labels:
                raise location.error(f'duplicate label {name!r}', pc)
            labels[name] = pc
    if instructions is not None or pending_encoding or encoding:
        raise bundle_location.error('unclosed bundle', len(bundles))
    # 片段可以没有 bundle：只删除原 bundle 的 BundleInsertion 只有 .target 声明。
    if not fragment and (not bundles or len(bundles) % hardware.bundles_per_block):
        raise statements[-1][1].error(f'bundle count must be a positive multiple of {hardware.bundles_per_block}', len(bundles))
    return AssemblyProgram(tuple(bundles), labels, hardware)

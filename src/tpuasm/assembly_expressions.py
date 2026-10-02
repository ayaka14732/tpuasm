"""与目标无关的类型化操作数表达式构造器。"""
from __future__ import annotations

from .assembly_model import Expression

def literal(text: str) -> Expression:
    return ('literal', text)

def register(field: str, prefix: str) -> Expression:
    return ('register', field, prefix)

def scalar_register(field: str) -> Expression:
    return register(field, 's')

def numeric(kind: str, width: int, parts: tuple[tuple[str, int], ...] = (), base: int = 0, bias: int = 0, *, part_width: int | None = None) -> Expression:
    return ('number', kind, width, tuple((name, shift, part_width) for name, shift in parts), base, bias)

def packed(kind: str, width: int, parts: tuple[tuple[str, int, int], ...]) -> Expression:
    """若干字段按位拼成一个数，``parts`` 每项为 (字段, 在数中的起始 bit, 位宽)。"""
    return ('number', kind, width, parts, 0, 0)

def direct(field: str, width: int, kind: str = 'u32', bias: int = 0) -> Expression:
    return numeric(kind, width, ((field, 0),), bias=bias)

def choice(field: str, alternatives: list[tuple[int, Expression]]) -> Expression:
    return ('choice', field, tuple(alternatives))

def memory(space: str, base: Expression, offset: Expression | None = None, *, modifiers: tuple[tuple[str, str, Expression], ...] = ()) -> Expression:
    """``modifiers`` 的每项为 (地址修饰名, 字段, 操作数)；省略该修饰时字段取 0。"""
    return ('memory', space, base, offset, modifiers)

def table(entries: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]) -> Expression:
    """固定文本与若干字段取值一一对应，例如由单元号决定的目的寄存器组。"""
    return ('table', entries)

def pattern(template: str, parts: tuple[Expression, ...]) -> Expression:
    """固定文本中嵌入若干操作数，``template`` 中的 ``{N}`` 依次对应 ``parts``；汇编时忽略空白。"""
    return ('pattern', template, parts)

def predicate(field: str) -> Expression:
    alternatives = [(i, literal(f'p{i}')) for i in range(15)] + [(16 + i, literal(f'!p{i}')) for i in range(15)]
    return choice(field, alternatives + [(15, literal('1')), (31, literal('0'))])

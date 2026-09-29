"""ISA 字段、指令签名及可合并的指令包物理位约束。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

Expression = tuple[Any, ...]

@dataclass(frozen=True)
class Bits:
    mask: int = 0
    value: int = 0
    immediates: int = 0
    scalars: int = 0

    def merge(self, other: Bits) -> Bits | None:
        if (self.value ^ other.value) & self.mask & other.mask:
            return None
        return Bits(self.mask | other.mask, self.value | other.value, self.immediates | other.immediates, self.scalars | other.scalars)

@dataclass(frozen=True)
class Field:
    number: int
    start: int
    width: int

    @property
    def mask(self) -> int:
        return ((1 << self.width) - 1) << self.start

    def read(self, word: int) -> int:
        return word >> self.start & ((1 << self.width) - 1)

    def bind(self, value: int) -> Bits:
        if not 0 <= value < 1 << self.width:
            raise ValueError(f'value {value} exceeds {self.width}-bit field')
        return Bits(self.mask, value << self.start)

@dataclass(frozen=True)
class Form(ABC):
    slot: str
    branch: int
    name: str
    fields: dict[str, Field]

    @property
    @abstractmethod
    def predicate_field(self) -> Field:
        raise NotImplementedError

    @abstractmethod
    def bind(self, name: str, value: int, *, consumed: bool = False) -> Bits:
        raise NotImplementedError

    @abstractmethod
    def fixed(self, predicate: int) -> Bits:
        raise NotImplementedError

@dataclass(frozen=True)
class Signature:
    form: Form
    mnemonic: str
    operands: tuple[Expression, ...]
    fixed_fields: tuple[tuple[str, int], ...] = ()
    keywords: tuple[str, ...] = ()
    # 可省略的具名操作数及其省略时的文本；反汇编时取默认值的具名操作数不打印。
    defaults: tuple[tuple[str, str], ...] = ()

def combine(left: list[Bits], right: list[Bits]) -> list[Bits]:
    merged = (a.merge(b) for a in left for b in right)
    return list(dict.fromkeys(bits for bits in merged if bits is not None))

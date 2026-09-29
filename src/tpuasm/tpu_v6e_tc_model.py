"""TPU v6e TC 的形式登记与物理字段模型。"""
from __future__ import annotations

from dataclasses import dataclass

from .assembly_model import Bits, Field, Form
from .tpu_v6e_tc_isa_data import FIELD_LAYOUTS, INSTRUCTION_FORMS, SHARED_FIELDS, SLOTS

ALWAYS = 14
# 共享立即数与标量操作数不属于任何槽，由编码器在每个指令包中写出。
SHARED = {name: Field(0, start, width) for name, _, _, start, width in SHARED_FIELDS}
SHARED_MASK = sum(field.mask for field in SHARED.values())
IMMEDIATES = tuple(name for name in SHARED if name.startswith('imm'))
SCALARS = tuple(name for name in SHARED if name.startswith('vs'))
# 标量子 bundle 与 DMA 是同一个 protobuf oneof 的两个分支。
_ONEOF = {'s0': ('dma',), 's1': ('dma',), 'dma': ('s0', 's1')}

@dataclass(frozen=True)
class GlForm(Form):
    """一个槽内形式；``fields`` 同时包含本形式的字段和共享立即数、标量操作数。"""
    layout: tuple[str, ...]
    enums: dict[str, str]
    fixed_bits: tuple[tuple[int, int, int], ...]
    excludes: tuple[str, ...]
    formatter_mnemonic: str | None
    formatter_tokens: tuple[str, ...]

    @property
    def predicate_field(self) -> Field:
        return Field(1, SLOTS[self.slot][1], 4)

    @property
    def inversion_field(self) -> Field | None:
        inversion = SLOTS[self.slot][2]
        return Field(2, inversion, 1) if inversion is not None else None

    @property
    def own_fields(self) -> tuple[Field, ...]:
        return tuple(self.fields[name] for name in self.layout)

    @property
    def fixed_mask(self) -> int:
        return sum(((1 << width) - 1) << start for start, width, _ in self.fixed_bits)

    @property
    def writable_mask(self) -> int:
        mask = self.fixed_mask | self.predicate_field.mask
        if self.inversion_field is not None:
            mask |= self.inversion_field.mask
        for field in self.own_fields:
            mask |= field.mask
        return mask

    def bind(self, name: str, value: int, *, consumed: bool = False) -> Bits:
        bits = self.fields[name].bind(value)
        if consumed and name in IMMEDIATES:
            return Bits(bits.mask, bits.value, 1 << IMMEDIATES.index(name), 0)
        if consumed and name in SCALARS:
            return Bits(bits.mask, bits.value, 0, 1 << SCALARS.index(name))
        return bits

    def fixed(self, predicate: int) -> Bits:
        """汇编谓词：15 为无谓词，0..13 为 pN，16 + N 为 !pN；p14 与 ALWAYS 编码相同，不能使用。"""
        register, inverted = predicate & 15, predicate >> 4
        if register == 15 and not inverted:
            register = ALWAYS
        elif register > 13:
            raise ValueError('predicate register must be p0..p13')
        if inverted and self.inversion_field is None:
            raise ValueError(f'{self.slot} has no inverted predicate')
        mask, value = self.predicate_field.mask, register << self.predicate_field.start
        if self.inversion_field is not None:
            mask |= self.inversion_field.mask
            value |= inverted << self.inversion_field.start
        for start, width, fixed in self.fixed_bits:
            mask |= ((1 << width) - 1) << start
            value |= fixed << start
        return Bits(mask, value)

    def predicate(self, word: int) -> int:
        """把机器字中的谓词还原为汇编谓词编号。"""
        register = self.predicate_field.read(word)
        inverted = self.inversion_field.read(word) if self.inversion_field is not None else 0
        return 15 if register == ALWAYS and not inverted else register + 16 * inverted

def _form(slot: str, branch: int, name: str, layout: int, fixed: tuple[tuple[int, int, int], ...], excludes: tuple[str, ...], mnemonic: str | None, tokens: tuple[str, ...]) -> GlForm:
    entries = FIELD_LAYOUTS[layout]
    fields = {**SHARED, **{field: Field(number, start, width) for field, number, start, width, _ in entries}}
    enums = {field: enum for field, _, _, _, enum in entries if enum}
    return GlForm(slot, branch, name, fields, tuple(field for field, *_ in entries), enums, fixed, excludes + _ONEOF.get(slot, ()), mnemonic, tokens)

FORMS = tuple(_form(*entry) for entry in INSTRUCTION_FORMS)
FORMS_BY_BRANCH = {(form.slot, form.branch): form for form in FORMS}
FORMS_BY_NAME = {(form.slot, form.name): form for form in FORMS}

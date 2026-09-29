"""TPU v4 TC 的形式登记与物理字段模型。"""
from __future__ import annotations

from dataclasses import dataclass

from .assembly_model import Bits, Field, Form
from .tpu_v4_tc_isa_data import FIELD_LAYOUTS, INSTRUCTION_FORMS, OPCODES

@dataclass(frozen=True)
class TcForm(Form):
    opcode: int
    mnemonic: str
    skeleton: tuple[str, ...]
    extra: tuple[tuple[str, int, int, int], ...]

    @property
    def opcode_field(self) -> Field:
        return Field(0, *OPCODES[self.slot])

    @property
    def predicate_field(self) -> Field:
        start, width = OPCODES[self.slot]
        return Field(1, start + width, 5)

    @property
    def writable_mask(self) -> int:
        mask = self.opcode_field.mask | self.predicate_field.mask
        for field in self.fields.values():
            mask |= field.mask
        for _, start, width, _ in self.extra:
            mask |= ((1 << width) - 1) << start
        return mask

    def bind(self, name: str, value: int, *, consumed: bool = False) -> Bits:
        bits = self.fields[name].bind(value)
        field = self.fields[name]
        if consumed and field.width == 16 and field.start in (338, 320, 304, 288, 272, 256):
            bits = Bits(bits.mask, bits.value, 1 << (338, 320, 304, 288, 272, 256).index(field.start), 0)
        elif consumed and field.width == 5 and field.start in (251, 246, 241):
            bits = Bits(bits.mask, bits.value, 0, 1 << (251, 246, 241).index(field.start))
        return bits

    def fixed(self, predicate: int) -> Bits:
        bits = self.opcode_field.bind(self.opcode).merge(self.predicate_field.bind(predicate))
        assert bits is not None
        for _, start, width, value in self.extra:
            bits = bits.merge(Field(0, start, width).bind(value))
            assert bits is not None
        return bits

FORMS = tuple(
    TcForm(slot, branch, name, {name: Field(number, start, width) for name, number, start, width in FIELD_LAYOUTS[layout]}, opcode, mnemonic, skeleton, extra)
    for slot, branch, name, opcode, layout, mnemonic, skeleton, extra in INSTRUCTION_FORMS
)
FORMS_BY_BRANCH = {(form.slot, form.branch): form for form in FORMS}

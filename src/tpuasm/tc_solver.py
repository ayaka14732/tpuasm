"""整个指令包的约束求解，按资源占用和机器字节字典序选择确定编码；TC 各代际共用。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .assembly_model import Bits, Field, Form, Signature, combine
from .assembly_operands import encode_operand, ordered_operands
from .assembly_syntax import AssemblyBundle, AssemblyInstruction, AssemblyProgram, EncodingConstraint
from .targets import HardwareTarget

@dataclass(frozen=True)
class TcIsa:
    """一个 TC 代际交给求解器的硬件描述。

    ``always_writable`` 是不论哪些槽存在都由编码器写入的位，例如 v6e 的共享立即数；``excludes`` 返回某个形式发射时不能同时占用的其他物理槽。``prefer_low_lanes`` 为真时，在占用数相同的编码中优先使用编号小的立即数与标量槽，再比较机器字节。
    """
    target: HardwareTarget
    word_bytes: int
    empty_word: int
    always_writable: int
    global_fields: dict[str, Field]
    bind_constraint: Callable[[EncodingConstraint, Form | None], Bits]
    signatures_by_mnemonic: dict[tuple[str, str], list[Signature]]
    writable_mask: Callable[[Form], int]
    excludes: Callable[[Form], tuple[str, ...]]
    prefer_low_lanes: bool = False

    def cost(self, bits: Bits) -> tuple[int, ...]:
        cost = (bits.immediates.bit_count(), bits.scalars.bit_count())
        return cost + (bits.immediates, bits.scalars) if self.prefer_low_lanes else cost

@dataclass(frozen=True)
class Candidate:
    form: Form
    bits: Bits

def candidates(isa: TcIsa, instruction: AssemblyInstruction, program: AssemblyProgram, pc: int) -> list[Candidate]:
    signatures = isa.signatures_by_mnemonic.get((instruction.slot, instruction.mnemonic), ())
    if not signatures:
        raise instruction.location.error(f'unsupported slot/mnemonic {instruction.mnemonic!r}', pc, instruction.slot)
    operands = instruction.operands
    if instruction.mnemonic in ('sbr.rel', 'sbr.abs', 'scall.rel', 'scall.abs'):
        index = 1 if instruction.mnemonic.startswith('scall') else 0
        if len(operands) > index:
            value = program.target(operands[index], pc, instruction.mnemonic.endswith('.rel'), instruction.location)
            operands = operands[:index] + (str(value),) + operands[index + 1:]
    result: list[Candidate] = []
    errors = []
    for signature in signatures:
        form = signature.form
        try:
            ordered = ordered_operands(signature, operands)
            if len(ordered) != len(signature.operands):
                raise ValueError(f'expected {len(signature.operands)} operands, got {len(ordered)}')
            bits = [form.fixed(instruction.predicate)]
            for name, value in signature.fixed_fields:
                bits = combine(bits, [form.bind(name, value)])
            for index, (expression, operand) in enumerate(zip(signature.operands, ordered)):
                matches = encode_operand(expression, operand, form)
                if not matches:
                    raise ValueError(f'operand {index + 1} {operand!r} does not match its type or range')
                bits = combine(bits, matches)
            result.extend(Candidate(form, value) for value in bits)
        except ValueError as error:
            errors.append(str(error))
    if not result:
        raise instruction.location.error('; '.join(dict.fromkeys(errors)) or 'operands require conflicting physical fields', pc, instruction.slot)
    return result

class BundleSolver:
    def __init__(self, isa: TcIsa, bundle: AssemblyBundle, program: AssemblyProgram, pc: int) -> None:
        self.isa = isa
        self.bundle = bundle
        self.pc = pc
        present = {instruction.slot for instruction in bundle.instructions}
        self.domains = []
        for instruction in bundle.instructions:
            found = candidates(isa, instruction, program, pc)
            allowed = [candidate for candidate in found if not present.intersection(isa.excludes(candidate.form))]
            if not allowed:
                blocked = ', '.join(sorted({slot for candidate in found for slot in isa.excludes(candidate.form) if slot in present}))
                raise instruction.location.error(f'{instruction.mnemonic} cannot issue in the same bundle as {blocked}', pc, instruction.slot)
            self.domains.append((instruction, allowed))

    def solve(self, constraints: tuple[EncodingConstraint, ...] | None = None) -> tuple[int, tuple[Form, ...]]:
        isa = self.isa
        pins = self.bundle.constraints if constraints is None else constraints
        global_bits = Bits()
        for pin in pins:
            if pin.name not in isa.global_fields:
                continue
            try:
                bound = isa.bind_constraint(pin, None)
                merged = global_bits.merge(bound)
                if merged is None:
                    raise ValueError(f'conflicting shared resource {pin.name}')
                global_bits = merged
            except ValueError as error:
                raise pin.location.error(str(error), self.pc) from error
        present = {instruction.slot for instruction, _ in self.domains}
        for pin in pins:
            if pin.name not in isa.global_fields and pin.name.split('.')[0] not in present:
                raise pin.location.error(f'unknown constraint or missing slot for {pin.name!r}', self.pc)
        domains = []
        for instruction, candidates_for_slot in self.domains:
            matching = []
            errors = []
            relevant = [pin for pin in pins if pin.name.startswith(instruction.slot + '.')]
            for candidate in candidates_for_slot:
                bits = candidate.bits.merge(global_bits)
                for pin in relevant:
                    try:
                        bound = isa.bind_constraint(pin, candidate.form)
                        bits = bits.merge(bound) if bits is not None else None
                    except ValueError as error:
                        errors.append(str(error))
                        bits = None
                    if bits is None:
                        break
                if bits is not None:
                    matching.append(Candidate(candidate.form, bits))
            if not matching:
                detail = '; '.join(dict.fromkeys(errors)) or 'encoding constraints conflict with visible operands'
                involved = [pin for pin in pins if pin.name in isa.global_fields or pin in relevant]
                detail += '; constraints: ' + ', '.join(f'{pin.name} = {pin.value} at {pin.location.line}:{pin.location.column}' for pin in involved)
                detail += f'; instruction: {instruction.mnemonic} {", ".join(instruction.operands)}'
                raise instruction.location.error(detail, self.pc, instruction.slot)
            domains.append((instruction, matching))
        domains.sort(key=lambda item: (len(item[1]), isa.target.slots.index(item[0].slot)))
        best_key = None
        best = None
        conflict_mask = 0
        all_writable = isa.always_writable
        for _, domain in domains:
            for candidate in domain:
                all_writable |= isa.writable_mask(candidate.form)
        if global_bits.mask & ~all_writable:
            names = ', '.join(pin.name for pin in pins if pin.name in isa.global_fields)
            raise self.bundle.location.error(f'shared constraint has no encodable field in this bundle: {names}', self.pc)
        default_lower = isa.empty_word & ~all_writable
        size = isa.word_bytes

        def search(index: int, bits: Bits, writable: int, selected: tuple[Form, ...]) -> None:
            nonlocal best, best_key, conflict_mask
            cost = isa.cost(bits)
            lower = (default_lower | bits.value).to_bytes(size, 'little')
            # Resource use only grows along the search, so the partial cost and word bound the result.
            if best_key is not None and (cost > best_key[0] or (cost == best_key[0] and lower >= best_key[1])):
                return
            if index == len(domains):
                if global_bits.mask & ~writable:
                    return
                word = (isa.empty_word & ~writable) | bits.value
                key = (cost, word.to_bytes(size, 'little'))
                if best_key is None or key < best_key:
                    best_key, best = key, (word, selected)
                return
            for candidate in sorted(domains[index][1], key=lambda item: (isa.cost(item.bits), item.bits.value.to_bytes(size, 'little'))):
                merged = bits.merge(candidate.bits)
                if merged is not None:
                    search(index + 1, merged, writable | isa.writable_mask(candidate.form), selected + (candidate.form,))
                else:
                    conflict_mask |= bits.mask & candidate.bits.mask & (bits.value ^ candidate.bits.value)

        search(0, global_bits, isa.always_writable, ())
        if best is None:
            locations = '; '.join(
                f'{instruction.slot} at {instruction.location.line}:{instruction.location.column}: {instruction.mnemonic} {", ".join(instruction.operands)}'
                for instruction, _ in domains
            )
            resources = ', '.join(sorted({pin.name for pin in pins} | {name for name, field in isa.global_fields.items() if field.mask & conflict_mask})) or 'shared hardware fields'
            raise self.bundle.location.error(f'conflicting {resources}; involved slots: {locations}', self.pc)
        return best

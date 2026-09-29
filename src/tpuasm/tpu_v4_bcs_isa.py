"""TPU v4 BCS 的物理字段与具名操作数；调度由程序作者负责。"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .assembly_model import Bits, Expression, Field, Form, Signature
from .tpu_v4_bcs_isa_data import ENUMS, FIELD_LAYOUTS, INSTRUCTION_FORMS
from .assembly_expressions import choice, direct, literal, memory, numeric, predicate, register, scalar_register

IMMEDIATES = {f'imm{i}': Field(0, start, 16) for i, start in enumerate((63, 47, 31, 15))}
EMPTY_WORD = (31 << 128) | (31 << 101)

@dataclass(frozen=True)
class BcsForm(Form):
    fixed_mask: int
    fixed_value: int
    enums: dict[str, str]

    @property
    def opcode_field(self) -> Field:
        return Field(0, 122 if self.slot == 's0' else 95, 6)

    @property
    def predicate_field(self) -> Field:
        return Field(1, 128 if self.slot == 's0' else 101, 5)

    def fixed(self, predicate: int) -> Bits:
        pred = self.predicate_field.bind(predicate)
        return Bits(self.fixed_mask | pred.mask, self.fixed_value | pred.value)

    def bind(self, name: str, value: int, *, consumed: bool = False) -> Bits:
        bits = self.fields[name].bind(value)
        used = 1 << int(name[-1]) if consumed and name in IMMEDIATES else 0
        return Bits(bits.mask, bits.value, used)

FORMS = tuple(
    BcsForm(
        slot,
        branch,
        name,
        {name: Field(number, start, width) for name, number, start, width, enum in FIELD_LAYOUTS[layout]},
        mask,
        value,
        {name: enum for name, number, start, width, enum in FIELD_LAYOUTS[layout] if enum},
    )
    for slot, branch, name, mask, value, layout in INSTRUCTION_FORMS
)
FORMS_BY_BRANCH = {(form.slot, form.branch): form for form in FORMS}

def enum_names(kind: str) -> dict[int, str]:
    prefix = re.sub(r'(?<!^)(?=[A-Z])', '_', kind).upper() + '_'
    return {value: name.removeprefix(prefix).lower() for name, value in ENUMS[kind].items()}

def enum_operand(field: str, kind: str) -> Expression:
    return choice(field, [(value, literal(name)) for value, name in enum_names(kind).items()])

def scalar_y(field: str) -> Expression:
    # Keep builtin selectors explicit: historical BCS documents disagree about
    # some constant values. Numeric literals use verified inline lane encodings.
    alternatives = [(i, literal(f's{i}')) for i in range(32)]
    for i in range(4):
        alternatives += [
            (32 + i, numeric('hex', 32, ((f'imm{i}', 0),))),
            (36 + i, numeric('hex', 32, ((f'imm{i}', 0),), 0xffff0000)),
            (40 + i, numeric('hex', 32, ((f'imm{i}', 16),))),
        ]
    alternatives += [
        (44, numeric('hex', 32, (('imm1', 16), ('imm0', 0)))),
        (45, numeric('hex', 32, (('imm3', 16), ('imm2', 0)))),
    ]
    alternatives.extend((value, literal('sy.' + name)) for value, name in enum_names('ScalarY').items() if value >= 46)
    return choice(field, alternatives)

_ALU = {
    'scalar_convert_int_to_float': 'sitofp',
    'scalar_convert_float_to_int': 'sftoi',
    'scalar_int_add': 'sadd.s32',
    'scalar_int_sub': 'ssub.s32',
    'scalar_and': 'sand.u32',
    'scalar_or': 'sor.u32',
    'scalar_xor': 'sxor.u32',
    'scalar_float_mul': 'smul.f32',
    'scalar_uint_mul': 'smul.u32',
    'scalar_float_add': 'sadd.f32',
    'scalar_float_sub': 'ssub.f32',
    'scalar_float_max': 'smax.f32',
    'scalar_float_min': 'smin.f32',
    'scalar_logical_shift_left': 'sshll.u32',
    'scalar_logical_shift_right': 'sshrl.u32',
    'scalar_arithmetic_shift_right': 'sshra.s32',
    'scalar_move': 'smov',
    'scalar_count_leading_zeros': 'sclz',
    'scalar_int_add_carry_out': 'sadd.carry',
    'scalar_is_inf_or_nan': 'sisinf_or_nan',
}
for _kind in ('int', 'float'):
    for _op, _suffix in (('equal', 'eq'), ('not_equal', 'ne'), ('greater', 'gt'), ('greater_equal', 'ge'), ('less', 'lt'), ('less_equal', 'le')):
        _ALU[f'scalar_{_kind}_{_op}'] = f'scmp.{_suffix}.{"s32" if _kind == "int" else "f32"}'

def signatures(form: BcsForm) -> tuple[Signature, ...]:
    name, fields = form.name, form.fields
    sreg = scalar_register
    if name in _ALU:
        comparison = name.startswith(('scalar_int_', 'scalar_float_')) and name.endswith(('equal', 'greater', 'less'))
        dest = register('dest', 'p' if comparison or name in ('scalar_is_inf_or_nan', 'scalar_int_add_carry_out') else 's')
        operands = [dest]
        # Subtraction has Y - X order in the existing BCS presentation contract.
        order = ('sy', 'sx') if name in ('scalar_int_sub', 'scalar_float_sub') else ('sx', 'sy')
        operands.extend(scalar_y(field) if field == 'sy' else sreg(field) for field in order if field in fields)
        return (Signature(form, _ALU[name], tuple(operands)),)
    if name == 'scalar_predicate_or':
        return (Signature(form, 'por', (register('dest', 'p'), predicate('px'), predicate('py'))),)
    if name.startswith(('scalar_branch_', 'scalar_call_')):
        suffix = name.rsplit('_', 1)[1]
        mnemonic = ('scall' if 'call' in name else 'sbr') + '.' + {'absolute': 'abs', 'relative': 'rel', 'reg': 'reg'}[suffix]
        operands = [sreg('return_address')] if 'call' in name else []
        operands.append(sreg('target_address') if suffix == 'reg' else direct(suffix + '_address', 16, 's32' if suffix == 'relative' else 'u32'))
        return (Signature(form, mnemonic, tuple(operands)),)
    if name in ('scalar_load_smem', 'scalar_load_smem_offset', 'scalar_store_smem_absolute'):
        address = memory('smem', scalar_y('address'))
        if name == 'scalar_load_smem_offset':
            # Keep the register offset separate so selector operands remain unambiguous.
            return (Signature(form, 'sld.offset', (sreg('dest'), address, sreg('offset'))),)
        return (Signature(form, 'sst' if 'store' in name else 'sld', (address, sreg('data')) if 'store' in name else (sreg('dest'), address)),)
    if name == 'scalar_read_registers':
        return (Signature(form, 'srdreg', (sreg('dest'), enum_operand('reg', 'BcsReadRegister'))),)
    if name == 'issue_fsm':
        return (Signature(form, 'issue.fsm', (enum_operand('dest', 'FsmInstruction'), scalar_y('fsm_address'))),)
    if name in ('read_done', 'read_public_access', 'write_done', 'write_public_access'):
        address = scalar_y('sflag_memory_offset')
        mnemonic = ('sdone' if name.endswith('done') else 'spublic') + ('.read' if name.startswith('read') else '.write')
        pair = (sreg('dest'), address) if name.startswith('read') else (address, sreg('write_data'))
        return (Signature(form, mnemonic, pair),)
    if name.startswith('sync_'):
        suffix = {'done': 'done', 'equal_to': 'eq', 'not_equal_to': 'ne', 'greater_than': 'gt', 'greater_or_equal_to': 'ge', 'less_than': 'lt', 'add': 'add'}[name[5:]]
        if suffix == 'add':
            return (Signature(form, 'ssyncadd.s32', (scalar_y('sync_flag'), sreg('increment'))),)
        wait_operands = (sreg('sync_flag'),) if suffix == 'done' else (sreg('sync_flag'), scalar_y('comparison_value'))
        return (Signature(form, 'swait.' + suffix, wait_operands),)
    fixed = {
        'scalar_halt': ('shalt', ()),
        'host_interrupt': ('interrupt', ()),
        'trace': ('trace', (scalar_y('operand'),)),
        'scalar_pop_hmf': ('spop.hmf', (sreg('dest'),)),
        'scalar_delay': ('sdelay', (direct('count', 11),)),
        'scalar_fence': ('sfence', (scalar_y('mask_bitmap'),)),
        'scalar_set_tag_register': ('ssettag', (scalar_y('reg_value'),)),
        'scalar_set_tracemark_register': ('ssettracemark', (scalar_y('reg_value'),)),
    }
    if name in fixed:
        mnemonic, fixed_operands = fixed[name]
        return (Signature(form, mnemonic, fixed_operands),)
    if name in ('scalar_dma_simple', 'scalar_dma_single_strided', 'scalar_general_dma'):
        # Hardware endpoint enums stay named; BCS runtime owns address domains.
        operands = []
        keys = []
        for field, layout in fields.items():
            if field in IMMEDIATES:
                continue
            kind = form.enums.get(field)
            operand = scalar_y(field) if kind == 'ScalarY' else enum_operand(field, kind) if kind else direct(field, layout.width)
            if not kind and field not in ('stride_count', 'outfeed_queue_id', 'trace'):
                operand = sreg(field)
            operands.append(operand)
            keys.append(field)
        mnemonic = {'scalar_dma_simple': 'dma.simple', 'scalar_dma_single_strided': 'dma.strided', 'scalar_general_dma': 'dma.general'}[name]
        return (Signature(form, mnemonic, tuple(operands), keywords=tuple(keys)),)
    raise ValueError(f'BCS instruction lacks an operand signature: {name}')

SIGNATURES = tuple(signature for form in FORMS for signature in signatures(form))
SIGNATURES_BY_MNEMONIC = {(signature.form.slot, signature.mnemonic): signature for signature in SIGNATURES}
SIGNATURES_BY_BRANCH = {(signature.form.slot, signature.form.branch): signature for signature in SIGNATURES}

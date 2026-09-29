"""TPU v4 TC 的助记符签名与共享操作数选择器。"""
from __future__ import annotations

from .assembly_expressions import literal, register, scalar_register, numeric, direct, choice, memory, predicate
from .assembly_model import Expression, Signature
from .tpu_v4_tc_model import FORMS, TcForm as Form

SCALAR_CONSTANTS = {46: 0x100, 47: 0xffffffff, **{48 + i: i for i in range(9)}, 57: 0x10, 58: 0x18, 59: 0x20, 60: 0x30, 61: 0x40, 62: 0x60, 63: 0x80}
VECTOR_CONSTANTS = {1: 1, 2: 0xffffffff, 3: 0, 4: 0x3f800000, 5: 0xbf800000, 6: 0x40000000, 7: 0x3f000000}
READ_PORTS = ('v0_x', 'v0_y_vreg', 'v1_x', 'v1_y_vreg', 'vst_source')
WRITE_PORTS = ('v0_dest', 'v1_dest', 'vld_dest', 'aux_dest')

def scalar_y(field: str, kind: str = 's32', mode: str = 'any') -> Expression:
    alternatives: list[tuple[int, Expression]] = []
    if mode != 'immediate':
        alternatives.extend((i, literal(f's{i}')) for i in range(32))
    if mode != 'register':
        for i in range(4):
            alternatives.extend((
                (32 + i, numeric(kind, 32, ((f'imm{i}', 0),))),
                (36 + i, numeric(kind, 32, ((f'imm{i}', 0),), 0xffff0000)),
                (40 + i, numeric(kind, 32, ((f'imm{i}', 16),))),
            ))
        alternatives.extend((
            (44, numeric(kind, 32, (('imm1', 16), ('imm0', 0)))),
            (45, numeric(kind, 32, (('imm3', 16), ('imm2', 0)))),
        ))
        alternatives.extend((selector, numeric(kind, 32, base=value)) for selector, value in SCALAR_CONSTANTS.items())
    return choice(field, alternatives)

def vector_y(kind: str = 's32', mode: str = 'any') -> Expression:
    alternatives: list[tuple[int, Expression]] = []
    if mode != 'immediate':
        alternatives.append((0, register('y_src_vreg', 'v')))
        alternatives.extend((29 + i, register(f'vs{i}', 's')) for i in range(3))
    if mode != 'register':
        alternatives.extend((selector, numeric(kind, 32, base=value)) for selector, value in VECTOR_CONSTANTS.items())
        for i in range(6):
            alternatives.extend((
                (8 + i, numeric(kind, 32, ((f'imm{i}', 0),))),
                (14 + i, numeric(kind, 32, ((f'imm{i}', 0),), 0xffff0000)),
                (20 + i, numeric(kind, 32, ((f'imm{i}', 16),))),
            ))
        alternatives.extend((26 + i, numeric(kind, 32, ((f'imm{2 * i + 1}', 16), (f'imm{2 * i}', 0)))) for i in range(3))
    return choice('y_src', alternatives)

def msc(field: str, kind: str = 'u32', zero: int = 0, *, no_zero: bool = False) -> Expression:
    alternatives = [] if no_zero else [(0, numeric(kind, 16, base=zero))]
    alternatives.extend((i + 1, register(f'vs{i}', 's')) for i in range(3))
    alternatives.extend((i + 4, numeric(kind, 16, ((f'imm{i + 2}', 0),))) for i in range(4))
    return choice(field, alternatives)

def vector_address(form: Form, space: str) -> Expression:
    base = choice('base_address', [(0, numeric('u32', 16, base=0))] + [(i + 1, register(f'vs{i}', 's')) for i in range(3)])
    offset = choice('offset', [(i, direct(f'imm{i + 2}', 16, 'hex')) for i in range(4)]) if 'offset' in form.fields else None
    modifiers = tuple((key, field, msc(field, zero=zero, no_zero=True)) for key, field, zero in (('sm', 'sublane_mask', 0), ('ss', 'stride', 1)) if field in form.fields)
    return memory(space, base, offset, modifiers=modifiers)

def read_port() -> Expression:
    return choice('read_port', [(i, register(field, 'v')) for i, field in enumerate(READ_PORTS)])

def write_port() -> Expression:
    return choice('dest', [(i, register(field, 'v')) for i, field in enumerate(WRITE_PORTS)])

def _scalar(form: Form) -> list[Signature]:
    f, name, mnemonic = form.fields, form.name, form.mnemonic
    sreg = scalar_register
    if name == 'ScalarSetRegister':
        return [Signature(form, 'ssettag', (scalar_y('reg_value', 'hex'),), (('target', 1),))]
    if name == 'ScalarReadRegisters':
        names = {0: 'lcclo', 1: 'lcchi', 2: 'gtclo', 3: 'gtchi'}
        return [Signature(form, 'srdreg.' + suffix, (sreg('dest'),), (('reg', value),)) for value, suffix in names.items()]
    if name in ('ScalarDmaSimple', 'ScalarDmaSingleStrided', 'ScalarGeneralDma'):
        operands = [('dma_address', True), ('dma_address', False), scalar_y('length', 'u32')]
        keys = ['length']
        if name == 'ScalarDmaSingleStrided':
            operands.extend((sreg('dest_stride'), sreg('source_stride'), sreg('elements_per_stride')))
            keys.extend(('dst_stride', 'src_stride', 'elements_per_stride'))
        elif name == 'ScalarGeneralDma':
            operands.extend((memory('smem', scalar_y('stride_descriptor', 'hex')), direct('stride_count', 2), memory('sflag', sreg('source_sync_flag'))))
            keys.extend(('stride_descriptor', 'stride_count', 'src_flag'))
        flag = sreg('dest_sync_flag') if name == 'ScalarGeneralDma' else scalar_y('dest_sync_flag', 'u32')
        operands.append(memory('sflag', flag))
        keys.append('dst_flag')
        if name == 'ScalarGeneralDma':
            operands.append(sreg('ici_dest'))
            keys.append('ici_dest')
        return [Signature(form, mnemonic + suffix, tuple(operands), (('trace', trace),), tuple(keys)) for trace, suffix in ((0, ''), (1, '.trace'))]
    if name == 'ScalarMove':
        return [Signature(form, op, (sreg('dest'), scalar_y('sy', mode=mode))) for op, mode in (('smov', 'register'), ('simm.s32', 'immediate'))]
    if name == 'ScalarPredicateOr':
        return [Signature(form, mnemonic, (register('dest', 'p'), predicate('px'), predicate('py')))]
    if 'Branch' in name or 'Call' in name:
        operands = [sreg('return_address')] if 'Call' in name else []
        if 'relative_address' in f:
            operands.append(direct('relative_address', 16, 's32'))
        elif 'absolute_address' in f:
            operands.append(direct('absolute_address', 16))
        else:
            operands.append(sreg('target_address'))
        return [Signature(form, mnemonic, tuple(operands))]
    if name in ('ScalarLoadSmem', 'ScalarLoadSmemOffset', 'ScalarStoreSmemAbsolute', 'ScalarDescriptorDma'):
        if name == 'ScalarLoadSmemOffset':
            address = memory('smem', sreg('offset'), scalar_y('address', 'hex'))
        else:
            address = memory('smem', scalar_y('smem_address' if name == 'ScalarDescriptorDma' else 'address', 'hex'))
        memory_operands = (address,) if name == 'ScalarDescriptorDma' else ((address, sreg('data')) if name == 'ScalarStoreSmemAbsolute' else (sreg('dest'), address))
        return [Signature(form, mnemonic, memory_operands)]
    if name == 'ScalarDelay':
        return [Signature(form, mnemonic, (direct('count', 11),))]
    if name == 'ScalarPopV2s':
        return [Signature(form, mnemonic, (sreg('dest'), literal('v2sf')))]
    if name in ('ScalarHalt', 'ScalarFence'):
        return [Signature(form, mnemonic, ())]
    kind = 'f32' if 'Float' in name and 'Convert' not in name else ('hex' if mnemonic.endswith('.u32') or 'Shift' in name or name == 'ScalarConvertFloatToInt' else 's32')
    if name == 'ScalarUintMul':
        kind = 's32'
    dest = register('dest', 'p' if form.skeleton[0].startswith('p') else 's')
    operands = [dest]
    if 'sx' in f:
        operands.append(sreg('sx'))
    if 'sy' in f:
        operands.append(scalar_y('sy', kind))
    y_first = ('ScalarIntAdd', 'ScalarIntSub', 'ScalarAnd', 'ScalarOr', 'ScalarXor', 'ScalarFloatMul', 'ScalarUintMul', 'ScalarFloatAdd', 'ScalarFloatSub')
    if name in y_first:
        operands[1:] = reversed(operands[1:])
    return [Signature(form, mnemonic, tuple(operands))]

def _vector_alu(form: Form) -> list[Signature]:
    name, f = form.name, form.fields
    if name == 'VectorMove':
        return [Signature(form, op, (register('dest', 'v'), vector_y(mode=mode))) for op, mode in (('vmov.8x128', 'register'), ('vimm.8x128.s32', 'immediate'))]
    kind = 'f32' if ('Float' in name or name in ('VectorRelux', 'VectorClampSymmetric')) and name not in ('VectorConvertFloatToInt', 'VectorComposeFloat') else 's32'
    if name in ('VectorAnd', 'VectorOr', 'VectorXor', 'VectorConvertFloatToInt', 'VectorComposeFloat', 'CreateSublaneMask', 'CreateLaneMask') or 'Shift' in name or name.startswith('VectorSelect'):
        kind = 'hex'
    operands = []
    if 'dest' in f:
        operands.append(register('dest', 'v'))
    elif 'vmdest' in f:
        operands.append(register('vmdest', 'vm'))
    elif form.skeleton and form.skeleton[0] == 'erf':
        operands.append(literal('erf'))
    if name.startswith('VectorSelectVmsk'):
        operands.append(literal('vm' + name[-1]))
    sources = []
    if 'vx' in f:
        sources.append(register('vx', 'v'))
    if 'y_src' in f:
        sources.append(vector_y(kind))
    y_first = name.startswith(('VectorSelect', 'VectorPack')) or name in (
        'VectorIntAdd', 'VectorIntSub', 'VectorAnd', 'VectorOr', 'VectorXor', 'VectorFloatMul', 'VectorFloatAdd', 'VectorFloatSub', 'VectorComposeFloat',
    )
    if y_first:
        sources.reverse()
    return [Signature(form, form.mnemonic, tuple(operands + sources))]

def _vector_memory(form: Form) -> list[Signature]:
    f, name = form.fields, form.name
    if name.startswith('SetIar'):
        return [Signature(form, form.mnemonic, (register('iar', 'iar'), register('vsrc', 'v')))]
    if name == 'PushV2s':
        return [Signature(form, form.mnemonic, (literal('v2sf'), register('source_vreg', 'v')))]
    address = vector_address(form, 'cmem' if name.startswith('Cmem') else 'vmem')
    if form.slot == 'cld':
        operands = [literal('crf'), address]
    elif form.slot == 'vld':
        operands = [register('dest', 'v'), address]
        if 'shuffle' in f:
            alternatives = [(i + 1, register(f'vs{i}', 's')) for i in range(3)]
            alternatives.extend((i + 4, numeric('hex', 32, ((f'imm{2 * i}', 0), (f'imm{2 * i + 1}', 16)))) for i in range(3))
            operands.append(choice('shuffle', alternatives))
    else:
        operands = [address]
        if 'iar' in f:
            operands.append(register('iar', 'iar'))
        if 'Vmsk' in name:
            operands.append(literal('vm' + name[-1]))
        operands.append(register('source', 'v'))
    return [Signature(form, form.mnemonic, tuple(operands))]

def _extended(form: Form) -> list[Signature]:
    f, name = form.fields, form.name
    if name.startswith('DoneWithGains'):
        return [Signature(form, form.mnemonic, (register('mxu', 'gmr'), register('mxu', 'gsft' if name.endswith('Gsft') else 'gsfn')))]
    variants = []
    for unit in range(4) if 'xlu_and_source_bus' in f else (None,):
        mnemonic = form.mnemonic.replace('.0.', f'.{unit}.', 1) if unit is not None else form.mnemonic
        dest = literal(form.skeleton[0])
        if 'mxu' in f:
            dest = register('mxu', 'gsft' if 'Transposed' in name else 'gsfn')
        elif unit is not None:
            dest = literal(form.skeleton[0][:-1] + str(unit & 1))
        operands = [dest]
        if 'vmsk' in f:
            operands.append(register('vmsk', 'vm'))
        operands.append(read_port())
        if 'matrix_width' in f:
            operands.append(msc('matrix_width', zero=128))
        if 'rotate_count' in f:
            operands.append(msc('rotate_count', 'hex', zero=1))
        fixed = (('xlu_and_source_bus', unit),) if unit is not None else ()
        variants.append(Signature(form, mnemonic, tuple(operands), fixed))
    return variants

def _result(form: Form) -> list[Signature]:
    source = register('mrf_number', 'mrf') if 'mrf_number' in form.fields else (register('trf', 'trf') if 'trf' in form.fields else literal(form.skeleton[1]))
    return [Signature(form, form.mnemonic, (write_port(), source))]

def _misc(form: Form) -> list[Signature]:
    name, mnemonic, f = form.name, form.mnemonic, form.fields
    if name == 'Trace':
        return [Signature(form, mnemonic, (('trace',),))]
    if name == 'MoveVmsk':
        return [Signature(form, 'vnop', ())]
    if name == 'DelayFixed':
        return [Signature(form, 'vdelay', (direct('delay_count', 4, bias=1),))]
    if name == 'CmemFence':
        return [Signature(form, mnemonic, ())]
    if name == 'SetTracemark':
        return [Signature(form, mnemonic, (msc('operand'),))]
    if name == 'HostInterrupt':
        return [Signature(form, mnemonic, (msc('interrupt_number'),))]
    if name == 'Delay':
        # Selector 0 matches immediate 1 in device cycle probes, despite the formatter's 0.
        return [Signature(form, mnemonic, (msc('delay_count', zero=1),))]
    if name.endswith('Vmsk'):
        operands = [register('vmdest', 'vm'), register('vmsrc1', 'vm')]
        if 'vmsrc2' in f:
            operands.append(register('vmsrc2', 'vm'))
        return [Signature(form, mnemonic, tuple(operands))]
    flag = 'sync_flag_number' if 'sync_flag_number' in f else 'dest'
    operands = [memory('sflag', msc(flag))]
    if name in ('ReadSyncFlag', 'ReadSyncDone'):
        operands.insert(0, literal('v2sf'))
    elif 'operand' in f and name != 'DelayUntilDone':
        operands.append(msc('operand', 's32'))
    if 'done_control' in f:
        return [Signature(form, mnemonic.replace('.s32', suffix + '.s32'), tuple(operands), (('done_control', i),)) for i, suffix in enumerate(('', '.done', '.notdone'))]
    return [Signature(form, mnemonic, tuple(operands))]

def signatures(form: Form) -> list[Signature]:
    if form.slot in ('s0', 's1'):
        return _scalar(form)
    if form.slot in ('va0', 'va1'):
        return _vector_alu(form)
    if form.slot in ('vld', 'vst', 'cld'):
        return _vector_memory(form)
    if form.slot in ('vx0', 'vx1'):
        return _extended(form)
    if form.slot in ('vr0', 'vr1'):
        return _result(form)
    return _misc(form)

SIGNATURES = tuple(signature for form in FORMS for signature in signatures(form))
SIGNATURES_BY_MNEMONIC: dict[tuple[str, str], list[Signature]] = {}
SIGNATURES_BY_BRANCH: dict[tuple[str, int], list[Signature]] = {}
for _signature in SIGNATURES:
    SIGNATURES_BY_MNEMONIC.setdefault((_signature.form.slot, _signature.mnemonic), []).append(_signature)
    SIGNATURES_BY_BRANCH.setdefault((_signature.form.slot, _signature.form.branch), []).append(_signature)

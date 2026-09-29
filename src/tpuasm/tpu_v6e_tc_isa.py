"""TPU v6e TC 的助记符签名与共享操作数选择器。

助记符、操作数顺序和由单元号决定的固定文本取自 libtpu formatter（见 tpu_v6e_tc_isa_data），操作数的写法由字段类型决定：寄存器写作 sN、vN、vmN、pN，选择器写作寄存器或数值，数值所需的共享立即数由求解器分配。
"""
from __future__ import annotations

import os
import re

from .assembly_expressions import choice, direct, literal, memory, numeric, register, table
from .assembly_model import Expression, Signature
from .tpu_v6e_tc_isa_data import ENUMS, MNEMONIC_VARIANTS, TOKEN_FIELDS, TOKEN_TABLES
from .tpu_v6e_tc_model import FORMS, GlForm, SHARED

LANE = SHARED['imm0'].width
# formatter 不输出、只作固定文本的目的操作数：助记符已经说明写入 PC、tag 或 tracemark。
IMPLIED = frozenset(('(pc)', '(tag)', '(tm)'))
# DMA 端点：(core id, memory id) 在 formatter 中的空间名；其余组合没有 formatter 名称。
DMA_SPACES = {
    (0, 0): 'vmem',
    (0, 1): 'smem',
    (0, 2): 'imem',
    (1, 0): 'hbm',
    (1, 1): 'host',
    (1, 2): 'vmem_all',
    (2, 0): 'vmem0',
    (2, 1): 'smem0',
    (2, 2): 'imem0',
    **{(4 + core, memory): f'{name}{core}' for core in range(2) for memory, name in enumerate(('spmem', 'ssmem', 'simem', 'timem'))},
}
# formatter 没有输出的形式使用的助记符；按同族已有助记符的写法命名。
FALLBACK_MNEMONICS = {
    'vector_f32_remap': 'vremap.8x128.f32',
    'vector_bf16_remap': 'vremap.8x128.bf16',
    'vector_u16_carry': 'vc.8x128.u16',
    'vector_bf16_inf_or_nan': 'vweird.8x128.bf16',
    'vector_u16_multiply': 'vmul.8x128.u16',
    'set_pattern_register_pcr_half_sublanes': 'vsetperm.half.u8',
    'lane_rotate_packed_8_bit': 'vrot.lane.packed.b8.8x128',
    'lane_broadcast_packed_8_bit': 'vbcast.lane.packed.b8.8x128',
    **{f'sync_{name}_yieldable': f'vwait.{suffix}.yield' for name, suffix in (('equal', 'eq'), ('not_equal', 'ne'), ('greater_than', 'gt'), ('greater_equal', 'ge'), ('less_than', 'lt'), ('done', 'done'), ('not_done', 'notdone'))},
}
_WORDS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'sixteen': 16}

def value_names(kind: str) -> dict[int, str]:
    """descriptor 枚举值去掉公共前缀后的小写名称，用作命名编码约束的取值。"""
    names = ENUMS[kind]
    prefix = os.path.commonprefix(list(names.values()))
    prefix = prefix[:prefix.rfind('_') + 1]
    return {value: name[len(prefix):].lower() for value, name in names.items()}

def constant(name: str) -> int | None:
    """选择器的内置常量；名称取自 descriptor，数值经 v6e 设备核对。"""
    name = re.sub(r'^(?:width_|raw_|int_)', '', name)
    if name in _WORDS:
        return _WORDS[name]
    if name == 'negative_one':
        return -1
    if re.fullmatch(r'[0-9]+', name):
        return int(name)
    if match := re.fullmatch(r'hex_([0-9a-f]+)', name):
        return int(match[1], 16)
    if match := re.fullmatch(r'x([0-9a-f]+)', name):
        return int(match[1], 16)
    if match := re.fullmatch(r'descending_pattern_([0-7])', name):
        shift = 4 * int(match[1])
        return (0x76543210 >> shift | 0x76543210 << (32 - shift)) & 0xffffffff
    return {'float_one': 0x3f800000, 'float_negative_one': 0xbf800000, 'float_one_half': 0x3f000000}.get(name)

def source(name: str, kind: str, width: int) -> Expression | None:
    """一个选择器取值对应的操作数；共享立即数的位宽为 20；拼接只取各字段低 16 位，已在 v6e 上核对。"""
    if match := re.fullmatch(r'sreg([0-9]+)', name):
        return literal(f's{match[1]}')
    if match := re.fullmatch(r'(?:width_)?vs_?([0-3])', name):
        return register(f'vs{match[1]}', 's')
    if match := re.fullmatch(r'(?:width_)?imm_?([0-5])', name):
        return numeric(kind, LANE, ((f'imm{match[1]}', 0),))
    if match := re.fullmatch(r'zero_imm([0-5])', name):
        return numeric(kind, 32, ((f'imm{match[1]}', 0),))
    if match := re.fullmatch(r'ones_imm([0-5])', name):
        return numeric(kind, 32, ((f'imm{match[1]}', 0),), (0xffffffff << LANE) & 0xffffffff)
    if match := re.fullmatch(r'imm([0-5])_zero', name):
        return numeric(kind, 32, ((f'imm{match[1]}', 32 - LANE),))
    if match := re.fullmatch(r'imm([0-5])_imm([0-5])', name):
        return numeric(kind, 32, ((f'imm{match[1]}', 16), (f'imm{match[2]}', 0)), part_width=16)
    value = constant(name)
    if value is not None:
        return numeric(kind, width, base=value & ((1 << width) - 1))
    return None

def selector(field: str, kind_name: str, kind: str, *, skip: tuple[int, ...] = ()) -> Expression:
    names = value_names(kind_name)
    width = 32 if any(re.fullmatch(r'zero_imm.|ones_imm.|imm._zero|imm._imm.|descending_pattern_.|float_.*', name) for name in names.values()) else LANE
    alternatives = []
    for value, name in names.items():
        operand = source(name, kind, width)
        if operand is not None and value not in skip:
            alternatives.append((value, operand))
    return choice(field, alternatives)

def scalar_y(field: str, kind: str = 's32', mode: str = 'any') -> Expression:
    """ScalarY：0..31 为标量寄存器本身，其余为共享立即数或内置常量。"""
    alternatives = []
    for value, name in value_names('ScalarY').items():
        if (mode == 'register') != name.startswith('sreg') and mode != 'any':
            continue
        operand = source(name, kind, 32)
        assert operand is not None
        alternatives.append((value, operand))
    return choice(field, alternatives)

def vector_y(kind: str = 's32', mode: str = 'any') -> Expression:
    """VectorY：y 寄存器、标量操作数槽、共享立即数或内置常量。"""
    alternatives = []
    for value, name in value_names('VectorY').items():
        operand = register('y_vreg', 'v') if name == 'vreg' else source(name, kind, 32)
        assert operand is not None
        is_register = operand[0] == 'register'
        if mode == 'any' or (mode == 'register') == is_register:
            alternatives.append((value, operand))
    return choice('y_src', alternatives)

def predicate_or(field: str) -> Expression:
    """PredicationOr：pN、!pN 与常量；取反的常量是同值别名，放在后面只用于解码。"""
    names = value_names('PredicationOr')
    alternatives: list[tuple[int, Expression]] = []
    for value, name in names.items():
        if match := re.fullmatch(r'preg([0-9]+)_is_([01])', name):
            alternatives.append((value, literal(('!' if match[2] == '0' else '') + f'p{match[1]}')))
    alternatives.extend(((14, literal('1')), (15, literal('0')), (30, literal('0')), (31, literal('1'))))
    return choice(field, alternatives)

def predicate_register(field: str) -> Expression:
    names = value_names('PredicateDest')
    return choice(field, [(value, literal(f'p{value}')) for value, name in names.items() if name.startswith('preg')])

def lane_register(field: str) -> Expression:
    """两位字段选择一个标量操作数槽，操作数是该槽中的标量寄存器。"""
    return choice(field, [(index, register(f'vs{index}', 's')) for index in range(4)])

def port(field: str, kind_name: str) -> Expression:
    """经由其他槽读端口读取的向量寄存器；端口名与本形式中保存寄存器号的字段同名。"""
    return choice(field, [(value, register(name, 'v')) for value, name in value_names(kind_name).items()])

def vector_address(form: GlForm, space: str = 'vmem') -> Expression:
    base = selector('base_address', 'VectorBase', 'u32') if 'base_address' in form.layout else numeric('u32', LANE, base=0)
    offset = selector('offset', 'VectorOffset', 'hex') if 'offset' in form.layout else None
    modifiers = []
    # The default selector (all sublanes, unit stride) is written by omitting the modifier.
    if 'sublane_mask' in form.layout:
        modifiers.append(('sm', 'sublane_mask', selector('sublane_mask', 'SublaneMask', 'hex', skip=(0,))))
    if 'stride' in form.layout:
        modifiers.append(('ss', 'stride', selector('stride', 'VectorStride', 's32', skip=(0,))))
    return memory(space, base, offset, modifiers=tuple(modifiers))

def mnemonic_of(form: GlForm) -> str:
    mnemonic = form.formatter_mnemonic or FALLBACK_MNEMONICS[form.name]
    return re.sub(r'<alu[0-9]>', '', mnemonic)

def value_kind(mnemonic: str) -> str:
    """数值操作数的显示类型：浮点运算写十进制浮点数，按位运算和打包类型写十六进制。"""
    parts = mnemonic.split('.')
    types = [part for part in parts[1:] if re.fullmatch(r'[subf]+(?:f)?[0-9]+|if8|bf8|hf16|bf16', part)]
    operation = parts[0]
    if operation in ('scvt', 'vcvt') and types:
        return 'f32' if types[0] == 'f32' else 's32'
    if any(item in types for item in ('bf16', 'hf16', 'bf8', 'if8')) or operation in ('sand', 'sor', 'sxor', 'sshll', 'sshrl', 'sshra', 'sshla', 'sclz', 'ssettag', 'vand', 'vor', 'vxor', 'vshll', 'vshrl', 'vshra', 'vsel', 'vnsel', 'vcmask', 'vsmask', 'vnez', 'vsetacc'):
        return 'hex'
    if 'f32' in types:
        return 'f32'
    if any(item.startswith(('u', 'b')) for item in types):
        return 'hex'
    return 's32'

def _plain(text: str) -> str:
    """去掉单个隐含寄存器外的括号，例如 `(v2sf)` 写作 `v2sf`；多个寄存器保留为元组。"""
    return text[1:-1] if re.fullmatch(r'\([a-z0-9_]+\)', text) else text

Operand = tuple[tuple[str, ...], Expression]

def arrange(form: GlForm, operands: list[Operand]) -> tuple[Expression, ...]:
    """按 formatter 的记号顺序排列操作数；没有 formatter 输出时按给定顺序。"""
    owners = TOKEN_FIELDS.get((form.slot, form.name), {})
    tables = TOKEN_TABLES.get((form.slot, form.name), {})
    result: list[Expression] = []
    used: set[int] = set()
    for index, text in enumerate(form.formatter_tokens):
        if index in tables:
            names, entries = tables[index]
            result.append(table(tuple((_plain(label), tuple(zip(names, values))) for values, label in entries)))
            continue
        names = owners.get(index, ())
        position = next((position for position, (fields, _) in enumerate(operands) if set(fields) & set(names) or (fields == ('',) and text.startswith('$'))), None)
        if position is None and text.startswith('['):
            position = next((position for position, (fields, expression) in enumerate(operands) if expression[0] == 'memory' and position not in used), None)
        if position is None:
            if names:
                raise ValueError(f'{form.slot} {form.name}: no operand for formatter token {text!r} of {names}')
            if text not in IMPLIED:
                result.append(literal(_plain(text)))
            continue
        if position not in used:
            used.add(position)
            result.append(operands[position][1])
    result.extend(expression for position, (_, expression) in enumerate(operands) if position not in used)
    return tuple(result)

def _variants(form: GlForm, operands: tuple[Expression, ...], mnemonic: str, keywords: tuple[str, ...] = (), defaults: tuple[tuple[str, str], ...] = ()) -> list[Signature]:
    """字段只改变 formatter 助记符时，每种取值组合对应一个带固定字段的签名。"""
    if (form.slot, form.name) not in MNEMONIC_VARIANTS:
        return [Signature(form, mnemonic, operands, keywords=keywords, defaults=defaults)]
    names, entries = MNEMONIC_VARIANTS[(form.slot, form.name)]
    return [Signature(form, re.sub(r'<alu[0-9]>', '', text), operands, tuple(zip(names, values)), keywords, defaults) for values, text in entries]

def _scalar(form: GlForm) -> list[Signature]:
    name, mnemonic = form.name, mnemonic_of(form)
    kind = value_kind(mnemonic)
    if name == 'move_y':
        return [Signature(form, 'smov', (register('dest', 's'), scalar_y('y', mode='register'))), Signature(form, 'simm.s32', (register('dest', 's'), scalar_y('y', mode='immediate')))]
    operands: list[Operand] = []
    for field in form.layout:
        enum = form.enums.get(field)
        if name == 'scalar_load_smem_x_y' and field in ('x', 'y'):
            if field == 'y':
                operands.append((('x', 'y'), memory('smem', register('x', 's'), scalar_y(field, 'hex'))))
        elif field in ('dest', 'x'):
            operands.append(((field,), register(field, 's')))
        elif field == 'pdst':
            operands.append(((field,), predicate_register(field)))
        elif enum == 'PredicationOr':
            operands.append(((field,), predicate_or(field)))
        elif field == 'delay_count':
            operands.append(((field,), direct(field, form.fields[field].width)))
        elif enum == 'ScalarY' and name in ('scalar_load_smem_y', 'descriptor_based_dma', 'scalar_store_x_to_smem_y'):
            operands.append(((field,), memory('smem', scalar_y(field, 'hex'))))
        elif enum == 'ScalarY':
            operands.append(((field,), scalar_y(field, 'hex' if name == 'set_tag' else kind)))
        else:
            raise ValueError(f'unhandled scalar field {name}.{field}')
    # Branch targets sit in imm0 without a selector field; '' matches the formatter's '$' token.
    if name in ('branch_absolute', 'call_absolute'):
        operands.append((('',), direct('imm0', LANE)))
    elif name in ('branch_relative', 'call_relative'):
        operands.append((('',), numeric('s32', LANE, (('imm0', 0),))))
    return [Signature(form, mnemonic, arrange(form, operands))]

def _dma_address(prefix: str, sreg_field: str, vreg_field: str | None, in_vreg: str | None) -> Expression:
    """DMA 端点：core id 与 memory id 选出空间，地址寄存器可以是标量或向量寄存器。"""
    def inner(space: str) -> Expression:
        scalar = memory(space, register(sreg_field, 's'))
        if in_vreg is None or vreg_field is None:
            return scalar
        return choice(in_vreg, [(0, scalar), (1, memory(space, register(vreg_field, 'v')))])

    cores = []
    for core in range(8):
        spaces = [(item, inner(DMA_SPACES.get((core, item), f'core{core}_mem{item}'))) for item in range(4)]
        cores.append((core, choice(f'{prefix}_mem_mem_id', spaces)))
    return choice(f'{prefix}_mem_core_id', cores)

def _named(field: str, kind_name: str) -> Expression:
    return choice(field, [(value, literal(name)) for value, name in value_names(kind_name).items()])

def _dma(form: GlForm) -> list[Signature]:
    """DMA 的两个端点按位置写出，其余字段写作具名操作数；formatter 不显示的字段可以省略，省略时取 0。"""
    fields = form.layout
    positional = [
        _dma_address('dst', 'dma_sreg_dest_offset', 'dma_vreg_dest_offset' if 'dest_in_vreg' in fields else None, 'dest_in_vreg' if 'dest_in_vreg' in fields else None),
        _dma_address('src', 'dma_sreg_source_offset', 'dma_vreg_source_offset' if 'source_in_vreg' in fields else None, 'source_in_vreg' if 'source_in_vreg' in fields else None),
    ]
    named: list[tuple[str, Expression]] = [('length', scalar_y('dma_length', 'u32'))]
    if 'dest_sync_flags' in fields:
        named.append(('dst_flag', memory('sflag', scalar_y('dest_sync_flags', 'u32'))))
    if 'dest_sync_flags_vs1' in fields:
        named.append(('dst_flag', memory('sflag', register('dest_sync_flags_vs1', 's'))))
    if form.name == 'single_strided_dma':
        named.extend((
            ('dst_stride', register('destination_stride', 's')),
            ('src_stride', register('source_stride', 's')),
            ('elements_per_stride', register('elements_per_stride', 's')),
            ('inner_vector_length', direct('inner_vector_length', form.fields['inner_vector_length'].width)),
        ))
    if form.name == 'general_dma':
        named.extend((
            ('src_flag', memory('sflag', register('source_sync_flag_number', 's'))),
            ('stride_descriptor', memory('smem', scalar_y('stride_info_location', 'hex'))),
            ('stride_count', direct('nondefault_stride_dimensions', form.fields['nondefault_stride_dimensions'].width)),
            ('ici_dest', register('destination_id', 's')),
        ))
    optional: list[tuple[str, Expression, str]] = [
        ('opcode', _named('dst_opcode', 'DmaDestOpcode'), 'write'),
        ('sync', _named('sync_mode', 'SyncMode'), '32b_word'),
        ('thread', direct('thread_select', 1), '0'),
        ('relaxed', direct('relaxed_ordering_or_thread_select_1', 1), '0'),
        ('host_upper', register('upper_host_addr', 's'), 's0'),
    ]
    if 'dma_vreg_source_operand' in fields:
        optional.append(('src_operand', _named('dma_vreg_source_operand', 'VectorY'), 'vreg'))
    keywords = tuple(key for key, _ in named) + tuple(key for key, _, _ in optional)
    operands = tuple(positional) + tuple(expression for _, expression in named) + tuple(expression for _, expression, _ in optional)
    defaults = tuple((key, default) for key, _, default in optional)
    return _variants(form, operands, mnemonic_of(form), keywords, defaults)

def _vector_alu(form: GlForm) -> list[Signature]:
    name, mnemonic = form.name, mnemonic_of(form)
    kind = value_kind(mnemonic)
    if name == 'vector_move':
        return [Signature(form, 'vmov.8x128', (register('dest', 'v'), vector_y(mode='register'))), Signature(form, 'vimm.8x128.s32', (register('dest', 'v'), vector_y(mode='immediate')))]
    if name == 'dynamic_vector_unpack':
        # FormatterGl prints the lane number as sN; the device reads the SREG held in vsN.
        return _variants(form, (register('dest', 'v'), register('x', 'v'), lane_register('vs')), mnemonic)
    owners = TOKEN_FIELDS.get((form.slot, name), {})
    mask_result = any(form.formatter_tokens[index].startswith('vm') for index, names in owners.items() if 'dest' in names) or (not form.formatter_tokens and re.search(r'carry|inf_or_nan', name) is not None)
    operands: list[Operand] = []
    for field in form.layout:
        if field == 'dest':
            operands.append(((field,), register(field, 'vm' if mask_result else 'v')))
        elif field in ('dest_upper', 'x'):
            operands.append(((field,), register(field, 'v')))
        elif field == 'y_src':
            operands.append((('y_src', 'y_vreg'), vector_y(kind)))
        elif field == 'y_vreg' and 'y_src' not in form.layout:
            operands.append(((field,), register(field, 'v')))
        elif field == 'vmsk':
            operands.append(((field,), register(field, 'vm')))
        elif field == 'vs':
            operands.append(((field,), lane_register(field)))
        elif field not in ('y_vreg', 'packing_format'):
            raise ValueError(f'unhandled vector ALU field {name}.{field}')
    return _variants(form, arrange(form, operands), mnemonic)

def _vector_memory(form: GlForm) -> list[Signature]:
    operands: list[Operand] = []
    address_fields = tuple(field for field in ('base_address', 'offset', 'stride', 'sublane_mask') if field in form.layout)
    if address_fields:
        operands.append((address_fields, vector_address(form)))
    for field in form.layout:
        enum = form.enums.get(field)
        if field in address_fields:
            continue
        if enum == 'VmiscSourcePortEncoding':
            ports = tuple(value_names(enum).values())
            operands.append(((field, *ports), port(field, enum)))
        elif field in ('dest_vreg', 'source_vreg'):
            operands.append(((field,), register(field, 'v')))
        elif field == 'vmsk':
            operands.append(((field,), register(field, 'vm')))
        elif field == 'shuffle':
            operands.append(((field,), selector(field, 'VectorShuffle', 'hex')))
        elif not re.fullmatch(r'v[0-3]_(?:x|y_vreg)', field):
            raise ValueError(f'unhandled vector memory field {form.name}.{field}')
    return [Signature(form, mnemonic_of(form), arrange(form, operands))]

def _trace() -> Expression:
    """vtrace 的 32 位值：高半取自 upper_operand_field 选出的立即数左移 16 位，低半取自 operand；两半读同一标量槽时就是该标量寄存器。"""
    lower_names = value_names('Operand')
    upper = []
    for upper_value, upper_name in value_names('VectorSource').items():
        if upper_name != 'zero' and not re.fullmatch(r'imm[0-5]|vs[0-3]', upper_name):
            continue
        lower = []
        for lower_value, lower_name in lower_names.items():
            if upper_name.startswith('vs'):
                if lower_name == upper_name:
                    lower.append((lower_value, register(upper_name, 's')))
            elif lower_name == 'zero' or re.fullmatch(r'imm[0-5]', lower_name):
                parts = tuple(part for part in ((f'imm{upper_name[-1]}', 16) if upper_name != 'zero' else None, (f'imm{lower_name[-1]}', 0) if lower_name != 'zero' else None) if part is not None)
                lower.append((lower_value, numeric('hex', 32, parts)))
        upper.append((upper_value, choice('operand', lower)))
    return choice('upper_operand_field', upper)

def _misc(form: GlForm) -> list[Signature]:
    name, mnemonic = form.name, mnemonic_of(form)
    if name in ('vector_load', 'vector_load_base', 'vector_load_shuffled', 'vector_load_shuffled_base') or name.startswith('vector_misc_store'):
        return _vector_memory(form)
    if name == 'trace':
        return [Signature(form, mnemonic, (_trace(),))]
    if name == 'vmsk_move':
        registers = (register('destination', 'vm'), register('first_operand', 'vm'))
        return [Signature(form, 'vnop', (), (('destination', 0), ('first_operand', 0))), Signature(form, mnemonic, registers)]
    if name == 'delay_short_immediate':
        # The field holds the cycle count minus one; one more bit keeps the largest count in range.
        return [Signature(form, mnemonic, (direct('delay_cycles', form.fields['delay_cycles'].width + 1, bias=1),))]
    kind = 's32' if mnemonic.startswith(('vsync', 'vwait')) and not mnemonic.startswith('vsyncpa') else 'hex'
    operands: list[Operand] = []
    for field in form.layout:
        enum = form.enums.get(field)
        if field in ('sync_flag_number', 'destination_address'):
            operands.append(((field,), memory('sflag', selector(field, 'VectorSource', 'u32'))))
        elif enum == 'Operand':
            operands.append(((field,), selector(field, enum, kind)))
        elif enum == 'TensorCoreVectorMisc.VectorDelay.Operand':
            # FormatterGl prints OPERAND_ONE as 0; device cycle probes match immediate 1.
            operands.append(((field,), selector(field, enum, 'u32')))
        elif field in ('destination', 'first_operand', 'second_operand'):
            operands.append(((field,), register(field, 'vm')))
        elif field == 'dest_vreg':
            operands.append(((field,), register(field, 'v')))
        else:
            raise ValueError(f'unhandled misc field {name}.{field}')
    return [Signature(form, mnemonic, arrange(form, operands))]

def _extended(form: GlForm) -> list[Signature]:
    name, mnemonic = form.name, mnemonic_of(form)
    operands: list[Operand] = []
    for field in form.layout:
        enum = form.enums.get(field)
        if enum == 'VexSourcePortEncoding':
            if not name.startswith('matrix_multiply_lmr'):
                operands.append(((field, *value_names(enum).values()), port(field, enum)))
        elif field == 'vmsk':
            operands.append(((field,), register(field, 'vm')))
        elif enum == 'TransposeMatrixWidthEncoding':
            operands.append(((field,), selector(field, enum, 'u32')))
        elif enum == 'SourceSpecifierEncoding':
            operands.append(((field,), selector(field, enum, 'hex' if field == 'rotate_specifier' else 'u32')))
        elif field not in ('mxu', 'xlu_unit', 'target', 'transpose', 'variant', 'lmr_width', 'vst_source') and not re.fullmatch(r'v[0-3]_(?:x|y_vreg)', field):
            raise ValueError(f'unhandled extended field {name}.{field}')
    return _variants(form, arrange(form, operands), mnemonic)

def _result(form: GlForm) -> list[Signature]:
    return [Signature(form, mnemonic_of(form), arrange(form, [(('dest',), register('dest', 'v'))]))]

def signatures(form: GlForm) -> list[Signature]:
    if form.slot in ('s0', 's1'):
        return _scalar(form)
    if form.slot == 'dma':
        return _dma(form)
    if form.slot.startswith('va'):
        return _vector_alu(form)
    if form.slot in ('vst', 'vld0', 'vld1'):
        return _vector_memory(form)
    if form.slot == 'misc':
        return _misc(form)
    if form.slot.startswith('vx'):
        return _extended(form)
    return _result(form)

SIGNATURES = tuple(signature for form in FORMS for signature in signatures(form))
SIGNATURES_BY_MNEMONIC: dict[tuple[str, str], list[Signature]] = {}
SIGNATURES_BY_BRANCH: dict[tuple[str, int], list[Signature]] = {}
for _signature in SIGNATURES:
    SIGNATURES_BY_MNEMONIC.setdefault((_signature.form.slot, _signature.mnemonic), []).append(_signature)
    SIGNATURES_BY_BRANCH.setdefault((_signature.form.slot, _signature.form.branch), []).append(_signature)

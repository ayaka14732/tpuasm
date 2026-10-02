"""从当前 ISA 签名生成文档站点的 Markdown 指令索引；不加载 libtpu。"""
from __future__ import annotations

from pathlib import Path
import re

from tpuasm.assembly_model import Expression, Form, Signature
from tpuasm.tpu_v4_tc_model import FORMS
from tpuasm.assembly_values import display_number
from tpuasm.tpu_v4_tc_isa import SIGNATURES
from tpuasm.targets import TPU_V4_TC

def alternatives(values: list[str]) -> str:
    unique = list(dict.fromkeys(values))
    # Fold enumerated register literals into ranges without losing holes.
    for prefix in ('s', 'p', '!p'):
        matching = [value for value in unique if re.fullmatch(re.escape(prefix) + r'[0-9]+', value)]
        if len(matching) < 2:
            continue
        numbers = sorted(int(value[len(prefix):]) for value in matching)
        if numbers == list(range(numbers[0], numbers[-1] + 1)):
            position = unique.index(matching[0])
            unique = [value for value in unique if value not in matching]
            unique.insert(position, f'{prefix}[{numbers[0]}..{numbers[-1]}]')
    return ' / '.join(unique)

def leaves(expression: Expression) -> list[Expression]:
    if expression[0] != 'choice':
        return [expression]
    return [leaf for _, operand in expression[2] for leaf in leaves(operand)]

def dma_endpoints(expression: Expression, form: Form) -> str:
    """把按 core id 与 memory id 嵌套选择的 DMA 端点合并为 `[端点:(基址)]`，端点名称在文档开头列出。"""
    addresses = [leaf for leaf in leaves(expression) if leaf[0] == 'memory']
    if len(addresses) != len(leaves(expression)) or len({address[1] for address in addresses}) <= 8:
        return ''
    bases: dict[str, list[str]] = {}
    for address in addresses:
        bases.setdefault(address[1], []).append(describe(address[2], form))
    groups: dict[str, list[str]] = {}
    for space, values in bases.items():
        groups.setdefault(alternatives(values), []).append(space)
    if len(groups) != 1:
        raise ValueError(f'DMA endpoints of {form.slot} {form.name} accept different base registers')
    return f'[端点:({next(iter(groups))})]'

def describe(expression: Expression, form: Form) -> str:
    kind = expression[0]
    if kind == 'literal':
        return str(expression[1])
    if kind == 'register':
        _, field, prefix = expression
        return f'{prefix}[0..{(1 << form.fields[field].width) - 1}]'
    if kind == 'number':
        _, number_kind, width, parts, base, bias = expression
        if not parts:
            return display_number(base + bias, number_kind, width)
        label = 'f32' if number_kind == 'f32' else ('i' if number_kind.startswith('s') else 'u') + str(width)
        return label + (f'(>= {bias})' if bias else '')
    if kind == 'choice':
        choices = [operand for selector, operand in expression[2]]
        numeric_types = {(operand[1], operand[2]) for operand in choices if operand[0] == 'number' and operand[3]}
        visible = [operand for operand in choices if not (operand[0] == 'number' and not operand[3] and (operand[1], operand[2]) in numeric_types)]
        endpoints = dma_endpoints(expression, form)
        if endpoints:
            return endpoints
        return alternatives([describe(operand, form) for operand in visible])
    if kind == 'table':
        return alternatives([label for label, _ in expression[1]])
    if kind == 'memory':
        _, space, base, offset, modifiers = expression
        address = describe(base, form)
        if offset is not None:
            separator = ' + ' if form.name == 'ScalarLoadSmemOffset' else ' [+ '
            address += separator + '(' + describe(offset, form) + ')' + ('' if form.name == 'ScalarLoadSmemOffset' else ']')
        for key, _, modifier in modifiers:
            address += f' [, {key}=(' + describe(modifier, form) + ')]'
        return f'[{space}:({address})]'
    if kind == 'dma_address':
        dest = expression[1]
        spaces = ['vmem', 'smem', 'imem', 'hbm', 'cmem']
        if dest:
            spaces.append(f'oq[0..{(1 << form.fields["outfeed_queue_id"].width) - 1}]')
        else:
            spaces.extend(('memseti', 'memsetd'))
        register = describe(('register', 'dest_address' if dest else 'source_address', 's'), form)
        return '[' + '(' + ' / '.join(spaces) + '):' + register + ']'
    if kind == 'pattern':
        return expression[1].format(*(describe(part, form) for part in expression[2]))
    if kind == 'trace':
        return 's[0..31] / u32'
    raise ValueError(f'unhandled operand expression {kind!r}')

def operands(signature: Signature) -> str:
    values = [describe(expression, signature.form) for expression in signature.operands]
    if signature.mnemonic in ('sbr.rel', 'sbr.abs', 'scall.rel', 'scall.abs'):
        index = 1 if signature.mnemonic.startswith('scall') else 0
        values[index] = 'label / ' + values[index]
    positional = len(values) - len(signature.keywords)
    optional = dict(signature.defaults)
    for index, keyword in enumerate(signature.keywords, positional):
        values[index] = keyword + '=' + '(' + values[index] + ')'
        if keyword in optional:
            values[index] = '[' + values[index] + ']'
    return ', '.join(values) if values else '无操作数'

def slot_tables(slots: tuple[str, ...], signatures: tuple[Signature, ...]) -> list[str]:
    lines: list[str] = []
    for slot in slots:
        entries: dict[str, list[str]] = {}
        for signature in signatures:
            if signature.form.slot == slot:
                entries.setdefault(signature.mnemonic, []).append(operands(signature))
        lines.extend((f'## {slot}', '', '| 助记符 | 按顺序排列的操作数类型 |', '|---|---|'))
        for mnemonic, variants in sorted(entries.items()):
            text = '<br>'.join('`' + value.replace('|', '\\|') + '`' for value in dict.fromkeys(variants))
            lines.append(f'| `{mnemonic}` | {text} |')
        lines.append('')
    return lines

def render_tpu_v4_tc() -> str:
    pairs = {(signature.form.slot, signature.mnemonic) for signature in SIGNATURES}
    lines = [
        '# TPU v4 TC 指令索引',
        '',
        f'当前登记 **{len(FORMS)} 个非 Noop descriptor 形式、{len(SIGNATURES)} 条展开签名、{len(pairs)} 个槽与助记符组合**。这些数字含不同槽、编码变体与别名，不等于独立助记符数量，也不是设备执行覆盖率。',
        '',
        '## 如何阅读',
        '',
        '表内操作数按源码顺序排列，用法为 `{ 槽: [@pN 或 @!pN] 助记符 操作数 }`。`s[0..31]` 等表示可选寄存器编号，`/` 表示类型或拼写的选择；这些记号是索引表示法，不是可直接粘贴的源码。多目的元组如 `(gmr0, gsfn0, mrf0)` 则是实际语法。',
        '',
        '`iN/uN` 表示 N 位有符号/无符号数值类型，`f32` 表示 float32；非负十六进制数还可以表示该位宽的原始位型，`f32bits(...)` 直接给出 float32 位型。数值类型限定输入位宽，并不保证范围内每个值都满足某条指令的选择器、立即数拆分和整个指令包的资源条件。表中合并了同类型的编码候选，最终可编码性由汇编器检查。',
        '',
        '`label` 表示源码标签；直接分支与 call 的 `.rel` 数值是相对当前指令包的位移，`.abs` 数值是 bundle PC。',
        '',
        '地址的外层方括号、冒号和空间名是实际语法，内部圆括号用来分组类型，`[+ …]`、`[, sm=…]`、`[, ss=…]` 表示可选部分。允许零基址的向量存取可直接写数值偏移，如 `[vmem:0x40]`。操作数的数值单位沿用 ISA，不统一解释为字节。`vmem` 指 TC VMEM，`cmem` 指 Megacore Shared CMEM。',
        '',
        '相同槽与助记符下不同签名分行显示；相同操作数类型的编码变体合并。所有具名参数都必须提供，允许换序。谓词前缀的范围为 `p0..p14`，不由表内其他谓词寄存器字段的位宽推导。`vm[0..31]` 只是 5-bit 目的字段的编码范围，真机只有 `vm0..vm7`；实际执行写入 `vm8..vm31` 的指令会使 TensorCore halt。共享通路、跨操作数字段相等关系以及 `.encoding` 约束仍按[格式参考](tpu_v4_tc.md)处理。',
        '',
        '未登记的形式会被拒绝；尚不支持的 9 个形式见[未登记的形式](../design/assembly.md#未登记的形式)。',
        '',
        '## 按槽查找',
        '',
        ' · '.join(f'[{slot}](#{slot})' for slot in TPU_V4_TC.slots),
        '',
    ]
    lines.extend(slot_tables(TPU_V4_TC.slots, SIGNATURES))
    return '\n'.join(lines)

def render_tpu_v4_bcs() -> str:
    from tpuasm.tpu_v4_bcs_isa import SIGNATURES as bcs_signatures
    lines = [
        '# TPU v4 BCS 指令索引',
        '',
        f'登记 {len(bcs_signatures)} 个非 Noop 槽内形式（S0 57 个、S1 55 个），共 66 种非 Noop operation；空槽用省略该槽或空 bundle 表示。这里的覆盖是编码与解码覆盖，不是设备执行覆盖。',
        '',
        '使用 `.target tpu-v4-bcs` 和 `{ s0: mnemonic operands ; s1: mnemonic operands }`。表中 `s[0..31]`、`u32` 等是类型说明，不能照抄为源码。谓词前缀使用 `@p0..@p14` 或 `@!p0..@!p14`。目的字段的硬件位宽不证明全部编号都可用。',
        '',
        'ScalarY 数字按 32-bit 位型编码；内置 selector 写作 `sy.NAME`，名称来自当前 descriptor，不将其自动改写为推测的常量值。浮点 ALU 的立即数也使用位型。`smov` 同时接受寄存器与数值；`ssub.s32` / `ssub.f32` 的源顺序是 Y、X，其余二源 ALU 是 X、Y。完整规则见 [BCS 使用与互操作](tpu_v4_bcs.md)。',
        '',
        'DMA 使用具名字段；所有列出的字段均须给出，顺序可变。目标端的 outfeed queue 与 memory/core ID 共享物理位，必须一致。字段枚举只选择编码；地址域、allocation、completion 和生命周期由 runtime 管理。',
        '',
    ]
    for slot in ('s0', 's1'):
        lines.extend((f'## {slot}', '', '| 助记符 | 操作数 |', '|---|---|'))
        for signature in sorted((item for item in bcs_signatures if item.form.slot == slot), key=lambda item: item.mnemonic):
            lines.append(f'| `{signature.mnemonic}` | `{operands(signature)}` |')
        lines.append('')
    return '\n'.join(lines)

def render_tpu_v6e_tc() -> str:
    from tpuasm.tpu_v6e_tc_isa import DMA_SPACES, FALLBACK_MNEMONICS, SIGNATURES as v6e_signatures
    from tpuasm.tpu_v6e_tc_isa_data import REJECTED
    from tpuasm.tpu_v6e_tc_model import FORMS as v6e_forms
    from tpuasm.targets import TPU_V6E_TC
    pairs = {(signature.form.slot, signature.mnemonic) for signature in v6e_signatures}
    names = {form.name for form in v6e_forms}
    rejected = sum(len(slots) for slots in REJECTED.values())
    fallback = sorted({FALLBACK_MNEMONICS[form.name] for form in v6e_forms if not form.formatter_mnemonic})
    lines = [
        '# TPU v6e TC 指令索引',
        '',
        f'由 libtpu 0.0.49 的 descriptor、encoder 与 formatter 生成：登记 **{len(names)} 种形式、{len(v6e_forms)} 个槽与形式组合、{len(v6e_signatures)} 条展开签名、{len(pairs)} 个槽与助记符组合**。encoder 拒绝的 {rejected} 个槽与形式组合不登记。这里的覆盖是编码、解码与重汇编覆盖，不是设备执行覆盖。',
        '',
        '## 如何阅读',
        '',
        '`s[0..31]`、`v[0..63]` 表示寄存器编号，`/` 表示可选的拼写，`iN/uN/f32` 是数值类型，地址内部的圆括号用于分组，`[+ …]`、`[, sm=…]` 与 `[, ss=…]` 是可选部分。数值类型只限定输入位宽，立即数槽和整个指令包的资源是否够用由汇编器检查。',
        '',
        '谓词前缀为 `@p0..@p13`；只有 `s0`、`s1`、`dma` 接受 `@!pN`。某个助记符只出现在部分槽的表中，表示 encoder 在其他槽拒绝该形式。`vmul.8x128.u32.u64` 在 `va0` 时占用 `va1`、在 `va2` 时占用 `va3`；`dma` 不能与 `s0`、`s1` 同时出现。',
        '',
        'DMA 的 `端点` 是以下地址空间之一：' + '、'.join(f'`{name}`' for name in DMA_SPACES.values()) + '，其余 core id 与 memory id 组合写作 `coreC_memM`。DMA 的具名参数可以换序；方括号括起的具名参数可以省略，省略时取 0。',
        '',
        '以下助记符的形式没有 formatter 输出，名称由 tpuasm 按同族写法给出，操作数按 descriptor 字段顺序排列，没有 formatter 文本可对照，执行结果已在设备上核对：' + '、'.join(f'`{name}`' for name in fallback) + '。其余助记符与操作数顺序取自 libtpu formatter。写法见[格式参考](tpu_v6e_tc.md)。',
        '',
        '## 按槽查找',
        '',
        ' · '.join(f'[{slot}](#{slot})' for slot in TPU_V6E_TC.slots),
        '',
    ]
    lines.extend(slot_tables(TPU_V6E_TC.slots, v6e_signatures))
    return '\n'.join(lines)

def render_tpu_v6e_tec() -> str:
    from tpuasm.tpu_v6e_tec_isa import FALLBACKS, SIGNATURES as all_signatures
    from tpuasm.tpu_v6e_tec_isa_data import INSTRUCTIONS, REJECTED
    from tpuasm.tpu_v6e_tec_model import FORMS as tec_forms
    from tpuasm.targets import TPU_V6E_TEC
    printed = {(entry[0], entry[1]) for entry in INSTRUCTIONS}
    # The named-field syntax exists for every form; the tables list it only where the printer has none.
    tec_signatures = tuple(signature for signature in all_signatures if signature not in FALLBACKS or (signature.form.slot, signature.form.name) not in printed)
    pairs = {(signature.form.slot, signature.mnemonic) for signature in tec_signatures}
    names = {form.name for form in tec_forms}
    rejected = sum(len(slots) for slots in REJECTED.values())
    fallback = sorted({form.name for form in tec_forms if (form.slot, form.name) not in printed})
    lines = [
        '# TPU v6e TEC 指令索引',
        '',
        f'由 libtpu 0.0.49 的 descriptor、encoder、LLVM TPU printer 与 TEC emitter 生成：登记 **{len(names)} 种形式、{len(tec_forms)} 个槽与形式组合、{len(tec_signatures)} 条展开签名、{len(pairs)} 个槽与助记符组合**。encoder 拒绝的 {rejected} 个槽与形式组合不登记。这里的覆盖是编码、解码与重汇编覆盖，没有在设备上执行。',
        '',
        '## 如何阅读',
        '',
        '`s[0..31]`、`v[0..63]` 表示寄存器编号，`/` 表示可选的拼写，`iN/uN/f32` 是数值类型，其余字符照写。操作数顺序与 LLVM TPU printer 相同，目的在前。数值类型只限定输入位宽，立即数槽和整个指令包的资源是否够用由汇编器检查。写法见[格式参考](tpu_v6e_tec.md)。',
        '',
        '每个形式另有具名字段写法：助记符是 descriptor 中的形式名，操作数是按字段顺序排列的具名参数，用于 printer 写法表达不了的编码。表中只对没有 printer 写法的槽与形式组合列出这种写法，涉及以下形式：' + '、'.join(f'`{name}`' for name in fallback) + '。',
        '',
        '## 按槽查找',
        '',
        ' · '.join(f'[{slot}](#{slot})' for slot in TPU_V6E_TEC.slots),
        '',
    ]
    lines.extend(slot_tables(TPU_V6E_TEC.slots, tec_signatures))
    return '\n'.join(lines)

def main() -> None:
    destination = Path(__file__).resolve().parents[1] / 'docs' / 'references'
    destination.mkdir(exist_ok=True)
    for filename, content in (
        ('tpu_v4_tc_isa.md', render_tpu_v4_tc()),
        ('tpu_v4_bcs_isa.md', render_tpu_v4_bcs()),
        ('tpu_v6e_tc_isa.md', render_tpu_v6e_tc()),
        ('tpu_v6e_tec_isa.md', render_tpu_v6e_tec()),
    ):
        path = destination / filename
        if not path.exists() or path.read_text(encoding='utf-8') != content:
            path.write_text(content, encoding='utf-8')

if __name__ == '__main__':
    main()

"""按程序声明的硬件目标呈现汇编文本。"""
from __future__ import annotations

from .assembly_syntax import AssemblyProgram, branch_target
from .tc_source_mapping import ProgramSourceMap, source_comments

def render_program(program: AssemblyProgram, pins: list[dict[str, str]], source_map: ProgramSourceMap | None = None) -> str:
    labels = {pc: label for label, pc in program.labels.items()}
    lines = [f'.target {program.hardware.identifier}']
    outside, inline = source_comments(source_map) if source_map is not None else ({}, {})
    for pc, bundle in enumerate(program.bundles):
        lines.extend('# ' + text for text in outside.get(pc, ()))
        if pc in labels:
            lines.append(labels[pc] + ':')
        statements = []
        comments = []
        for instruction in sorted(bundle.instructions, key=lambda inst: program.hardware.slots.index(inst.slot)):
            predicate = instruction.predicate
            prefix = '' if predicate == 15 else f'@{"!" if predicate >= 16 else ""}p{predicate % 16} '
            operands = instruction.operands
            branch = branch_target(instruction.mnemonic, operands, pc)
            if branch is not None:
                index, target = branch
                if target in labels:
                    operands = operands[:index] + (labels[target],) + operands[index + 1:]
            tail = ' ' + ', '.join(operands) if operands else ''
            statements.append(f'{instruction.slot}: {prefix}{instruction.mnemonic}{tail}')
            comments.append(inline.get((pc, instruction.slot), ''))
        if pins[pc]:
            statements.append('.encoding { ' + ' ; '.join(f'{name} = {value}' for name, value in sorted(pins[pc].items())) + ' }')
            comments.append('')
        for index, (statement, comment) in enumerate(zip(statements, comments)):
            prefix = '{ ' if index == 0 else '  '
            suffix = ' }' if index == len(statements) - 1 else ' ;'
            lines.append(prefix + statement + suffix + ('  # ' + comment if comment else ''))
        if not statements:
            lines.append('{}')
    lines.extend('# ' + text for text in outside.get(len(program.bundles), ()))
    return '\n'.join(lines) + '\n'

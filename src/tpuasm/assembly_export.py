"""共享的可逆汇编导出与分支标签恢复。"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from .assembler import assemble_listing
from .assembly_printer import render_program
from .assembly_syntax import AssemblyBundle, AssemblyProgram, EncodingConstraint, branch_target

if TYPE_CHECKING:
    from .tc_source_mapping import ProgramSourceMap

SolveBundle = Callable[[AssemblyBundle, int, tuple[EncodingConstraint, ...]], int]
ValidateCanonical = Callable[[bytes], None]
Form = TypeVar('Form')

def decoded_labels(bundles: list[AssemblyBundle]) -> dict[str, int]:
    targets = {0}
    for pc, bundle in enumerate(bundles):
        for instruction in bundle.instructions:
            branch = branch_target(instruction.mnemonic, instruction.operands, pc)
            if branch is not None:
                _, target = branch
                if 0 <= target < len(bundles):
                    targets.add(target)
    return {('entry' if pc == 0 else f'L_{pc:04x}'): pc for pc in sorted(targets)}

def validate_encoding(encoding: str) -> None:
    if encoding not in ('exact', 'canonical'):
        raise ValueError(f'unknown assembly encoding mode {encoding!r}')

def export_program(
    image: bytes,
    program: AssemblyProgram,
    solved: list[tuple[int, tuple[Form, ...]]],
    *,
    encoding: str,
    solve: SolveBundle,
    list_constraints: Callable[[tuple[Form, ...], int], dict[str, str]],
    source_map: ProgramSourceMap | None = None,
    validate_canonical: ValidateCanonical | None = None,
) -> str:
    """导出可重汇编的源码；exact 模式为每个 bundle 保留恢复原机器字所需的最少约束。

    ``solve`` 在给定约束下求解 bundle，返回机器字。``list_constraints`` 由 ``solved`` 中解码得到的 forms 和原机器字列出全部命名约束；解码出的分支形式可能与规范别名不同，所以不能改用规范求解得到的 forms。
    """
    pins = []
    for pc, (bundle, (word, forms)) in enumerate(zip(program.bundles, solved)):
        retained: dict[str, str] = {}
        if encoding == 'exact' and solve(bundle, pc, ()) != word:
            retained = list_constraints(forms, word)

            def constraints(values: dict[str, str]) -> tuple[EncodingConstraint, ...]:
                return tuple(EncodingConstraint(name, value, bundle.location) for name, value in values.items())

            if solve(bundle, pc, constraints(retained)) != word:
                raise ValueError(f'bundle {pc:#x}: named fields do not cover the original encoding')
            changed = True
            while changed:
                changed = False
                for name in tuple(retained):
                    trial = {key: value for key, value in retained.items() if key != name}
                    try:
                        candidate = solve(bundle, pc, constraints(trial))
                    except ValueError:
                        continue
                    if candidate == word:
                        retained = trial
                        changed = True
        pins.append(retained)
    source = render_program(program, pins, source_map)
    rebuilt = assemble_listing(source)
    if encoding == 'exact' and rebuilt != image:
        raise ValueError('printed source does not reproduce the original program image bytes')
    if encoding == 'canonical' and validate_canonical is not None:
        validate_canonical(rebuilt)
    return source

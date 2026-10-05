"""只调整 JAX 生成的静态 MLIR location，不改 trace op 或计算。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable, Iterator

from jax._src.pallas.mosaic import lowering
from jax._src.source_info_util import NameStack

_SOURCE_SHA256 = {
    # https://github.com/jax-ml/jax/blob/886d2370c1c959d210f522e352e3ddc6bcff7d6c/jax/_src/pallas/mosaic/lowering.py#L1746
    'jaxpr_subcomp': 'f9919ea845980f1917e973b5abb6974258074e3b1ec588151beb63697b6815b1',
    # https://github.com/jax-ml/jax/blob/886d2370c1c959d210f522e352e3ddc6bcff7d6c/jax/_src/pallas/mosaic/lowering.py#L4413
    '_lower_jaxpr_to_for_loop': '4415e0369cc23b917a7d84b35eaf7e8ae3713a64b9d58f22486fe89a16a460f1',
}

_parent: ContextVar[NameStack | None] = ContextVar('tpuasm_source_parent', default=None)
_parent_traceback: ContextVar[Any | None] = ContextVar('tpuasm_source_parent_traceback', default=None)

def _stack(context_stack: NameStack, equation_stack: NameStack) -> NameStack:
    parent = _parent.get()
    return (context_stack if parent is None else parent) + equation_stack

def _traceback(traceback: Any | None) -> Any | None:
    parent = _parent_traceback.get()
    if traceback is None:
        return parent
    if parent is None or traceback is parent:
        return traceback
    return traceback + parent

@contextmanager
def _scope(stack: NameStack, traceback: Any | None) -> Iterator[None]:
    stack_token = _parent.set(stack)
    traceback_token = _parent_traceback.set(traceback)
    try:
        yield
    finally:
        _parent_traceback.reset(traceback_token)
        _parent.reset(stack_token)

def _patched(original: Callable[..., Any], edits: tuple[tuple[str, str], ...], namespace: dict[str, Any]) -> Callable[..., Any]:
    """核对函数源码字节的 SHA-256 后做唯一匹配替换，以原文件名和行号重新编译。"""
    name = original.__name__
    try:
        lines, start = inspect.getsourcelines(original)
        file_lines = Path(inspect.getfile(original)).read_bytes().splitlines(keepends=True)
    except (OSError, TypeError) as error:
        raise RuntimeError(f'cannot read JAX {name} source for compiler source lowering validation') from error
    raw = b''.join(file_lines[start - 1:start - 1 + len(lines)])
    if hashlib.sha256(raw).hexdigest() != _SOURCE_SHA256[name]:
        raise RuntimeError(f'JAX {name} source differs from the validated implementation; compiler source lowering needs validation for this source')
    source = raw.decode('utf-8')
    for before, after in edits:
        if source.count(before) != 1:
            raise RuntimeError(f'JAX {name} source differs from the validated implementation')
        source = source.replace(before, after)
    source = '\n' * (original.__code__.co_firstlineno - 1) + source
    exec(compile(source, original.__code__.co_filename, 'exec'), namespace)
    return namespace[name]

@contextmanager
def lowering_sources() -> Iterator[None]:
    original_subcomp = lowering.jaxpr_subcomp
    original_loop = lowering._lower_jaxpr_to_for_loop
    subcomp = _patched(
        original_subcomp,
        (
            (
                'eqn_name_stack = ctx.name_stack + eqn.source_info.name_stack\n    loc = mlir.source_info_to_location(\n        ctx, eqn.primitive, eqn_name_stack, eqn.source_info.traceback\n    )',
                'eqn_name_stack = _tpuasm_stack(ctx.name_stack, eqn.source_info.name_stack)\n    eqn_traceback = _tpuasm_traceback(eqn.source_info.traceback)\n    loc = mlir.source_info_to_location(\n        ctx, eqn.primitive, eqn_name_stack, eqn_traceback\n    )',
            ),
            (
                'with (source_info_util.user_context(eqn.source_info.traceback), loc,',
                'with (_tpuasm_scope(eqn_name_stack, eqn_traceback), source_info_util.user_context(eqn_traceback), loc,',
            ),
            # The per-equation lowering cache emits the first equation as a detached
            # function and inlines it for later equations with the same key, so their
            # ops carry callsite(first equation at later equation). Lower each
            # equation with its own rule and location instead.
            # https://github.com/jax-ml/jax/issues/41153
            (
                'can_cache = (eqn.primitive not in _uncacheable_primitives and',
                'can_cache = (False and eqn.primitive not in _uncacheable_primitives and',
            ),
        ),
        {**vars(lowering), '_tpuasm_stack': _stack, '_tpuasm_traceback': _traceback, '_tpuasm_scope': _scope},
    )
    # inlined_func_call gives every cloned op the loop's primitive. Lower each
    # fully unrolled iteration directly so each op keeps its own location.
    loop = _patched(
        original_loop,
        (('if unroll > 1 and (is_full_static_unroll or not supports_late_unroll):', 'if unroll > 1 and not supports_late_unroll:'),),
        {**vars(lowering), 'jaxpr_subcomp': subcomp},
    )
    lowering.jaxpr_subcomp = subcomp
    lowering._lower_jaxpr_to_for_loop = loop
    try:
        yield
    finally:
        lowering.jaxpr_subcomp = original_subcomp
        lowering._lower_jaxpr_to_for_loop = original_loop

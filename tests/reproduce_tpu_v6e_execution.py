"""在单芯片 TPU v6e 上核对 selector、formatter 无文本形式和动态 unpack。

在仓库根目录运行：PYTHONPATH=src python tests/reproduce_tpu_v6e_execution.py --output /tmp/tpuasm-v6e-execution
需要 libtpu 0.0.49。只使用真实设备，不进入 CPU CI。数值案例保存输入、设备输出、汇编片段和 JSONL 摘要，周期案例另存 JSON；不保存可重建的 executable。
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
from typing import cast

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxlib.xla_client import LoadedExecutable
import numpy as np

from tpuasm import BundleInsertion, assemble_listing, executable_programs, format_assembly, insert_executable_bundles, load_executable, replace_executable_programs
from tpuasm.assembly_syntax import parse_assembly
from tpuasm.backends import select_backend
from tpuasm.tpu_v6e_tc_codec import decode_program, decoded_signature
from tpuasm.tpu_v6e_tc_isa import value_names
from tpuasm.tpu_v6e_tc_isa_data import EMPTY_WORD
from tpuasm.tpu_v6e_tc_model import FORMS_BY_NAME

TARGET = 'tpu-v6e-tc'

@pl.kernel(
    out_type=jax.ShapeDtypeStruct((8, 128), jnp.uint32),
    mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
    scratch_types=(pltpu.VMEM((256, 128), jnp.uint32), pltpu.SemaphoreType.DMA),
    name='isa_probe',
    compiler_params=pltpu.CompilerParams(
        disable_bounds_checks=True,
        disable_semaphore_checks=True,
    ),
)
def kernel(x_hbm: Ref, out_hbm: Ref, data: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, data, sem).wait()
    data[:8, :] = data[:8, :] ^ jnp.uint32(0x13579BDF)
    pltpu.async_copy(data.at[:8, :], out_hbm, sem).wait()

def instruction(slot: str, name: str, **values: int) -> str:
    form = FORMS_BY_NAME[slot, name]
    bits = form.fixed(15)
    word = (EMPTY_WORD & ~bits.mask) | bits.value
    for key, value in values.items():
        field = form.fields[key]
        assert 0 <= value < (1 << field.width), (key, value)
        word = (word & ~field.mask) | (value << field.start)
    image = word.to_bytes(64, 'little') + EMPTY_WORD.to_bytes(64, 'little') * 7
    return '\n'.join(line for line in format_assembly(image, target=TARGET).splitlines()[1:] if not line.endswith(':'))

def op(text: str) -> str:
    return '{ ' + text + ' }\n.empty 16\n'

def place(host: np.ndarray) -> jax.Array:
    data = np.zeros((256, 128), dtype=np.uint32)
    data[: host.shape[0], :] = host
    return jax.device_put(data)

class Probe:
    def __init__(self, output: Path) -> None:
        self.output = output
        output.mkdir(exist_ok=True, parents=True)
        assert jax.devices()[0].device_kind == 'TPU v6 lite'
        assert version('libtpu') == '0.0.49'
        self.host = np.arange(1024, dtype=np.uint32).reshape(8, 128)
        self.x = place(self.host)
        self.compiled = jax.jit(kernel, compiler_options={'xla_msa_enable': 'false', 'xla_tpu_vmem_scavenging_mode': 'NONE'}).lower(self.x).compile()
        np.testing.assert_array_equal(np.asarray(self.compiled(self.x)), self.host ^ 0x13579BDF)
        raw = bytes(cast(LoadedExecutable, self.compiled.runtime_executable()).serialize())
        ((record, index, image),) = executable_programs(raw)
        self.key = record, index
        decoded = decode_program(image)
        markers = [
            (pc, decoded_signature(form, word))
            for pc, (word, forms) in enumerate(decoded)
            for form, _ in forms
            if form.name == 'vector_bitwise_xor' and '0x13579bdf' in decoded_signature(form, word)[1]
        ]
        assert len(markers) == 1
        self.pc, (signature, operands) = markers[0]
        assert len(decoded[self.pc][1]) == 1
        self.dest = operands[0]
        (self.input,) = [v for v in operands[1:] if v.startswith('v')]
        replacement = assemble_listing(f'.target {TARGET}\n' + op(f'{signature.form.slot}: vmov.8x128 {self.dest}, v10') + '.empty 7\n')[:64]
        image = image[: 64 * self.pc] + replacement + image[64 * (self.pc + 1) :]
        raw = replace_executable_programs(raw, {self.key: image})
        self.capacity = 512
        self.raw = insert_executable_bundles(raw, {self.key: [BundleInsertion(self.pc, f'.target {TARGET}\n.empty {self.capacity}\n')]})
        ((_, _, self.image),) = executable_programs(self.raw)
        print('carrier ready', self.pc, self.input, self.dest, flush=True)
        self.log = (output / 'results.jsonl').open('a', buffering=1)
        backend, _ = select_backend(TARGET)
        environment = {
            'jax': jax.__version__,
            'libtpu': version('libtpu'),
            'device': str(jax.devices()[0]),
            'kind': jax.devices()[0].device_kind,
            'python': f'{sys.version_info.major}.{sys.version_info.minor}{sys.abiflags}',
            'jaxlib': version('jaxlib'),
            'libtpu_build_id': backend.build_id,
        }
        self.log.write(json.dumps({'environment': environment}) + '\n')

    def run(self, name: str, body: str, expected: np.ndarray | np.uint32, host: np.ndarray | None = None) -> np.ndarray:
        source = f'.target {TARGET}\n.empty 16\n' + body + '\n.empty 16\n'
        fragment = parse_assembly(source, fragment=True)
        assert len(fragment.bundles) <= self.capacity
        code = assemble_listing(source + f'\n.empty {self.capacity - len(fragment.bundles)}\n')
        image = self.image[: 64 * self.pc] + code + self.image[64 * (self.pc + self.capacity) :]
        patched = replace_executable_programs(self.raw, {self.key: image})
        x = self.x if host is None else place(host)
        y = np.asarray(load_executable(patched, self.compiled)(x))
        np.save(self.output / f'{name}.npy', y)
        np.save(self.output / f'{name}-input.npy', self.host if host is None else host)
        (self.output / f'{name}.tpuasm').write_text(source)
        okay = bool(np.array_equal(y, np.broadcast_to(expected, y.shape)))
        self.log.write(json.dumps({'name': name, 'match': okay, 'sample': [int(v) for v in y.flatten()[:16]], 'unique': [int(v) for v in np.unique(y)[:32]]}) + '\n')
        print(name, 'OK' if okay else 'MISMATCH', [hex(int(v)) for v in y.flatten()[:8]], flush=True)
        np.testing.assert_array_equal(y, np.broadcast_to(expected, y.shape), err_msg=name)
        return y

def selector_expected(name: str, imm: list[int]) -> int:
    constants = {
        'negative_one': 0xFFFFFFFF,
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'int_one': 1,
        'int_negative_one': 0xFFFFFFFF,
        'float_one': 0x3F800000,
        'float_negative_one': 0xBF800000,
        'float_one_half': 0x3F000000,
    }
    if name in constants:
        return constants[name]
    if name.startswith('hex_'):
        return int(name[4:], 16)
    if name.startswith('zero_imm'):
        return imm[int(name[-1])]
    if name.startswith('ones_imm'):
        return imm[int(name[-1])] | 0xFFF00000
    if name.endswith('_zero'):
        return imm[int(name[3])] << 12
    if re.fullmatch(r'imm[0-5]_imm[0-5]', name):
        return ((imm[int(name[3])] << 16) | (imm[int(name[-1])] & 0xFFFF)) & 0xFFFFFFFF
    raise ValueError(name)

def selectors(probe: Probe) -> None:
    imm = [0xA1357, 0x5B246, 0xC369A, 0x7D48B, 0xE5ABC, 0x9F6DE]
    shared = dict(zip((f'imm{i}' for i in range(6)), imm))
    for kind, slot, form, key in [('ScalarY', 's0', 'move_y', 'y'), ('VectorY', 'va0', 'vector_move', 'y_src')]:
        for value, name in value_names(kind).items():
            if name.startswith(('sreg', 'vs')) or name == 'vreg':
                continue
            body = instruction(slot, form, dest=20 if slot == 's0' else 10, **{key: value}, **shared) + '\n.empty 16\n'
            if slot == 's0':
                body += op('va0: vmov.8x128 v10, s20')
            probe.run(f'{kind}-{name}', body, np.uint32(selector_expected(name, imm)))
    for lane in range(4):
        body = op(f's0: simm.s32 s{20 + lane}, {0xABC001 + lane}')
        body += instruction('va0', 'vector_move', dest=10, y_src=28 + lane, **{f'vs{lane}': 20 + lane})
        probe.run(f'VectorY-vs{lane}', body, np.uint32(0xABC001 + lane))

def remap_expected(bits: np.ndarray, control: int | np.ndarray, width: int) -> np.ndarray:
    sign = 1 << (width - 1)
    exponent = 0x7F800000 if width == 32 else 0x7F80
    fraction = 0x7FFFFF if width == 32 else 0x7F
    categories = np.where(
        (bits & exponent) == exponent,
        np.where((bits & fraction) != 0, 6, np.where((bits & sign) != 0, 0, 5)),
        np.where((bits & exponent) == 0, np.where((bits & sign) != 0, 2, 3), np.where((bits & sign) != 0, 1, 4)),
    ).astype(np.uint32)
    code = (np.uint32(control) >> (4 * categories)) & 15
    one = 0x3F800000 if width == 32 else 0x3F80
    quiet_nan = 0x7FC00000 if width == 32 else 0x7FC0
    values = np.array([0, 0, sign, one, sign | one, sign | exponent, exponent, 0, quiet_nan, sign | (exponent - 1), exponent - 1, sign - 1, sign, 0, 1, (1 << width) - 1], np.uint32)
    return np.where(code == 0, bits, np.where(code == 7, bits ^ sign, np.where(code == 13, bits & (sign - 1), values[code]))).astype(np.uint32)

def remap_word(host: np.ndarray, control: int | np.ndarray, fmt: str) -> np.ndarray:
    if fmt == 'f32':
        return remap_expected(host, control, 32)
    return remap_expected(host & 65535, control, 16) | (remap_expected(host >> 16, control, 16) << 16)

def arithmetic(p: Probe) -> None:
    bits = np.array(
        [0, 0x80000000, 0x3F800000, 0xBF800000, 0x40000000, 0xC0000000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F801234, 1, 0x007FFFFF, 0x3F803F80, 0x7F80FF80, 0xFFFF0001, 0x1234FFFF],
        np.uint32,
    )
    host = np.resize(bits, (8, 128))
    for fmt in ('f32', 'bf16'):
        for operand in [0, *(1 << bit for bit in range(32)), 0xFFFFFFFF]:
            body = op(f'va0: vimm.8x128.s32 v11, {operand:#x}') + op(f'va0: vremap.8x128.{fmt} v10, {p.input}, v11')
            p.run(f'remap-{fmt}-{operand:x}', body, remap_word(host, operand, fmt), host=host)
    x = np.arange(1024, dtype=np.uint32).reshape(8, 128) * np.uint32(0x10001) + np.uint32(0xFFFE0000)
    y = np.uint32(0x1234ABCD)
    expected = ((x & 65535) * (y & 65535) & 65535) | (((x >> 16) * (y >> 16) & 65535) << 16)
    p.run('multiply-u16', op(f'va0: vimm.8x128.s32 v11, {int(y):#x}') + op(f'va0: vmul.8x128.u16 v10, {p.input}, v11'), expected, host=x)
    # 16 位 mask 的高、低位分别选择输出半字。
    pre = op('va0: vimm.8x128.s32 v12, 0xffffffff') + op('va0: vimm.8x128.s32 v13, 0')
    select = op('va0: vsel.8x128 v10, vm7, v12, v13')
    expected = np.where((x & 65535) + 65535 > 65535, 65535, 0).astype(np.uint32) | (np.where((x >> 16) + 2 > 65535, 65535, 0).astype(np.uint32) << 16)
    p.run('carry-u16', pre + op('va0: vimm.8x128.s32 v11, 0x0002ffff') + op(f'va0: vc.8x128.u16 vm7, {p.input}, v11') + select, expected, host=x)
    expected = np.where((host & 0x7F80) == 0x7F80, 65535, 0).astype(np.uint32) | (np.where((host & 0x7F800000) == 0x7F800000, 65535, 0).astype(np.uint32) << 16)
    p.run('weird-bf16', pre + op(f'va0: vweird.8x128.bf16 vm7, {p.input}') + select, expected, host=host)

def unpack_expected(host: np.ndarray, packing: int, part: int) -> np.ndarray:
    if packing == 0:
        return ((host >> (16 * part)) & 65535) << 16
    if packing in (1, 2):
        expanded = np.stack((host & 65535, host >> 16), axis=1).reshape(16, 128)[8 * part : 8 * (part + 1)]
        return expanded << 16 if packing == 1 else expanded.astype(np.uint16).view(np.int16).astype(np.int32).view(np.uint32)
    expanded = np.stack(tuple((host >> (8 * i)) & 255 for i in range(4)), axis=1).reshape(32, 128).astype(np.uint8)
    values = expanded.view(np.int8) if packing != 4 else expanded
    if packing == 5:
        return values[8 * part : 8 * (part + 1)].astype(np.int32).view(np.uint32)
    bf16 = values[16 * part : 16 * (part + 1)].astype(np.float32).view(np.uint32) >> 16
    return bf16[::2] | (bf16[1::2] << 16)

def unpack(p: Probe) -> None:
    host = np.arange(1024, dtype=np.uint32).reshape(8, 128) * np.uint32(0x10001) + np.uint32(0x81003F00)
    for packing, mnemonic in [(0, 'i.bf16'), (1, 'c.bf16'), (2, 'c.s16'), (3, 'c.s8.bf16'), (4, 'c.u8.bf16'), (5, 'c.s8')]:
        for half in range(4 if packing == 5 else 2):
            static = f'vunpack.{"c." + str(half) if packing == 5 else mnemonic.split(".")[0] + (".u" if half else ".l")}.8x128.' + '.'.join(mnemonic.split('.')[1:])
            reference = p.run(f'unpack-reference-{packing}-{half}', op(f'va0: {static} v10, {p.input}'), unpack_expected(host, packing, half), host=host)
            for lane in range(4):
                body = op(f's0: simm.s32 s{20 + lane}, {half}')
                body += instruction('va0', 'dynamic_vector_unpack', packing_format=packing, vs=lane, dest=10, x=int(p.input[1:]), **{f'vs{lane}': 20 + lane})
                p.run(f'unpack-{packing}-{half}-vs{lane}', body, reference, host=host)

def packed_lane_expected(host: np.ndarray, amount: int, broadcast: bool) -> np.ndarray:
    result = np.zeros_like(host)
    for slane, row in enumerate(host):
        for part in range(4):
            shift = (amount & 255) + (4 * slane + part) * ((amount >> 8) & 255)
            source = row[shift % 128] if broadcast else np.roll(row, shift)
            result[slane] |= ((source >> (8 * part)) & 255) << (8 * part)
    return result

def xlu(p: Probe) -> None:
    host = np.arange(1024, dtype=np.uint32).reshape(8, 128) * np.uint32(0x04040404) + np.uint32(0x03020100)
    for shift in (0, 1, 3, 8, 127, 128, 255, 511, 0x102, 0x203, 0x7F02):
        body = op(f'vx0: vrot.lane.packed.b8.8x128 {p.input}, {shift}') + op('vr0: vpop.8x128 v10, trf0')
        expected = packed_lane_expected(host, shift, broadcast=False)
        p.run(f'rotate-b8-{shift}', body, expected, host=host)
        body = op(f'vx0: vbcast.lane.packed.b8.8x128 {p.input}, {shift}') + op('vr0: vpop.8x128 v10, trf0')
        expected = packed_lane_expected(host, shift, broadcast=True)
        p.run(f'broadcast-b8-{shift}', body, expected, host=host)
    # 用每个 sublane 都不同的源，观察 half-sublanes 对 PCR 的装载。
    for pattern in (0, 1, 0x03020100, 0x07060504, 0x7F7E7D7C):
        body = op(f'va0: vimm.8x128.s32 v11, {pattern:#x}') + op('vx0: vsetperm.half.u8 v11') + op(f'vx0: vperm.lane.8x128 trf0, {p.input}') + op('vr0: vpop.8x128 v10, trf0')
        expected = ((host[:, pattern & 127, None] & 65535) | (host[:, (pattern >> 16) & 127, None] & 0xFFFF0000)).astype(np.uint32)
        p.run(f'perm-half-{pattern:x}', body, expected, host=host)

def waits(p: Probe) -> None:
    for condition, actual, threshold, setop in [
        ('eq', 7, 7, 's32'),
        ('ne', 7, 8, 's32'),
        ('gt', 7, 6, 's32'),
        ('ge', 7, 7, 's32'),
        ('lt', 7, 8, 's32'),
        ('done', 7, None, 'doneinv.s32'),
        ('notdone', 7, None, 'done.s32'),
    ]:
        body = op(f'misc: vsyncset.{setop} [sflag:100], {actual}')
        body += op(f'misc: vwait.{condition}.yield [sflag:100]' + (f', {threshold}' if threshold is not None else ''))
        body += op(f'va0: vmov.8x128 v10, {p.input}') + op('misc: vsyncset.s32 [sflag:100], 0')
        p.run(f'wait-{condition}', body, p.host)

def doneflags(p: Probe) -> None:
    for form in ('s32', 'done.s32', 'doneinv.s32'):
        for value in (-1, 0, 1, 7):
            body = op(f'misc: vsyncset.{form} [sflag:100], {value}') + op('misc: vsyncdonemov sfrf, [sflag:100]') + op('s0: spop s20, sfrf') + op('va0: vmov.8x128 v10, s20')
            p.run(f'doneflag-{form}-{value}', body, np.uint32(form == 'doneinv.s32'))

def memory(p: Probe) -> None:
    host = np.arange(256 * 128, dtype=np.uint32).reshape(256, 128)
    for encoded, name in value_names('VectorStride').items():
        if name not in ('one', 'negative_one', 'two', 'four'):
            continue
        body = instruction('vld0', 'vector_load', dest_vreg=10, stride=encoded, offset=1, imm0=64)
        stride = {'one': 1, 'negative_one': -1, 'two': 2, 'four': 4}[name]
        p.run(f'stride-{name}', body, host[64 + np.arange(8) * stride], host=host)
    for encoded, name in value_names('SublaneMask').items():
        if name not in ('all_ones', 'one', 'x0f', 'xf0', 'three', 'sixteen'):
            continue
        body = op('va0: vimm.8x128.s32 v10, 0xdeadbeef') + instruction('vld0', 'vector_load', dest_vreg=10, sublane_mask=encoded, offset=1, imm0=64)
        mask = {'all_ones': 255, 'one': 1, 'x0f': 15, 'xf0': 240, 'three': 3, 'sixteen': 16}[name]
        expected = np.where(((mask >> np.arange(8)) & 1)[:, None], host[64:72], 0).astype(np.uint32)
        p.run(f'mask-{name}', body, expected, host=host)
    for encoded, name in value_names('VectorShuffle').items():
        if not name.startswith('descending_pattern'):
            continue
        body = instruction('vld0', 'vector_load_shuffled', dest_vreg=10, shuffle=encoded, offset=1, imm0=64)
        p.run(f'shuffle-{name}', body, np.roll(host[64:72], -int(name[-1]), axis=0), host=host)
    for pair in range(3):
        body = op('s0: simm.s32 s20, 64') + instruction('vld0', 'vector_load_shuffled_base', dest_vreg=10, shuffle=5 + pair, base_address=0, vs0=20, **{f'imm{2 * pair}': 0x23210, f'imm{2 * pair + 1}': 0xA7654})
        p.run(f'shuffle-pair-{pair}', body, host[64:72], host=host)
    for lane in range(6):
        body = instruction('vld0', 'vector_load', dest_vreg=10, offset=lane + 1, **{f'imm{lane}': 64})
        p.run(f'offset-imm{lane}', body, host[64:72], host=host)
    for lane in range(4):
        body = op(f's0: simm.s32 s{20 + lane}, 64') + instruction('vld0', 'vector_load_base', dest_vreg=10, base_address=lane, **{f'vs{lane}': 20 + lane})
        p.run(f'base-vs{lane}', body, host[64:72], host=host)

def source_constants(p: Probe) -> None:
    for selector, amount in ((0, 1), (11, 64), (12, 48), (13, 32), (14, 16), (15, 8)):
        body = instruction('vx0', 'lane_rotate_32_bit', vex_source=0, vst_source=int(p.input[1:]), rotate_specifier=selector) + '\n.empty 16\n' + op('vr0: vpop.8x128 v10, trf0')
        p.run(f'source-constant-{amount}', body, np.roll(p.host, amount, axis=1))

def widths(p: Probe) -> None:
    for encoded, width in ((14, 8), (13, 16), (12, 32), (11, 64), (0, 128)):
        drain = op('vr0: vpop.8x128 v10, trf0') * (width // 8)
        body = instruction('vx0', 'transpose_start_end', vex_source=0, vst_source=int(p.input[1:]), matrix_width=5, imm0=width) + '\n.empty 16\n' + drain
        expected = np.zeros((8, 128), np.uint32)
        expected[:, :8] = p.host[:, width - 8 : width].T
        reference = p.run(f'width-immediate-{width}', body, expected)
        body = instruction('vx0', 'transpose_start_end', vex_source=0, vst_source=int(p.input[1:]), matrix_width=encoded) + '\n.empty 16\n' + drain
        p.run(f'width-constant-{width}', body, reference)

def remap_codes(p: Probe) -> None:
    host = np.resize(
        np.array(
            [0, 0x80000000, 0x3F800000, 0xBF800000, 0x40000000, 0xC0000000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F801234, 1, 0x007FFFFF, 0x80000001, 0x807FFFFF, 0x3F803F80, 0x7F80FF80],
            np.uint32,
        ),
        (8, 128),
    )
    for fmt in ('f32', 'bf16'):
        for code in range(16):
            mode = code * 0x1111111
            body = op(f'va0: vimm.8x128.s32 v11, {mode:#x}') + op(f'va0: vremap.8x128.{fmt} v10, {p.input}, v11')
            p.run(f'remap-code-{fmt}-{code}', body, remap_word(host, mode, fmt), host=host)

def sync_selectors(p: Probe) -> None:
    read = op('misc: vsyncmov sfrf, [sflag:100]') + op('s0: spop s21, sfrf') + op('va0: vmov.8x128 v10, s21')
    p.run('operand-zero', op('misc: vsyncset.s32 [sflag:100], 0') + read, np.uint32(0))
    for lane in range(6):
        flag_lane = (lane + 1) % 6
        body = instruction('misc', 'set_sync', sync_flag_number=5 + flag_lane, operand=5 + lane, **{f'imm{lane}': 0xABCDE, f'imm{flag_lane}': 100}) + '\n.empty 16\n' + read
        p.run(f'operand-imm{lane}', body, np.uint32(0xFFFABCDE))
    for lane in range(4):
        body = op('s0: simm.s32 s20, 0x12345678') + instruction('misc', 'set_sync', sync_flag_number=5, operand=1 + lane, imm0=100, **{f'vs{lane}': 20}) + '\n.empty 16\n' + read
        p.run(f'operand-vs{lane}', body, np.uint32(0x12345678))
    for selector in range(11):
        body = op('misc: vsyncset.s32 [sflag:100], 6543')
        fields = {'sync_flag_number': selector}
        if selector == 0:
            body = op('misc: vsyncmov sfrf, [sflag:0]') + op('s0: spop s22, sfrf')
        elif selector <= 4:
            fields[f'vs{selector - 1}'] = 20
            body += op('s0: simm.s32 s20, 100')
        else:
            fields[f'imm{selector - 5}'] = 100
        body += instruction('misc', 'read_sync', **fields) + '\n.empty 16\n' + op('s0: spop s21, sfrf')
        if selector == 0:
            body += op('s0: ssub.s32 s21, s21, s22')
        body += op('va0: vmov.8x128 v10, s21')
        p.run(f'vector-source-{selector}', body, np.uint32(0 if selector == 0 else 6543))

def remap_random(p: Probe) -> None:
    rng = np.random.default_rng(4051)
    host = rng.integers(0, 1 << 32, (8, 128), dtype=np.uint32)
    host[:, :4] = np.array([0x80000001, 0x807FFFFF, 1, 0x007FFFFF], np.uint32)
    controls = rng.integers(0, 1 << 32, (8, 128), dtype=np.uint32)
    for fmt in ('f32', 'bf16'):
        for slot in ('va0', 'va1', 'va2', 'va3'):
            body = op('vld0: vld.8x128 v11, [vmem:0x8]') + op(f'{slot}: vremap.8x128.{fmt} v10, {p.input}, v11')
            p.run(f'remap-random-{fmt}-{slot}', body, remap_word(host, controls, fmt), host=np.concatenate((host, controls)))

def strides_large(p: Probe) -> None:
    host = np.arange(256 * 128, dtype=np.uint32).reshape(256, 128)
    for encoded, stride in ((14, 8), (15, 16)):
        body = instruction('vld0', 'vector_load', dest_vreg=10, stride=encoded, sublane_mask=15, offset=1, imm0=64)
        expected = np.zeros((8, 128), np.uint32)
        expected[4] = host[64 + 4 * stride]
        p.run(f'stride-masked-{stride}', body, expected, host=host)

def delays(p: Probe) -> None:
    # 由 vector FIFO 读回形成屏障；多次采样的最小值减少宿主及队列调度抖动。
    records = []
    minima = {}
    for count in (1, 16, 64, 128):
        for operand in ('builtin', 0, 1, 2, 16):
            delay = instruction('misc', 'vector_delay', operand=0 if operand == 'builtin' else 5, imm0=0 if operand == 'builtin' else int(operand))
            delay = '\n'.join(line for line in delay.splitlines() if line != '{}' and not line.startswith('.empty')) + '\n'
            body = op('misc: vsyncset.s32 [sflag:100], 0') + op('s0: srdreg.lcclo s20') + delay * count
            body += op('misc: vsyncmov sfrf, [sflag:100]') + op('s0: spop s22, sfrf') + op('s0: srdreg.lcclo s21')
            body += op('s0: ssub.s32 s20, s21, s20') + op('va0: vmov.8x128 v10, s20')
            source = f'.target {TARGET}\n.empty 16\n' + body + '\n.empty 16\n'
            fragment = parse_assembly(source, fragment=True)
            assert len(fragment.bundles) < p.capacity
            code = assemble_listing(source + f'\n.empty {p.capacity - len(fragment.bundles)}\n')
            image = p.image[: 64 * p.pc] + code + p.image[64 * (p.pc + p.capacity) :]
            function = load_executable(replace_executable_programs(p.raw, {p.key: image}), p.compiled)
            samples = []
            for _ in range(20):
                actual = np.asarray(function(p.x))
                assert np.all(actual == actual[0, 0])
                samples.append(int(actual[0, 0]))
            records.append({'count': count, 'operand': operand, 'cycles': samples})
            minima[count, operand] = min(samples)
            (p.output / f'delay-{count}-{operand}.tpuasm').write_text(source)
            print('delay', count, operand, samples, flush=True)
    (p.output / 'delay-timings.json').write_text(json.dumps(records, indent=2) + '\n')
    for count in (64, 128):
        assert abs(minima[count, 'builtin'] - minima[count, 1]) <= 16
        for operand in (1, 2, 16):
            assert abs(minima[count, operand] - minima[count, 0] - count * operand) <= 16

GROUPS = {group.__name__: group for group in (selectors, arithmetic, remap_codes, remap_random, unpack, memory, strides_large, source_constants, xlu, widths, sync_selectors, doneflags, waits, delays)}

def archive_results(output: Path, destination: Path) -> None:
    """只读取已记录的运行环境与采样，打包为 JSON/NPZ，不初始化设备。"""
    records = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
    environment = next(record['environment'] for record in records if 'environment' in record)
    cases = {record['name']: record for record in records if 'name' in record}
    arrays: dict[str, np.ndarray] = {}
    inputs: dict[tuple[tuple[int, ...], str, bytes], str] = {}
    archived = []
    for name in sorted(cases):
        host = np.load(output / f'{name}-input.npy')
        identity = (host.shape, host.dtype.str, host.tobytes())
        if identity not in inputs:
            key = f'input_{len(inputs)}'
            inputs[identity] = key
            arrays[key] = host
        arrays[name] = np.load(output / f'{name}.npy')
        archived.append({'name': name, 'input': inputs[identity], 'match': cases[name]['match'], 'assembly': (output / f'{name}.tpuasm').read_text()})
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / 'execution.npz', allow_pickle=False, **arrays)
    report = {
        'environment': environment,
        'arrays': 'execution.npz',
        'array_format': 'uint32; output keys are case names; input keys are referenced per case',
        'original_executable_baseline_passed': records[-1].get('original_executable_baseline_passed', False),
        'cases': archived,
    }
    (destination / 'execution.json').write_text(json.dumps(report, indent=2) + '\n')
    if (output / 'delay-timings.json').exists():
        (destination / 'execution-delay-timings.json').write_text((output / 'delay-timings.json').read_text())
    print(f'Archived {len(archived)} cases and {len(inputs)} distinct inputs to {destination}.', flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('all', *GROUPS), default='all')
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpuasm-v6e-execution'))
    parser.add_argument('--archive', type=Path, help='运行后把原始输出打包到指定目录')
    parser.add_argument('--archive-only', action='store_true', help='只打包 --output 中已有的记录，不使用设备；需指定 --archive')
    args = parser.parse_args()
    if args.archive_only:
        if args.archive is None:
            parser.error('--archive-only requires --archive')
        archive_results(args.output, args.archive)
        return
    probe = Probe(args.output)
    for group in GROUPS if args.group == 'all' else (args.group,):
        GROUPS[group](probe)
    np.testing.assert_array_equal(np.asarray(probe.compiled(probe.x)), probe.host ^ 0x13579BDF)
    probe.log.write(json.dumps({'original_executable_baseline_passed': True}) + '\n')
    probe.log.close()
    if args.archive is not None:
        archive_results(args.output, args.archive)
    print('All selected device checks passed; original executable still returns its baseline result.', flush=True)

if __name__ == '__main__':
    main()

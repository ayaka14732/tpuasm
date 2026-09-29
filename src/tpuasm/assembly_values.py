"""汇编操作数的精确数值转换，不依赖原生 formatter 的显示精度。"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
import re
import struct

from .assembly_syntax import integer

_FLOAT = re.compile(r'[+-]?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)\Z')
_BITS = re.compile(r'f32bits\(\s*(0x[0-9a-fA-F]+)\s*\)\Z')

def _nearest(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient + (2 * remainder > denominator or (2 * remainder == denominator and quotient % 2 == 1))

def float32_bits(text: str) -> int:
    """将精确十进制舍入一次到 binary32（nearest, ties-to-even）。"""
    if text in ('inf', '+inf', '-inf'):
        return 0xff800000 if text.startswith('-') else 0x7f800000
    if not _FLOAT.fullmatch(text):
        integer(text)
    try:
        decimal = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f'invalid float32 literal {text!r}') from error
    if not decimal.is_finite():
        raise ValueError('use f32bits(...) for NaN and inf for infinity')
    sign = int(decimal.is_signed()) << 31
    value = Fraction(decimal.copy_abs())
    if not value:
        return sign
    n, d = value.numerator, value.denominator
    exponent = n.bit_length() - d.bit_length()
    if (n < d << exponent) if exponent >= 0 else (n << -exponent < d):
        exponent -= 1
    step = max(exponent, -126) - 23
    mantissa = _nearest(n, d << step) if step >= 0 else _nearest(n << -step, d)
    if exponent < -126:
        return sign | mantissa
    if mantissa == 1 << 24:
        exponent += 1
        mantissa >>= 1
    if exponent > 127:
        raise ValueError('finite literal overflows float32; use inf explicitly')
    return sign | ((exponent + 127) << 23) | (mantissa - (1 << 23))

def number(text: str, kind: str = 's32', width: int = 32) -> int:
    """返回有界原始位型；十进制检查数值范围，非负十六进制允许原始位型。"""
    text = text.strip()
    bits = _BITS.fullmatch(text)
    if bits:
        if width != 32:
            raise ValueError('f32bits requires a 32-bit operand')
        value = integer(bits[1])
        if value >= 1 << 32:
            raise ValueError('f32bits exceeds 32 bits')
        return value
    if kind == 'f32' and '0x' not in text:
        return float32_bits(text)
    value = integer(text)
    signed = kind.startswith('s')
    raw_hex = text.lstrip('+').startswith('0x')
    low = -(1 << (width - 1)) if signed else 0
    high = (1 << width) - 1 if raw_hex or not signed else (1 << (width - 1)) - 1
    if not low <= value <= high:
        raise ValueError(f'{text} exceeds {kind} operand range for {width} bits')
    return value & ((1 << width) - 1)

def display_number(value: int, kind: str = 's32', width: int = 32) -> str:
    value &= (1 << width) - 1
    if kind == 'f32':
        scalar = struct.unpack('<f', value.to_bytes(4, 'little'))[0]
        if math.isnan(scalar) or value == 0x80000000:
            return f'f32bits({value:#010x})'
        return repr(scalar)
    if kind == 'hex':
        return hex(value)
    if kind.startswith('s') and value >> (width - 1):
        value -= 1 << width
    return str(value)

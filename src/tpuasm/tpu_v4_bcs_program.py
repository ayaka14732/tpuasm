"""BCS semantic protobuf 互操作与 libtpu 程序容器提取。"""
from __future__ import annotations

from ._protobuf import fields
from .assembly_model import Bits
from .tpu_v4_bcs_isa import FORMS_BY_BRANCH
from .printer import _program_proto, _verify_image
from .targets import TPU_V4_BCS
from .program_container import executable_records

TARGET = TPU_V4_BCS.identifier

def _semantic_fields(program: bytes) -> list[Bits]:
    bundles = fields(program)
    if not bundles or len(bundles) % TPU_V4_BCS.bundles_per_block:
        raise ValueError('BCS semantic program must contain a positive multiple of 16 bundles; link fragments before encoding')
    words = []
    for pc, (number, wire, payload) in enumerate(bundles):
        if number != 1 or wire != 2 or not isinstance(payload, bytes):
            raise ValueError('BCS semantic program must contain only repeated field-1 bundles')
        bound = Bits()
        occupied = set()
        for number, wire, instruction in fields(payload):
            if number not in (1, 2) or number in occupied or wire != 2 or not isinstance(instruction, bytes):
                raise ValueError(f'invalid or repeated BCS slot at bundle {pc:#x}')
            occupied.add(number)
            slot = TPU_V4_BCS.slots[number - 1]
            common = fields(instruction)
            predicates = [value for key, kind, value in common if key == 1 and kind == 0 and isinstance(value, int)]
            operations = [(key, value) for key, kind, value in common if key >= 5 and kind == 2 and isinstance(value, bytes)]
            if len(predicates) > 1 or len(operations) != 1:
                raise ValueError(f'invalid BCS predicate or operation oneof at bundle {pc:#x}')
            predicate = predicates[0] if predicates else 0
            branch, arguments = operations[0]
            if any(key not in (1, 2, 3, 4, branch) or (key in (1, 2, 4) and kind != 0) or (key in (3, branch) and kind != 2) for key, kind, _ in common):
                raise ValueError(f'unknown BCS instruction metadata at bundle {pc:#x}')
            if branch == 5:
                if arguments:
                    raise ValueError('BCS Noop must have no operands')
                continue
            form = FORMS_BY_BRANCH.get((slot, branch))
            if form is None:
                raise ValueError(f'unsupported BCS instruction {slot}.{branch}')
            requested = form.fixed(predicate)
            seen = set()
            for key, kind, value in fields(arguments):
                field = next((field for field in form.fields.values() if field.number == key), None)
                if field is None or key in seen or kind != 0 or not isinstance(value, int):
                    raise ValueError(f'unknown or repeated BCS operand at bundle {pc:#x}')
                seen.add(key)
                signed = value - (1 << 64) if value >= 1 << 63 else value
                if not -(1 << (field.width - 1)) <= signed < 1 << field.width:
                    raise ValueError(f'BCS operand exceeds its hardware width at bundle {pc:#x}')
                merged = requested.merge(field.bind(signed & ((1 << field.width) - 1)))
                if merged is None:
                    raise ValueError(f'BCS operand aliases require conflicting bits at bundle {pc:#x}')
                requested = merged
            merged = bound.merge(requested)
            if merged is None:
                raise ValueError(f'BCS slots require conflicting shared fields at bundle {pc:#x}')
            bound = merged
        words.append(bound)
    return words

def encode_tpu_v4_bcs_program(program: bytes) -> bytes:
    """将完整 BCS semantic protobuf 编码为机器程序映像，不读写文件或访问设备。

    ``program`` 是 BarnaCoreSequencerProgram（repeated field-1 bundles），不是外层的 TpuCoreProgramProto。bundle 数必须为 16 的正整数倍；不会自动补齐 fragment 或隐式覆盖双槽的共享字段。使用与当前环境匹配的 libtpu 后端，并核对原生编码和实际硬件字段。latency/resource_usage/bit_width 是辅助元数据，不属于机器映像。返回机器字节。非法结构、字段范围、共享字段冲突或不可逆编码抛出 ValueError/RuntimeError。调用方继续负责内存分配、装载、生命周期及运行时对程序长度的限制。
    """
    constraints = _semantic_fields(program)
    image = _program_proto(program, encode=True, target=TARGET)
    if len(image) != len(constraints) * 32:
        raise ValueError('BCS native codec changed the bundle count')
    for pc, bound in enumerate(constraints):
        word = int.from_bytes(image[pc * 32:(pc + 1) * 32], 'little')
        if word & bound.mask != bound.value:
            raise ValueError(f'BCS native codec normalized requested fields at bundle {pc:#x}')
    _verify_image(image, target=TARGET)
    return image

def decode_tpu_v4_bcs_program(image: bytes) -> bytes:
    """将完整 BCS 机器映像解码为当前后端的 semantic protobuf。

    不读写文件或访问设备。输入须为非空 512 字节对齐的完整映像，且通过原生 decode→encode 逐字节检查。返回值可交回 :func:`encode_tpu_v4_bcs_program`；保证机器映像固定点，不保证恢复此前 protobuf 的字段顺序、presence 或辅助元数据。非法输入抛出 ValueError/RuntimeError。
    """
    program = _program_proto(image, encode=False, target=TARGET)
    if encode_tpu_v4_bcs_program(program) != image:
        raise ValueError('BCS semantic protobuf does not reproduce its machine image')
    return program

def _nested(message: bytes, number: int, context: str) -> bytes:
    values = [(wire, value) for key, wire, value in fields(message) if key == number]
    if len(values) != 1 or values[0][0] != 2 or not isinstance(values[0][1], bytes):
        raise ValueError(f'{context}: expected one message field {number}')
    return values[0][1]

def extract_tpu_v4_bcs_program(core_program: bytes) -> bytes:
    """从一个 TpuCoreProgramProto 提取 BCS semantic protobuf，不选择 codec。

    当前已核对的 libtpu 容器路径为 ``barna_core(6) → sequencer(1) → pufferfish(10)``。输入是单个 protobuf，区别于长度分隔的 serialized executable。不提取 Channel Controller，不解码为机器字节，不修改容器。缺失或重复路径抛出 ValueError；后续用 encode_tpu_v4_bcs_program 验证完整程序。
    """
    core_kinds = [key for key, wire, value in fields(core_program) if key in (5, 6, 7)]
    if core_kinds != [6]:
        raise ValueError('expected a single BarnaCore core-program alternative')
    barna = _nested(core_program, 6, 'TpuCoreProgramProto')
    sequencer = _nested(barna, 1, 'BarnaCore')
    kinds = [value for key, wire, value in fields(sequencer) if key == 3 and wire == 0]
    if kinds != [2]:
        raise ValueError('expected a BarnaCore sequencer (sequencer_type=2)')
    alternatives = [key for key, wire, value in fields(sequencer) if 7 <= key <= 23]
    if alternatives != [10]:
        raise ValueError('expected the Pufferfish BCS semantic program alternative')
    return _nested(sequencer, 10, 'BarnaCore sequencer')

def executable_bcs_programs(serialized: bytes) -> list[tuple[int, int, bytes]]:
    programs = []
    for record, raw in enumerate(executable_records(serialized)):
        try:
            outer = fields(raw)
        except ValueError:
            outer = []
        # platform_type=1 and the BarnaCore oneof identify candidate core records;
        # field 2 is NOT a core-kind enum. Validate the nested sequencer separately.
        if any(key == 2 and wire == 0 and value == 1 for key, wire, value in outer) and any(key == 6 and wire == 2 for key, wire, value in outer):
            programs.append((record, 0, encode_tpu_v4_bcs_program(extract_tpu_v4_bcs_program(raw))))
    if not programs:
        raise ValueError('no BCS programs found in serialized executable')
    return programs

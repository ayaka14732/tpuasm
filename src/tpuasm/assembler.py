"""按明确声明的硬件目标分派汇编与反汇编。"""
from __future__ import annotations

from .assembly_syntax import parse_assembly
from .targets import HardwareTarget, TPU_V4_TC, TPU_V4_BCS, TPU_V6E_TC, TPU_V6E_TEC, hardware_target
from .tc_source_mapping import ProgramSourceMap

def supports_source_mapping(hardware: HardwareTarget) -> bool:
    """编译来源映射目前只适配 TensorCore。"""
    return hardware in (TPU_V4_TC, TPU_V6E_TC)

def check_source_mapping_supported(hardware: HardwareTarget) -> None:
    if not supports_source_mapping(hardware):
        raise ValueError(f'{hardware.identifier} compiler source mapping is not supported')

def assemble_listing(text: str, *, filename: str = '<assembly>') -> bytes:
    """将完整的 TPU 汇编源码转换为所声明目标的程序映像字节。

    汇编包括整个指令包的资源求解、编码结果重解码核对及原生字节往返校验。本函数不读写文件。

    Args:
        text: 完整的 .tpuasm 源码，必须声明 ``.target``；bundle 总数是该目标每块 bundle 数的正整数倍：v4 TC 为 10，v4 BCS 为 16，v6e TC 为 8，v6e TEC 为 1。
        filename: 错误诊断中的文件名，默认 ``'<assembly>'``；不会打开对应路径。

    Returns:
        完整程序映像的字节，不是 serialized executable。

    Raises:
        ValueError: 源码语法、操作数类型、范围或编码约束冲突，或编码结果与求解结果不一致。
        RuntimeError: 原生后端不可用或原生校验失败。

    Examples:
        在仓库根目录汇编现成样例::

            from pathlib import Path
            from tpuasm import assemble_listing

            source = Path('tests/data/tpu_v4_tc/slots.tpuasm')
            image = assemble_listing(source.read_text(encoding='utf-8'), filename=str(source))
            Path('/tmp/slots.bin').write_bytes(image)
    """
    program = parse_assembly(text, filename=filename)
    if program.hardware == TPU_V4_TC:
        from .tpu_v4_tc_assembler import assemble_program
    elif program.hardware == TPU_V4_BCS:
        from .tpu_v4_bcs_assembler import assemble_program
    elif program.hardware == TPU_V6E_TC:
        from .tpu_v6e_tc_assembler import assemble_program
    elif program.hardware == TPU_V6E_TEC:
        from .tpu_v6e_tec_assembler import assemble_program
    return assemble_program(program)

def format_assembly(image: bytes, *, target: str, encoding: str = 'exact', source_map: ProgramSourceMap | None = None) -> str:
    """将完整程序映像导出为可独立汇编的源码，不读写文件。

    Args:
        image: 非空、按目标块大小对齐的完整程序映像。TPU v4 TC、v6e TC 和 BCS 每块 512 字节，分别含 10、8、16 个 bundle；v6e TEC 每块 64 字节，含 1 个 bundle。
        target: 必填，``'tpu-v4-tc'``、``'tpu-v4-bcs'``、``'tpu-v6e-tc'`` 或 ``'tpu-v6e-tec'``。raw image 没有目标标记，不根据字节长度或尝试不同 codec 猜测。
        encoding: ``'exact'`` （默认）保存原机器编码，必要时附加指令包级命名 ``.encoding`` 约束，实际重汇编并逐字节比较成功后才返回源码。``'canonical'`` 按确定规则重新分配共享资源，生成便于编辑的源码，不承诺与输入程序映像的字节相同。
        source_map: 可选的 TC 编译来源，用于在清单中添加注释；BCS 与 v6e TEC 不支持。

    Returns:
        含目标声明和完整 bundle 清单的 .tpuasm 源码。可汇编性不代表流水线调度或设备执行已经验证。

    Raises:
        ValueError: target / encoding 无效、目标不支持 source_map、指令形式不支持、编码无法恢复，或重汇编校验失败。
        RuntimeError: 原生后端不可用、程序映像格式无效或原生校验失败。

    Examples:
        从已保存的程序映像生成精确源码和便于编辑的源码::

            from pathlib import Path
            from tpuasm import assemble_listing, format_assembly

            image = Path('/tmp/slots.bin').read_bytes()
            exact_source = format_assembly(image, target='tpu-v4-tc')
            editable_source = format_assembly(image, target='tpu-v4-tc', encoding='canonical')
            assert assemble_listing(exact_source) == image
    """
    hardware = hardware_target(target)
    if source_map is not None:
        check_source_mapping_supported(hardware)
    if hardware == TPU_V4_TC:
        from .tpu_v4_tc_assembler import format_program
        return format_program(image, encoding=encoding, source_map=source_map)
    if hardware == TPU_V6E_TC:
        from .tpu_v6e_tc_assembler import format_program as format_v6e_program
        return format_v6e_program(image, encoding=encoding, source_map=source_map)
    if hardware == TPU_V6E_TEC:
        from .tpu_v6e_tec_assembler import format_program as format_tec_program
        return format_tec_program(image, encoding=encoding)
    from .tpu_v4_bcs_assembler import format_program as format_bcs_program
    return format_bcs_program(image, encoding=encoding)

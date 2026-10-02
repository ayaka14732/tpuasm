"""按 TPU 代际与执行单元定义硬件程序格式与固定分支延迟，供不同 libtpu 后端共享。"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class HardwareTarget:
    """一种 TPU 代际与执行单元的程序格式。

    Attributes:
        identifier: 目标名称，例如 ``'tpu-v4-tc'``。
        image_block_size: 程序映像块的字节数。
        bundles_per_block: 每块的 bundle 数。
        slots: 物理槽名称，按清单中的书写顺序排列。
        branch_delay_bundles: 分支的延迟 bundle 数；None 表示该目标尚未确定。
    """
    identifier: str
    image_block_size: int
    bundles_per_block: int
    slots: tuple[str, ...]
    # Verified from target constructors and bundle_for_delay_slots; see docs/design/executable_replacement.md.
    # None means not established for this target.
    branch_delay_bundles: int | None = None

TPU_V4_TC = HardwareTarget(
    identifier='tpu-v4-tc',
    image_block_size=512,
    bundles_per_block=10,
    slots=('s0', 's1', 'va0', 'va1', 'vst', 'vld', 'cld', 'vx0', 'vx1', 'vr0', 'vr1', 'misc'),
    branch_delay_bundles=1,
)

TPU_V4_BCS = HardwareTarget(
    identifier='tpu-v4-bcs',
    image_block_size=512,
    bundles_per_block=16,
    slots=('s0', 's1'),
)

TPU_V6E_TC = HardwareTarget(
    identifier='tpu-v6e-tc',
    image_block_size=512,
    bundles_per_block=8,
    slots=('s0', 's1', 'dma', 'va0', 'va1', 'va2', 'va3', 'vst', 'vld0', 'vld1', 'misc', 'vx0', 'vx1', 'vr0', 'vr1'),
    branch_delay_bundles=4,
)

TPU_V6E_TEC = HardwareTarget(
    identifier='tpu-v6e-tec',
    image_block_size=64,
    bundles_per_block=1,
    slots=('s0', 's1', 'dma', 'misc', 'va0', 'va1', 'va2', 'vld', 'vst', 'stream', 'vr', 'vx'),
)

TARGETS = {target.identifier: target for target in (TPU_V4_TC, TPU_V4_BCS, TPU_V6E_TC, TPU_V6E_TEC)}

def hardware_target(identifier: str) -> HardwareTarget:
    """查找明确指定的硬件目标；不从机器字节长度推测执行单元。"""
    try:
        return TARGETS[identifier]
    except KeyError:
        raise ValueError(f'unsupported hardware target {identifier!r}; expected one of {tuple(TARGETS)}') from None

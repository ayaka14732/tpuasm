#!/usr/bin/env bash
# run_all.sh tpu-v4-tc|tpu-v6e-tc|tpu-v6e-tec：在该目标的 TPU 上运行该目标的全部示例，检查数值并导出清单。
# run_all.sh --aot [tpu-v4-tc|tpu-v6e-tc|tpu-v6e-tec]：不需要 TPU，按参考拓扑离线编译并导出清单；省略目标时依次处理三个目标。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
aot=false
if [[ "${1:-}" == --aot ]]; then
    aot=true
    export TPUASM_EXAMPLES_AOT=1
    export JAX_PLATFORMS=cpu
    shift
fi
targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
    if ! $aot; then
        printf 'usage: %s tpu-v4-tc|tpu-v6e-tc|tpu-v6e-tec\n       %s --aot [tpu-v4-tc|tpu-v6e-tc|tpu-v6e-tec]\n' "$0" "$0" >&2
        exit 2
    fi
    targets=(tpu-v4-tc tpu-v6e-tc tpu-v6e-tec)
fi

for target in "${targets[@]}"; do
    export TPUASM_EXAMPLES_TARGET="$target"
    if [[ "$target" == tpu-v4-tc ]] && ! $aot; then
        # 在多 host 的 v4 切片中只使用本 host 的四颗芯片。
        export TPU_CHIPS_PER_PROCESS_BOUNDS='2,2,1'
        export TPU_PROCESS_BOUNDS='1,1,1'
        export TPU_VISIBLE_CHIPS='0,1,2,3'
    fi
    for example in examples/pallas/*.py; do
        [[ "$example" == */common.py ]] && continue
        # SparseCore 示例（sc_*.py）只属于 tpu-v6e-tec，其余示例只属于 TC 目标。
        if [[ "$example" == */sc_*.py ]]; then
            [[ "$target" == tpu-v6e-tec ]] || continue
        else
            [[ "$target" != tpu-v6e-tec ]] || continue
        fi
        printf 'Running %s for %s\n' "$example" "$target"
        python "$example"
    done
done

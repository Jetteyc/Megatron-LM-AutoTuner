#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

if [ -f .secrets/env.sh ]; then
    source .secrets/env.sh
else
    echo "Warning: .secrets/env.sh not found. Continuing with current environment."
fi

source "${SCRIPT_DIR}/test_env.sh"

export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-6010}"
export NUM_NODES="${NUM_NODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export CP_INTRANODE_BACKEND="${CP_INTRANODE_BACKEND:-torch_dist}"
export CP_INTERNODE_BACKEND="${CP_INTERNODE_BACKEND:-torch_dist}"
export EP_INTRANODE_BACKEND="${EP_INTRANODE_BACKEND:-torch_dist}"
export EP_INTERNODE_BACKEND="${EP_INTERNODE_BACKEND:-torch_dist}"
export TP_INTRANODE_BACKEND="${TP_INTRANODE_BACKEND:-torch_dist}"
export PP_INTERNODE_BACKEND="${PP_INTERNODE_BACKEND:-torch_dist}"
export DP_INTERNODE_BACKEND="${DP_INTERNODE_BACKEND:-torch_dist}"

CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/runtime_baseline_config.json}"
if [[ $# -ge 1 ]] && [[ "$1" != -* ]]; then
    CONFIG_FILE="$1"
    shift
fi

detect_num_gpus_per_node() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        python - <<'PY'
import os
devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
print(len(devices))
PY
        return
    fi

    "${PYTHON_BIN}" - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
}

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-$(detect_num_gpus_per_node)}"

if [[ "${NUM_GPUS_PER_NODE}" -lt 1 ]]; then
    echo "Error: failed to detect available GPUs on this node." >&2
    exit 1
fi

PY_ARGS=(--config "${CONFIG_FILE}")
if [[ -n "${MODEL_FILTER:-}" ]]; then
    PY_ARGS+=(--model-filter "${MODEL_FILTER}")
fi
PY_ARGS+=(
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
    --num-nodes "${NUM_NODES}"
    --node-rank "${NODE_RANK}"
    --nproc-per-node "${NUM_GPUS_PER_NODE}"
)

echo "[runtime-debug] CONFIG_FILE=${CONFIG_FILE}"
echo "[runtime-debug] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NUM_NODES=${NUM_NODES} NODE_RANK=${NODE_RANK}"
echo "[runtime-debug] NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[runtime-debug] CP_INTRANODE_BACKEND=${CP_INTRANODE_BACKEND} CP_INTERNODE_BACKEND=${CP_INTERNODE_BACKEND}"
echo "[runtime-debug] EP_INTRANODE_BACKEND=${EP_INTRANODE_BACKEND} EP_INTERNODE_BACKEND=${EP_INTERNODE_BACKEND}"
echo "[runtime-debug] TP_INTRANODE_BACKEND=${TP_INTRANODE_BACKEND} PP_INTERNODE_BACKEND=${PP_INTERNODE_BACKEND} DP_INTERNODE_BACKEND=${DP_INTERNODE_BACKEND}"

"$PYTHON_BIN" "${SCRIPT_DIR}/runtime_baseline_run_from_config.py" "${PY_ARGS[@]}" "$@"

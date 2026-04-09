#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

if [ -f .secrets/env.sh ]; then
    source .secrets/env.sh
else
    echo "Warning: .secrets/env.sh not found. Continuing with current environment."
fi

ENV_FILE="tests/functional_test/runtime/test_env_schedule_plan_nsys.sh"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
else
    echo "Warning: ${ENV_FILE} not found. Using defaults."
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
TEST_CASES_FILE="${TEST_CASES_FILE:-qwen3_0_6b.json}"
OVERRIDE_MODEL_CONFIG_FILE="${OVERRIDE_MODEL_CONFIG_FILE:-override_model_config.json}"
OVERRIDE_TF_CONFIG_FILE="${OVERRIDE_TF_CONFIG_FILE:-override_tf_config.json}"
TEST_CASES_DIR="${TEST_CASES_DIR:-AutoTuner/testbench/profile/cases/local}"
CONFIG_DIR="${CONFIG_DIR:-AutoTuner/testbench/profile/configs/local}"

TP_SIZE="${TP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-2}"
ETP_SIZE="${ETP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-2}"
VPP_SIZE="${VPP_SIZE:-2}"

NUM_NODES="${NUM_NODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6010}"

NUM_TEST_CASES="${NUM_TEST_CASES:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-8}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-1}"
RUN_ONE_DATA="${RUN_ONE_DATA:-true}"

OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-outputs}"
NSYS_BIN="${NSYS_BIN:-nsys}"
ENABLE_GPU_METRICS="${ENABLE_GPU_METRICS:-auto}"

WORLD_SIZE_EXPECTED=$((TP_SIZE * CP_SIZE * EP_SIZE * ETP_SIZE * PP_SIZE))

if [[ -z "${GPUS_PER_NODE:-}" ]]; then
    if (( WORLD_SIZE_EXPECTED % NUM_NODES != 0 )); then
        echo "Error: WORLD_SIZE=${WORLD_SIZE_EXPECTED} is not divisible by NUM_NODES=${NUM_NODES}."
        exit 1
    fi
    GPUS_PER_NODE=$((WORLD_SIZE_EXPECTED / NUM_NODES))
fi

if (( GPUS_PER_NODE * NUM_NODES != WORLD_SIZE_EXPECTED )); then
    echo "Error: GPUS_PER_NODE*NUM_NODES != TP*CP*EP*ETP*PP"
    echo "  GPUS_PER_NODE=${GPUS_PER_NODE} NUM_NODES=${NUM_NODES}"
    echo "  TP=${TP_SIZE} CP=${CP_SIZE} EP=${EP_SIZE} ETP=${ETP_SIZE} PP=${PP_SIZE}"
    exit 1
fi

if [[ "${NUM_NODES}" -gt 1 && "${MASTER_ADDR}" == "localhost" ]]; then
    echo "Error: NUM_NODES=${NUM_NODES} but MASTER_ADDR=localhost."
    echo "Please set MASTER_ADDR to rank-0 node IP/hostname in ${ENV_FILE}."
    exit 1
fi

# Megatron interleaved pipeline (PP>1 and VPP enabled) requires enough micro-batches.
# Forcing run-one-data makes M=1 and will fail with
# "The number of contiguous micro-batches ... should range in [PP, M]".
if [[ "${RUN_ONE_DATA}" == "true" || "${RUN_ONE_DATA}" == "True" ]]; then
    if [[ "${VPP_SIZE}" != "None" ]] && (( PP_SIZE > 1 )); then
        echo "Error: RUN_ONE_DATA=true is incompatible with interleaved pipeline (PP=${PP_SIZE}, VPP=${VPP_SIZE})."
        echo "Please set RUN_ONE_DATA=false in ${ENV_FILE}."
        exit 1
    fi
fi

TIMESTAMP_VAR=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="${OUTPUT_ROOT_DIR:-outputs}/${TIMESTAMP_VAR}/${MODEL_NAME}/schedule_plan_nsys_compare"
mkdir -p "${OUTPUT_DIR}"

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NUM_NODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --context-parallel-size "${CP_SIZE}"
    --expert-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size "${ETP_SIZE}"
)

if [[ "${VPP_SIZE}" != "None" ]]; then
    PARALLEL_ARGS+=(--virtual-pipeline-model-parallel-size "${VPP_SIZE}")
fi

RUNTIME_ARGS=(
    --model-name "${MODEL_NAME}"
    --test-cases-dir "${TEST_CASES_DIR}"
    --test-cases-file "${TEST_CASES_FILE}"
    --config-dir "${CONFIG_DIR}"
    --override-model-config-file "${OVERRIDE_MODEL_CONFIG_FILE}"
    --override-tf-config-file "${OVERRIDE_TF_CONFIG_FILE}"
    --num-test-cases "${NUM_TEST_CASES:-1}"
    --max-iterations "${MAX_ITERATIONS:-8}"
    --warmup-iterations "${WARMUP_ITERATIONS:-1}"
    --output-dir "${OUTPUT_ROOT_DIR:-outputs}"
)

if [[ "${RUN_ONE_DATA:-true}" == "true" || "${RUN_ONE_DATA:-true}" == "True" ]]; then
    RUNTIME_ARGS+=(--run-one-data)
fi

# Enable rich timeline markers.
export NVTE_NVTX_ENABLED=${NVTE_NVTX_ENABLED:-1}
export NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN:-1}
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# Keep staggered debug logs visible in terminal (not required for nsys timeline itself).
export NE_STAGGERED_1F1B_LOG=${NE_STAGGERED_1F1B_LOG:-1}
export NE_STAGGERED_1F1B_LOG_MAX_CALLS=${NE_STAGGERED_1F1B_LOG_MAX_CALLS:-128}

# Network Engine backend routing (can be overridden in env file).
export CP_INTRANODE_BACKEND=${CP_INTRANODE_BACKEND:-torch_dist}
export CP_INTERNODE_BACKEND=${CP_INTERNODE_BACKEND:-torch_dist}
export EP_INTRANODE_BACKEND=${EP_INTRANODE_BACKEND:-torch_dist}
export EP_INTERNODE_BACKEND=${EP_INTERNODE_BACKEND:-torch_dist}
export TP_INTRANODE_BACKEND=${TP_INTRANODE_BACKEND:-torch_dist}
export PP_INTERNODE_BACKEND=${PP_INTERNODE_BACKEND:-torch_dist}
export DP_INTERNODE_BACKEND=${DP_INTERNODE_BACKEND:-torch_dist}

NSYS_TRACE_ARGS=(
    -w true
    -f true
    -x true
    -t cuda,nvtx,cudnn,cublas,osrt,python-gil
    --sample=none
    --cpuctxsw=none
    --cuda-memory-usage=true
)

ENABLE_GPU_METRICS=${ENABLE_GPU_METRICS:-auto}
if [[ "${ENABLE_GPU_METRICS}" == "auto" ]]; then
    NSYS_HELP_OUTPUT=$("${NSYS_BIN:-nsys}" profile --gpu-metrics-devices=help 2>&1 || true)
    if ! echo "${NSYS_HELP_OUTPUT}" | grep -q "Insufficient privilege"; then
        NSYS_TRACE_ARGS+=(--gpu-metrics-devices=all --cuda-event-trace=false)
    else
        echo "[nsys] GPU metrics disabled due to insufficient privilege."
    fi
elif [[ "${ENABLE_GPU_METRICS}" == "true" ]]; then
    NSYS_TRACE_ARGS+=(--gpu-metrics-devices=all --cuda-event-trace=false)
fi

run_one_case() {
    local plan_name="$1"
    local enable_staggered="$2"
    local report_base="${OUTPUT_DIR}/${plan_name}"

    echo "============================================================"
    echo "Running case: ${plan_name}"
    echo "STAGGERED_1F1B=${enable_staggered}"
    echo "Report: ${report_base}.nsys-rep"
    echo "============================================================"

    export STAGGERED_1F1B="${enable_staggered}"

    "${NSYS_BIN:-nsys}" profile \
        "${NSYS_TRACE_ARGS[@]}" \
        -o "${report_base}" \
        torchrun "${DISTRIBUTED_ARGS[@]}" -m AutoTuner.runtime.baseline.main \
            "${RUNTIME_ARGS[@]}" \
            "${PARALLEL_ARGS[@]}"
}

echo "[schedule-plan-nsys] ===== launch config ====="
echo "[schedule-plan-nsys] MODEL_NAME=${MODEL_NAME}"
echo "[schedule-plan-nsys] TEST_CASES_FILE=${TEST_CASES_FILE}"
echo "[schedule-plan-nsys] TEST_CASES_DIR=${TEST_CASES_DIR}"
echo "[schedule-plan-nsys] CONFIG_DIR=${CONFIG_DIR}"
echo "[schedule-plan-nsys] TP=${TP_SIZE} CP=${CP_SIZE} EP=${EP_SIZE} ETP=${ETP_SIZE} PP=${PP_SIZE} VPP=${VPP_SIZE}"
echo "[schedule-plan-nsys] WORLD_SIZE=${WORLD_SIZE_EXPECTED} NUM_NODES=${NUM_NODES} GPUS_PER_NODE=${GPUS_PER_NODE} NODE_RANK=${NODE_RANK}"
echo "[schedule-plan-nsys] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[schedule-plan-nsys] MAX_ITERATIONS=${MAX_ITERATIONS:-8} WARMUP_ITERATIONS=${WARMUP_ITERATIONS:-1} RUN_ONE_DATA=${RUN_ONE_DATA:-true}"
echo "[schedule-plan-nsys] CP_INTRANODE_BACKEND=${CP_INTRANODE_BACKEND} CP_INTERNODE_BACKEND=${CP_INTERNODE_BACKEND}"
echo "[schedule-plan-nsys] EP_INTRANODE_BACKEND=${EP_INTRANODE_BACKEND} EP_INTERNODE_BACKEND=${EP_INTERNODE_BACKEND}"
echo "[schedule-plan-nsys] TP_INTRANODE_BACKEND=${TP_INTRANODE_BACKEND} PP_INTERNODE_BACKEND=${PP_INTERNODE_BACKEND} DP_INTERNODE_BACKEND=${DP_INTERNODE_BACKEND}"
echo "[schedule-plan-nsys] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[schedule-plan-nsys] =================================="

# run_one_case "baseline_transformer_model_chunk" 0
run_one_case "staggered_transformer_model_chunk" 1

echo ""
echo "Done. Compare these reports in Nsight Systems:"
echo "  ${OUTPUT_DIR}/baseline_transformer_model_chunk.nsys-rep"
echo "  ${OUTPUT_DIR}/staggered_transformer_model_chunk.nsys-rep"

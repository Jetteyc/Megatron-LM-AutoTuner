#!/bin/bash

set -euo pipefail

source .secrets/env.sh

if [ -f tests/functional_test/runtime/test_env.sh ]; then
    source tests/functional_test/runtime/test_env.sh
else
    echo "Warning: tests/functional_test/runtime/test_env.sh not found. Using defaults."
    MODEL_NAME="Qwen/Qwen3-0.6B"
    TEST_CASES_FILE="qwen3_0_6b.json"
    OVERRIDE_MODEL_CONFIG_FILE="override_model_config.json"
    OVERRIDE_TF_CONFIG_FILE="override_tf_config.json"
    NUM_TEST_CASES=1
    MAX_ITERATIONS=10
    WARMUP_ITERATIONS=3

    SHARE_EMB=None

    TP_SIZE=1
    CP_SIZE=1
    EP_SIZE=1
    ETP_SIZE=1
    PP_SIZE=1
    VPP_SIZE=None
fi

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6010}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($TP_SIZE*$CP_SIZE*$EP_SIZE*$ETP_SIZE*$PP_SIZE))
GPUS_PER_NODE=$(($WORLD_SIZE / $NUM_NODES))

if [[ "$NUM_NODES" -gt 1 && "$MASTER_ADDR" == "localhost" ]]; then
    echo "Error: NUM_NODES=${NUM_NODES} but MASTER_ADDR is localhost."
    echo "Please set MASTER_ADDR to rank-0 node IP/hostname."
    exit 1
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --context-parallel-size $CP_SIZE
    --expert-parallel-size $EP_SIZE
    --expert-tensor-parallel-size $ETP_SIZE
)

if [[ "${VPP_SIZE}" != "None" ]]; then
    PARALLEL_ARGS+=(--virtual-pipeline-model-parallel-size $VPP_SIZE)
fi

RUNTIME_ARGS=(
    --model-name $MODEL_NAME
    --test-cases-file $TEST_CASES_FILE
)

if [[ -n "${OVERRIDE_MODEL_CONFIG_FILE:-}" ]]; then
    RUNTIME_ARGS+=(--override-model-config-file "$OVERRIDE_MODEL_CONFIG_FILE")
fi

if [[ -n "${OVERRIDE_TF_CONFIG_FILE:-}" ]]; then
    RUNTIME_ARGS+=(--override-tf-config-file "$OVERRIDE_TF_CONFIG_FILE")
fi

if [[ "${SHARE_EMB}" != "None" ]]; then
    RUNTIME_ARGS+=(--share-embeddings-and-output-weights $SHARE_EMB)
fi

if [[ -n "${NUM_TEST_CASES:-}" ]]; then
    RUNTIME_ARGS+=(--num-test-cases $NUM_TEST_CASES)
fi

if [[ -n "${MAX_ITERATIONS:-}" ]]; then
    RUNTIME_ARGS+=(--max-iterations $MAX_ITERATIONS)
fi

if [[ -n "${WARMUP_ITERATIONS:-}" ]]; then
    RUNTIME_ARGS+=(--warmup-iterations $WARMUP_ITERATIONS)
fi

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0

# Network Engine routing overrides from test env.
export CP_INTRANODE_BACKEND=${CP_INTRANODE_BACKEND:-torch_dist}
export CP_INTERNODE_BACKEND=${CP_INTERNODE_BACKEND:-torch_dist}
export EP_INTRANODE_BACKEND=${EP_INTRANODE_BACKEND:-torch_dist}
export EP_INTERNODE_BACKEND=${EP_INTERNODE_BACKEND:-torch_dist}
export TP_INTRANODE_BACKEND=${TP_INTRANODE_BACKEND:-torch_dist}
export PP_INTERNODE_BACKEND=${PP_INTERNODE_BACKEND:-torch_dist}
export DP_INTERNODE_BACKEND=${DP_INTERNODE_BACKEND:-torch_dist}

echo "[runtime-debug] ===== launch config ====="
echo "[runtime-debug] MODEL_NAME=${MODEL_NAME}"
echo "[runtime-debug] TEST_CASES_FILE=${TEST_CASES_FILE}"
echo "[runtime-debug] TP=${TP_SIZE} CP=${CP_SIZE} EP=${EP_SIZE} ETP=${ETP_SIZE} PP=${PP_SIZE} VPP=${VPP_SIZE}"
echo "[runtime-debug] GPUS_PER_NODE=${GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE} NUM_NODES=${NUM_NODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[runtime-debug] NVTE_ENABLE_NVSHMEM=${NVTE_ENABLE_NVSHMEM:-<unset>} NVSHMEM_HOME=${NVSHMEM_HOME:-<unset>}"
echo "[runtime-debug] CP_INTRANODE_BACKEND=${CP_INTRANODE_BACKEND:-<unset>} CP_INTERNODE_BACKEND=${CP_INTERNODE_BACKEND:-<unset>}"
echo "[runtime-debug] EP_INTRANODE_BACKEND=${EP_INTRANODE_BACKEND:-<unset>} EP_INTERNODE_BACKEND=${EP_INTERNODE_BACKEND:-<unset>}"
echo "[runtime-debug] TP_INTRANODE_BACKEND=${TP_INTRANODE_BACKEND:-<unset>} PP_INTERNODE_BACKEND=${PP_INTERNODE_BACKEND:-<unset>} DP_INTERNODE_BACKEND=${DP_INTERNODE_BACKEND:-<unset>}"
echo "[runtime-debug] NVSHMEM_DEBUG=${NVSHMEM_DEBUG:-<unset>} NVSHMEM_IB_ADDR_FAMILY=${NVSHMEM_IB_ADDR_FAMILY:-<unset>} NVSHMEM_IB_ADDR_RANGE=${NVSHMEM_IB_ADDR_RANGE:-<unset>}"
echo "[runtime-debug] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>} NCCL_DEBUG=${NCCL_DEBUG:-<unset>}"
echo "[runtime-debug] CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-<unset>} TORCH_USE_CUDA_DSA=${TORCH_USE_CUDA_DSA:-<unset>} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<unset>}"
echo "[runtime-debug] ========================="

torchrun ${DISTRIBUTED_ARGS[@]} -m AutoTuner.runtime.baseline.main \
    ${RUNTIME_ARGS[@]} \
    ${PARALLEL_ARGS[@]}

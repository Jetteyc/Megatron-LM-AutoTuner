# Schedule plan nsys compare sample config
# Copy to tests/functional_test/runtime/test_env_schedule_plan_nsys.sh and edit.

MODEL_NAME="Qwen/Qwen3-30B-A3B-Base_8layers"
TEST_CASES_FILE="qwen3_30b_a3b.json"

# 5D parallel shape used for overlap comparison.
TP_SIZE=1
CP_SIZE=2
EP_SIZE=2
ETP_SIZE=1
PP_SIZE=2
VPP_SIZE=2

# Multi-node launch config.
NUM_NODES=2
NODE_RANK=0
MASTER_ADDR="10.156.154.35"

# Runtime iterations: keep enough steady-state windows for timeline observation.
NUM_TEST_CASES=1
MAX_ITERATIONS=1
WARMUP_ITERATIONS=0
RUN_ONE_DATA=false

OUTPUT_ROOT_DIR="outputs"
NSYS_BIN="nsys"
ENABLE_GPU_METRICS="auto"

# Network Engine backend routing (override per platform if needed).
CP_INTRANODE_BACKEND=torch_dist
CP_INTERNODE_BACKEND=torch_dist
EP_INTRANODE_BACKEND=torch_dist
EP_INTERNODE_BACKEND=torch_dist
TP_INTRANODE_BACKEND=torch_dist
PP_INTERNODE_BACKEND=torch_dist
DP_INTERNODE_BACKEND=torch_dist

# Optional knobs.
NVTE_NVTX_ENABLED=1
NVTE_FLASH_ATTN=1
NVTE_FUSED_ATTN=0
CUDA_DEVICE_MAX_CONNECTIONS=1
NE_STAGGERED_1F1B_LOG=1
NE_STAGGERED_1F1B_LOG_MAX_CALLS=128
NE_STAGGERED_1F1B_DEBUG=1

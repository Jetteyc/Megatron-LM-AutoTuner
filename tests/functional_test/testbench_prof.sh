#!/bin/bash

MEGATRON_LM_HASH=$(git -C "Megatron-LM" rev-parse --short=6 HEAD)
TRANSFORMER_ENGINE_HASH=$(git -C "TransformerEngine" rev-parse --short=6 HEAD)
VERL_HASH=$(git -C "verl" rev-parse --short=6 HEAD)

MODEL_NAME="Qwen/Qwen3-0.6B"
TEST_CASES_FILE="qwen3_0_6b.json"

TEST_OPS_LIST=None
TEST_CASE_IDXES=None

TIMESTAMP_VAR=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR=outputs/$(TIMESTAMP_VAR)


ARGS=(
    --model-name $MODEL_NAME
    --test-cases-file $TEST_CASES_FILE
    --test-ops-list $TEST_OPS_LIST
    --test-case-idxes $TEST_CASE_IDXES
    --profile-mode
    --output-dir $OUTPUT_DIR
)

python3 -m AutoTuner.testbench.nsys_main \
    ${ARGS[@]} \
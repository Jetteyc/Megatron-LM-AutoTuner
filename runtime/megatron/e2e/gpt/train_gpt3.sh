#!/bin/bash

SECRET_ENV_FILE=".secrets/env.sh"
if [[ -f $SECRET_ENV_FILE ]]; then
    source $SECRET_ENV_FILE
else
    echo "[WARNING] Please create a .secrets/env.sh file which contains necessary environment variables."
fi

BASE_DIR=${BASE_DIR:-"${BASE_DATA_DIR}/dataset/pretrain"}
POST_PROCESS_DATA_DIR=${POST_PROCESS_DATA_DIR:-"${BASE_DIR}/post_processed"}

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=checkpoints
TENSORBOARD_LOGS_PATH=logs/tensorboard
VOCAB_FILE=$BASE_DIR/gpt2-vocab.json
MERGE_FILE=$BASE_DIR/gpt2-merges.txt
DATA_PATH=$POST_PROCESS_DATA_DIR/text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

CONFIG_YAML=runtime/megatron/e2e/gpt/gpt_config.yaml

CONFIG_ARGS=(
    --yaml-cfg $CONFIG_YAML
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

#!/bin/bash

## Follow the https://help.aliyun.com/zh/ecs/use-cases/use-the-megatron-deepspeed-training-gpt-2-and-generate-text

SECRET_ENV_FILE=".secrets/env.sh"
if [[ -f $SECRET_ENV_FILE ]]; then
    source $SECRET_ENV_FILE
else
    echo "[WARNING] Please create a .secrets/env.sh file which contains necessary environment variables."
fi

BASE_DIR=${BASE_DIR:-"${BASE_DATA_DIR}/dataset/pretrain"}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/post_processed"}

python3 Megatron-LM/tools/preprocess_data.py \
    --input $BASE_DIR/oscar-1GB.jsonl \
    --output-prefix $OUTPUT_DIR/meg-gpt2 \
    --vocab-file $BASE_DIR/gpt2-vocab.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file $BASE_DIR/gpt2-merges.txt \
    --append-eod \
    --workers 8
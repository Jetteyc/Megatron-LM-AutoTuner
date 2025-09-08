#!/bin/bash

MARK=${1:-"0"}

docker create --rm -it --gpus all --shm-size=25GB --name megatron-lm-autotuner-$MARK -v $(pwd):/workspace/Megatron-LM-AutoTuner nvcr.io/nvidia/pytorch:24.08-py3
docker start megatron-lm-autotuner-$MARK
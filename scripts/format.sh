#!/bin/bash

DIR_LIST=(
    AutoTuner
)

for dir in "${DIR_LIST[@]}"; do
    isort $dir
    black $dir
done

#!/bin/bash

DIR_LIST=(
    testbench
    utils
    AutoTunner
)

for dir in "${DIR_LIST[@]}"; do
    isort $dir
    black $dir
done

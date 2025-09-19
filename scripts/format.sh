#!/bin/bash

DIR_LIST=(
    AutoTunner
)

for dir in "${DIR_LIST[@]}"; do
    isort $dir
    black $dir
done

#!/bin/bash

task=$1
model=$2
method=$3

declare -a gpus=(1 2 3)

num_gpus=${#gpus[@]}

declare -a parts=()

for ((i=0; i<num_gpus; i++)); do
    parts+=($i)
done

for i in "${!gpus[@]}"; do
  command="CUDA_VISIBLE_DEVICES=${gpus[$i]} python run.py --chunk_id ${parts[$i]} --total_chunks $num_gpus --model $model --task $task --method $method"
  echo "Executing: $command"
  eval $command &
  sleep 10
done

wait
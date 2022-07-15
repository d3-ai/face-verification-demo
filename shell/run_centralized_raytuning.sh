#!/bin/bash

. ./shell/path.sh

dataset="CIFAR10"
model="ResNet18"
max_epochs=10
seed=1234
yaml_path="./conf/${dataset}/Centralized_${model}/search_space.yaml"

ray start --head --min-worker-port 20000 --max-worker-port 29999 
sleep 1

python ./local/centralized_raytuning.py \
--dataset ${dataset} \
--model ${model} \
--seed ${seed} &
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait
ray stop -f
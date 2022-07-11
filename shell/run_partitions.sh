#!/bin/bash

. ./shell/path.sh

dataset="CIFAR10"
target="iid"

# fl configuration
num_clients=10

seed=1234

python ./local/partitions.py \
--dataset ${dataset} \
--target ${target} \
--num_clients ${num_clients} \
--seed ${seed} 
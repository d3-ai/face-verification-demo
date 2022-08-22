#!/bin/bash

. ./shell/path.sh
export CUDA_VISIBLE_DEVICES="MIG-1802aebf-f92f-5ef7-a5ab-ed496882fdaf"
dataset="CIFAR10"
model="ResNet18"
seed=1234

# training configuration
max_epochs=5
batch_size=256
lr=0.05
momentum=0.9
weight_decay=5e-4

python ./local/centralized.py \
--dataset ${dataset} \
--model ${model} \
--max_epochs ${max_epochs} \
--batch_size ${batch_size} \
--lr ${lr} \
--momentum ${momentum} \
--weight_decay ${weight_decay} \
--seed ${seed} &
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait
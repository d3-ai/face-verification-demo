#!/bin/bash

. ./shell/path.sh

if [ ! -z $CUDA_VISIBLE_DEVICES_FILE ]; then
while read line
do
export CUDA_VISIBLE_DEVICES="$line"
done < $CUDA_VISIBLE_DEVICES_FILE
fi

dataset="CIFAR10"
target="iid"
model="ResNet18"

# fl configuration
num_rounds=10
num_clients=10

# fit configuration
batch_size=128
local_epochs=1
weight_decay=0.0005
lr=0.01

seed=1234

exp_dir="./res/simulation/${dataset}/FedAvg_${model}/"${target}"/R_${num_rounds}_B_${batch_size}_E_${local_epochs}_lr_${lr}_S_${seed}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi

ray start --head --min-worker-port 20000 --max-worker-port 29999 --num-cpus 20 --num-gpus 10
sleep 1 

python ./local/simulation.py \
--num_rounds ${num_rounds} \
--num_clients ${num_clients} \
--dataset ${dataset} \
--target ${target} \
--model ${model} \
--local_epochs ${local_epochs} \
--batch_size ${batch_size} \
--lr ${lr} \
--weight_decay ${weight_decay} \
--seed ${seed} \
2>"${exp_dir}/logs/flower.log" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
ray stop
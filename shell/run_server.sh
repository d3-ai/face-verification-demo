#!/bin/bash

. ./shell/path.sh

args=()
for arg in $@; do
args+=($arg)
done
server_address=${args[1]}

dataset="CelebA"
target="small"
model="GNResNet18"
pretrained="CelebA"

# fl configuration
num_rounds=2
num_clients=3
strategy="FedAwS"

# fit configuration
batch_size=10
local_epochs=1
lr=0.05
momentum=0.9
weight_decay=5e-4

seed=1234

exp_dir="./exp/${dataset}/${strategy}_${model}/"${target}"/R_${num_rounds}_B_${batch_size}_E_${local_epochs}_lr_${lr}_S_${seed}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi

python ./local/server.py --server_address ${server_address} \
--strategy ${strategy} \
--num_rounds ${num_rounds} \
--num_clients ${num_clients} \
--dataset ${dataset} \
--target ${target} \
--model ${model} \
--local_epochs ${local_epochs} \
--batch_size ${batch_size} \
--lr ${lr} \
--momentum ${momentum} \
--weight_decay ${weight_decay} \
--seed ${seed} \
2>"${exp_dir}/logs/server_flower.log" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
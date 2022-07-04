#!/bin/bash

. ./shell/path.sh

args=()
for arg in $@; do
args+=($arg)
done
server_address=${args[1]}
cid=${args[3]}

dataset="CelebA"
target="Eyeglasses"
model="tiny_CNN"

# fl configuration
num_rounds=10
num_clients=5

# fit configuration
batch_size=10
local_epochs=1
lr=0.05

seed=1234

exp_dir="./exp/${dataset}/FedAvg_${model}/"${target}"/R_${num_rounds}_B_${batch_size}_E_${local_epochs}_lr_${lr}_S_${seed}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi


python ./local/client.py --server_address ${server_address} \
--cid ${cid} \
--dataset ${dataset} \
--target ${target} \
--model ${model} \
--seed ${seed} \
2>"${exp_dir}/logs/client${cid}_flower.log" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
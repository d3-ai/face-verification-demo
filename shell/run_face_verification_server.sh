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
criterion="ArcFace"
save_model=0

# fl configuration
strategy="FedAvg"
num_rounds=2
num_clients=3

# fit configuration
batch_size=10
local_epochs=1
lr=0.05
weight_decay=1e-4
scale=3.0
margin=0.01

seed=1234

exp_dir="./exp/${dataset}/${strategy}_${model}/"${target}"/R_${num_rounds}_B_${batch_size}_E_${local_epochs}_lr_${lr}_S_${seed}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi

python ./face_verification/server.py --server_address ${server_address} \
--strategy ${strategy} \
--num_rounds ${num_rounds} \
--num_clients ${num_clients} \
--dataset ${dataset} \
--target ${target} \
--model ${model} \
--pretrained ${pretrained} \
--local_epochs ${local_epochs} \
--batch_size ${batch_size} \
--lr ${lr} \
--weight_decay ${weight_decay} \
--save_model ${save_model} \
--seed ${seed} \
2>"${exp_dir}/logs/server_flower.log" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
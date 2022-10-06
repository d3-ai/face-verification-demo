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
save_model=0

# fl configuration
strategy="FedAvg"
num_rounds=2
num_clients=2

# fit configuration
batch_size=4
local_epochs=1
lr=0.005
weight_decay=1e-4
scale=32.0
margin=0.1

seed=1234

time=`date '+%Y%m%d%H%M'`
exp_dir="./exp/${dataset}/${strategy}_${model}/"${target}"/run_${time}"

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
--save_dir ${exp_dir} \
--seed ${seed} \
2>"${exp_dir}/logs/server_flower.log" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
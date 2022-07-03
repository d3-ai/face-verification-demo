#!/bin/bash

. ./path.sh

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

python ./local/server.py \
--num_rounds ${num_rounds} \
--dataset ${dataset} \
--target ${target} \
--model ${model} \
--local_epochs ${local_epochs} \
--batch_size ${batch_size} \
--seed ${seed} \
2>"${exp_dir}/logs/server_flower.log" &
sleep 5 # Sleep for 2s to give the server enough time to start

for i in $(seq 1 $num_clients); do
    echo "Starting client $i"
    python ./local/client.py --cid ${i} \
    --dataset ${dataset} \
    --target ${target} \
    --model ${model} \
    --seed ${seed} \
    2> "${exp_dir}/logs/client${i}_flower.log" &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
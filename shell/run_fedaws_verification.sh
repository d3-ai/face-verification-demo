#!/bin/bash

. ./shell/path.sh

if [ ! -z $CUDA_VISIBLE_DEVICES_FILE ]; then
while read line
do
export CUDA_VISIBLE_DEVICES="$line"
done < ${CUDA_VISIBLE_DEVICES_FILE}
fi

dataset="CelebA"
target="small"
model="GNResNet18"
pretrained="CelebA"
criterion="CCL"
save_model=1

# fl configuration
num_rounds=50
num_clients=10
fraction_fit=1

# fit configuration
batch_size=4
local_epochs=1
lr=0.005
weight_decay=0

seed=1234

exp_dir="./res/simulation/${dataset}/FedAwS_${model}/"${target}"/R_${num_rounds}_B_${batch_size}_E_${local_epochs}_lr_${lr}_S_${seed}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi

ray start --head --min-worker-port 20000 --max-worker-port 29999 --num-cpus 20 --num-gpus 10
sleep 1 

python ./face_verification/fedaws_verification.py \
--num_rounds ${num_rounds} \
--num_clients ${num_clients} \
--fraction_fit ${fraction_fit} \
--dataset ${dataset} \
--target ${target} \
--model ${model} \
--pretrained ${pretrained} \
--local_epochs ${local_epochs} \
--batch_size ${batch_size} \
--criterion ${criterion} \
--lr ${lr} \
--weight_decay ${weight_decay} \
--save_model ${save_model} \
--seed ${seed} \
2>"${exp_dir}/logs/flower.log" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
ray stop
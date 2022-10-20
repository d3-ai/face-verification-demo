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
criterion="ArcFace"
save_model=0

# fl configuration
strategy="FedAvg"
num_rounds=2
num_clients=10
fraction_fit=1

# fit configuration
batch_size=2
local_epochs=1
scale=32
margin=0.1
nu=0.9
lam=10.0
lr=0.005
weight_decay=0

seed=1234
time=`date '+%Y%m%d%H%M'`
exp_dir="./sim/${dataset}/${strategy}_${model}/"${target}"/run_${time}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi

ray start --head --min-worker-port 20000 --max-worker-port 29999 --num-cpus 20 --num-gpus 10
sleep 1 

python ./face_verification/simulation.py \
--strategy ${strategy} \
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
--scale ${scale} \
--margin ${margin} \
--nu ${nu} \
--lam ${lam} \
--lr ${lr} \
--weight_decay ${weight_decay} \
--save_model ${save_model} \
--save_dir ${exp_dir} \
--seed ${seed} \
2>"${exp_dir}/logs/flower.log" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
ray stop
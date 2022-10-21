#!/bin/bash

. ./shell/path.sh

args=()
for arg in $@; do
args+=($arg)
done
server_address=${args[1]}
cid=${args[3]}
dataset${args[5]}

# model config
dataset="CelebA"
target="small"
model="GNResNet18"

seed=1234

strategy="FedAwS"
time=`date '+%Y%m%d%H%M'`
exp_dir="./exp/${dataset}/${strategy}_${model}/${target}/run_${time}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
fi

python ./face_verification/client.py --server_address ${server_address} \
--strategy ${strategy} \
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
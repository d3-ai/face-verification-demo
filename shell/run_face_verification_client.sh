#!/bin/bash

. ./shell/path.sh

args=()
for arg in $@; do
args+=($arg)
done
server_address=${args[1]}
cid=${args[3]}

# model config
dataset="usbcam"
target="small"
model="GNResNet18"
pretrained="None"

seed=1234

python ./face_verification/client.py --server_address ${server_address} \
--cid ${cid} \
--dataset ${dataset} \
--target ${target} \
--model ${model} \
--pretrained ${pretrained} \
--seed ${seed} &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
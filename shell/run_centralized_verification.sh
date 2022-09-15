#!/bin/bash

. ./shell/path.sh

# dataset config
dataset="CelebA"
target="large"

# model config
model="ResNet18"
pretrained="IMAGENET1K_V1"
criterion="ArcFace"
save_model=1
seed=1234

# training configuration
max_epochs=300
batch_size=$1
scale=$2
margin=0.1
lr=$3
momentum=0.9
weight_decay=1e-4

python ./face_verification/centralized_verification.py \
--dataset ${dataset} \
--targe ${target} \
--model ${model} \
--pretrained ${pretrained} \
--max_epochs ${max_epochs} \
--batch_size ${batch_size} \
--criterion ${criterion} \
--margin ${margin} \
--scale ${scale} \
--lr ${lr} \
--momentum ${momentum} \
--weight_decay ${weight_decay} \
--save_model ${save_model} \
--seed ${seed} &
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait
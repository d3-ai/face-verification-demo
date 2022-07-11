#!/bin/bash

. ./shell/path.sh

dataset="CIFAR10"
model="tiny_CNN"
seed=1234

python ./local/centralized.py \
--dataset ${dataset} \
--model ${model} \
--seed ${seed}
#!/bin/bash
if [ ! -z $CUDA_VISIBLE_DEVICES_FILE ]; then
while read line
do
info=${line}
done < ${CUDA_VISIBLE_DEVICES_FILE}
fi
CUDA_VISIBLE_DEVICES_LIST=(${info//,/ })

for ((i=0; i < 9; i++)); do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]} . ./shell/run_fv_client.sh --server_address SERVER_IP:8080 --cid ${i} --dataset CelebA &
done
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[9]} . ./shell/run_fv_client.sh --server_address SERVER_IP:8080 --cid 9 --dataset usbcam &
wait
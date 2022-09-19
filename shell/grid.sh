#!/bin/bash
if [ ! -z $CUDA_VISIBLE_DEVICES_FILE ]; then
while read line
do
info=${line}
done < ${CUDA_VISIBLE_DEVICES_FILE}
fi
CUDA_VISIBLE_DEVICES_LIST=(${info//,/ })

for ((i=0; i < 9; i++)); do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]} . ./shell/run_face_verification_client.sh --server_address SERVER_IP:8080 --cid ${i} &
done
wait
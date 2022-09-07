#!/bin/bash
if [ ! -z $CUDA_VISIBLE_DEVICES_FILE ]; then
while read line
do
info=${line}
done < ${CUDA_VISIBLE_DEVICES_FILE}
fi
CUDA_VISIBLE_DEVICES_LIST=(${info//,/ })

lr=(0.1 0.05 0.1 0.05 0.1 0.05 0.1 0.05 0.1 0.05)
scale=(11.0 11.0 11.0 11.0 11.0 6.5 6.5 6.5 6.5 6.5)
margin=(0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5)

for ((i=0; i < 10; i++)); do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]} . ./shell/run_centralized_verification.sh ${scale[$i]} ${margin[$i]} ${lr[$i]} 1> /dev/null &
done
# . ./shell/run_centralized_verification.sh 11.0 0.005 0.01 &
# pid1=$!
# wait $pid1
# . ./shell/run_centralized_verification.sh 3.0 0.005 0.01 &
# pid1=$!
# wait $pid1
# . ./shell/run_centralized_verification.sh 3.0 0.01 0.01 &
# pid1=$!
# wait $pid1
# . ./shell/run_centralized_verification.sh 3.0 0.05 0.01 &
# pid1=$!
# wait $pid1
# . ./shell/run_centralized_verification.sh 3.0 0.005 0.005 &
# pid1=$!
# wait $pid1
# . ./shell/run_centralized_verification.sh 3.0 0.01 0.005 &
# pid1=$!
# wait $pid1
# . ./shell/run_centralized_verification.sh 3.0 0.05 0.005 &
# pid1=$!
# wait $pid1
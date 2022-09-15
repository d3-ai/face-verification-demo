#!/bin/bash
if [ ! -z $CUDA_VISIBLE_DEVICES_FILE ]; then
while read line
do
info=${line}
done < ${CUDA_VISIBLE_DEVICES_FILE}
fi
CUDA_VISIBLE_DEVICES_LIST=(${info//,/ })

lrs=(0.01 0.01 0.01 0.01 0.05 0.05 0.05 0.05)
scales=(11.0 11.0 11.0 11.0 11.0 11.0 11.0 11.0)

batch=(128 64 32 16 128 64 32 16)
for ((i=0; i < 8; i++)); do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]} . ./shell/run_centralized_verification.sh ${batch[$i]} ${scales[$i]} ${lrs[$i]} 1> /dev/null &
done
wait

# batch=(16 16 16 16 16 16 16 16 16 16)
# for ((i=0; i < 10; i++)); do
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]} . ./shell/run_centralized_verification.sh ${batch[$i]} ${scales[$i]} ${lrs[$i]} 1> /dev/null &
# done
# wait

# batch=(8 8 8 8 8 8 8 8 8 8)
# for ((i=0; i < 10; i++)); do
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]} . ./shell/run_centralized_verification.sh ${batch[$i]} ${scalse[$i]} ${lrs[$i]} 1> /dev/null &
# done
# wait

# batch=(4 4 4 4 4 4 4 4 4 4)
# for ((i=0; i < 10; i++)); do
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]} . ./shell/run_centralized_verification.sh ${batch[$i]} ${scales[$i]} ${lrs[$i]} 1> /dev/null &
# done

# batch_sizes=(2 4)
# lrs=(0.1 0.05)
# scales=(11.0 6.5)
# margins=(0.05 0.01)

# for b in "${batch_sizes[@]}"; do
# for l in "${lrs[@]}"; do
# for s in "${scales[@]}"; do
# for m in "${margins[@]}"; do
# echo $b $l $s $m
# . ./shell/run_federated_verification.sh $b $l $s $m & wait
# done
# done
# done
# done

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
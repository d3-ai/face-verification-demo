#!/usr/bin/bash
base=20

for ((i=0; i < 10; i++)); do
    # echo (($base + $i))
    mkdir -p "./tmp/pi${i}"
    ip=$((base + i))
    echo "192.168.199.${ip}"
    sudo mount -t nfs "192.168.199.${ip}":/home/yamasaki/PythonProject/project/tmp/ /home/sakiyama/PythonProject/project/tmp/pi${i}
done

# Flower_mockup
## Dataset
Extract `img_align_celeba.zip` to `./data/celeba/raw/`
```=bash
apt-get install -y zip
mkdir -p ./data/celeba/raw
unzip ./img_align_celeba.zip ./data/celeba/raw
```
Server
```
. ./run_server.sh
```
Client
```
$ docker build -t flower_mockup:latest ./
$ docker run -it \
  --rm \
  --name hoge \
  --mount type=bind,source="$(pwd)"/data/celeba/raw,target=/project/data/celeba/raw,readonly \
  flower_mockup:latest
```

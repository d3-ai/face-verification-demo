# Flower_mockup
## Dataset
Extract `img_align_celeba.zip` to `./data/celeba/raw/`
```=bash
apt-get install -y zip
mkdir -p ./data/celeba/raw
unzip ./img_align_celeba.zip ./data/celeba/raw
```
## Server
```=bash
$ docker build -t {image_name}:latest ./
$ docker run -it \
  -p 8080:8080 \
  --name test \
  --mount type=bind,source="$(pwd)"/shell/,target=/project/shell/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/local/,target=/project/local/ \
  --mount type=bind,source="$(pwd)"/src/,target=/project/src/,readonly \
  flower_mockup:latest
root@{containerid}:/project# pipenv update
root@{containerid}:/project# pipenv shell
(project)root@{containerid}:/project# . ./shell/run_server.sh --server_address "CONTAINER_IPADDRESS:8080"
```
When GPU is availabe
```=bash
$ docker build -t {image_name}:latest ./
$ docker run \
  --gpus=all \
  --name test \
  --mount type=bind,source="$(pwd)"/shell/,target=/project/shell/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/local/,target=/project/local/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/project/src/,readonly \
  flower_mockup:latest
root@{containerid}:/project# pipenv update
root@{containerid}:/project# pipenv shell
(project)root@{containerid}:/project# . ./shell/run_server.sh --server_address "CONTAINER_IPADDRESS:8080"
```
## Client
```=bash
$ docker run -it \
  --rm \
  --name test \
  --mount type=bind,source="$(pwd)"/shell/,target=/project/shell/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/local/,target=/project/local/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/project/src/,readonly \
  flower_mockup_pi4:latest
root@{containerid}:/project# pipenv update
root@{containerid}:/project# pipenv shell
(project)root@{containerid}:/project# . ./run_client.sh --server_address "SERVER_IPADDRESS:8080" --cid {client_id}
```

# Flower_mockup
## Docker image
How to build Docker image for rapberry pi4 (Ubuntu20.04 LTS)

Edit a json Docker daemon configuration file
```=josn
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": true, // change here if false
  "features": {
    "buildkit": true
  }
}
```
Build image for `linux/arm/v8`
```=bash
$ docker buildx build --platform linux/arm/v8 --tag {image_name}:latest'
$ docker save -o {image_name}.tar {image_name}
```
Install `{image_name}.tar` to Raspberry Pi
```=bash
$ docker load -i {image_name}.tar
```
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
  --rm \
  -p 8080:8080 \
  --name {container_name} \
  --mount type=bind,source="$(pwd)"/Pipfile,target=/project/,readonly \
  --mount type=bind,source="$(pwd)"/*.sh,target=/project/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/local/,target=/project/local/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/project/src/,readonly \
  {image_name}:latest
root@{containerid}:/project# pipenv update
root@{containerid}:/project# pipenv shell
(project)root@{containerid}:/project# . ./run_server.sh --server_address "SERVER_IP:8080"
```
## Client
```=bash
$ docker run -it \
  --rm \
  --name {container_name} \
  --mount type=bind,source="$(pwd)"/Pipfile,target=/project/,readonly \
  --mount type=bind,source="$(pwd)"/*.sh,target=/project/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/local/,target=/project/local/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/project/src/,readonly \
  {image_name}:latest
root@{containerid}:/project# pipenv update
root@{containerid}:/project# pipenv shell
(project)root@{containerid}:/project# . ./run_client.sh --server_address "SERVER_IP:8080" --cid {client_id}
```

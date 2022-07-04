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
$ docker buildx build --platform linux/arm64/v8 --tag {image_name}:latest --load ./Dockerfile
$ docker save -o {image_name}.tar {image_name}:latest
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
  --name test \
  --mount type=,source="$(pwd)"/shell/,target=/project/shell/,readonly \
  --mount type=bind,source="$(pwd)"/shell/,target=/project/shell/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/local/,target=/project/local/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/project/src/,readonly \
  flower_mockup:latest
root@{containerid}:/project# pipenv update
root@{containerid}:/project# pipenv shell
(project)root@{containerid}:/project# . ./shell/run_server.sh --server_address "SERVER_IPADDRESS:8080"
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

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
Server
```
$ docker build -t flower_mockup:latest ./
$ docker run -it \
  --rm \
  -p 8080:8080 \
  --name hoge \
  --mount type=bind,source="$(pwd)"/data/,target=/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/local/,target=/project/local/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/project/src/,readonly \
  flower_mockup:latest
root@{containerid}:/project# pipenv shell
(project)root@{containerid}:/project# . ./run_server.sh "SERVER_IP:8080"
```

# Face Verification Example (PyTorch)
## Setup
Start by cloning the example project.
```bash
$ git clone https://github.com/d3-ai/Flower_mockup.git ./project && cd ./project && rm -rf ./local
```
Build docker image compatible with your environment.
```bash
$ docker build --tag flower_mockup:latest ./baseimages/gpu_machine/ # gpu is available
$ docker build --tag flower_mockup:latest ./baseimages/cpu_machine/ 
$ docker build --tag flower_mockup:latest ./baseimages/pi/ # Raspberry Pi 4B 
```
If 
## Python requirements
Project deoendencies are defined in `Pipfile`.
At least, you should statisfy the following requirements.
```bash
Python 3.8.0
torch == 1.12.0
torchvision == 0.13.0
flwr == 1.0.0
scikit-image == *
opencv-python==4.6.0.66
facenet-pytorch
tqdm == *
matplotlib == *
```
Optional
```bash
ray
ray[tune]
wandb
optuna
pandas
```
## Preprocessing
Crop and transform face images using the detected five landmarks.
Face landmarks detection is performed by MTCNN from [face-pytorch](https://github.com/timesler/facenet-pytorch).
Put the raw images at `./data/{dataset_name}/img_align_{dataset_name}` and simply run as follows:
```bash
python ./face_verification/preprocess_face.py --dataset {DATASET_NAME}
```
When you face OpenBLAS warning, you should set `OMP_NUM_THREADS` as follows:
```bash
export OMP_NUM_THREADS=1
python ./face_verification/preprocess_face.py --dataset {DATASET_NAME}
```
# Run Federated Learing
The included `./shell/run_face_verification_server.sh` will start Flower server (using `./face_verification/server.py`). After the server is up, and then you should start 10 clients (using `./face_verification_client.sh`).

## Server
In `./shell/run_face_verification_server.sh`, you can specify server strategy FedAvg or [FedAwS](http://proceedings.mlr.press/v119/yu20f/yu20f.pdf).
First, you should run container as follows:
```bash
$ docker build -t {image_name}:latest ./
$ docker run -it \
  -p 8080:8080 \
  -u ${UID}:${GID}
  --name test \
  --mount type=bind,source="$(pwd)"/shell/,target=/home/"${USER}"/project/shell/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/home/"${USER}"/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/models/,target=/home/"${USER}"/project/models/,readonly \
  --mount type=bind,source="$(pwd)"/face_verification/,target=/home/"${USER}"/project/face_verification/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/home/"${USER}"/project/src/,readonly \
  flower_mockup:latest
```
In container, you can simply run
```bash
$ pipenv shell
(project) . ./shell/run_server.sh --server_address "CONTAINER_IPADDRESS:8080"
```

## Client
First, you should run container as follows:
```bash
$ docker build -t flower_mockup:latest ./baseimages/{*}/
$ docker run -it --rm\
  -u ${UID}:${GID} \
  --name test \
  --mount type=bind,source="$(pwd)"/shell/,target=/home/"${USER}"/project/shell/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/home/"${USER}"/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/local/,target=/home/"${USER}"/project/local/,readonly \
  --mount type=bind,source="$(pwd)"/models/,target=/home/"${USER}"/project/models/,readonly \
  --mount type=bind,source="$(pwd)"/face_verification/,target=/home/"${USER}"/project/face_verification/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/home/"${USER}"/project/src/,readonly \
  flower_mockup_pi4:latest
```
If you want to run container via VNC server 
```bash
$ docker build -t flower_mockup:latest ./baseimages/{*}/
$ xhost +
$ docker run -it --rm\
  -u ${UID}:${GID} \
  --name test \
  --mount type=bind,source="/tmp/.X11-unix",target="/tmp/.X11-unix",readonly\
  --mount type=bind,source="$(pwd)"/shell/,target=/home/"${USER}"/project/shell/,readonly \
  --mount type=bind,source="$(pwd)"/data/,target=/home/"${USER}"/project/data/,readonly \
  --mount type=bind,source="$(pwd)"/local/,target=/home/"${USER}"/project/local/,readonly \
  --mount type=bind,source="$(pwd)"/models/,target=/home/"${USER}"/project/models/,readonly \
  --mount type=bind,source="$(pwd)"/face_verification/,target=/home/"${USER}"/project/face_verification/,readonly \
  --mount type=bind,source="$(pwd)"/src/,target=/home/"${USER}"/project/src/,readonly \
  --env DISPLAY="$DISPLAY" \
  flower_mockup_pi4:latest
```

In container, you can simply run
```bash
$ pipenv shell
(project) . ./shell/run_client.sh --server_address "SERVER_IPADDRESS:8080" --cid "CLIENT_ID"
```
# Run Federated Simulation
The included `./shell/run_fedavg_verification.sh` and `./shell/run_fedaws_verification.sh` will start Flower simulation using Ray.
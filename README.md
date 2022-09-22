# Face Verification Tutorial
In this project, a face verification model is trained on CelebA and wild face images (e.g. taken with a USB camera) using [Flower](https://github.com/adap/flower) and PyTorch. Face verification can determine whether people are really who they say they are based on the obtained Cosine similarity. 

FL system comprises one server and multiple clients (ten clients is the default in this example). Each client has face images corresponding to one identity. There is no overlapping among the clients.

The face verification model comprises a backbone network and a followed classification matrix. The goal of FL is to train a global model without focusing on data collection due to privacy concerns.
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
For the details of how to build image (ex, usage of `buildx`) refer to [here](https://github.com/d3-ai/Flower_mockup/tree/main/baseimages).
## Python requirements
Project deoendencies are described in `Pipfile`.
At least, you should statisfy the following requirements to run the tutorial example.
> Note: Regarding to `torch` and `torchvision`, you should specify the version compatible with your environment from [here](https://download.pytorch.org/whl/torch_stable.html).
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
Put the raw images at `./data/{dataset_name}/img_align_{dataset_name}` and you can simply run as follows:
```bash
python ./face_verification/preprocess_face.py --dataset {DATASET_NAME}
```
When you face OpenBLAS warning, you should set `OMP_NUM_THREADS` as follows:
```bash
export OMP_NUM_THREADS=1
python ./face_verification/preprocess_face.py --dataset {DATASET_NAME}
```
# Run Federated Learning
The included `./shell/run_face_verification_server.sh` will start Flower server (using `./face_verification/server.py`). After the server is up, and then you should start 10 clients (using `./face_verification_client.sh`).

## Server
In `./shell/run_face_verification_server.sh`, you can modify the configuration of Federated Learning as follows:
|  Item  |  Default  | Options | Description |
| --- | :---: | :---: | :--- |
|  dataset  | `CelebA` | N/A | The dataset for training |
|  target  |  `small`  | `small`, `medium` | The size of FL system, `small` represents less than 10 clients and `medium` less than 100. |
|  model  |  `GNResNet18`  | `GNResNet18`, `ResNet18` | The model architecture. `GNResNet18` represents ResNet18 including GroupNorm instead of BatchNorm. |
|  pretrained  |  `CelebA`  | `CelebA`, `None` | The argument determine pretrained weights. `CelebA` represents wights of the corresponding model derived from centralized training using CelebA dataset partitioning available at `./data/celeba/identities/large/train_data.json`. Threre is no overlap between `./data/celeba/identities/small/train_data.json` and `./data/celeba/identities/large/train_data.json`. |
|  criterion  |  `ArcFace`  | `ArcFace`, `CCL`, `CrossEntropy` | The loss function for client training. `ArcFace` and `CrossEntropy` can be used with `FedAvg`, `CCL` with `FesAwS`. |
|  strategy  |  `FedAvg`  | `FedAvg`, `FedAwS` | The server aggregation strategy. `FedAwS` is proposed in [here](http://proceedings.mlr.press/v119/yu20f/yu20f.pdf). |
| num_rounds | 10 | int | The number of aggregation rounds that server performs. |
| num_clients | 10 | int | The number of clients in FL system. |
| batch_size | 2 | up to 24 | The batch size for clients training. |
| local_epochs | 1 | int | The number of training epochs for clients training. |
| lr | 0.005 | float | The learning rate for clients training. |
| weight_decay | 0.0001 | float | The weight decay coefficient for clients training. |


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
In the container, you can simply run
```bash
$ pipenv shell
(project) . ./shell/run_server.sh --server_address "CONTAINER_IPADDRESS:8080"
```

## Client
In `./shell/run_face_verification_client.sh`, you should specify the configuration of Federated Learning compatible with one defined in `./shell/run_face_verification_client.sh`. The following configuration is only used for initialization. 
|  Item  |  Default  | Options | Description |
| --- | :---: | :---: | :--- |
|  dataset  | `CelebA` | `CelebA`, `usbcam` | The dataset for training. |
|  target  |  `small`  | `small`, `medium` | The size of FL system, `small` represents less than 10 clients and `medium` less than 100. |
|  model  |  `GNResNet18`  | `GNResNet18`, `ResNet18` | The model architecture. `GNResNet18` represents ResNet18 including GroupNorm instead of BatchNorm. |
|  pretrained  |  `None`  | N/A | The argument determine pretrained weights. |

Note: Client model weights will be derived from the model that server distributes at each round. Therefore, client should not have to specify pretrained weights. 

Note: When you specify `FedAwS` as the server strategy, you should modify line 44 in `./src/client_app/face_client.py` as follows:
```python
self.net: Net = load_arcface_model(name=self.model, input_spec=dataset_config['input_spec'], out_dims=1, pretrained=self.pretrained)
```

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
If you want to run container on raspberry pi 4B via VNC server 
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

In the container, you can simply run
```bash
$ pipenv shell
(project) . ./shell/run_client.sh --server_address "SERVER_IPADDRESS:8080" --cid "CLIENT_ID"
```
Each client should specify a unique `CLIENT_ID` over the whole FL system.
# Run Simulation of Federated Learning
The included `./shell/run_fedavg_verification.sh` and `./shell/run_fedaws_verification.sh` will start Flower simulation utilizing [virtual client engine](https://flower.dev/docs/tutorial/Flower-1-Intro-to-FL-PyTorch.html#Using-the-Virtual-Client-Engine). 

After running container, you can simply run
```bash
$ pipenv shell
(project) . ./shell/run_fedavg_verification.sh
```
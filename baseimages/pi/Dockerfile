FROM ubuntu:20.04

RUN apt-get clean && apt-get update && apt-get install -y locales

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en_US
ENV LC_ALL en_US.UTF-8

RUN apt-get clean && apt-get update && apt-get install -y iputils-ping \
net-tools \
dnsutils \
tmux \
vim

LABEL maintainer="Y. Yamasaki <yamasaki@hal.ipc.i.u-tokyo.ac.jp>"

SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y tzdata 
ENV TZ=Asia/Tokyo
RUN apt-get update && apt-get install -y git \
make \
build-essential \
libssl-dev \
zlib1g-dev \
libbz2-dev \
libreadline-dev \
libsqlite3-dev \
wget \
curl \
llvm \
libncursesw5-dev \
xz-utils \
tk-dev \
libxml2-dev \
libxmlsec1-dev \
libffi-dev \
liblzma-dev \
python3-distutils \
python3-tk \
libpng-dev \
libjpeg-dev

ARG USER
ENV USER=${USER}
RUN useradd -m ${USER}
USER ${USER}
ENV HOME=/home/${USER}
WORKDIR ${HOME}/project

RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv

ENV PYENV_ROOT ${HOME}/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
ENV PATH $PYENV_ROOT/shims:$PATH

RUN eval "$(pyenv init -)" && pyenv install 3.8.0 && pyenv global 3.8.0
RUN pip install pipenv

RUN pipenv install --python 3.8.0 
RUN pipenv install http://download.pytorch.org/whl/cpu/torch-1.12.0-cp38-cp38-manylinux2014_aarch64.whl --skip-lock && \
pipenv install http://download.pytorch.org/whl/cpu/torchvision-0.13.0-cp38-cp38-manylinux2014_aarch64.whl --skip-lock && \
pipenv install flwr==1.0.0 --skip-lock && \
pipenv install tqdm --skip-lock && \
pipenv install matplotlib --skip-lock && \
pipenv install scikit-image && \
pipenv install opencv-python==4.6.0.66 --skip-lock && \
pipenv install facenet-pytorch && \
pipenv lock
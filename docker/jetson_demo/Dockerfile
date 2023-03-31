FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3

RUN apt-get clean && apt-get update && apt-get install -y locales

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en_US
ENV LC_ALL en_US.UTF-8

RUN apt-get clean && apt-get update && apt-get install -y git \
    iputils-ping \
    net-tools \
    iperf3 \
    dnsutils \
    tmux \
    python3-tk \
    libpng-dev \
    libjpeg-dev

LABEL maintainer="Y. Yamasaki <yamasaki@hal.ipc.i.u-tokyo.ac.jp>"

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y tzdata
ENV TZ=Asia/Tokyo

ARG USER
ENV USER=${USER}
RUN groupadd -f -r gpio
RUN useradd -m ${USER}
RUN gpasswd -a ${USER} video
RUN usermod -a -G gpio ${USER}

RUN pip3 install Jetson.GPIO
RUN cp /usr/local/lib/python3.8/dist-packages/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/
# RUN udevadm control --reload-rules && udevadm trigger

USER ${USER}
ENV HOME=/home/${USER}
WORKDIR ${HOME}/project
ENTRYPOINT [ "./shell/entrypoint.sh" ]

RUN pip3 install matplotlib==3.5
RUN pip3 install scipy==1.8
RUN pip3 install scikit-image
RUN pip3 install facenet-pytorch
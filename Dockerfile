FROM --platform=linux/x86_64 ubuntu:20.04

RUN apt-get clean && apt-get update && apt-get install -y locales

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en_US
ENV LC_ALL en_US.UTF-8

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
python3-distutils

# RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv

# RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
# echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
# echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
# exec "$SHELL"

# ENV PYENV_ROOT /root/.pyenv
# ENV PATH $PYENV_ROOT/bin:$PATH
# ENV PATH $PYENV_ROOT/shims:$PATH

# RUN pyenv install 3.8.0
# RUN pyenv global 3.8.0
# RUN pip install --upgrade pip 
# RUN pip3 install pipenv

# WORKDIR /project
# COPY ./Pipfile /project/
# COPY ./Pipfile.lock /project/
# RUN pipenv install

# COPY ./local/ /project/local
# COPY ./src/ /project/src
# COPY ./path.sh /project/
# COPY ./run.sh /project/
# RUN . ./path.sh
# RUN pipenv run python ./src/dataset_app/celeba_json.py
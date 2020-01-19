FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y \
  software-properties-common \
  && \
  rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update
RUN apt-get install -y \
  git \
  python3.7-dev \
  python3-pip \
  sudo \
  && rm -rf /var/lib/apt/lists/*

# Add user with valid passwrd
RUN useradd -ms /bin/bash user
RUN (echo user ; echo user) | passwd user

# Configure sudo
RUN usermod -a -G sudo user

# Install necessary python libraries
RUN python3.7 -m pip install pip --upgrade
RUN python3.7 -m pip install torch===1.4.0
RUN python3.7 -m pip install torchvision===0.5.0
RUN python3.7 -m pip install memcnn

# Set MemCNN config file for user environement
RUN python3.7 -c "import os, shutil, memcnn; path=os.path.join(os.path.dirname(memcnn.__file__), 'config'); shutil.copy(os.path.join(path, 'config.json.example'), os.path.join(path, 'config.json'));"

# Change user and prepare user data folders
USER user
WORKDIR /home/user
RUN mkdir data
RUN mkdir experiments

ENTRYPOINT /bin/bash

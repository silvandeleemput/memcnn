FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y \
  ca-certificates \
  libjasper-runtime \
  openssh-server \
  git \
  wget \
  libopenblas-dev \
  python-dev \
  python-pip \
  python-nose \
  python-numpy \
  python-scipy \
  python-skimage \
  python-sklearn \
  python-yaml \
  gcc \
  g++ \
  make \
  gfortran \
  libatlas-base-dev \
  libblas-dev \
  liblapack-dev \
  build-essential \
  curl \
  libfreetype6 \
  libfreetype6-dev \
  libjpeg62-dev \
  libjpeg8 \
  libpng12-dev \
  libzmq3-dev \
  pkg-config \
  software-properties-common \
  unzip \
  zip \
  sudo \
  && \
  rm -rf /var/lib/apt/lists/*

# install latest cmake 3.*
RUN add-apt-repository ppa:george-edison55/cmake-3.x
RUN apt-get update && apt-get install -y cmake && \
  rm -rf /var/lib/apt/lists/*

# Setup ssh
RUN mkdir /var/run/sshd
RUN service ssh stop

# Add user with valid passwrd
RUN useradd -ms /bin/bash user
RUN (echo user ; echo user) | passwd user

# Configure sudo
RUN usermod -a -G sudo user

RUN pip install --upgrade pip
RUN pip install --upgrade Cython
RUN pip install --upgrade tqdm
RUN pip install --upgrade setuptools
RUN pip install --upgrade six
RUN pip install --upgrade h5py
RUN pip install nibabel
RUN pip install SimpleITK
RUN pip install --upgrade pillow

# install PyTorch
RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl
RUN pip install torchvision
RUN pip install git+https://github.com/lanpa/tensorboard-pytorch

# Clone into memcnn library & setup
WORKDIR /home/user
RUN mkdir data
RUN mkdir experiments
RUN git clone https://github.com/silvandeleemput/memcnn.git
RUN mv /home/user/memcnn/memcnn/config/config.json.example /home/user/memcnn/memcnn/config/config.json

ENV PYTHONPATH $PYTHONPATH:/home/user/memcnn:/home/user/memcnn/memcnn

WORKDIR /home/user/memcnn/memcnn
ENTRYPOINT /bin/bash

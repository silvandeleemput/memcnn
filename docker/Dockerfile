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

RUN pip install -Iv pip==18.1
RUN pip install -Iv Cython==0.27.3
RUN pip install -Iv tqdm==4.19.5
RUN pip install -Iv setuptools==38.5.1
RUN pip install -Iv six==1.11.0
RUN pip install -Iv h5py==2.7.1
RUN pip install -Iv nibabel==2.2.1
RUN pip install -Iv SimpleITK==1.0.1
RUN pip install -Iv pillow==5.0.0
RUN pip install -Iv tensorflow==1.11.0

# install PyTorch
RUN pip install torch==0.4.0
RUN pip install torchvision==0.2.1
RUN pip install tensorboardX==1.4

# Clone into memcnn library & setup
WORKDIR /home/user
RUN mkdir data
RUN mkdir experiments
RUN git clone https://github.com/silvandeleemput/memcnn.git
RUN cd memcnn && git reset --hard 01730d35f8b3ecc47b6a1f9a2c037f6858080e82
RUN mv /home/user/memcnn/memcnn/config/config.json.example /home/user/memcnn/memcnn/config/config.json

ENV PYTHONPATH $PYTHONPATH:/home/user/memcnn:/home/user/memcnn/memcnn

WORKDIR /home/user/memcnn/memcnn
ENTRYPOINT /bin/bash

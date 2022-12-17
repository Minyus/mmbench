FROM continuumio/miniconda3:4.9.2
# Debian GNU/Linux 10 (buster)
# Python 3.8.5

RUN apt-get --allow-releaseinfo-change update \
    && apt-get -y dist-upgrade \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    bash-completion \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    graphviz \
    htop \
    jq \
    libgraphviz-dev \
    ncdu \
    net-tools \
    openssh-server \
    sudo \
    tar \
    tmux \
    tree \
    unzip \
    vim \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN true \
    && conda install \
    pytorch=1.10.0 \
    torchvision=0.11.1 \
    -c pytorch -c nvidia -c conda-forge -y \
    && conda clean -ya

RUN pip --no-cache-dir install \
    opencv-python-headless \
&& true

RUN pip --no-cache-dir install \
    mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10/index.html \
&& true

RUN pip --no-cache-dir install \
    mmcls==0.25.0 \
&& true

RUN pip --no-cache-dir install \
    onnx \
    onnxruntime \
    onnx-simplifier \
    timm \
    future \
    tensorboard \
    pydot \
    torchinfo \
    black \
    isort \
    flake8 \
    pandas \
    scikit-learn \
    plotly \
    flatten-dict \
    nested-lookup \
    pipdeptree \
    && true

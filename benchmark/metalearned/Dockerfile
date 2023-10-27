FROM nvidia/cuda:10.0-cudnn7-devel

# Install system packages
RUN apt-get update  -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    unrar \
    libfreetype6-dev \
    git && \
    apt-get clean -y

# Install Python
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update  -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    python3.7 \
    python3.7-dev \
    python3-pip && \
    apt-get clean -y

RUN python3.7 -m pip install --upgrade pip setuptools wheel

# Enable GPU access from ssh login into the Docker container
RUN echo "ldconfig" >> /etc/profile


#
# Jupyterlab
#
#
#RUN APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn curl -sL https://deb.nodesource.com/setup_12.x | bash - && \
#    apt install -y nodejs && \
#    apt-get clean -y && \
#    python3.7 -m pip install 'jupyterlab' nbdime jupytext ipympl jupyterlab-git ipywidgets && \
#    jupyter labextension install --no-build @jupyter-widgets/jupyterlab-manager \
#    jupyter-matplotlib \
#    @jupyterlab/toc \
#    jupyterlab_bokeh \
#    nbdime-jupyterlab && \
#    jupyter lab build && \
#    mkdir -p /home/jupyter/.jupyter/lab/user-settings/@jupyterlab/extensionmanager-extension && \
#    echo '{"enabled": true}' > /home/jupyter/.jupyter/lab/user-settings/@jupyterlab/extensionmanager-extension/plugin.jupyterlab-settings && \
#    chmod -R a=rwx /home/jupyter

#
# Tensorboard (assuming tensorboard in installed from project dependencies)
#

RUN echo '#!/usr/bin/env bash\nnohup /usr/bin/python /usr/local/bin/tensorboard \
--port $1 --logdir $2 &>/tensorboard/tensorboard.log &\nsleep 5\n' > /start_tensorboard.sh
RUN mkdir /tensorboard && chmod 777 /tensorboard

RUN chmod +x /start_tensorboard.sh

#
# Project dependencies
#
COPY requirements.txt .
RUN python3.7 -m pip install -r requirements.txt
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.7 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1


FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

#Versions
ARG python_version=3.6

# Update package list and install dependencies
RUN apt-get -qq update && \
    apt-get -y install sudo && \
    apt-get -qq install -y --no-install-recommends build-essential \
    ca-certificates libgtk2.0-0 sqlite3 wget git libhdf5-dev g++ graphviz unzip x11-apps sudo vim && \
    apt-get -qq clean
    
    
# Install conda (Usar Miniconda3 se python 3)
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh
    
    
# Set PATH and PYTHONPATH environmental variables 
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH    
ENV PYTHONPATH='/project/:$PYTHONPATH'

# Install Python libraries 
RUN conda update -n base conda &&\
    conda install --quiet --yes python=${python_version} &&\
    conda install tensorflow-gpu=1.12.0 keras=2.1.6 opencv=3.4.5 Pillow scikit-learn notebook pandas matplotlib mkl nose  pyyaml six h5py pygpu && \
    conda clean -yt   

RUN apt-get update && apt-get install sudo

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER docker
ENV HOME /home/docker

WORKDIR /home/docker/

RUN git clone https://github.com/gpspelle/personal
RUN mv personal/.vimrc ~/

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

# Install miniconda (necessary for python-pdal library...)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN mkdir -p ~/miniconda3 && \
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
	bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
	rm -rf ~/miniconda3/miniconda.sh
	
# Initialize conda env
RUN ~/miniconda3/bin/conda init bash && bash ~/.bashrc && . ~/.bashrc && \
	conda install -y -c conda-forge pdal python-pdal gdal && \
	conda install -y -c conda-forge gdal sqlite && \
	apt-get update && apt-get install -y --no-install-recommends build-essential && \
	rm -rf /var/lib/apt/lists/* && \
	python -m pip install --upgrade pip && \
 	pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 && \
	pip install python-dotenv hydra-core tqdm comet-ml lightning h5py pandas potpourri3d laspy matplotlib torch_geometric --upgrade && \
	pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html && \
	pip install pytorch_metric_learning
	
ENV conda ~/miniconda3/bin/conda
ENV bashrc /root/.bashrc
ENV LOGS_DIR=logs
# ENV OMPI_MCA_opal_cuda_support=true

WORKDIR /app


# Use the official NVIDIA CUDA base image
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Set the working directory in the container
WORKDIR /app

# Install Miniconda and Git
RUN apt-get update && apt-get install -y wget git && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init bash

# Add Conda to the PATH
ENV PATH=/opt/conda/bin:$PATH

# Clone the Git repository
RUN git clone https://github.com/MaximilianKr/eval_pipeline.git /app

# Copy the environment.yml file into the container
COPY dockerenv.yml /app/dockerenv.yml

# Create the Conda environment from environment.yml
RUN conda env create -f /app/dockerenv.yml

# Specify the command to start a shell
CMD ["/bin/bash"]

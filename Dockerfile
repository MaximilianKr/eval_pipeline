# Use official PyTorch + CUDA base image
# Python 3.10.14 pre-installed
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment variables for non-interactive apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    wget \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Clone the application repository
RUN git clone https://github.com/MaximilianKr/eval_pipeline.git /app

# Install necessary dependencies into the base Conda environment
RUN conda install -n base -c conda-forge pip \
    transformers accelerate tqdm pandas pillow requests urllib3 openai tenacity && \
    pip install minicons --no-deps && \
    conda clean -a -y

# Ensure the default shell is Bash and source the base environment
SHELL ["/bin/bash", "-c"]

# Expose port 80
EXPOSE 80

# Start a shell by default
CMD ["/bin/bash"]

# Test if container runs properly
# python -c "import torch; print(torch.cuda.is_available())"

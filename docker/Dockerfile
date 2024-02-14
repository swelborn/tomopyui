# Use an official NVIDIA CUDA  image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04   

# Install some basic utilities, Python, and necessary libraries for Conda
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev wget bzip2 git \
    && rm -rf /var/lib/apt/lists/*

# Determine platform (x86_64 vs. ARM) and fetch the appropriate Miniconda installer
RUN case $(uname -m) in \
    x86_64)     ARCH="x86_64";; \
    aarch64)    ARCH="aarch64";; \
    *)          echo "Unsupported architecture"; exit 1 ;; \
    esac && \
    wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh" -O miniconda.sh && \
    bash miniconda.sh -b -p /miniconda && \
    rm miniconda.sh


# Update PATH so that pip and conda are available
ENV PATH="/miniconda/bin:$PATH"

# Create a Conda environment from environment.yml. Assuming environment.yml is at the Docker build context root
ARG ENV_FILE=environment.yml
COPY ./${ENV_FILE} ./environment.yml
RUN conda env create -f environment.yml

# Activate the Conda environment. Any command run in this layer or subsequent layers will use the Conda environment by default
SHELL ["conda", "run", "-n", "tomopyui", "/bin/bash", "-c"]

# Set working directory
WORKDIR /workspace

# Expose the port JupyterLab will run on
EXPOSE 8888

# Command to run JupyterLab within the Conda environment
CMD ["conda", "run", "-n", "tomopyui", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

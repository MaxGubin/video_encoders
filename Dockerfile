ARG CUDA="12.8.0"
ARG TAG="devel"
ARG OS="ubuntu24.04"
FROM nvidia/cuda:${CUDA}-cudnn-${TAG}-${OS}
# FROM python:3.12

RUN apt-get update && \
    apt-get install -y \
        git \
        vim \
        htop \
        python3 \
        python3-pip \
	python3-venv && \
    rm -rf /var/lib/apt/lists/*

ARG USER_ID
ARG GROUP_ID
ARG NAME
#RUN groupadd --gid ${GROUP_ID} ${NAME}
#RUN useradd \
#    --no-log-init \
#    --create-home \
#    --uid ${USER_ID} \
#    --gid ${GROUP_ID} \
#    -s /bin/sh ${NAME}

ARG WORKDIR_PATH
WORKDIR ${WORKDIR_PATH}

ARG JAX_CUDA_CUDNN="cuda12"

# Enable virtual environment to avoid pip failure
RUN python3 -m venv venv


RUN . ${WORKDIR_PATH}/venv/bin/activate && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install "jax[$JAX_CUDA_CUDNN]" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    python3 -m pip install -r requirements.txt

CMD ["jupyter notebook"]

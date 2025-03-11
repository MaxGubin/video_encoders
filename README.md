# Introduction

"From scratch" implementations of image processing algorithms using Jax

## Goals

1. Learn about the algorithms (my background is NLP, learning the new area)
2. Refresh my knowledge of the latest changes in JAX
3. Have fun
4. This is not production quality implementations, the focus is on simplicity and algorithm understanding

## Reproducing

Everything was tried using Ubuntu 24.04  CUDA 12.7

Setting up and running

```bash
python3 -m venv .venv:wa
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Using Docker

```bash
docker run --runtime=nvidia -p 8888:8888 -it jax-cuda
```

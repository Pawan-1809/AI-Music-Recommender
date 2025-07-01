#!/bin/bash

export PORT=3000

export TRANSFORMERS_CACHE="./cache"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

uvicorn backend.main:app --host 0.0.0.0 --port $PORT


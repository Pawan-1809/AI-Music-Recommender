#!/bin/bash

# Set a fixed port so Render can detect it
export PORT=10000

# Optional: Limit memory usage to avoid hitting the 512MiB cap
export TRANSFORMERS_CACHE="./cache"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

uvicorn backend.main:app --host 0.0.0.0 --port 10000

#!/bin/bash
set -e

# Start FastAPI backend
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
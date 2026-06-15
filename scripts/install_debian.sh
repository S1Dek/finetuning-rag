#!/bin/bash

set -e

echo "=== SYSTEM UPDATE ==="
sudo apt update && sudo apt upgrade -y

echo "=== BASE TOOLS ==="
sudo apt install -y \
    git curl wget build-essential \
    python3 python3-pip python3-venv \
    htop nvtop tmux

echo "=== CUDA CHECK ==="
nvidia-smi

echo "=== PYTHON ENV ==="
python3 -m venv ai-env
source ai-env/bin/activate

pip install --upgrade pip

echo "=== INSTALL PYTORCH (CUDA 12.1) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== DONE ==="
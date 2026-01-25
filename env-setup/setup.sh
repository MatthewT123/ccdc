#!/usr/bin/env bash
set -euo pipefail

# Create env from yml
conda env create -f environment.yml

# Activate
conda activate MolFLAE2

# Ensure pip build uses the env
export PIP_NO_BUILD_ISOLATION=1
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.9.1+cpu torchvision==0.16.1+cpu torchaudio==2.0.2+cpu

# Install PyG compiled wheels 
  torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.9.1+cpu.html

# Install the rest from requirements file (including torch-geometric)
python -m pip install -r requirements-pip.txt

# Create activation script to set LD_LIBRARY_PATH
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh <<'EOF'
#!/bin/sh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo "Environment MolFLAE2 ready."

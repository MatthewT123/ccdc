#!/usr/bin/env bash
set -euo pipefail

# Create an env
conda create -n MolFLAE2 python=3.10 -y
conda activate MolFLAE2

# Step 1: Install OpenBLAS-based scientific stack (NO MKL)
conda install -c conda-forge \
  "blas=*=openblas" \
  numpy \
  scipy \
  matplotlib \
  pandas \
  pyyaml \
  -y

# Step 2: Install PyTorch via pip (uses OpenBLAS from conda)
pip install \
  torch==2.12.0 \
  torchvision==0.27.0 \
  torchaudio==2.12.0 \
  --index-url https://download.pytorch.org/whl/cpu

# Step 3: CRITICAL - Set library path to avoid segfault
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Step 4: Test PyTorch
python << 'EOF'
import torch
import numpy as np
print("✓ PyTorch:", torch.__version__)
print("✓ NumPy:", np.__version__)
x = torch.randn(3, 3)
print("✓ Tensor creation works!")
print("✓ Matrix mult:", torch.mm(x, x.t()).shape)
EOF

# Make LD_LIBRARY_PATH permanent
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/sh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF

# Deactivate and reactivate to test
conda deactivate
conda activate MolFLAE2

# Verify it still works
python -c "import torch; print('✓ Still working:', torch.__version__)"

# Install chemistry libraries
conda install -c conda-forge rdkit=2023.09.5 openbabel=3.1.1 -y

# Install PyG
# make sure pip build uses the active env
export PIP_NO_BUILD_ISOLATION=1
python -m pip install -U pip "setuptools<82" wheel

# install the compiled extension wheels for torch 2.9.1 (CPU)
python -m pip install --no-deps --no-build-isolation \
  torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.12.0+cpu.html

# then install the high-level package
python -m pip install torch-geometric

# Test PyG
python << 'EOF'
import torch
import torch_geometric
import torch_scatter
import torch_sparse

print("Torch:", torch.__version__)
print("PyG:", torch_geometric.__version__)
print("✓ PyG working")
EOF

# Install MolFLAE dependencies
pip install \
  pytorch-lightning \
  torchmetrics \
  flow-matching \
  torchdiffeq \
  torch-ema \
  datamol \
  selfies \
  biopython \
  biotite \
  wandb \
  loguru \
  tqdm \
  seaborn \
  scikit-learn

# Final verification
python << 'EOF'
import torch
import torch_geometric
import rdkit
import pytorch_lightning as pl
import numpy as np

print("=" * 60)
print("MOLFLAE ENVIRONMENT READY")
print("=" * 60)
print(f"✓ Python: 3.10")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ PyTorch Geometric: {torch_geometric.__version__}")
print(f"✓ RDKit: {rdkit.__version__}")
print(f"✓ PyTorch Lightning: {pl.__version__}")
print(f"✓ NumPy: {np.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print("=" * 60)

# Quick functional test
x = torch.randn(10, 10)
y = torch.mm(x, x.t())
print("✓ Computation test passed")
print("\nReady to run MolFLAE!")
EOF

# Additional installs 
conda install -y -c conda-forge absl-py fire ipykernel
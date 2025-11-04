#!/bin/bash

# Setup script for music-text embedding training

set -e  # Exit on error

echo "=========================================="
echo "Music-Text Embedding Training Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: CUDA not detected. Training will be slower on CPU."
fi

echo ""
echo "=========================================="
echo "Installing Dependencies"
echo "=========================================="

# Install requirements
pip install -r requirements.txt

# Install python_services dependencies
echo ""
echo "Installing python_services dependencies..."
cd ../python_services
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi
cd ../python_training

echo ""
echo "=========================================="
echo "Creating Directory Structure"
echo "=========================================="

# Create data directories
mkdir -p ../data/datasets
mkdir -p ../data/embeddings/muq
mkdir -p ../data/embeddings/mert
mkdir -p ../data/embeddings/music2latent

# Create checkpoint directories
mkdir -p checkpoints/muq
mkdir -p checkpoints/mert
mkdir -p checkpoints/music2latent

echo "Created directories:"
echo "  ../data/datasets         - For downloaded datasets"
echo "  ../data/embeddings       - For audio embeddings"
echo "  checkpoints              - For model checkpoints"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Download datasets:"
echo "   cd datasets"
echo "   python download_all.py --phase 1 --output-dir ../../data/datasets"
echo ""
echo "2. Generate audio embeddings:"
echo "   python preprocessing/audio_embedding_generator.py \\"
echo "       ../data/datasets/musiccaps \\"
echo "       ../data/embeddings/muq/musiccaps.h5 \\"
echo "       --model muq"
echo ""
echo "3. Train a model:"
echo "   python training/train.py --config configs/muq_config.json"
echo ""
echo "For more information, see README.md"
echo ""

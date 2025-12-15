#!/bin/bash
# setup_linux.sh - Complete setup for Linux + AMD 7900 XT
# ECM3401 Individual Project
# Run: bash setup_linux.sh

set -e  # Exit on error

echo "======================================================================"
echo "ECM3401 PROJECT SETUP - LINUX + AMD 7900 XT"
echo "======================================================================"

# Check system
echo ""
echo "System Information:"
echo "  OS: $(uname -s)"
echo "  Kernel: $(uname -r)"
echo "  CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "  RAM: $(free -h | awk '/^Mem:/ {print $2}')"

# Check for GPU
if command -v rocm-smi &> /dev/null; then
    echo "  GPU: $(rocm-smi --showproductname | grep 'GPU\[0\]' | cut -d: -f2 | xargs)"
else
    echo "  GPU: AMD GPU detected (rocm-smi not installed yet)"
fi

# Step 1: Install Python dependencies
echo ""
echo "======================================================================"
echo "Step 1/5: Installing Python dependencies"
echo "======================================================================"

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Python version: $PYTHON_VERSION"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Step 2: Install PyTorch with ROCm
echo ""
echo "======================================================================"
echo "Step 2/5: Installing PyTorch with ROCm (AMD GPU support)"
echo "======================================================================"
echo "This will download ~2GB..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Step 3: Install other requirements
echo ""
echo "======================================================================"
echo "Step 3/5: Installing other dependencies"
echo "======================================================================"

pip install -r requirements.txt

# Step 4: Verify GPU
echo ""
echo "======================================================================"
echo "Step 4/5: Verifying GPU setup"
echo "======================================================================"

python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: GPU not detected!")
EOF

# Step 5: Create .env file
echo ""
echo "======================================================================"
echo "Step 5/5: Environment setup"
echo "======================================================================"

if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << 'ENVFILE'
# OpenAI API Key (for GPT-4o-mini)
OPENAI_API_KEY=sk-proj-your-key-here

# HuggingFace Token (for Llama model download)
HUGGINGFACE_TOKEN=hf_your-token-here
ENVFILE
    echo "✓ Created .env file"
    echo "⚠️  IMPORTANT: Edit .env and add your API keys!"
else
    echo "✓ .env file already exists"
fi

# Set ROCm environment variables for RX 7900 XT
echo ""
echo "Setting ROCm environment variables..."
cat >> ~/.bashrc << 'BASHRC'

# ROCm settings for AMD RX 7900 XT
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
BASHRC

echo "✓ Added ROCm settings to ~/.bashrc"

# Final summary
echo ""
echo "======================================================================"
echo "✓ SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env file and add your API keys"
echo "  2. Copy dataset to ~/datasets/write-and-improve-corpus-2024-v2/"
echo "  3. Run: source venv/bin/activate"
echo "  4. Run: python scripts/01_explore_dataset.py"
echo ""
echo "To activate environment in future sessions:"
echo "  source venv/bin/activate"
echo ""

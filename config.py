"""
config.py - Configuration for ECM3401 Individual Project (LINUX VERSION)

Project: Measuring Semantic Robustness in LLM-Based Essay Scoring
Author: Sansiri Charoenpong (Siemon)
Supervisor: Dr. Rodrigo Souza Wilkens
Last updated: December 14, 2025
Hardware: Linux + AMD Radeon RX 7900 XT + Intel i5-12600K + 64GB RAM
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
load_dotenv(PROJECT_ROOT / '.env')

# =============================================================================
# PROJECT PATHS (LINUX)
# =============================================================================
# UPDATE THIS to match your Linux dataset location
DATASET_ROOT = Path.home() / "datasets" / "write-and-improve-corpus-2024-v2"
# Or use absolute path: Path('/home/your-username/datasets/write-and-improve-corpus-2024-v2')

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Prompts directory
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Dataset files
CORPUS_FILE = DATASET_ROOT / "whole-corpus" / "en-writeandimprove2024-corpus.tsv"
PROMPTS_FILE = DATASET_ROOT / "whole-corpus" / "en-writeandimprove2024-prompts-info"

# Create directories
for d in [DATA_DIR, PROCESSED_DIR, RESULTS_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR, PROMPTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Validate paths
if not CORPUS_FILE.exists():
    print(f"❌ ERROR: Corpus file not found at {CORPUS_FILE}")
    print(f"Expected location: {DATASET_ROOT}")
    print(f"Please copy dataset to this location or update DATASET_ROOT in config.py")
else:
    print(f"✓ Dataset found: {DATASET_ROOT}")

# =============================================================================
# GPU DETECTION (AUTO-DETECT CUDA/ROCm FOR AMD)
# =============================================================================
if torch.cuda.is_available():
    DEVICE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU detected: {GPU_NAME}")
    print(f"✓ VRAM: {GPU_MEMORY_GB:.1f} GB")
else:
    DEVICE = "cpu"
    print("⚠️ WARNING: No GPU detected! Llama will run on CPU (very slow)")

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================
RANDOM_SEED = 42
PHASE1_SAMPLE_SIZE = 100
ESSAYS_PER_LEVEL = 20
CEFR_LEVELS = ['A2', 'B1', 'B2', 'C1', 'C2']

# CEFR level mapping (combine + levels with base levels)
CEFR_MAPPING = {
    'A1': 'A2', 'A1+': 'A2',
    'A2': 'A2', 'A2+': 'A2',
    'B1': 'B1', 'B1+': 'B1',
    'B2': 'B2', 'B2+': 'B2',
    'C1': 'C1', 'C1+': 'C1',
    'C2': 'C2', 'C2+': 'C2'
}

def map_cefr_level(raw_level):
    """Convert raw CEFR label to research label"""
    return CEFR_MAPPING.get(raw_level, raw_level)

# =============================================================================
# MODEL CONFIGURATION - COMMERCIAL (OpenAI GPT-4o-mini)
# =============================================================================
GPT_MODEL = "gpt-4o-mini"
GPT_TEMPERATURE = 0.0      # Deterministic (CRITICAL for research)
GPT_MAX_TOKENS = 50

# API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not set!")
    print("Add to .env file: OPENAI_API_KEY=sk-proj-...")
else:
    print(f"✓ API key loaded from .env")

# Cost tracking (GPT-4o-mini pricing)
GPT_COST_PER_1M_INPUT = 0.150 / 1_000_000
GPT_COST_PER_1M_OUTPUT = 0.600 / 1_000_000

# =============================================================================
# MODEL CONFIGURATION - OPEN SOURCE (Llama on AMD 7900 XT)
# =============================================================================
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # or Llama-3.1-8B-Instruct
LLAMA_DEVICE = DEVICE  # Auto-detected: "cuda" for AMD GPU via ROCm
LLAMA_TEMPERATURE = 0.0  # Deterministic
LLAMA_MAX_TOKENS = 50

# GPU settings for AMD 7900 XT (24GB VRAM)
LLAMA_TORCH_DTYPE = torch.float16  # Use float16 for speed
LLAMA_USE_4BIT = True  # Enable 4-bit quantization (optional, saves VRAM)

# HuggingFace configuration
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
LLAMA_CACHE_DIR = PROJECT_ROOT / "models" / "llama_cache"
LLAMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PROMPTING STRATEGIES
# =============================================================================
N_PARAPHRASES = 3
PROMPTING_STRATEGIES = ['minimal', 'rubric_guided', 'chain_of_thought', 'few_shot']

# Few-shot methods
FEW_SHOT_METHODS = ['random', 'semantic', 'centroid', 'mmr']
FEW_SHOT_N_EXAMPLES = 3

# =============================================================================
# STATISTICAL SETTINGS
# =============================================================================
ALPHA = 0.05
ROBUSTNESS_THRESHOLD_GOOD = 3.0
ROBUSTNESS_THRESHOLD_MODERATE = 5.0

print("✓ Config loaded successfully")
"""
config.py - Configuration for ECM3401 Individual Project (MacBook M2 Pro)

Project: Measuring Semantic Robustness in LLM-Based Essay Scoring
Author: Sansiri Charoenpong (Siemon)
Supervisor: Dr. Rodrigo Souza Wilkens
Last updated: December 15, 2025
Hardware: MacBook M2 Pro + 16GB RAM
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
# PROJECT PATHS (MACOS)
# =============================================================================
DATASET_ROOT = Path('/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/write-and-improve-corpus-2024-v2')

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
else:
    print(f"✓ Dataset found: {DATASET_ROOT}")

# =============================================================================
# GPU DETECTION (Apple Silicon MPS)
# =============================================================================
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print(f"✓ Apple Silicon GPU (MPS) detected")
else:
    DEVICE = "cpu"
    print("⚠️ WARNING: MPS not available, using CPU")

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================
RANDOM_SEED = 42
PHASE1_SAMPLE_SIZE = 135
ESSAYS_PER_LEVEL = 27
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
    print(f"✓ OpenAI API key loaded")

# Cost tracking (GPT-4o-mini pricing)
GPT_COST_PER_1M_INPUT = 0.150 / 1_000_000
GPT_COST_PER_1M_OUTPUT = 0.600 / 1_000_000

# =============================================================================
# MODEL CONFIGURATION - OPEN SOURCE (Phi-3-Mini on M2 Pro)
# =============================================================================
LLAMA_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # 3.8B params, fast on M2 Pro!
LLAMA_DEVICE = DEVICE  # Auto-detected: "mps" for Apple Silicon
LLAMA_TEMPERATURE = 0.0  # Deterministic
LLAMA_MAX_TOKENS = 50

# M2 Pro settings
LLAMA_TORCH_DTYPE = torch.float16  # Use float16 for speed

# HuggingFace configuration
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
LLAMA_CACHE_DIR = PROJECT_ROOT / "models" / "llama_cache"
LLAMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PROMPTING STRATEGIES
# =============================================================================
N_PARAPHRASES = 3
PROMPTING_STRATEGIES = ['minimal', 'rubric', 'cot']

# =============================================================================
# STATISTICAL SETTINGS
# =============================================================================
ALPHA = 0.05
ROBUSTNESS_THRESHOLD_GOOD = 3.0
ROBUSTNESS_THRESHOLD_MODERATE = 5.0

print("✓ Config loaded successfully")
print(f"✓ Model: {LLAMA_MODEL}")
print(f"✓ Device: {DEVICE}")
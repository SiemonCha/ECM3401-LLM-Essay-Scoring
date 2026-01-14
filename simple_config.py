"""
Simple Configuration for ECM3401 Project
Everything you need in one place
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load API keys
load_dotenv()

# =============================================================================
# PATHS (Absolute - no confusion!)
# =============================================================================

# Your actual paths
DATASET_FILE = Path("/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/write-and-improve-corpus-2024-v2/whole-corpus/en-writeandimprove2024-corpus.tsv")
PROJECT_ROOT = Path("/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/ECM3401-LLM-Essay-Scoring")

# Project directories
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories
for d in [DATA_DIR, PROCESSED_DIR, RESULTS_DIR, PROMPTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FILES
# =============================================================================

# Sample
SAMPLE_FILE = PROCESSED_DIR / "sample_100.csv"

# Results
PHASE1_RESULTS = RESULTS_DIR / "phase1_results.csv"
PHASE2_RESULTS = RESULTS_DIR / "phase2_results.csv"

# Analysis outputs
PHASE1_METRICS = TABLES_DIR / "phase1_metrics.csv"
PHASE2_METRICS = TABLES_DIR / "phase2_metrics.csv"
COMPARISON = TABLES_DIR / "phase_comparison.csv"

# Model cache
MODEL_CACHE = MODELS_DIR / "phi3_cache"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

# API Keys
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')  # Optional - Phi-3 is public

# Models
GPT_MODEL = "gpt-4o-mini"
PHI3_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "mps"  # Apple Silicon

# Sampling
RANDOM_SEED = 42
SAMPLE_SIZE = 100
ESSAYS_PER_LEVEL = 20

# CEFR levels
CEFR_LEVELS = ['A2', 'B1', 'B2', 'C1', 'C2']

# Map intermediate levels to base
def map_cefr(level):
    mapping = {
        'A1': 'A2', 'A1+': 'A2',
        'A2': 'A2', 'A2+': 'A2',
        'B1': 'B1', 'B1+': 'B1',
        'B2': 'B2', 'B2+': 'B2',
        'C1': 'C1', 'C1+': 'C1',
        'C2': 'C2', 'C2+': 'C2'
    }
    return mapping.get(level, level)

# Length categories
def categorize_length(word_count):
    if word_count < 100:
        return 'short'
    elif word_count < 200:
        return 'medium'
    else:
        return 'long'

# Prompts
PHASE1_PROMPTS = ['minimal_v1', 'minimal_v2', 'minimal_v3',
                  'rubric_v1', 'rubric_v2', 'rubric_v3',
                  'cot_v1', 'cot_v2', 'cot_v3']

PHASE2_PROMPTS = ['minimal_v4', 'minimal_v5', 'minimal_v6',
                  'rubric_v4', 'rubric_v5', 'rubric_v6',
                  'cot_v4', 'cot_v5', 'cot_v6']

# =============================================================================
# VALIDATION
# =============================================================================

def validate():
    """Check everything is ready"""
    issues = []
    
    # Dataset
    if not DATASET_FILE.exists():
        issues.append(f"Dataset not found: {DATASET_FILE}")
    
    # API keys (only OPENAI required)
    if not OPENAI_KEY:
        issues.append("OPENAI_API_KEY not set in .env")
    
    # Note: HF_TOKEN is optional - Phi-3 is public
    
    return issues

# =============================================================================
# INFO
# =============================================================================

def print_config():
    """Print configuration"""
    print("="*70)
    print("ECM3401 CONFIGURATION")
    print("="*70)
    print(f"Dataset:     {DATASET_FILE}")
    print(f"Project:     {PROJECT_ROOT}")
    print(f"Sample size: {SAMPLE_SIZE} essays")
    print(f"Models:      {GPT_MODEL}, {PHI3_MODEL}")
    print("="*70)

if __name__ == "__main__":
    print_config()
    issues = validate()
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ All checks passed!")
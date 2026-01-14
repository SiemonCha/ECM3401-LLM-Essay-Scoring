#!/usr/bin/env python3
"""
SCRIPT 1: ONE-TIME SETUP
Downloads model, creates sample, validates everything

Run: python setup.py
Time: 10-30 minutes (mostly downloading model)
"""

import sys
from pathlib import Path

# Fix imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from simple_config import *

# =============================================================================
# STEP 1: VALIDATE ENVIRONMENT
# =============================================================================

def step1_validate():
    """Check everything is ready"""
    print("\n" + "="*70)
    print("STEP 1: VALIDATING ENVIRONMENT")
    print("="*70)
    
    issues = validate()
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues and run again.")
        sys.exit(1)
    
    print("\n✓ Environment validated")
    
    # Check PyTorch MPS
    if torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon GPU) available")
    else:
        print("⚠️  MPS not available, will use CPU (slower)")
    
    return True

# =============================================================================
# STEP 2: CREATE SAMPLE
# =============================================================================

def step2_create_sample():
    """Create stratified sample of 100 essays"""
    print("\n" + "="*70)
    print("STEP 2: CREATING SAMPLE")
    print("="*70)
    
    if SAMPLE_FILE.exists():
        print(f"\n✓ Sample already exists: {SAMPLE_FILE}")
        df = pd.read_csv(SAMPLE_FILE)
        print(f"  {len(df)} essays")
        return True
    
    print(f"\nLoading dataset from: {DATASET_FILE}")
    
    # Load full dataset
    df = pd.read_csv(DATASET_FILE, sep='\t')
    print(f"✓ Loaded {len(df):,} essays")
    
    # Filter usable essays
    usable = df[
        (df['is_final_version'] == True) &
        (df['humannotator_cefr_level'].notna()) &
        (df['split'].isin(['train', 'dev']))
    ].copy()
    
    print(f"✓ Usable essays: {len(usable):,}")
    
    # Map CEFR levels
    usable['cefr_mapped'] = usable['humannotator_cefr_level'].apply(map_cefr)
    
    # Add metadata
    usable['word_count'] = usable['text'].str.split().str.len()
    usable['length_category'] = usable['word_count'].apply(categorize_length)
    
    # Stratified sample
    np.random.seed(RANDOM_SEED)
    
    samples = []
    for level in CEFR_LEVELS:
        level_essays = usable[usable['cefr_mapped'] == level]
        if len(level_essays) < ESSAYS_PER_LEVEL:
            print(f"⚠️  Only {len(level_essays)} {level} essays available (need {ESSAYS_PER_LEVEL})")
            sample = level_essays
        else:
            sample = level_essays.sample(n=ESSAYS_PER_LEVEL, random_state=RANDOM_SEED)
        samples.append(sample)
        print(f"  Sampled {len(sample)} {level} essays")
    
    sample_df = pd.concat(samples, ignore_index=True)
    
    # Save
    sample_df.to_csv(SAMPLE_FILE, index=False)
    print(f"\n✓ Saved sample: {SAMPLE_FILE}")
    print(f"  Total: {len(sample_df)} essays")
    
    return True

# =============================================================================
# STEP 3: DOWNLOAD MODEL
# =============================================================================

def step3_download_model():
    """Download Phi-3-Mini model"""
    print("\n" + "="*70)
    print("STEP 3: DOWNLOADING PHI-3-MINI MODEL")
    print("="*70)
    
    # Check if already downloaded (transformers creates nested dirs)
    if MODEL_CACHE.exists() and any(MODEL_CACHE.rglob("*.safetensors")):
        print("\n✓ Model already downloaded")
        return True
    
    print(f"\nDownloading {PHI3_MODEL}...")
    print("This will take 10-30 minutes (~8GB)")
    print("This is ONE-TIME only!\n")
    
    try:
        # Download tokenizer (HF_TOKEN is optional for public models)
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            PHI3_MODEL,
            cache_dir=MODEL_CACHE,
            token=HF_TOKEN if HF_TOKEN else None,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded")
        
        # Download model
        print("\nDownloading model (~8GB)...")
        model = AutoModelForCausalLM.from_pretrained(
            PHI3_MODEL,
            cache_dir=MODEL_CACHE,
            token=HF_TOKEN if HF_TOKEN else None,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✓ Model downloaded")
        
        print(f"\n✓ Cached at: {MODEL_CACHE}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Check disk space (~10GB needed)")
        print("  3. Try again (downloads resume automatically)")
        return False

# =============================================================================
# STEP 4: VERIFY PROMPTS
# =============================================================================

def step4_verify_prompts():
    """Check that Phase 1 prompts exist"""
    print("\n" + "="*70)
    print("STEP 4: VERIFYING PROMPTS")
    print("="*70)
    
    missing = []
    for prompt_name in PHASE1_PROMPTS:
        prompt_file = PROMPTS_DIR / f"{prompt_name}.txt"
        if prompt_file.exists():
            print(f"✓ {prompt_name}.txt")
        else:
            print(f"❌ {prompt_name}.txt - MISSING")
            missing.append(prompt_name)
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} prompts!")
        print(f"  You need to create these files in: {PROMPTS_DIR}/")
        print(f"  Each prompt should contain '{{essay_text}}' placeholder")
        return False
    
    print(f"\n✓ All {len(PHASE1_PROMPTS)} Phase 1 prompts found")
    return True

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete setup"""
    print("="*70)
    print("ECM3401 PROJECT - ONE-TIME SETUP")
    print("="*70)
    
    print("\nThis will:")
    print("  1. Validate environment")
    print("  2. Create sample (100 essays)")
    print("  3. Download Phi-3-Mini model (~8GB, 10-30 min)")
    print("  4. Verify prompts exist")
    
    input("\nPress Enter to continue (Ctrl+C to cancel)...")
    
    # Run all steps
    steps = [
        ("Validate", step1_validate),
        ("Create Sample", step2_create_sample),
        ("Download Model", step3_download_model),
        ("Verify Prompts", step4_verify_prompts)
    ]
    
    for i, (name, func) in enumerate(steps, 1):
        try:
            success = func()
            if not success:
                print(f"\n❌ Setup failed at step {i}: {name}")
                sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Setup interrupted at step {i}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error in step {i} ({name}): {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Success!
    print("\n" + "="*70)
    print("✓ SETUP COMPLETE!")
    print("="*70)
    
    print("\nYou're ready to run experiments!")
    print("\nNext steps:")
    print("  python run_experiment.py --phase 1")
    print("\nThis will take 3-5 hours (unattended)")

if __name__ == "__main__":
    main()
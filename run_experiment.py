#!/usr/bin/env python3
"""
SCRIPT 2: RUN EXPERIMENT
Runs Phase 1 or Phase 2 experiments

Usage:
  python run_experiment.py --phase 1
  python run_experiment.py --phase 2

Time: 3-5 hours per phase (unattended)
"""

import sys
from pathlib import Path

# Fix imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import time
from datetime import datetime
from tqdm import tqdm

from simple_config import *

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

def parse_args():
    """Get phase from command line"""
    if len(sys.argv) < 3 or sys.argv[1] != '--phase':
        print("Usage: python run_experiment.py --phase [1|2]")
        sys.exit(1)
    
    phase = int(sys.argv[2])
    if phase not in [1, 2]:
        print("Error: Phase must be 1 or 2")
        sys.exit(1)
    
    return phase

# =============================================================================
# VALIDATION
# =============================================================================

def validate_ready(phase):
    """Check everything is ready"""
    print("\nValidating...")
    
    issues = []
    
    # Sample
    if not SAMPLE_FILE.exists():
        issues.append(f"Sample not found. Run: python setup.py")
    
    # API keys (only OpenAI required)
    if not OPENAI_KEY:
        issues.append("OPENAI_API_KEY not set")
    
    # Prompts
    prompt_list = PHASE1_PROMPTS if phase == 1 else PHASE2_PROMPTS
    for prompt_name in prompt_list:
        if not (PROMPTS_DIR / f"{prompt_name}.txt").exists():
            issues.append(f"Missing prompt: {prompt_name}.txt")
    
    # Model (check if cache directory has files)
    if not MODEL_CACHE.exists() or not any(MODEL_CACHE.rglob("*.json")):
        issues.append("Model not downloaded. Run: python setup.py")
    
    if issues:
        print("\n❌ Not ready:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    
    print("✓ All prerequisites met")

# =============================================================================
# LOAD PROMPTS
# =============================================================================

def load_prompts(phase):
    """Load prompt templates"""
    prompt_list = PHASE1_PROMPTS if phase == 1 else PHASE2_PROMPTS
    
    prompts = {}
    for name in prompt_list:
        file = PROMPTS_DIR / f"{name}.txt"
        prompts[name] = file.read_text()
    
    return prompts

# =============================================================================
# GPT PREDICTOR
# =============================================================================

class GPTPredictor:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_KEY)
    
    def predict(self, essay_text, prompt_template):
        """Get prediction from GPT"""
        prompt = prompt_template.replace("{essay_text}", essay_text)
        
        try:
            response = self.client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            prediction = response.choices[0].message.content.strip()
            
            # Extract CEFR level
            for level in CEFR_LEVELS:
                if level in prediction.upper():
                    return level
            
            return prediction[:10]
            
        except Exception as e:
            print(f"\n⚠️  GPT error: {e}")
            return "ERROR"

# =============================================================================
# PHI-3 PREDICTOR
# =============================================================================

class Phi3Predictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = DEVICE
    
    def load(self):
        """Load model (one time)"""
        if self.model is not None:
            return
        
        print("\nLoading Phi-3-Mini...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            PHI3_MODEL,
            cache_dir=MODEL_CACHE,
            token=HF_TOKEN if HF_TOKEN else None,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            PHI3_MODEL,
            cache_dir=MODEL_CACHE,
            token=HF_TOKEN if HF_TOKEN else None,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print("✓ Phi-3-Mini loaded")
    
    def predict(self, essay_text, prompt_template):
        """Get prediction from Phi-3"""
        self.load()
        
        prompt = prompt_template.replace("{essay_text}", essay_text)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = response[len(prompt):].strip()
            
            # Extract CEFR level
            for level in CEFR_LEVELS:
                if level in prediction.upper():
                    return level
            
            return prediction[:10]
            
        except Exception as e:
            print(f"\n⚠️  Phi-3 error: {e}")
            return "ERROR"

# =============================================================================
# RUN EXPERIMENT
# =============================================================================

def run_experiment(phase):
    """Run complete experiment"""
    
    print("="*70)
    print(f"PHASE {phase} EXPERIMENT")
    print("="*70)
    
    # Validate
    validate_ready(phase)
    
    # Load data
    print(f"\nLoading sample...")
    sample = pd.read_csv(SAMPLE_FILE)
    print(f"✓ {len(sample)} essays")
    
    # Load prompts
    print(f"\nLoading Phase {phase} prompts...")
    prompts = load_prompts(phase)
    print(f"✓ {len(prompts)} prompts")
    
    # Setup
    results_file = PHASE1_RESULTS if phase == 1 else PHASE2_RESULTS
    total_predictions = len(sample) * len(prompts) * 2  # 2 models
    
    print(f"\nConfiguration:")
    print(f"  Essays: {len(sample)}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Models: 2 (GPT, Phi-3)")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Estimated time: 3-5 hours")
    
    input("\nPress Enter to start (Ctrl+C to cancel)...")
    
    # Initialize
    gpt = GPTPredictor()
    phi3 = Phi3Predictor()
    
    results = []
    start_time = time.time()
    
    # Progress bar
    pbar = tqdm(total=total_predictions, desc="Progress")
    
    # Run experiments
    for _, essay in sample.iterrows():
        essay_id = essay['public_essay_id']
        essay_text = essay['text']
        true_label = essay['cefr_mapped']
        word_count = essay['word_count']
        length_cat = essay['length_category']
        
        for prompt_name, prompt_template in prompts.items():
            # Extract strategy and variant
            strategy = prompt_name.rsplit('_', 1)[0]
            variant = prompt_name.rsplit('_', 1)[1]
            
            # GPT prediction
            gpt_pred = gpt.predict(essay_text, prompt_template)
            results.append({
                'phase': phase,
                'essay_id': essay_id,
                'true_label': true_label,
                'word_count': word_count,
                'length_category': length_cat,
                'model': 'gpt-4o-mini',
                'prompt_name': prompt_name,
                'strategy': strategy,
                'variant': variant,
                'prediction': gpt_pred,
                'timestamp': datetime.now().isoformat()
            })
            pbar.update(1)
            
            # Phi-3 prediction
            phi3_pred = phi3.predict(essay_text, prompt_template)
            results.append({
                'phase': phase,
                'essay_id': essay_id,
                'true_label': true_label,
                'word_count': word_count,
                'length_category': length_cat,
                'model': 'phi-3-mini',
                'prompt_name': prompt_name,
                'strategy': strategy,
                'variant': variant,
                'prediction': phi3_pred,
                'timestamp': datetime.now().isoformat()
            })
            pbar.update(1)
            
            # Save incrementally
            pd.DataFrame(results).to_csv(results_file, index=False)
    
    pbar.close()
    
    # Done!
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"✓ PHASE {phase} EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"Time: {elapsed/3600:.1f} hours")
    print(f"Results: {results_file}")
    print(f"Predictions: {len(results)}")
    
    # Check errors
    errors = sum(1 for r in results if r['prediction'] == 'ERROR')
    if errors > 0:
        print(f"\n⚠️  {errors} predictions failed")
    
    print(f"\nNext step:")
    print(f"  python analyze.py --phase {phase}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        phase = parse_args()
        run_experiment(phase)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Progress has been saved!")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
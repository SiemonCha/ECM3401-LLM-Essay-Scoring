# scripts/08_run_full_experiment.py
"""
Complete Experiment Runner for ECM3401 Project
Runs all 2,430 predictions (135 essays × 9 prompts × 2 models)
Expected runtime: ~21 hours total

Features:
- Checkpointing (can resume if crashes)
- Progress tracking with ETA
- Automatic retry on failures
- Incremental saving (results saved after each essay)
- Both GPT-4o-mini and Phi-3-Mini

Run: python scripts/08_run_full_experiment.py
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable MPS fallback FIRST

import pandas as pd
import numpy as np
from pathlib import Path
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import json

from config import (
    PROCESSED_DIR, RESULTS_DIR, PROMPTS_DIR,
    OPENAI_API_KEY, GPT_MODEL, GPT_TEMPERATURE,
    LLAMA_MODEL, LLAMA_DEVICE, LLAMA_CACHE_DIR, LLAMA_TEMPERATURE,
    HF_TOKEN, PROMPTING_STRATEGIES
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CHECKPOINT_FILE = RESULTS_DIR / "experiment_checkpoint.json"
RESULTS_FILE = RESULTS_DIR / "full_experiment_results.csv"
LOG_FILE = RESULTS_DIR / "experiment_log.txt"

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# =============================================================================
# LOGGING
# =============================================================================

def log(message):
    """Log message to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + "\n")

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def load_checkpoint():
    """Load checkpoint if exists"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {
        'completed_essays': [],
        'completed_count': 0,
        'start_time': None,
        'gpt_completed': False,
        'phi3_completed': False
    }

def save_checkpoint(checkpoint):
    """Save checkpoint"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def clear_checkpoint():
    """Clear checkpoint file"""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

# =============================================================================
# LOAD PROMPTS
# =============================================================================

def load_prompts():
    """Load all 9 prompts"""
    prompts = {}
    for strategy in PROMPTING_STRATEGIES:
        for variant in [1, 2, 3]:
            prompt_file = PROMPTS_DIR / f"{strategy}_v{variant}.txt"
            if not prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
            prompts[f"{strategy}_v{variant}"] = prompt_file.read_text()
    
    log(f"✓ Loaded {len(prompts)} prompts")
    return prompts

# =============================================================================
# GPT-4O-MINI INFERENCE
# =============================================================================

def run_gpt_inference(essay_text, prompt_template, max_retries=MAX_RETRIES):
    """Run GPT-4o-mini inference with retry logic"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = prompt_template.replace("{essay_text}", essay_text)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GPT_TEMPERATURE,
                max_tokens=10  # Force brevity
            )
            
            prediction = response.choices[0].message.content.strip()
            
            # Extract CEFR level
            for level in ['A2', 'B1', 'B2', 'C1', 'C2']:
                if level in prediction.upper():
                    return level
            
            # If no level found, return raw (will be flagged in analysis)
            return prediction[:10]
            
        except Exception as e:
            if attempt < max_retries - 1:
                log(f"  ⚠️ GPT error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                log(f"  ❌ GPT failed after {max_retries} attempts: {e}")
                return "ERROR"
    
    return "ERROR"

# =============================================================================
# PHI-3-MINI INFERENCE
# =============================================================================

class Phi3Predictor:
    """Phi-3-Mini predictor with caching"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = LLAMA_DEVICE
        
    def load(self):
        """Load model (one-time)"""
        if self.model is not None:
            return  # Already loaded
        
        log("Loading Phi-3-Mini model...")
        start = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_MODEL,
            cache_dir=LLAMA_CACHE_DIR,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL,
            cache_dir=LLAMA_CACHE_DIR,
            token=HF_TOKEN,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        load_time = time.time() - start
        log(f"✓ Phi-3-Mini loaded in {load_time:.1f}s")
    
    def predict(self, essay_text, prompt_template, max_retries=MAX_RETRIES):
        """Run Phi-3 inference"""
        self.load()  # Load if not already loaded
        
        prompt = prompt_template.replace("{essay_text}", essay_text)
        
        for attempt in range(max_retries):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,  # Deterministic
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = response[len(prompt):].strip()
                
                # Extract CEFR level
                for level in ['A2', 'B1', 'B2', 'C1', 'C2']:
                    if level in prediction.upper():
                        return level
                
                return prediction[:10]
                
            except Exception as e:
                if attempt < max_retries - 1:
                    log(f"  ⚠️ Phi-3 error (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(RETRY_DELAY)
                else:
                    log(f"  ❌ Phi-3 failed after {max_retries} attempts: {e}")
                    return "ERROR"
        
        return "ERROR"

# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_experiment():
    """Run complete experiment"""
    
    print("="*70)
    print("EXPERIMENT RUNNER - ECM3401 PROJECT")
    print("="*70)
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    
    if checkpoint['start_time'] is None:
        checkpoint['start_time'] = datetime.now().isoformat()
        log(f"Starting new experiment at {checkpoint['start_time']}")
    else:
        log(f"Resuming experiment from checkpoint")
        log(f"  Completed: {checkpoint['completed_count']} essays")
        log(f"  GPT done: {checkpoint['gpt_completed']}")
        log(f"  Phi-3 done: {checkpoint['phi3_completed']}")
    
    # Load data
    log("Loading sample...")
    sample = pd.read_csv(PROCESSED_DIR / "phase1_sample_100.csv")
    log(f"✓ Loaded {len(sample)} essays")
    
    # Load prompts
    prompts = load_prompts()
    
    # Initialize results list
    if RESULTS_FILE.exists() and checkpoint['completed_count'] > 0:
        log("Loading existing results...")
        results_df = pd.read_csv(RESULTS_FILE)
        results = results_df.to_dict('records')
        log(f"✓ Loaded {len(results)} existing predictions")
    else:
        results = []
    
    # Total predictions
    total_essays = len(sample)
    total_prompts = len(prompts)
    total_predictions_per_model = total_essays * total_prompts
    
    log(f"\nExperiment Configuration:")
    log(f"  Essays: {total_essays}")
    log(f"  Prompts: {total_prompts}")
    log(f"  Predictions per model: {total_predictions_per_model}")
    log(f"  Total predictions: {total_predictions_per_model * 2}")
    
    # ==========================================================================
    # RUN GPT-4O-MINI
    # ==========================================================================
    
    if not checkpoint['gpt_completed']:
        log("\n" + "="*70)
        log("PHASE 1: GPT-4O-MINI PREDICTIONS")
        log("="*70)
        
        gpt_start = time.time()
        gpt_count = 0
        
        for essay_idx, essay_row in tqdm(sample.iterrows(), total=len(sample), desc="GPT-4o-mini"):
            essay_id = essay_row['public_essay_id']
            
            # Skip if already completed
            if essay_id in checkpoint['completed_essays'] and checkpoint['gpt_completed']:
                continue
            
            essay_text = essay_row['text']
            true_label = essay_row['cefr_mapped']
            word_count = len(essay_text.split())
            
            # Categorize length
            if word_count < 100:
                length_category = 'short'
            elif word_count < 200:
                length_category = 'medium'
            else:
                length_category = 'long'
            
            log(f"\n[GPT] Essay {essay_idx+1}/{total_essays}: {essay_id} ({word_count} words, {true_label})")
            
            # Test with all prompts
            for prompt_name, prompt_template in prompts.items():
                start_time = time.time()
                
                prediction = run_gpt_inference(essay_text, prompt_template)
                
                inference_time = time.time() - start_time
                
                # Save result
                result = {
                    'essay_id': essay_id,
                    'essay_index': essay_idx,
                    'true_label': true_label,
                    'word_count': word_count,
                    'length_category': length_category,
                    'model': 'gpt-4o-mini',
                    'prompt_name': prompt_name,
                    'strategy': prompt_name.rsplit('_', 1)[0],
                    'variant': prompt_name.rsplit('_', 1)[1],
                    'prediction': prediction,
                    'inference_time_sec': inference_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                gpt_count += 1
                
                log(f"  {prompt_name}: {prediction} ({inference_time:.2f}s)")
            
            # Save after each essay (incremental)
            pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
            
            # Update checkpoint
            if essay_id not in checkpoint['completed_essays']:
                checkpoint['completed_essays'].append(essay_id)
            checkpoint['completed_count'] = len(checkpoint['completed_essays'])
            save_checkpoint(checkpoint)
        
        gpt_time = time.time() - gpt_start
        checkpoint['gpt_completed'] = True
        save_checkpoint(checkpoint)
        
        log(f"\n✓ GPT-4o-mini complete!")
        log(f"  Predictions: {gpt_count}")
        log(f"  Total time: {gpt_time/60:.1f} minutes")
        log(f"  Avg per essay: {gpt_time/total_essays:.1f}s")
    
    else:
        log("\n✓ GPT-4o-mini already completed (skipping)")
    
    # ==========================================================================
    # RUN PHI-3-MINI
    # ==========================================================================
    
    if not checkpoint['phi3_completed']:
        log("\n" + "="*70)
        log("PHASE 2: PHI-3-MINI PREDICTIONS")
        log("="*70)
        
        phi3_predictor = Phi3Predictor()
        phi3_start = time.time()
        phi3_count = 0
        
        for essay_idx, essay_row in tqdm(sample.iterrows(), total=len(sample), desc="Phi-3-Mini"):
            essay_id = essay_row['public_essay_id']
            essay_text = essay_row['text']
            true_label = essay_row['cefr_mapped']
            word_count = len(essay_text.split())
            
            # Categorize length
            if word_count < 100:
                length_category = 'short'
            elif word_count < 200:
                length_category = 'medium'
            else:
                length_category = 'long'
            
            log(f"\n[Phi-3] Essay {essay_idx+1}/{total_essays}: {essay_id} ({word_count} words, {true_label})")
            
            # Test with all prompts
            for prompt_name, prompt_template in prompts.items():
                start_time = time.time()
                
                prediction = phi3_predictor.predict(essay_text, prompt_template)
                
                inference_time = time.time() - start_time
                
                # Save result
                result = {
                    'essay_id': essay_id,
                    'essay_index': essay_idx,
                    'true_label': true_label,
                    'word_count': word_count,
                    'length_category': length_category,
                    'model': 'phi-3-mini',
                    'prompt_name': prompt_name,
                    'strategy': prompt_name.rsplit('_', 1)[0],
                    'variant': prompt_name.rsplit('_', 1)[1],
                    'prediction': prediction,
                    'inference_time_sec': inference_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                phi3_count += 1
                
                log(f"  {prompt_name}: {prediction} ({inference_time:.2f}s)")
            
            # Save after each essay (incremental)
            pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
        
        phi3_time = time.time() - phi3_start
        checkpoint['phi3_completed'] = True
        save_checkpoint(checkpoint)
        
        log(f"\n✓ Phi-3-Mini complete!")
        log(f"  Predictions: {phi3_count}")
        log(f"  Total time: {phi3_time/3600:.1f} hours")
        log(f"  Avg per essay: {phi3_time/total_essays:.1f}s")
    
    else:
        log("\n✓ Phi-3-Mini already completed (skipping)")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    
    log("\n" + "="*70)
    log("EXPERIMENT COMPLETE!")
    log("="*70)
    
    # Load final results
    final_results = pd.read_csv(RESULTS_FILE)
    
    log(f"\nFinal Statistics:")
    log(f"  Total predictions: {len(final_results)}")
    log(f"  GPT predictions: {len(final_results[final_results['model']=='gpt-4o-mini'])}")
    log(f"  Phi-3 predictions: {len(final_results[final_results['model']=='phi-3-mini'])}")
    log(f"  Unique essays: {final_results['essay_id'].nunique()}")
    log(f"  Prompts tested: {final_results['prompt_name'].nunique()}")
    
    # Error check
    errors = final_results[final_results['prediction'] == 'ERROR']
    if len(errors) > 0:
        log(f"\n⚠️ WARNING: {len(errors)} predictions returned ERROR")
        log(f"  Check {RESULTS_FILE} for details")
    else:
        log(f"\n✓ All predictions successful!")
    
    log(f"\nResults saved to: {RESULTS_FILE}")
    log(f"Log saved to: {LOG_FILE}")
    
    # Clear checkpoint
    clear_checkpoint()
    log("✓ Checkpoint cleared")
    
    log("\nNext steps:")
    log("  1. Review results: head -20 " + str(RESULTS_FILE))
    log("  2. Run analysis: python scripts/09_analyze_results.py")
    log("  3. Create plots: python scripts/10_create_plots.py")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        run_experiment()
    except KeyboardInterrupt:
        log("\n\n⚠️ Experiment interrupted by user")
        log("Progress saved in checkpoint. Run again to resume.")
    except Exception as e:
        log(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        log("Progress saved in checkpoint. Fix error and run again to resume.")

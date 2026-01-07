# scripts/08_run_experiment.py
"""
Unified Experiment Runner (Works for Both Phases)

Usage:
  Phase 1: python scripts/08_run_experiment.py --phase 1
  Phase 2: python scripts/08_run_experiment.py --phase 2

Auto-detects which prompts to use based on phase argument.
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import pandas as pd
import sys
from pathlib import Path
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from datetime import datetime
from tqdm import tqdm

from config import (
    PROCESSED_DIR, RESULTS_DIR, PROMPTS_DIR,
    OPENAI_API_KEY, GPT_MODEL, GPT_TEMPERATURE,
    LLAMA_MODEL, LLAMA_DEVICE, LLAMA_CACHE_DIR,
    HF_TOKEN
)

# =============================================================================
# CONFIGURATION
# =============================================================================

PHASE1_PROMPTS = ['minimal_v1', 'minimal_v2', 'minimal_v3',
                  'rubric_v1', 'rubric_v2', 'rubric_v3',
                  'cot_v1', 'cot_v2', 'cot_v3']

PHASE2_PROMPTS = ['minimal_v4', 'minimal_v5', 'minimal_v6',
                  'rubric_v4', 'rubric_v5', 'rubric_v6',
                  'cot_v4', 'cot_v5', 'cot_v6']

MAX_RETRIES = 3
RETRY_DELAY = 5

# =============================================================================
# PARSE ARGS
# =============================================================================

def parse_args():
    """Parse command line arguments"""
    if len(sys.argv) < 3 or sys.argv[1] != '--phase':
        print("Usage: python scripts/08_run_experiment.py --phase [1|2]")
        sys.exit(1)
    
    phase = int(sys.argv[2])
    if phase not in [1, 2]:
        print("Error: Phase must be 1 or 2")
        sys.exit(1)
    
    return phase

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(phase):
    """Setup logging for experiment"""
    log_file = RESULTS_DIR / f"phase{phase}_experiment_log.txt"
    return log_file

def log(message, log_file):
    """Log to console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(log_file, 'a') as f:
        f.write(log_msg + "\n")

# =============================================================================
# LOAD PROMPTS
# =============================================================================

def load_prompts(prompt_list):
    """Load prompt files from prompts/ directory"""
    prompts = {}
    for prompt_name in prompt_list:
        prompt_file = PROMPTS_DIR / f"{prompt_name}.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_file}")
        prompts[prompt_name] = prompt_file.read_text()
    
    return prompts

# =============================================================================
# MODEL INFERENCE
# =============================================================================

def run_gpt_inference(essay_text, prompt_template, max_retries=MAX_RETRIES):
    """Run GPT-4o-mini inference"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = prompt_template.replace("{essay_text}", essay_text)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GPT_TEMPERATURE,
                max_tokens=10
            )
            
            prediction = response.choices[0].message.content.strip()
            
            # Extract CEFR level
            for level in ['A2', 'B1', 'B2', 'C1', 'C2']:
                if level in prediction.upper():
                    return level
            
            return prediction[:10]
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                return "ERROR"
    
    return "ERROR"

class Phi3Predictor:
    """Phi-3-Mini predictor"""
    
    def __init__(self, log_file):
        self.model = None
        self.tokenizer = None
        self.device = LLAMA_DEVICE
        self.log_file = log_file
        
    def load(self):
        """Load model once"""
        if self.model is not None:
            return
        
        log("Loading Phi-3-Mini model...", self.log_file)
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
        
        log(f"✓ Phi-3-Mini loaded in {time.time() - start:.1f}s", self.log_file)
    
    def predict(self, essay_text, prompt_template, max_retries=MAX_RETRIES):
        """Run inference"""
        self.load()
        prompt = prompt_template.replace("{essay_text}", essay_text)
        
        for attempt in range(max_retries):
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
                
                for level in ['A2', 'B1', 'B2', 'C1', 'C2']:
                    if level in prediction.upper():
                        return level
                
                return prediction[:10]
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    return "ERROR"
        
        return "ERROR"

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(phase):
    """Run experiment for specified phase"""
    
    # Setup
    log_file = setup_logging(phase)
    results_file = RESULTS_DIR / f"phase{phase}_experiment_results.csv"
    
    prompt_list = PHASE1_PROMPTS if phase == 1 else PHASE2_PROMPTS
    
    print("="*70)
    print(f"PHASE {phase} EXPERIMENT")
    print("="*70)
    
    # Load data
    log(f"\nLoading sample...", log_file)
    sample = pd.read_csv(PROCESSED_DIR / "phase1_sample_100.csv")
    log(f"✓ Loaded {len(sample)} essays", log_file)
    
    # Load prompts
    log(f"\nLoading Phase {phase} prompts...", log_file)
    prompts = load_prompts(prompt_list)
    log(f"✓ Loaded {len(prompts)} prompts: {list(prompts.keys())}", log_file)
    
    # Stats
    total_predictions = len(sample) * len(prompts) * 2
    log(f"\nConfiguration:", log_file)
    log(f"  Phase: {phase}", log_file)
    log(f"  Essays: {len(sample)}", log_file)
    log(f"  Prompts: {len(prompts)}", log_file)
    log(f"  Models: 2 (GPT-4o-mini, Phi-3-Mini)", log_file)
    log(f"  Total predictions: {total_predictions}", log_file)
    
    results = []
    
    # =========================================================================
    # GPT-4O-MINI
    # =========================================================================
    
    log("\n" + "="*70, log_file)
    log("RUNNING GPT-4O-MINI", log_file)
    log("="*70, log_file)
    
    gpt_start = time.time()
    
    for essay_idx, essay_row in tqdm(sample.iterrows(), total=len(sample), desc="GPT"):
        essay_id = essay_row['public_essay_id']
        essay_text = essay_row['text']
        true_label = essay_row['cefr_mapped']
        word_count = len(essay_text.split())
        
        # Length category
        if word_count < 100:
            length_category = 'short'
        elif word_count < 200:
            length_category = 'medium'
        else:
            length_category = 'long'
        
        log(f"\n[GPT] Essay {essay_idx+1}/{len(sample)}: {essay_id}", log_file)
        
        for prompt_name, prompt_template in prompts.items():
            start_time = time.time()
            prediction = run_gpt_inference(essay_text, prompt_template)
            inference_time = time.time() - start_time
            
            # Extract strategy
            strategy = prompt_name.rsplit('_', 1)[0]
            variant = prompt_name.rsplit('_', 1)[1]
            
            results.append({
                'phase': phase,
                'essay_id': essay_id,
                'true_label': true_label,
                'word_count': word_count,
                'length_category': length_category,
                'model': 'gpt-4o-mini',
                'prompt_name': prompt_name,
                'strategy': strategy,
                'variant': variant,
                'prediction': prediction,
                'inference_time_sec': inference_time,
                'timestamp': datetime.now().isoformat()
            })
            
            log(f"  {prompt_name}: {prediction}", log_file)
        
        # Save incrementally
        pd.DataFrame(results).to_csv(results_file, index=False)
    
    gpt_time = time.time() - gpt_start
    log(f"\n✓ GPT complete: {gpt_time/60:.1f} min", log_file)
    
    # =========================================================================
    # PHI-3-MINI
    # =========================================================================
    
    log("\n" + "="*70, log_file)
    log("RUNNING PHI-3-MINI", log_file)
    log("="*70, log_file)
    
    phi3_predictor = Phi3Predictor(log_file)
    phi3_start = time.time()
    
    for essay_idx, essay_row in tqdm(sample.iterrows(), total=len(sample), desc="Phi-3"):
        essay_id = essay_row['public_essay_id']
        essay_text = essay_row['text']
        true_label = essay_row['cefr_mapped']
        word_count = len(essay_text.split())
        
        if word_count < 100:
            length_category = 'short'
        elif word_count < 200:
            length_category = 'medium'
        else:
            length_category = 'long'
        
        log(f"\n[Phi-3] Essay {essay_idx+1}/{len(sample)}: {essay_id}", log_file)
        
        for prompt_name, prompt_template in prompts.items():
            start_time = time.time()
            prediction = phi3_predictor.predict(essay_text, prompt_template)
            inference_time = time.time() - start_time
            
            strategy = prompt_name.rsplit('_', 1)[0]
            variant = prompt_name.rsplit('_', 1)[1]
            
            results.append({
                'phase': phase,
                'essay_id': essay_id,
                'true_label': true_label,
                'word_count': word_count,
                'length_category': length_category,
                'model': 'phi-3-mini',
                'prompt_name': prompt_name,
                'strategy': strategy,
                'variant': variant,
                'prediction': prediction,
                'inference_time_sec': inference_time,
                'timestamp': datetime.now().isoformat()
            })
            
            log(f"  {prompt_name}: {prediction}", log_file)
        
        pd.DataFrame(results).to_csv(results_file, index=False)
    
    phi3_time = time.time() - phi3_start
    log(f"\n✓ Phi-3 complete: {phi3_time/60:.1f} min", log_file)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    log("\n" + "="*70, log_file)
    log(f"PHASE {phase} EXPERIMENT COMPLETE!", log_file)
    log("="*70, log_file)
    
    final_results = pd.DataFrame(results)
    
    log(f"\nResults:", log_file)
    log(f"  Total predictions: {len(final_results)}", log_file)
    log(f"  GPT predictions: {len(final_results[final_results['model']=='gpt-4o-mini'])}", log_file)
    log(f"  Phi-3 predictions: {len(final_results[final_results['model']=='phi-3-mini'])}", log_file)
    log(f"  Errors: {len(final_results[final_results['prediction']=='ERROR'])}", log_file)
    log(f"\nTotal runtime: {(gpt_time + phi3_time)/60:.1f} minutes", log_file)
    log(f"\nResults saved: {results_file}", log_file)
    log(f"Log saved: {log_file}", log_file)
    
    log("\n" + "="*70, log_file)
    log("NEXT STEPS", log_file)
    log("="*70, log_file)
    log(f"Analyze results: python scripts/09_analyze_results.py --phase {phase}", log_file)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        phase = parse_args()
        run_experiment(phase)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

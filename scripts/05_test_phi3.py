# scripts/05_test_phi3.py - WITH MPS FALLBACK FIX
"""
Test Phi-3-Mini on M2 Pro MacBook
Automatically enables MPS fallback for unsupported operations
Expected performance: 20-40 seconds per essay
Run: python ./scripts/05_test_phi3.py
"""

import os
# Enable MPS fallback BEFORE importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from config import LLAMA_MODEL, LLAMA_CACHE_DIR, PROCESSED_DIR, HF_TOKEN, LLAMA_DEVICE
import time

def main():
    print("="*70)
    print("PHI-3-MINI TEST ON M2 PRO (MPS FALLBACK ENABLED)")
    print("="*70)
    
    # Check MPS
    if not torch.backends.mps.is_available():
        print("⚠️ MPS not available, using CPU (will be slower)")
        device = "cpu"
    else:
        device = LLAMA_DEVICE
        print(f"✓ Device: {device} (Apple Silicon GPU)")
        print("✓ MPS fallback enabled (for unsupported ops)")
    
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Load sample
    print("\nLoading test essay...")
    sample = pd.read_csv(PROCESSED_DIR / "phase1_sample_100.csv")
    test_essay = sample.iloc[0]
    
    print(f"  ID: {test_essay['public_essay_id']}")
    print(f"  True CEFR: {test_essay['cefr_mapped']}")
    print(f"  Words: {len(test_essay['text'].split())}")
    
    # Create prompt
    prompt = f"""Classify this essay's CEFR level (A2, B1, B2, C1, or C2):

{test_essay['text']}

CEFR Level:"""
    
    try:
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_MODEL,
            cache_dir=LLAMA_CACHE_DIR,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        # Set padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Tokenizer loaded")
        
        # Load model to MPS
        print(f"\nLoading Phi-3-Mini to {device}...")
        print("(First load may take 30-60 seconds)")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL,
            cache_dir=LLAMA_CACHE_DIR,
            token=HF_TOKEN,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.1f} seconds")
        
        # Prepare input
        print("\nRunning inference...")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        start_time = time.time()
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,  # Deterministic (temperature=0 equivalent)
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        inference_time = time.time() - start_time
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = response[len(prompt):].strip().split()[0] if response[len(prompt):].strip() else "N/A"
        
        print(f"\n✓ Phi-3 predicted: {prediction}")
        print(f"✓ True label: {test_essay['cefr_mapped']}")
        print(f"✓ Inference time: {inference_time:.2f} seconds")
        
        print("\n" + "="*70)
        print("✓ PHI-3-MINI WORKS ON M2 PRO!")
        print("="*70)
        
        print(f"\nPerformance:")
        print(f"  Model: Phi-3-Mini (3.8B params)")
        print(f"  Load time: {load_time:.1f}s (one-time per session)")
        print(f"  Inference time: {inference_time:.2f}s per essay")
        print(f"  Device: {device}")
        print(f"  MPS Fallback: Enabled (some ops use CPU)")
        
        # Estimate for full experiment
        total_predictions = 2430
        total_time_hours = (inference_time * total_predictions) / 3600
        
        if total_time_hours < 1:
            total_time_minutes = (inference_time * total_predictions) / 60
            print(f"\nFor 2,430 predictions: ~{total_time_minutes:.0f} minutes")
        else:
            print(f"\nFor 2,430 predictions: ~{total_time_hours:.1f} hours")
        
        # Performance assessment
        if inference_time <= 50:
            print("\n✅ Performance is EXCELLENT!")
            print("   Much faster than Llama-3-8B")
        elif inference_time <= 120:
            print("\n✅ Performance is GOOD!")
        else:
            print(f"\n⚠️ Slower than expected. Actual: {inference_time:.1f}s")
            print("   Consider upgrading PyTorch or using CPU mode")
        
        # Comparison
        llama_time = 720
        speedup = llama_time / inference_time
        print(f"\nSpeedup vs Llama-3-8B: {speedup:.0f}x faster!")
        
        # Note about fallback
        print("\nNote: Some operations fall back to CPU (MPS limitation)")
        print("      This is normal and performance is still very good!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Try: export PYTORCH_ENABLE_MPS_FALLBACK=1")
        print("2. Or use CPU mode: Change LLAMA_DEVICE='cpu' in config.py")
        print("3. Upgrade PyTorch: pip install --upgrade torch")

if __name__ == "__main__":
    main()
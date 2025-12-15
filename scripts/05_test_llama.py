# scripts/05_test_llama.py - OPTIMIZED FOR AMD 7900 XT
"""
Test Llama-3-8B on AMD Radeon RX 7900 XT (ROCm)
Expected performance: 1-3 seconds per essay
Run: python ./scripts/05_test_llama.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from config import LLAMA_MODEL, LLAMA_CACHE_DIR, PROCESSED_DIR, HF_TOKEN, LLAMA_DEVICE
import time

def main():
    print("="*70)
    print("LLAMA-3-8B TEST ON AMD 7900 XT (ROCm)")
    print("="*70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ GPU not detected!")
        print("\nInstall PyTorch with ROCm:")
        print("pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
        return
    
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"✓ Device: {LLAMA_DEVICE}")
    
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
            token=HF_TOKEN
        )
        
        # Set padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Tokenizer loaded")
        
        # Configure 4-bit quantization (saves VRAM, still fast on 7900 XT)
        print("\nConfiguring 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load model to GPU
        print(f"\nLoading model to {LLAMA_DEVICE}...")
        print("(First load may take 30-60 seconds)")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL,
            cache_dir=LLAMA_CACHE_DIR,
            token=HF_TOKEN,
            quantization_config=bnb_config,  # Use 4-bit
            device_map="auto",  # Automatically uses GPU
            torch_dtype=torch.float16
        )
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.1f} seconds")
        
        # Prepare input
        print("\nRunning inference...")
        inputs = tokenizer(prompt, return_tensors="pt").to(LLAMA_DEVICE)
        
        # Generate (should be FAST on 7900 XT!)
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
        
        print(f"\n✓ Llama predicted: {prediction}")
        print(f"✓ True label: {test_essay['cefr_mapped']}")
        print(f"✓ Inference time: {inference_time:.2f} seconds")
        
        print("\n" + "="*70)
        print("✓ LLAMA OPTIMIZED FOR AMD 7900 XT!")
        print("="*70)
        
        print(f"\nPerformance:")
        print(f"  Load time: {load_time:.1f}s (one-time)")
        print(f"  Inference time: {inference_time:.2f}s per essay")
        print(f"  Device: {LLAMA_DEVICE}")
        print(f"  Quantization: 4-bit NF4")
        print(f"  VRAM usage: ~6-8 GB (out of 24 GB)")
        
        # Estimate for full experiment
        total_predictions = 2400
        total_time_hours = (inference_time * total_predictions) / 3600
        total_time_minutes = (inference_time * total_predictions) / 60
        
        if total_time_hours < 1:
            print(f"\nFor 2,400 predictions: ~{total_time_minutes:.0f} minutes")
        else:
            print(f"\nFor 2,400 predictions: ~{total_time_hours:.1f} hours")
        
        if inference_time <= 5:
            print("\n✅ Performance is EXCELLENT! Ready for production!")
        elif inference_time <= 30:
            print("\n✅ Performance is GOOD! Ready for experiments!")
        else:
            print("\n⚠️ Performance slower than expected. Check GPU drivers.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Ensure PyTorch with ROCm is installed:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
        print("2. Set environment variable for RX 7900 XT:")
        print("   export HSA_OVERRIDE_GFX_VERSION=11.0.0")
        print("3. Check model is downloaded: ls models/llama_cache/")
        print("4. Try without quantization (remove quantization_config)")

if __name__ == "__main__":
    main()
# scripts/04_download_phi3.py
"""
Download Phi-3-Mini-4k-Instruct model to local cache
This is a one-time download (~8GB)
Much faster than Llama-3-8B on M2 Pro
Run: python ./scripts/04_download_phi3.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from config import LLAMA_MODEL, LLAMA_CACHE_DIR, HF_TOKEN
import torch

def main():
    print("="*70)
    print("PHI-3-MINI DOWNLOAD")
    print("="*70)
    
    print(f"\nModel: {LLAMA_MODEL}")
    print(f"Cache directory: {LLAMA_CACHE_DIR}")
    print(f"Download size: ~8GB")
    print(f"Expected speed on M2 Pro: 20-40 seconds per essay")
    print(f"This is a ONE-TIME download\n")
    
    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("⚠️ WARNING: MPS not available!")
        print("Model will run on CPU (much slower)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    else:
        print("✓ MPS (Apple Silicon GPU) detected")
    
    try:
        # Download tokenizer
        print("\n" + "="*70)
        print("Step 1/2: Downloading tokenizer...")
        print("="*70)
        
        tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_MODEL,
            cache_dir=LLAMA_CACHE_DIR,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded")
        
        # Download model
        print("\n" + "="*70)
        print("Step 2/2: Downloading model (~8GB, may take 10-30 min)...")
        print("="*70)
        
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL,
            cache_dir=LLAMA_CACHE_DIR,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✓ Model downloaded")
        
        print("\n" + "="*70)
        print("✓ PHI-3-MINI DOWNLOAD COMPLETE")
        print("="*70)
        
        print(f"\nModel cached at: {LLAMA_CACHE_DIR}")
        print("Next step: Run python ./scripts/05_test_phi3.py")
        
        print("\nExpected performance on M2 Pro:")
        print("  - 20-40 seconds per essay")
        print("  - 8-17 hours for 1,800 predictions")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Check disk space (~10GB needed)")
        print("3. Try again")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
# scripts/04_download_llama.py
"""
Download Llama-3-8B-Instruct model to local cache
This is a one-time download (~8GB)
Works on Linux with AMD GPU
Run: python ./scripts/04_download_llama.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from config import LLAMA_MODEL, LLAMA_CACHE_DIR, HF_TOKEN
import torch

def main():
    print("="*70)
    print("LLAMA-3-8B DOWNLOAD")
    print("="*70)
    
    print(f"\nModel: {LLAMA_MODEL}")
    print(f"Cache directory: {LLAMA_CACHE_DIR}")
    print(f"Download size: ~8GB")
    print(f"This is a ONE-TIME download\n")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ WARNING: GPU not detected!")
        print("Model will run on CPU (very slow)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    try:
        # Download tokenizer
        print("\n" + "="*70)
        print("Step 1/2: Downloading tokenizer...")
        print("="*70)
        
        tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_MODEL,
            cache_dir=LLAMA_CACHE_DIR,
            token=HF_TOKEN
        )
        print("✓ Tokenizer downloaded")
        
        # Download model (this is the big one)
        print("\n" + "="*70)
        print("Step 2/2: Downloading model (~8GB, may take 10-30 min)...")
        print("="*70)
        
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL,
            cache_dir=LLAMA_CACHE_DIR,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("✓ Model downloaded")
        
        print("\n" + "="*70)
        print("✓ LLAMA-3-8B DOWNLOAD COMPLETE")
        print("="*70)
        
        print(f"\nModel cached at: {LLAMA_CACHE_DIR}")
        print("Next step: Run python ./scripts/05_test_llama.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
        if "gated" in str(e).lower() or "401" in str(e):
            print("\n⚠️ This model is gated. You need:")
            print("1. Go to: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
            print("2. Click 'Access repository'")
            print("3. Accept terms")
            print("4. Get token from: https://huggingface.co/settings/tokens")
            print("5. Add to .env: HUGGINGFACE_TOKEN=hf_your_token_here")
        else:
            print("\nTroubleshooting:")
            print("1. Check internet connection")
            print("2. Check disk space (~10GB needed)")
            print("3. Try again")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
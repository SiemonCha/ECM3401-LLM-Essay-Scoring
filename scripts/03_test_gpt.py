# scripts/03_test_gpt.py
"""
Test GPT-4o-mini API with one essay
Works on both Linux and macOS
Run: python ./scripts/03_test_gpt.py
"""

from openai import OpenAI
import pandas as pd
from config import GPT_MODEL, OPENAI_API_KEY, PROCESSED_DIR

def main():
    print("="*70)
    print("GPT-4o-mini API TEST")
    print("="*70)
    
    # Check API key is set
    if not OPENAI_API_KEY:
        print("\n❌ API key not set!")
        print("Add to .env file: OPENAI_API_KEY=sk-proj-...")
        return
    
    # Initialize client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
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
        # Call GPT (FIXED: added comma after temperature)
        print("\nCalling GPT-4o-mini API...")
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # ← FIXED: was missing comma
            max_tokens=10
        )
        
        prediction = response.choices[0].message.content.strip()
        
        print(f"\n✓ GPT predicted: {prediction}")
        print(f"✓ True label: {test_essay['cefr_mapped']}")
        
        # Cost calculation
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # GPT-4o-mini pricing
        cost = (input_tokens * 0.150 + output_tokens * 0.600) / 1_000_000
        
        print(f"\nUsage:")
        print(f"  Input tokens: {input_tokens}")
        print(f"  Output tokens: {output_tokens}")
        print(f"  Cost: ${cost:.6f}")
        
        # Estimate for full experiment
        total_cost = cost * 2400
        print(f"\nEstimated cost for 2,400 predictions: ${total_cost:.2f}")
        
        print("\n" + "="*70)
        print("✓ GPT-4o-mini WORKS!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check API key in .env file")
        print("2. Check billing: https://platform.openai.com/account/billing")
        print("3. Ensure you have credits in your account")

if __name__ == "__main__":
    main()
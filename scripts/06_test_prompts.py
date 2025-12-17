# scripts/06_test_prompts.py
"""
Test all 9 prompt variants with one essay
Updated with better response extraction
Run: python scripts/06_test_prompts.py
"""

import pandas as pd
from pathlib import Path
from openai import OpenAI
from config import PROCESSED_DIR, OPENAI_API_KEY, GPT_MODEL
import re

def extract_cefr_level(response_text):
    """Extract CEFR level from response text"""
    # Look for CEFR levels in order of specificity
    patterns = [
        r'\b(A2|B1|B2|C1|C2)\b',  # Exact match
        r'level.*?(A2|B1|B2|C1|C2)',  # After "level"
        r'rating.*?(A2|B1|B2|C1|C2)',  # After "rating"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Fallback: first occurrence of any CEFR level
    for level in ['A2', 'B1', 'B2', 'C1', 'C2']:
        if level in response_text.upper():
            return level
    
    return response_text.strip()[:50]  # Return first 50 chars if no level found

def main():
    print("="*70)
    print("TESTING ALL 9 PROMPT VARIANTS")
    print("="*70)
    
    # Load one test essay
    sample = pd.read_csv(PROCESSED_DIR / "phase1_sample_100.csv")
    test_essay = sample.iloc[0]
    
    print(f"\nTest Essay:")
    print(f"  ID: {test_essay['public_essay_id']}")
    print(f"  True CEFR: {test_essay['cefr_mapped']}")
    print(f"  Length: {len(test_essay['text'].split())} words")
    print(f"\nText preview: {test_essay['text'][:200]}...")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Test each prompt
    prompt_dir = Path("prompts")
    strategies = ['minimal', 'rubric', 'cot']
    
    results = []
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"STRATEGY: {strategy.upper()}")
        print(f"{'='*70}")
        
        for variant in [1, 2, 3]:
            # Load prompt
            prompt_file = prompt_dir / f"{strategy}_v{variant}.txt"
            
            if not prompt_file.exists():
                print(f"⚠️ File not found: {prompt_file}")
                continue
            
            prompt_template = prompt_file.read_text()
            prompt = prompt_template.replace("{essay_text}", test_essay['text'])
            
            try:
                # Call GPT with reduced max_tokens to discourage verbosity
                response = client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10  # Reduced from 50 to force brevity
                )
                
                raw_response = response.choices[0].message.content.strip()
                prediction = extract_cefr_level(raw_response)
                
                # Show both raw and extracted
                if len(raw_response) > 20:
                    print(f"  {strategy}_v{variant}: {prediction} (extracted from: '{raw_response[:50]}...')")
                else:
                    print(f"  {strategy}_v{variant}: {prediction}")
                
                results.append({
                    'strategy': strategy,
                    'variant': variant,
                    'prediction': prediction,
                    'raw': raw_response
                })
                
            except Exception as e:
                print(f"  {strategy}_v{variant}: ERROR - {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nTrue label: {test_essay['cefr_mapped']}")
    print(f"\nPredictions by strategy:")
    
    for strategy in strategies:
        strategy_results = [r for r in results if r['strategy'] == strategy]
        predictions = [r['prediction'] for r in strategy_results]
        
        print(f"\n{strategy.upper()}:")
        for i, pred in enumerate(predictions, 1):
            match = "✓" if pred == test_essay['cefr_mapped'] else "✗"
            print(f"  v{i}: {pred} {match}")
        
        # Check consistency
        if len(set(predictions)) == 1:
            print(f"  → Consistent across paraphrases ✓")
        else:
            print(f"  → INCONSISTENT across paraphrases! ⚠️")
            print(f"     Unique predictions: {set(predictions)}")
    
    print(f"\n{'='*70}")
    print("✓ PROMPT TEST COMPLETE")
    print(f"{'='*70}")
    
    # Check if any prompts had issues
    clean_responses = [r for r in results if len(r['raw']) <= 10]
    verbose_responses = [r for r in results if len(r['raw']) > 10]
    
    if verbose_responses:
        print(f"\n⚠️ {len(verbose_responses)}/9 prompts returned verbose responses")
        print("   Consider updating those prompts to be more explicit")
    else:
        print(f"\n✓ All {len(clean_responses)}/9 prompts returned clean responses!")
    
    print("\nNext steps:")
    print("1. If still verbose, use the FIXED prompts in prompts_fixed/")
    print("2. Re-run this script to verify")
    print("3. Ready for full experiment!")

if __name__ == "__main__":
    main()

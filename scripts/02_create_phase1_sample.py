# scripts/02_create_phase1_sample.py
"""
Create Phase 1 sample (100 essays, 20 per CEFR level)
Run: python -m scripts.02_create_phase1_sample
"""

import pandas as pd
import numpy as np
from config import (
    CORPUS_FILE, PROCESSED_DIR, CEFR_LEVELS, 
    RANDOM_SEED, ESSAYS_PER_LEVEL, map_cefr_level
)

def main():
    np.random.seed(RANDOM_SEED)
    
    print("="*70)
    print("PHASE 1 SAMPLE CREATION")
    print("="*70)
    
    # Load corpus
    corpus = pd.read_csv(CORPUS_FILE, sep='\t')
    
    # Filter usable essays
    usable = corpus[
        (corpus['is_final_version'] == True) &
        (corpus['humannotator_cefr_level'].notna()) &
        (corpus['split'].isin(['train', 'dev']))
    ].copy()
    
    # Map CEFR levels (combine + levels)
    usable['cefr_mapped'] = usable['humannotator_cefr_level'].apply(map_cefr_level)
    
    print(f"\n✓ Loaded {len(usable):,} usable essays")
    
    # Show distribution after mapping
    print("\nCEFR distribution (after mapping + levels):")
    for level in CEFR_LEVELS:
        count = len(usable[usable['cefr_mapped'] == level])
        print(f"  {level}: {count:4d} essays")
    
    # Stratified sampling
    samples = []
    for level in CEFR_LEVELS:
        level_essays = usable[usable['cefr_mapped'] == level]
        sample = level_essays.sample(n=ESSAYS_PER_LEVEL, random_state=RANDOM_SEED)
        samples.append(sample)
        print(f"✓ Sampled {len(sample)} {level} essays")
    
    phase1_sample = pd.concat(samples, ignore_index=True)
    
    # Save
    output_file = PROCESSED_DIR / "phase1_sample_100.csv"
    phase1_sample.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    # Statistics
    print("\n" + "="*70)
    print("SAMPLE STATISTICS")
    print("="*70)
    print(f"\nTotal essays: {len(phase1_sample)}")
    print(f"CEFR distribution: {phase1_sample['cefr_mapped'].value_counts().sort_index().to_dict()}")
    
    word_counts = phase1_sample['text'].str.split().str.len()
    print(f"\nWord count:")
    print(f"  Mean: {word_counts.mean():.1f}")
    print(f"  Range: {word_counts.min()}-{word_counts.max()}")
    print(f"\nLanguages: {phase1_sample['language'].nunique()}")

if __name__ == "__main__":
    main()
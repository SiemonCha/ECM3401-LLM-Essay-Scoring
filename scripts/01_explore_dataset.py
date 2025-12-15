# scripts/01_explore_dataset.py
import pandas as pd
import sys
from config import CORPUS_FILE, CEFR_LEVELS

print("="*70)
print("DATASET EXPLORATION")
print("="*70)

# Load corpus
corpus = pd.read_csv(CORPUS_FILE, sep='\t')
print(f"\n✓ Loaded {len(corpus):,} essays")

# Show columns
print(f"\nColumns ({len(corpus.columns)}):")
for i, col in enumerate(corpus.columns, 1):
    print(f"  {i:2d}. {col}")

# Filter usable data (final versions with human labels)
usable = corpus[
    (corpus['is_final_version'] == True) &
    (corpus['humannotator_cefr_level'].notna()) &
    (corpus['split'].isin(['train', 'dev']))
]

print(f"\n✓ Usable for research: {len(usable):,} essays")

# CEFR distribution
print("\nCEFR distribution:")
print(usable['humannotator_cefr_level'].value_counts().sort_index())

# Check if enough for sampling
print("\nPhase 1 sampling check (need 20 per level):")
for level in CEFR_LEVELS:
    count = len(usable[usable['humannotator_cefr_level'] == level])
    status = "✓" if count >= 20 else "⚠️"
    print(f"  {status} {level}: {count:4d} essays")

# L1 languages
print(f"\nL1 languages: {usable['language'].nunique()}")
print(usable['language'].value_counts().head(10))
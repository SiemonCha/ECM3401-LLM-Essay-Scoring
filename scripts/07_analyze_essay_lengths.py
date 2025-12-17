# scripts/07_analyze_essay_lengths.py
"""
Analyze essay lengths in Phase 1 sample
Categorize into Short/Medium/Long for stratified analysis
Run: python scripts/07_analyze_essay_lengths.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import PROCESSED_DIR, FIGURES_DIR

def categorize_length(word_count):
    """Categorize essay by length"""
    if word_count < 100:
        return 'Short'
    elif word_count < 200:
        return 'Medium'
    else:
        return 'Long'

def main():
    print("="*70)
    print("ESSAY LENGTH ANALYSIS")
    print("="*70)
    
    # Load sample
    sample = pd.read_csv(PROCESSED_DIR / "phase1_sample_100.csv")
    
    # Calculate word counts
    sample['word_count'] = sample['text'].str.split().str.len()
    sample['length_category'] = sample['word_count'].apply(categorize_length)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Total essays: {len(sample)}")
    print(f"  Min length: {sample['word_count'].min()} words")
    print(f"  Max length: {sample['word_count'].max()} words")
    print(f"  Mean: {sample['word_count'].mean():.1f} words")
    print(f"  Median: {sample['word_count'].median():.1f} words")
    print(f"  Std Dev: {sample['word_count'].std():.1f} words")
    
    # By category
    print("\nBy Category:")
    print(f"  Short (<100 words): {(sample['length_category'] == 'Short').sum()} essays")
    print(f"  Medium (100-200): {(sample['length_category'] == 'Medium').sum()} essays")
    print(f"  Long (200+): {(sample['length_category'] == 'Long').sum()} essays")
    
    # By CEFR level
    print("\nBy CEFR Level:")
    for level in ['A2', 'B1', 'B2', 'C1', 'C2']:
        level_data = sample[sample['cefr_mapped'] == level]
        if len(level_data) > 0:
            print(f"  {level}: {len(level_data)} essays, "
                  f"mean length = {level_data['word_count'].mean():.1f} words")
    
    # Cross-tabulation
    print("\nLength Category × CEFR Level:")
    crosstab = pd.crosstab(sample['length_category'], sample['cefr_mapped'])
    print(crosstab)
    
    # Check if we have enough essays in each category
    print("\n" + "="*70)
    print("SAMPLE SIZE ASSESSMENT")
    print("="*70)
    
    min_per_category = crosstab.min().min()
    total_per_category = crosstab.sum(axis=1)
    
    print(f"\nEssays per length category:")
    for cat in ['Short', 'Medium', 'Long']:
        if cat in total_per_category.index:
            count = total_per_category[cat]
            print(f"  {cat}: {count} essays")
            
            # Check CEFR distribution
            if cat in crosstab.index:
                levels_with_data = (crosstab.loc[cat] > 0).sum()
                print(f"    Covers {levels_with_data}/5 CEFR levels")
    
    # Statistical recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    if min_per_category >= 3:
        print("✓ Each length×CEFR combination has ≥ 3 essays")
        print("✓ Sufficient for stratified analysis!")
        print("\nYou can analyze robustness separately for:")
        print("  - Short essays (<100 words)")
        print("  - Medium essays (100-200 words)")
        print("  - Long essays (200+ words)")
    else:
        print("⚠️ Some length×CEFR combinations have < 3 essays")
        print("Consider combining categories or noting as limitation")
    
    # Save categorized sample
    output_file = PROCESSED_DIR / "phase1_sample_with_lengths.csv"
    sample.to_csv(output_file, index=False)
    print(f"\n✓ Saved categorized sample to: {output_file}")
    
    # Create visualization
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Distribution of word counts
    axes[0].hist(sample['word_count'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(100, color='red', linestyle='--', label='Short/Medium')
    axes[0].axvline(200, color='red', linestyle='--', label='Medium/Long')
    axes[0].set_xlabel('Word Count')
    axes[0].set_ylabel('Number of Essays')
    axes[0].set_title('Distribution of Essay Lengths')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Box plot by CEFR level
    order = ['A2', 'B1', 'B2', 'C1', 'C2']
    present_levels = [l for l in order if l in sample['cefr_mapped'].values]
    sns.boxplot(data=sample, x='cefr_mapped', y='word_count', 
                order=present_levels, ax=axes[1])
    axes[1].set_xlabel('CEFR Level')
    axes[1].set_ylabel('Word Count')
    axes[1].set_title('Essay Length by CEFR Level')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = FIGURES_DIR / "essay_length_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {plot_file}")
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the crosstab above")
    print("2. If stratified analysis is feasible, I'll create the experiment script")
    print("3. Results will show robustness for each length category separately")

if __name__ == "__main__":
    main()
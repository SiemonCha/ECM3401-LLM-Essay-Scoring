# scripts/11_additional_analysis.py
"""
Additional Analysis for ECM3401 Project
Creates confusion matrices, prediction bias analysis, and detailed breakdowns

Analyses:
1. Confusion matrices (which levels are confused?)
2. Prediction bias (over/under prediction)
3. Per-CEFR level accuracy
4. Strategy effectiveness by CEFR level
5. Essay length vs accuracy correlation

Run: python scripts/11_additional_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
from pathlib import Path

from config import RESULTS_DIR, FIGURES_DIR, TABLES_DIR, CEFR_LEVELS

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_FILE = RESULTS_DIR / "full_experiment_results.csv"

# Plot styling
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    """Load results and clean"""
    print("="*70)
    print("ADDITIONAL ANALYSIS - ECM3401 PROJECT")
    print("="*70)
    
    print("\nLoading results...")
    if not RESULTS_FILE.exists():
        print(f"❌ Results file not found: {RESULTS_FILE}")
        print("Run script 08 first!")
        return None
    
    df = pd.read_csv(RESULTS_FILE)
    
    # Remove errors
    errors = df[df['prediction'] == 'ERROR']
    if len(errors) > 0:
        print(f"⚠️ Removing {len(errors)} ERROR predictions")
        df = df[df['prediction'] != 'ERROR']
    
    print(f"✓ Loaded {len(df)} valid predictions")
    print(f"  Essays: {df['essay_id'].nunique()}")
    print(f"  Models: {df['model'].nunique()}")
    
    return df

# =============================================================================
# ANALYSIS 1: CONFUSION MATRICES
# =============================================================================

def create_confusion_matrices(df):
    """Create confusion matrices for both models"""
    
    print("\n" + "="*70)
    print("CONFUSION MATRICES")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = ['gpt-4o-mini', 'phi-3-mini']
    titles = ['GPT-4o-mini', 'Phi-3-Mini']
    
    for ax, model, title in zip(axes, models, titles):
        model_df = df[df['model'] == model]
        
        # Confusion matrix
        cm = confusion_matrix(
            model_df['true_label'], 
            model_df['prediction'],
            labels=CEFR_LEVELS
        )
        
        # Calculate percentages
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both counts and percentages
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
        
        # Plot
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                   xticklabels=CEFR_LEVELS, yticklabels=CEFR_LEVELS,
                   cbar_kws={'label': 'Count'})
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted CEFR Level', fontweight='bold')
        ax.set_ylabel('True CEFR Level', fontweight='bold')
        
        # Print diagonal accuracy (per-class)
        print(f"\n{title}:")
        for i, level in enumerate(CEFR_LEVELS):
            correct = cm[i, i]
            total = cm[i, :].sum()
            acc = (correct / total * 100) if total > 0 else 0
            print(f"  {level}: {correct}/{total} = {acc:.1f}%")
    
    plt.suptitle('Confusion Matrices: True vs Predicted CEFR Levels', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = FIGURES_DIR / f"7_confusion_matrices.{FIGURE_FORMAT}"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {output_file}")
    
    # Save numerical confusion matrices
    for model in models:
        model_df = df[df['model'] == model]
        cm = confusion_matrix(
            model_df['true_label'],
            model_df['prediction'],
            labels=CEFR_LEVELS
        )
        cm_df = pd.DataFrame(cm, index=CEFR_LEVELS, columns=CEFR_LEVELS)
        cm_file = TABLES_DIR / f"confusion_matrix_{model.replace('-', '_')}.csv"
        cm_df.to_csv(cm_file)
        print(f"✓ Saved: {cm_file}")

# =============================================================================
# ANALYSIS 2: PREDICTION BIAS
# =============================================================================

def analyze_prediction_bias(df):
    """Analyze if models over/under predict certain levels"""
    
    print("\n" + "="*70)
    print("PREDICTION BIAS ANALYSIS")
    print("="*70)
    
    # Level to numeric mapping
    level_map = {'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
    
    bias_results = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        print(f"\n{model.upper()}:")
        
        # Overall distribution
        print("\n  Prediction Distribution:")
        pred_counts = model_df['prediction'].value_counts().sort_index()
        true_counts = model_df.groupby('essay_id')['true_label'].first().value_counts().sort_index()
        
        for level in CEFR_LEVELS:
            pred_count = pred_counts.get(level, 0)
            true_count = true_counts.get(level, 0) * 9  # 9 predictions per essay
            diff = pred_count - true_count
            print(f"    {level}: Predicted={pred_count}, Expected={true_count}, Diff={diff:+d}")
        
        # Numeric bias
        true_numeric = model_df['true_label'].map(level_map)
        pred_numeric = model_df['prediction'].map(level_map)
        
        bias = (pred_numeric - true_numeric).mean()
        bias_std = (pred_numeric - true_numeric).std()
        
        print(f"\n  Numeric Bias:")
        print(f"    Mean: {bias:+.3f} CEFR levels")
        print(f"    Std:  {bias_std:.3f}")
        
        if bias > 0.1:
            print(f"    → Tends to OVERPREDICT (assigns higher levels)")
        elif bias < -0.1:
            print(f"    → Tends to UNDERPREDICT (assigns lower levels)")
        else:
            print(f"    → Relatively UNBIASED")
        
        # Most confused pairs
        print(f"\n  Most Common Misclassifications:")
        misclass = model_df[model_df['true_label'] != model_df['prediction']]
        if len(misclass) > 0:
            confusions = misclass.groupby(['true_label', 'prediction']).size().sort_values(ascending=False)
            for (true, pred), count in confusions.head(5).items():
                pct = count / len(model_df[model_df['true_label'] == true]) * 100
                print(f"    {true} → {pred}: {count} times ({pct:.1f}%)")
        
        bias_results.append({
            'model': model,
            'mean_bias': bias,
            'std_bias': bias_std,
            'accuracy': (model_df['true_label'] == model_df['prediction']).mean() * 100
        })
    
    # Save bias summary
    bias_df = pd.DataFrame(bias_results)
    bias_file = TABLES_DIR / "prediction_bias_summary.csv"
    bias_df.to_csv(bias_file, index=False)
    print(f"\n✓ Saved: {bias_file}")

# =============================================================================
# ANALYSIS 3: PER-CEFR ACCURACY
# =============================================================================

def analyze_per_cefr_accuracy(df):
    """Detailed accuracy breakdown by CEFR level"""
    
    print("\n" + "="*70)
    print("PER-CEFR LEVEL ACCURACY")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    results = []
    
    for idx, (model, strategy) in enumerate([
        ('gpt-4o-mini', 'minimal'),
        ('gpt-4o-mini', 'rubric'),
        ('gpt-4o-mini', 'cot'),
        ('phi-3-mini', 'minimal'),
        ('phi-3-mini', 'rubric'),
        ('phi-3-mini', 'cot')
    ]):
        subset = df[(df['model'] == model) & (df['strategy'] == strategy)]
        
        accuracies = []
        for level in CEFR_LEVELS:
            level_data = subset[subset['true_label'] == level]
            if len(level_data) > 0:
                acc = (level_data['true_label'] == level_data['prediction']).mean() * 100
            else:
                acc = 0
            accuracies.append(acc)
            
            results.append({
                'model': model,
                'strategy': strategy,
                'cefr_level': level,
                'accuracy': acc,
                'n_predictions': len(level_data)
            })
        
        # Plot
        ax = axes[idx]
        bars = ax.bar(CEFR_LEVELS, accuracies, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Color code by accuracy
        for bar, acc in zip(bars, accuracies):
            if acc >= 50:
                bar.set_color('green')
            elif acc >= 30:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.axhline(20, color='gray', linestyle='--', alpha=0.5, label='Random (20%)')
        ax.set_ylim(0, 100)
        ax.set_xlabel('CEFR Level', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title(f"{model}\n{strategy}", fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{acc:.0f}%', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Accuracy by CEFR Level', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = FIGURES_DIR / f"8_accuracy_by_cefr.{FIGURE_FORMAT}"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Created: {output_file}")
    
    # Save table
    results_df = pd.DataFrame(results)
    results_file = TABLES_DIR / "per_cefr_accuracy.csv"
    results_df.to_csv(results_file, index=False)
    print(f"✓ Saved: {results_file}")

# =============================================================================
# ANALYSIS 4: LENGTH VS ACCURACY CORRELATION
# =============================================================================

def analyze_length_accuracy_correlation(df):
    """Analyze relationship between essay length and accuracy"""
    
    print("\n" + "="*70)
    print("LENGTH VS ACCURACY CORRELATION")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    models = ['gpt-4o-mini', 'phi-3-mini']
    titles = ['GPT-4o-mini', 'Phi-3-Mini']
    
    correlations = []
    
    for ax, model, title in zip(axes, models, titles):
        model_df = df[df['model'] == model]
        
        # Group by essay to get one accuracy per essay
        essay_stats = []
        for essay_id, essay_group in model_df.groupby('essay_id'):
            word_count = essay_group['word_count'].iloc[0]
            true_label = essay_group['true_label'].iloc[0]
            
            # Accuracy: are predictions correct?
            correct = (essay_group['prediction'] == true_label).sum()
            total = len(essay_group)
            accuracy = (correct / total) * 100
            
            essay_stats.append({
                'word_count': word_count,
                'accuracy': accuracy,
                'true_label': true_label
            })
        
        essay_df = pd.DataFrame(essay_stats)
        
        # Scatter plot
        for level, color in zip(CEFR_LEVELS, sns.color_palette("husl", len(CEFR_LEVELS))):
            level_data = essay_df[essay_df['true_label'] == level]
            ax.scatter(level_data['word_count'], level_data['accuracy'],
                      alpha=0.6, s=50, label=level, color=color)
        
        # Correlation
        corr, p_value = stats.pearsonr(essay_df['word_count'], essay_df['accuracy'])
        
        # Trend line
        z = np.polyfit(essay_df['word_count'], essay_df['accuracy'], 1)
        p = np.poly1d(z)
        ax.plot(essay_df['word_count'].sort_values(), 
               p(essay_df['word_count'].sort_values()),
               "r--", alpha=0.8, linewidth=2, label=f'Trend (r={corr:.3f})')
        
        ax.set_xlabel('Essay Length (words)', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title(f'{title}\nr = {corr:.3f}, p = {p_value:.4f}', fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
        
        correlations.append({
            'model': model,
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
        
        print(f"\n{title}:")
        print(f"  Correlation: r = {corr:.3f}")
        print(f"  P-value: {p_value:.4f}")
        if p_value < 0.05:
            if corr > 0:
                print(f"  → Significant POSITIVE correlation (longer = more accurate)")
            else:
                print(f"  → Significant NEGATIVE correlation (longer = less accurate)")
        else:
            print(f"  → No significant correlation")
    
    plt.suptitle('Essay Length vs Prediction Accuracy', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = FIGURES_DIR / f"9_length_accuracy_correlation.{FIGURE_FORMAT}"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Created: {output_file}")
    
    # Save correlations
    corr_df = pd.DataFrame(correlations)
    corr_file = TABLES_DIR / "length_accuracy_correlations.csv"
    corr_df.to_csv(corr_file, index=False)
    print(f"✓ Saved: {corr_file}")

# =============================================================================
# ANALYSIS 5: CLASSIFICATION REPORTS
# =============================================================================

def create_classification_reports(df):
    """Generate sklearn classification reports"""
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORTS")
    print("="*70)
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        print(f"\n{model.upper()}:")
        print("-" * 70)
        
        report = classification_report(
            model_df['true_label'],
            model_df['prediction'],
            labels=CEFR_LEVELS,
            target_names=CEFR_LEVELS,
            digits=3
        )
        
        print(report)
        
        # Save to file
        report_file = TABLES_DIR / f"classification_report_{model.replace('-', '_')}.txt"
        with open(report_file, 'w') as f:
            f.write(f"Classification Report: {model}\n")
            f.write("=" * 70 + "\n")
            f.write(report)
        
        print(f"✓ Saved: {report_file}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_additional_analysis():
    """Run all additional analyses"""
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Run analyses
    create_confusion_matrices(df)
    analyze_prediction_bias(df)
    analyze_per_cefr_accuracy(df)
    analyze_length_accuracy_correlation(df)
    create_classification_reports(df)
    
    # Summary
    print("\n" + "="*70)
    print("ADDITIONAL ANALYSIS COMPLETE!")
    print("="*70)
    
    print("\nGenerated Files:")
    print("\nFigures:")
    print("  7. outputs/figures/7_confusion_matrices.png")
    print("  8. outputs/figures/8_accuracy_by_cefr.png")
    print("  9. outputs/figures/9_length_accuracy_correlation.png")
    
    print("\nTables:")
    print("  - outputs/tables/confusion_matrix_*.csv (2 files)")
    print("  - outputs/tables/prediction_bias_summary.csv")
    print("  - outputs/tables/per_cefr_accuracy.csv")
    print("  - outputs/tables/length_accuracy_correlations.csv")
    print("  - outputs/tables/classification_report_*.txt (2 files)")
    
    print("\nAll additional analyses complete!")
    print("Total figures: 9 (including previous 6)")
    print("Ready for thesis inclusion!")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_additional_analysis()

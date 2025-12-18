# scripts/10_create_plots.py
"""
Visualization Creation for ECM3401 Project
Creates publication-quality plots for thesis

Plots created:
1. Robustness by strategy (bar chart)
2. Robustness by length category (grouped bar)
3. Model comparison (side-by-side bars)
4. Strategy × Length heatmap
5. Accuracy vs Robustness scatter
6. Per-essay variance distributions (box plots)

Run: python scripts/10_create_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import RESULTS_DIR, FIGURES_DIR, TABLES_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_FILE = RESULTS_DIR / "full_experiment_results.csv"
ROBUSTNESS_FILE = TABLES_DIR / "robustness_metrics.csv"

# Plot styling
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Colors
COLOR_GPT = '#FF6B6B'
COLOR_PHI3 = '#4ECDC4'
COLOR_MINIMAL = '#95E1D3'
COLOR_RUBRIC = '#F38181'
COLOR_COT = '#AA96DA'

ROBUSTNESS_THRESHOLD_GOOD = 3.0
ROBUSTNESS_THRESHOLD_MODERATE = 5.0

# =============================================================================
# PLOT 1: ROBUSTNESS BY STRATEGY (OVERALL)
# =============================================================================

def plot_robustness_by_strategy(robustness_df, output_dir):
    """Bar chart: Robustness by strategy"""
    
    # Filter overall only
    overall = robustness_df[robustness_df['length_category'] == 'overall'].copy()
    
    if len(overall) == 0:
        print("⚠️ No overall data for robustness plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pivot for grouped bar chart
    plot_data = overall.pivot(index='strategy', columns='model', values='robustness_sd')
    
    # Create grouped bars
    plot_data.plot(kind='bar', ax=ax, width=0.7, 
                   color=[COLOR_GPT, COLOR_PHI3])
    
    # Threshold lines
    ax.axhline(ROBUSTNESS_THRESHOLD_GOOD, color='green', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Deployment-Ready (SD < {ROBUSTNESS_THRESHOLD_GOOD})')
    ax.axhline(ROBUSTNESS_THRESHOLD_MODERATE, color='orange', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Acceptable (SD < {ROBUSTNESS_THRESHOLD_MODERATE})')
    
    ax.set_xlabel('Prompting Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Robustness (SD across paraphrases)', fontsize=12, fontweight='bold')
    ax.set_title('Robustness by Prompting Strategy', fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Model', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_file = output_dir / f"1_robustness_by_strategy.{FIGURE_FORMAT}"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file}")

# =============================================================================
# PLOT 2: ROBUSTNESS BY LENGTH CATEGORY
# =============================================================================

def plot_robustness_by_length(robustness_df, output_dir):
    """Grouped bar chart: Robustness by length and strategy"""
    
    # Filter out overall
    stratified = robustness_df[robustness_df['length_category'] != 'overall'].copy()
    
    if len(stratified) == 0:
        print("⚠️ No stratified data for length plot")
        return
    
    # Create figure with subplots for each model
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    models = ['gpt-4o-mini', 'phi-3-mini']
    titles = ['GPT-4o-mini', 'Phi-3-Mini']
    
    for ax, model, title in zip(axes, models, titles):
        model_data = stratified[stratified['model'] == model]
        
        # Pivot
        plot_data = model_data.pivot(index='strategy', columns='length_category', 
                                      values='robustness_sd')
        
        # Reorder columns
        plot_data = plot_data[['short', 'medium', 'long']]
        
        # Plot
        plot_data.plot(kind='bar', ax=ax, width=0.8)
        
        # Threshold lines
        ax.axhline(ROBUSTNESS_THRESHOLD_GOOD, color='green', linestyle='--', 
                   linewidth=1.5, alpha=0.5)
        ax.axhline(ROBUSTNESS_THRESHOLD_MODERATE, color='orange', linestyle='--', 
                   linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('Strategy', fontsize=11, fontweight='bold')
        ax.set_ylabel('Robustness (SD)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(title='Length', labels=['Short (<100)', 'Medium (100-200)', 'Long (200+)'])
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Robustness by Essay Length Category', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / f"2_robustness_by_length.{FIGURE_FORMAT}"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file}")

# =============================================================================
# PLOT 3: MODEL COMPARISON
# =============================================================================

def plot_model_comparison(robustness_df, output_dir):
    """Side-by-side comparison of GPT vs Phi-3"""
    
    overall = robustness_df[robustness_df['length_category'] == 'overall'].copy()
    
    if len(overall) == 0:
        print("⚠️ No data for model comparison plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Prepare data
    gpt_data = overall[overall['model'] == 'gpt-4o-mini'].set_index('strategy')
    phi3_data = overall[overall['model'] == 'phi-3-mini'].set_index('strategy')
    
    strategies = gpt_data.index.tolist()
    x = np.arange(len(strategies))
    width = 0.35
    
    # Plot 1: Robustness comparison
    ax1.bar(x - width/2, gpt_data['robustness_sd'], width, 
            label='GPT-4o-mini', color=COLOR_GPT, alpha=0.8)
    ax1.bar(x + width/2, phi3_data['robustness_sd'], width, 
            label='Phi-3-Mini', color=COLOR_PHI3, alpha=0.8)
    
    ax1.axhline(ROBUSTNESS_THRESHOLD_GOOD, color='green', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Strategy', fontweight='bold')
    ax1.set_ylabel('Robustness (SD)', fontweight='bold')
    ax1.set_title('Robustness Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Accuracy comparison
    ax2.bar(x - width/2, gpt_data['accuracy_pct'], width, 
            label='GPT-4o-mini', color=COLOR_GPT, alpha=0.8)
    ax2.bar(x + width/2, phi3_data['accuracy_pct'], width, 
            label='Phi-3-Mini', color=COLOR_PHI3, alpha=0.8)
    
    ax2.set_xlabel('Strategy', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Accuracy Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('GPT-4o-mini vs Phi-3-Mini', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / f"3_model_comparison.{FIGURE_FORMAT}"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file}")

# =============================================================================
# PLOT 4: HEATMAP (STRATEGY × LENGTH)
# =============================================================================

def plot_heatmap(robustness_df, output_dir):
    """Heatmap of robustness across strategy and length"""
    
    stratified = robustness_df[robustness_df['length_category'] != 'overall'].copy()
    
    if len(stratified) == 0:
        print("⚠️ No data for heatmap")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = ['gpt-4o-mini', 'phi-3-mini']
    titles = ['GPT-4o-mini', 'Phi-3-Mini']
    
    for ax, model, title in zip(axes, models, titles):
        model_data = stratified[stratified['model'] == model]
        
        # Pivot for heatmap
        heatmap_data = model_data.pivot(index='strategy', columns='length_category', 
                                         values='robustness_sd')
        
        # Reorder
        heatmap_data = heatmap_data[['short', 'medium', 'long']]
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                    vmin=0, vmax=ROBUSTNESS_THRESHOLD_MODERATE*2,
                    cbar_kws={'label': 'Robustness (SD)'},
                    ax=ax, linewidths=0.5)
        
        ax.set_xlabel('Essay Length', fontweight='bold')
        ax.set_ylabel('Strategy', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xticklabels(['Short\n(<100)', 'Medium\n(100-200)', 'Long\n(200+)'], rotation=0)
    
    plt.suptitle('Robustness Heatmap: Strategy × Length', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / f"4_robustness_heatmap.{FIGURE_FORMAT}"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file}")

# =============================================================================
# PLOT 5: ACCURACY VS ROBUSTNESS SCATTER
# =============================================================================

def plot_accuracy_vs_robustness(robustness_df, output_dir):
    """Scatter plot: Accuracy vs Robustness"""
    
    overall = robustness_df[robustness_df['length_category'] == 'overall'].copy()
    
    if len(overall) == 0:
        print("⚠️ No data for scatter plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot by model
    for model, color in [('gpt-4o-mini', COLOR_GPT), ('phi-3-mini', COLOR_PHI3)]:
        model_data = overall[overall['model'] == model]
        
        ax.scatter(model_data['robustness_sd'], model_data['accuracy_pct'],
                  s=200, alpha=0.7, color=color, label=model,
                  edgecolors='black', linewidth=1.5)
        
        # Add strategy labels
        for _, row in model_data.iterrows():
            ax.annotate(row['strategy'], 
                       (row['robustness_sd'], row['accuracy_pct']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)
    
    # Threshold lines
    ax.axvline(ROBUSTNESS_THRESHOLD_GOOD, color='green', linestyle='--', 
               linewidth=2, alpha=0.5, label=f'Robust (SD < {ROBUSTNESS_THRESHOLD_GOOD})')
    
    ax.set_xlabel('Robustness (SD across paraphrases)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Robustness Trade-off', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Quadrants
    ax.axhline(overall['accuracy_pct'].median(), color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / f"5_accuracy_vs_robustness.{FIGURE_FORMAT}"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file}")

# =============================================================================
# PLOT 6: BOX PLOTS (VARIANCE DISTRIBUTIONS)
# =============================================================================

def plot_variance_distributions(results_df, output_dir):
    """Box plots showing distribution of per-essay variances"""
    
    # Calculate per-essay SD for each strategy
    essay_variances = []
    
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]
        
        for strategy in results_df['strategy'].unique():
            strategy_df = model_df[model_df['strategy'] == strategy]
            
            for essay_id, essay_group in strategy_df.groupby('essay_id'):
                predictions = essay_group['prediction'].values
                
                pred_numeric = []
                for p in predictions:
                    if p == 'A2': pred_numeric.append(1)
                    elif p == 'B1': pred_numeric.append(2)
                    elif p == 'B2': pred_numeric.append(3)
                    elif p == 'C1': pred_numeric.append(4)
                    elif p == 'C2': pred_numeric.append(5)
                    else: pred_numeric.append(np.nan)
                
                if not any(np.isnan(pred_numeric)) and len(pred_numeric) > 1:
                    sd = np.std(pred_numeric, ddof=1)
                    essay_variances.append({
                        'model': model,
                        'strategy': strategy,
                        'sd': sd
                    })
    
    variance_df = pd.DataFrame(essay_variances)
    
    if len(variance_df) == 0:
        print("⚠️ No data for box plots")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    models = ['gpt-4o-mini', 'phi-3-mini']
    titles = ['GPT-4o-mini', 'Phi-3-Mini']
    
    for ax, model, title in zip(axes, models, titles):
        model_data = variance_df[variance_df['model'] == model]
        
        # Box plot
        sns.boxplot(data=model_data, x='strategy', y='sd', ax=ax,
                   palette=[COLOR_MINIMAL, COLOR_RUBRIC, COLOR_COT])
        
        # Threshold line
        ax.axhline(ROBUSTNESS_THRESHOLD_GOOD, color='green', linestyle='--', 
                   linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Strategy', fontweight='bold')
        ax.set_ylabel('Per-Essay SD', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Distribution of Per-Essay Variance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / f"6_variance_distributions.{FIGURE_FORMAT}"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def create_all_plots():
    """Create all visualization plots"""
    
    print("="*70)
    print("VISUALIZATION CREATION - ECM3401 PROJECT")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    
    if not ROBUSTNESS_FILE.exists():
        print(f"❌ Robustness file not found: {ROBUSTNESS_FILE}")
        print("Run script 09 first!")
        return
    
    robustness_df = pd.read_csv(ROBUSTNESS_FILE)
    print(f"✓ Loaded robustness metrics: {len(robustness_df)} rows")
    
    if not RESULTS_FILE.exists():
        print(f"❌ Results file not found: {RESULTS_FILE}")
        return
    
    results_df = pd.read_csv(RESULTS_FILE)
    results_df = results_df[results_df['prediction'] != 'ERROR']  # Remove errors
    print(f"✓ Loaded results: {len(results_df)} predictions")
    
    # Create plots
    print("\n" + "="*70)
    print("CREATING PLOTS")
    print("="*70 + "\n")
    
    plot_robustness_by_strategy(robustness_df, FIGURES_DIR)
    plot_robustness_by_length(robustness_df, FIGURES_DIR)
    plot_model_comparison(robustness_df, FIGURES_DIR)
    plot_heatmap(robustness_df, FIGURES_DIR)
    plot_accuracy_vs_robustness(robustness_df, FIGURES_DIR)
    plot_variance_distributions(results_df, FIGURES_DIR)
    
    print("\n" + "="*70)
    print("ALL PLOTS CREATED!")
    print("="*70)
    
    print(f"\nPlots saved to: {FIGURES_DIR}")
    print("\nGenerated files:")
    print("  1. 1_robustness_by_strategy.png")
    print("  2. 2_robustness_by_length.png")
    print("  3. 3_model_comparison.png")
    print("  4. 4_robustness_heatmap.png")
    print("  5. 5_accuracy_vs_robustness.png")
    print("  6. 6_variance_distributions.png")
    
    print("\nThese plots are ready for inclusion in your thesis!")
    print("All figures saved at 300 DPI for publication quality.")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    create_all_plots()

#!/usr/bin/env python3
"""
SCRIPT 3: ANALYZE RESULTS
Complete analysis: metrics, plots, reports

Usage:
  python analyze.py --phase 1
  python analyze.py --phase 2

Time: 2-3 minutes
"""

import sys
from pathlib import Path

# Fix imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

from simple_config import *

# Setup plotting
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# =============================================================================
# PARSE ARGS
# =============================================================================

def parse_args():
    if len(sys.argv) < 3 or sys.argv[1] != '--phase':
        print("Usage: python analyze.py --phase [1|2]")
        sys.exit(1)
    
    phase = int(sys.argv[2])
    if phase not in [1, 2]:
        print("Error: Phase must be 1 or 2")
        sys.exit(1)
    
    return phase

# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(df):
    """Calculate robustness and accuracy metrics"""
    
    metrics = []
    
    for model in df['model'].unique():
        for strategy in df['strategy'].unique():
            subset = df[(df['model'] == model) & (df['strategy'] == strategy)]
            
            if len(subset) == 0:
                continue
            
            # Calculate robustness (SD across variants)
            robustness_scores = []
            
            for essay_id in subset['essay_id'].unique():
                essay_preds = subset[subset['essay_id'] == essay_id]['prediction'].values
                
                # Convert to numeric
                pred_numeric = []
                for p in essay_preds:
                    if p == 'A2': pred_numeric.append(1)
                    elif p == 'B1': pred_numeric.append(2)
                    elif p == 'B2': pred_numeric.append(3)
                    elif p == 'C1': pred_numeric.append(4)
                    elif p == 'C2': pred_numeric.append(5)
                
                if len(pred_numeric) >= 2:
                    sd = np.std(pred_numeric, ddof=1)
                    robustness_scores.append(sd)
            
            mean_sd = np.mean(robustness_scores) if robustness_scores else np.nan
            
            # Calculate accuracy
            correct = (subset['prediction'] == subset['true_label']).sum()
            accuracy = (correct / len(subset)) * 100
            
            # Adjacent accuracy (±1 level)
            adjacent = 0
            for _, row in subset.iterrows():
                pred = row['prediction']
                true = row['true_label']
                
                level_map = {'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
                pred_num = level_map.get(pred, 0)
                true_num = level_map.get(true, 0)
                
                if abs(pred_num - true_num) <= 1:
                    adjacent += 1
            
            adjacent_acc = (adjacent / len(subset)) * 100
            
            metrics.append({
                'model': model,
                'strategy': strategy,
                'robustness_sd': mean_sd,
                'accuracy': accuracy,
                'adjacent_accuracy': adjacent_acc,
                'n_predictions': len(subset)
            })
    
    return pd.DataFrame(metrics)

# =============================================================================
# PLOTTING
# =============================================================================

def create_plots(df, metrics_df, phase):
    """Create all plots"""
    
    print("\nCreating plots...")
    
    # Plot 1: Robustness by strategy
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = metrics_df.groupby('strategy')['robustness_sd'].mean().sort_values()
    ax.barh(range(len(strategies)), strategies.values, color='skyblue', edgecolor='black')
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies.index)
    ax.set_xlabel('Mean SD (CEFR levels)')
    ax.set_title(f'Phase {phase}: Robustness by Strategy')
    ax.axvline(0.5, color='green', linestyle='--', label='Very Robust (<0.5)')
    ax.axvline(1.0, color='orange', linestyle='--', label='Acceptable (<1.0)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"phase{phase}_robustness.png")
    plt.close()
    print(f"  ✓ phase{phase}_robustness.png")
    
    # Plot 2: Model comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Robustness
    model_rob = metrics_df.groupby('model')['robustness_sd'].mean()
    ax1.bar(range(len(model_rob)), model_rob.values, color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
    ax1.set_xticks(range(len(model_rob)))
    ax1.set_xticklabels(model_rob.index)
    ax1.set_ylabel('Mean SD (CEFR levels)')
    ax1.set_title('Robustness by Model')
    ax1.grid(axis='y', alpha=0.3)
    
    # Accuracy
    model_acc = metrics_df.groupby('model')['accuracy'].mean()
    ax2.bar(range(len(model_acc)), model_acc.values, color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
    ax2.set_xticks(range(len(model_acc)))
    ax2.set_xticklabels(model_acc.index)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy by Model')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Phase {phase}: Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"phase{phase}_models.png")
    plt.close()
    print(f"  ✓ phase{phase}_models.png")
    
    # Plot 3: Accuracy vs Robustness
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model in metrics_df['model'].unique():
        model_data = metrics_df[metrics_df['model'] == model]
        ax.scatter(model_data['robustness_sd'], model_data['accuracy'], 
                  label=model, s=100, alpha=0.7)
    
    ax.set_xlabel('Robustness (SD, lower = better)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Phase {phase}: Accuracy vs Robustness Tradeoff')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"phase{phase}_tradeoff.png")
    plt.close()
    print(f"  ✓ phase{phase}_tradeoff.png")
    
    print("✓ All plots created")

# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(df, metrics_df, phase):
    """Generate markdown report"""
    
    print("\nGenerating report...")
    
    report = []
    report.append(f"# Phase {phase} Results\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    # Summary statistics
    report.append("## Summary\n")
    report.append(f"- Total predictions: {len(df):,}")
    report.append(f"- Essays: {df['essay_id'].nunique()}")
    report.append(f"- Prompts: {df['prompt_name'].nunique()}")
    report.append(f"- Models: {', '.join(df['model'].unique())}\n")
    
    # Overall metrics
    report.append("## Overall Performance\n")
    report.append("| Model | Robustness (SD) | Accuracy | Adjacent Accuracy |")
    report.append("|-------|----------------|----------|-------------------|")
    
    for model in metrics_df['model'].unique():
        model_data = metrics_df[metrics_df['model'] == model]
        rob = model_data['robustness_sd'].mean()
        acc = model_data['accuracy'].mean()
        adj = model_data['adjacent_accuracy'].mean()
        report.append(f"| {model} | {rob:.3f} | {acc:.1f}% | {adj:.1f}% |")
    
    report.append("")
    
    # Best strategies
    report.append("## Best Strategies\n")
    report.append("### Most Robust (lowest SD):\n")
    
    best_robust = metrics_df.nsmallest(5, 'robustness_sd')
    for _, row in best_robust.iterrows():
        report.append(f"- **{row['strategy']}** ({row['model']}): SD = {row['robustness_sd']:.3f}")
    
    report.append("\n### Most Accurate:\n")
    best_accurate = metrics_df.nlargest(5, 'accuracy')
    for _, row in best_accurate.iterrows():
        report.append(f"- **{row['strategy']}** ({row['model']}): {row['accuracy']:.1f}%")
    
    report.append("")
    
    # Deployment readiness
    report.append("## Deployment Readiness\n")
    
    deployable = metrics_df[metrics_df['robustness_sd'] < 0.5]
    if len(deployable) > 0:
        report.append(f"✓ **{len(deployable)} strategies are deployment-ready** (SD < 0.5)\n")
        for _, row in deployable.iterrows():
            report.append(f"- {row['strategy']} ({row['model']}): SD = {row['robustness_sd']:.3f}, Acc = {row['accuracy']:.1f}%")
    else:
        report.append("⚠️ No strategies meet deployment threshold (SD < 0.5)\n")
    
    # Save report
    report_file = TABLES_DIR / f"phase{phase}_report.md"
    report_file.write_text('\n'.join(report))
    print(f"✓ Report saved: {report_file}")

# =============================================================================
# MAIN
# =============================================================================

def analyze(phase):
    """Complete analysis"""
    
    print("="*70)
    print(f"PHASE {phase} ANALYSIS")
    print("="*70)
    
    # Load results
    results_file = PHASE1_RESULTS if phase == 1 else PHASE2_RESULTS
    
    if not results_file.exists():
        print(f"\n❌ Results not found: {results_file}")
        print(f"Run: python run_experiment.py --phase {phase}")
        sys.exit(1)
    
    print(f"\nLoading results...")
    df = pd.read_csv(results_file)
    print(f"✓ {len(df):,} predictions")
    
    # Remove errors
    errors = (df['prediction'] == 'ERROR').sum()
    if errors > 0:
        print(f"⚠️  Excluding {errors} ERROR predictions")
        df = df[df['prediction'] != 'ERROR']
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics_df = calculate_metrics(df)
    print(f"✓ Metrics calculated")
    
    # Save metrics
    metrics_file = PHASE1_METRICS if phase == 1 else PHASE2_METRICS
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ Saved: {metrics_file}")
    
    # Create plots
    create_plots(df, metrics_df, phase)
    
    # Generate report
    generate_report(df, metrics_df, phase)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nOverall Performance:")
    for model in metrics_df['model'].unique():
        model_data = metrics_df[metrics_df['model'] == model]
        rob = model_data['robustness_sd'].mean()
        acc = model_data['accuracy'].mean()
        print(f"  {model}:")
        print(f"    Robustness: {rob:.3f} SD")
        print(f"    Accuracy: {acc:.1f}%")
    
    print("\n✓ Analysis complete!")
    print(f"\nOutputs:")
    print(f"  Metrics: {metrics_file}")
    print(f"  Plots: {FIGURES_DIR}/phase{phase}_*.png")
    print(f"  Report: {TABLES_DIR}/phase{phase}_report.md")

if __name__ == "__main__":
    try:
        phase = parse_args()
        analyze(phase)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
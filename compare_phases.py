#!/usr/bin/env python3
"""
SCRIPT 4: COMPARE PHASES
Compare Phase 1 vs Phase 2 results

Usage:
  python compare_phases.py

Time: 2 minutes
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

from simple_config import *

# =============================================================================
# LOAD DATA
# =============================================================================

def load_both_phases():
    """Load Phase 1 and Phase 2 results"""
    
    print("Loading data...")
    
    if not PHASE1_METRICS.exists():
        print(f"\n❌ Phase 1 metrics not found")
        print("Run: python analyze.py --phase 1")
        sys.exit(1)
    
    if not PHASE2_METRICS.exists():
        print(f"\n❌ Phase 2 metrics not found")
        print("Run: python analyze.py --phase 2")
        sys.exit(1)
    
    phase1 = pd.read_csv(PHASE1_METRICS)
    phase1['phase'] = 1
    
    phase2 = pd.read_csv(PHASE2_METRICS)
    phase2['phase'] = 2
    
    print(f"✓ Phase 1: {len(phase1)} strategies")
    print(f"✓ Phase 2: {len(phase2)} strategies")
    
    return phase1, phase2

# =============================================================================
# COMPARISON METRICS
# =============================================================================

def compare_metrics(phase1, phase2):
    """Compare metrics between phases"""
    
    print("\nComparing metrics...")
    
    comparison = []
    
    # Overall comparison
    p1_rob = phase1['robustness_sd'].mean()
    p2_rob = phase2['robustness_sd'].mean()
    p1_acc = phase1['accuracy'].mean()
    p2_acc = phase2['accuracy'].mean()
    
    comparison.append({
        'metric': 'Overall Robustness (SD)',
        'phase1': p1_rob,
        'phase2': p2_rob,
        'improvement': p1_rob - p2_rob,  # Lower is better
        'better': 'Phase 2' if p2_rob < p1_rob else 'Phase 1'
    })
    
    comparison.append({
        'metric': 'Overall Accuracy (%)',
        'phase1': p1_acc,
        'phase2': p2_acc,
        'improvement': p2_acc - p1_acc,  # Higher is better
        'better': 'Phase 2' if p2_acc > p1_acc else 'Phase 1'
    })
    
    # By strategy
    for strategy in ['minimal', 'rubric', 'cot']:
        p1_strat = phase1[phase1['strategy'] == strategy]
        p2_strat = phase2[phase2['strategy'] == strategy]
        
        if len(p1_strat) > 0 and len(p2_strat) > 0:
            p1_rob = p1_strat['robustness_sd'].mean()
            p2_rob = p2_strat['robustness_sd'].mean()
            
            comparison.append({
                'metric': f'{strategy.capitalize()} Robustness',
                'phase1': p1_rob,
                'phase2': p2_rob,
                'improvement': p1_rob - p2_rob,
                'better': 'Phase 2' if p2_rob < p1_rob else 'Phase 1'
            })
    
    return pd.DataFrame(comparison)

# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def statistical_tests(phase1, phase2):
    """Run statistical tests"""
    
    print("\nRunning statistical tests...")
    
    tests = []
    
    # Test robustness difference
    t_stat, p_value = stats.ttest_ind(
        phase1['robustness_sd'].dropna(),
        phase2['robustness_sd'].dropna()
    )
    
    tests.append({
        'test': 'Robustness (Phase 1 vs Phase 2)',
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'result': 'Phase 2 more robust' if p_value < 0.05 and t_stat > 0 else 'No significant difference'
    })
    
    # Test accuracy difference
    t_stat, p_value = stats.ttest_ind(
        phase1['accuracy'].dropna(),
        phase2['accuracy'].dropna()
    )
    
    tests.append({
        'test': 'Accuracy (Phase 1 vs Phase 2)',
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'result': 'Phase 2 more accurate' if p_value < 0.05 and t_stat < 0 else 'No significant difference'
    })
    
    return pd.DataFrame(tests)

# =============================================================================
# PLOTS
# =============================================================================

def create_comparison_plots(phase1, phase2):
    """Create comparison plots"""
    
    print("\nCreating comparison plots...")
    
    # Plot 1: Side-by-side robustness
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Robustness
    strategies = phase1['strategy'].unique()
    x = np.arange(len(strategies))
    width = 0.35
    
    p1_rob = [phase1[phase1['strategy'] == s]['robustness_sd'].mean() for s in strategies]
    p2_rob = [phase2[phase2['strategy'] == s]['robustness_sd'].mean() for s in strategies]
    
    ax1.bar(x - width/2, p1_rob, width, label='Phase 1', color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax1.bar(x + width/2, p2_rob, width, label='Phase 2', color='#4ECDC4', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Robustness (SD)')
    ax1.set_title('Robustness Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Accuracy
    p1_acc = [phase1[phase1['strategy'] == s]['accuracy'].mean() for s in strategies]
    p2_acc = [phase2[phase2['strategy'] == s]['accuracy'].mean() for s in strategies]
    
    ax2.bar(x - width/2, p1_acc, width, label='Phase 1', color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax2.bar(x + width/2, p2_acc, width, label='Phase 2', color='#4ECDC4', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Phase 1 vs Phase 2 Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase_comparison.png")
    plt.close()
    print("  ✓ phase_comparison.png")
    
    # Plot 2: Improvement matrix
    fig, ax = plt.subplots(figsize=(10, 6))
    
    combined = pd.concat([phase1, phase2])
    pivot = combined.pivot_table(
        values='robustness_sd',
        index='strategy',
        columns='phase',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Robustness (SD)'})
    ax.set_title('Robustness by Strategy and Phase')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Strategy')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase_heatmap.png")
    plt.close()
    print("  ✓ phase_heatmap.png")
    
    print("✓ Comparison plots created")

# =============================================================================
# REPORT
# =============================================================================

def generate_comparison_report(comparison_df, tests_df):
    """Generate comparison report"""
    
    print("\nGenerating comparison report...")
    
    report = []
    report.append("# Phase 1 vs Phase 2 Comparison\n")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    # Summary
    report.append("## Key Findings\n")
    
    for _, row in comparison_df.iterrows():
        metric = row['metric']
        p1 = row['phase1']
        p2 = row['phase2']
        imp = row['improvement']
        better = row['better']
        
        if 'Robustness' in metric:
            report.append(f"- **{metric}**: Phase 1 = {p1:.3f}, Phase 2 = {p2:.3f} ({better} wins, Δ = {abs(imp):.3f})")
        else:
            report.append(f"- **{metric}**: Phase 1 = {p1:.1f}%, Phase 2 = {p2:.1f}% ({better} wins, Δ = {abs(imp):.1f}%)")
    
    report.append("\n## Statistical Significance\n")
    
    for _, row in tests_df.iterrows():
        test = row['test']
        p_val = row['p_value']
        sig = row['significant']
        result = row['result']
        
        sig_mark = "✓ Significant" if sig else "✗ Not significant"
        report.append(f"- **{test}**: p = {p_val:.4f} ({sig_mark})")
        report.append(f"  → {result}\n")
    
    # Recommendations
    report.append("## Recommendations\n")
    
    # Find best strategies
    overall_best = comparison_df[comparison_df['metric'] == 'Overall Robustness (SD)'].iloc[0]
    if overall_best['better'] == 'Phase 2':
        report.append("✓ **Phase 2 prompts are more robust overall**")
        report.append("  → Use Phase 2 prompts for deployment\n")
    else:
        report.append("⚠️ **Phase 1 prompts were more robust**")
        report.append("  → Consider refining Phase 2 approach\n")
    
    # Save
    report_file = TABLES_DIR / "comparison_report.md"
    report_file.write_text('\n'.join(report))
    print(f"✓ Report saved: {report_file}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Compare phases"""
    
    print("="*70)
    print("PHASE 1 VS PHASE 2 COMPARISON")
    print("="*70)
    
    # Load data
    phase1, phase2 = load_both_phases()
    
    # Compare metrics
    comparison_df = compare_metrics(phase1, phase2)
    
    # Statistical tests
    tests_df = statistical_tests(phase1, phase2)
    
    # Save comparison
    comparison_df.to_csv(COMPARISON, index=False)
    print(f"\n✓ Saved comparison: {COMPARISON}")
    
    # Create plots
    create_comparison_plots(phase1, phase2)
    
    # Generate report
    generate_comparison_report(comparison_df, tests_df)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nKey Findings:")
    for _, row in comparison_df.head(2).iterrows():
        print(f"  {row['metric']}: {row['better']} wins")
    
    print("\nStatistical Tests:")
    for _, row in tests_df.iterrows():
        sig = "✓" if row['significant'] else "✗"
        print(f"  {sig} {row['test']}: {row['result']}")
    
    print("\n✓ Comparison complete!")
    print(f"\nOutputs:")
    print(f"  Comparison: {COMPARISON}")
    print(f"  Plots: {FIGURES_DIR}/phase_*.png")
    print(f"  Report: {TABLES_DIR}/comparison_report.md")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

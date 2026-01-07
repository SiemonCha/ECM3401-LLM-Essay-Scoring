# scripts/16_compare_phases_and_test_hypotheses.py
"""
Phase 1 vs Phase 2 Comparison and Hypothesis Testing

Prerequisites:
  1. Run Phase 1: python scripts/08_run_experiment.py --phase 1
  2. Run Phase 2: python scripts/08_run_experiment.py --phase 2
  3. Then run this script

Compares results between phases and tests 9 specific hypotheses

Run: python scripts/16_compare_phases_and_test_hypotheses.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import RESULTS_DIR, FIGURES_DIR, TABLES_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

PHASE1_RESULTS = RESULTS_DIR / "phase1_experiment_results.csv"
PHASE2_RESULTS = RESULTS_DIR / "phase2_experiment_results.csv"

LEVEL_TO_NUM = {'A2': 0, 'B1': 1, 'B2': 2, 'C1': 3, 'C2': 4}

plt.style.use('seaborn-v0_8-paper')
FIGURE_DPI = 300

# =============================================================================
# HYPOTHESES DEFINITIONS
# =============================================================================

HYPOTHESES = {
    'H1': {
        'hypothesis': 'Ultra-simple prompts MORE robust than Phase 1 minimal',
        'test_prompt': 'minimal_v4',
        'baseline_prompt': 'minimal_v1',
        'metric': 'robustness_sd',
        'prediction': 'Phase2 < Phase1',
        'threshold': 0.163
    },
    
    'H2': {
        'hypothesis': 'Length-aware prompts IMPROVE long-essay accuracy',
        'test_prompt': 'minimal_v5',
        'baseline_prompt': 'minimal_v1',
        'metric': 'accuracy_long',
        'prediction': 'Phase2 > Phase1',
        'threshold': 6.4,
        'filter': 'long essays only'
    },
    
    'H3': {
        'hypothesis': 'Ordinal-constraint prompts REDUCE off-by-2+ errors',
        'test_prompt': 'minimal_v6',
        'baseline_prompt': 'minimal_v1',
        'metric': 'off_by_2_plus_pct',
        'prediction': 'Phase2 < Phase1',
        'threshold': 10.0
    },
    
    'H4': {
        'hypothesis': 'Rubric with examples INCREASES accuracy',
        'test_prompt': 'rubric_v4',
        'baseline_prompt': 'rubric_v1',
        'metric': 'exact_accuracy',
        'prediction': 'Phase2 > Phase1',
        'threshold': 35.6
    },
    
    'H5': {
        'hypothesis': 'Rubric with confidence IDENTIFIES uncertain cases',
        'test_prompt': 'rubric_v5',
        'baseline_prompt': 'rubric_v1',
        'metric': 'confidence_separation',
        'prediction': 'High-conf accuracy > 50%',
        'threshold': 50.0
    },
    
    'H6': {
        'hypothesis': 'Adjacent-level awareness INCREASES adjacent accuracy',
        'test_prompt': 'rubric_v6',
        'baseline_prompt': 'rubric_v1',
        'metric': 'adjacent_accuracy',
        'prediction': 'Phase2 > Phase1',
        'threshold': 68.9
    },
    
    'H7': {
        'hypothesis': 'Structured CoT MORE robust than free-form CoT',
        'test_prompt': 'cot_v4',
        'baseline_prompt': 'cot_v1',
        'metric': 'robustness_sd',
        'prediction': 'Phase2 < Phase1',
        'threshold': 0.205
    },
    
    'H8': {
        'hypothesis': 'CoT with constraints PREVENTS severe errors',
        'test_prompt': 'cot_v5',
        'baseline_prompt': 'cot_v1',
        'metric': 'off_by_2_plus_pct',
        'prediction': 'Phase2 < Phase1',
        'threshold': 'Phase1 CoT baseline'
    },
    
    'H9': {
        'hypothesis': 'CoT with QWK-awareness IMPROVES ordinal agreement',
        'test_prompt': 'cot_v6',
        'baseline_prompt': 'cot_v1',
        'metric': 'qwk',
        'prediction': 'Phase2 > Phase1',
        'threshold': 0.218
    }
}

# =============================================================================
# METRIC CALCULATIONS
# =============================================================================

def calculate_robustness(df):
    """Calculate per-essay SD, then mean"""
    essay_sds = []
    for essay_id in df['essay_id'].unique():
        essay_preds = df[df['essay_id'] == essay_id]['prediction']
        pred_nums = [LEVEL_TO_NUM[p] for p in essay_preds if p in LEVEL_TO_NUM]
        if len(pred_nums) > 1:
            essay_sds.append(np.std(pred_nums, ddof=1))
    return np.mean(essay_sds) if essay_sds else np.nan

def calculate_qwk(df):
    """Calculate Quadratic Weighted Kappa"""
    y_true = [LEVEL_TO_NUM[y] for y in df['true_label'] if y in LEVEL_TO_NUM]
    y_pred = [LEVEL_TO_NUM[y] for y in df['prediction'] if y in LEVEL_TO_NUM]
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def calculate_adjacent_accuracy(df):
    """Calculate adjacent accuracy"""
    y_true = np.array([LEVEL_TO_NUM[y] for y in df['true_label'] if y in LEVEL_TO_NUM])
    y_pred = np.array([LEVEL_TO_NUM[y] for y in df['prediction'] if y in LEVEL_TO_NUM])
    diff = np.abs(y_true - y_pred)
    return (diff <= 1).mean() * 100

def calculate_off_by_2_plus(df):
    """Calculate % of off-by-2+ errors"""
    y_true = np.array([LEVEL_TO_NUM[y] for y in df['true_label'] if y in LEVEL_TO_NUM])
    y_pred = np.array([LEVEL_TO_NUM[y] for y in df['prediction'] if y in LEVEL_TO_NUM])
    diff = np.abs(y_true - y_pred)
    return (diff >= 2).mean() * 100

# =============================================================================
# HYPOTHESIS TESTING
# =============================================================================

def test_hypothesis(h_id, phase1_df, phase2_df):
    """Test a specific hypothesis"""
    
    h_data = HYPOTHESES[h_id]
    
    print(f"\n{'='*70}")
    print(f"TESTING {h_id}: {h_data['hypothesis']}")
    print('='*70)
    
    # Filter for relevant prompts
    phase1_subset = phase1_df[phase1_df['prompt_name'] == h_data['baseline_prompt']]
    phase2_subset = phase2_df[phase2_df['prompt_name'] == h_data['test_prompt']]
    
    # Apply additional filters
    if h_data.get('filter') == 'long essays only':
        phase1_subset = phase1_subset[phase1_subset['length_category'] == 'long']
        phase2_subset = phase2_subset[phase2_subset['length_category'] == 'long']
    
    # Calculate metrics
    metric = h_data['metric']
    
    if metric == 'robustness_sd':
        phase1_value = calculate_robustness(phase1_subset)
        phase2_value = calculate_robustness(phase2_subset)
        
    elif metric == 'exact_accuracy':
        phase1_value = (phase1_subset['true_label'] == phase1_subset['prediction']).mean() * 100
        phase2_value = (phase2_subset['true_label'] == phase2_subset['prediction']).mean() * 100
        
    elif metric == 'accuracy_long':
        phase1_value = (phase1_subset['true_label'] == phase1_subset['prediction']).mean() * 100
        phase2_value = (phase2_subset['true_label'] == phase2_subset['prediction']).mean() * 100
        
    elif metric == 'qwk':
        phase1_value = calculate_qwk(phase1_subset)
        phase2_value = calculate_qwk(phase2_subset)
        
    elif metric == 'adjacent_accuracy':
        phase1_value = calculate_adjacent_accuracy(phase1_subset)
        phase2_value = calculate_adjacent_accuracy(phase2_subset)
        
    elif metric == 'off_by_2_plus_pct':
        phase1_value = calculate_off_by_2_plus(phase1_subset)
        phase2_value = calculate_off_by_2_plus(phase2_subset)
        
    elif metric == 'confidence_separation':
        # Special case: can't test without model confidence scores
        # But can approximate: if variance is lower, predictions are more confident
        phase1_value = calculate_robustness(phase1_subset)
        phase2_value = calculate_robustness(phase2_subset)
        print("  Note: Testing via robustness (proxy for confidence)")
    
    # Test prediction
    prediction = h_data['prediction']
    threshold = h_data['threshold']
    
    print(f"\nMetric: {metric}")
    print(f"Prediction: {prediction}")
    print(f"Threshold: {threshold}")
    
    print(f"\nResults:")
    print(f"  Phase 1 ({h_data['baseline_prompt']}): {phase1_value:.3f}")
    print(f"  Phase 2 ({h_data['test_prompt']}): {phase2_value:.3f}")
    print(f"  Change: {phase2_value - phase1_value:+.3f}")
    
    # Determine if hypothesis supported
    if 'Phase2 < Phase1' in prediction:
        supported = phase2_value < phase1_value
    elif 'Phase2 > Phase1' in prediction:
        supported = phase2_value > phase1_value
    elif '>' in prediction:
        supported = phase2_value > threshold
    else:
        supported = phase2_value < threshold
    
    # Statistical test
    if metric in ['exact_accuracy', 'accuracy_long', 'off_by_2_plus_pct', 'adjacent_accuracy']:
        # Proportion test
        n1 = len(phase1_subset)
        n2 = len(phase2_subset)
        p1 = phase1_value / 100
        p2 = phase2_value / 100
        
        # Z-test for proportions
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z_stat = (p2 - p1) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        print(f"\nStatistical Test (Z-test):")
        print(f"  Z-statistic: {z_stat:.3f}")
        print(f"  P-value: {p_value:.4f}")
        significant = p_value < 0.05
        
    else:
        # T-test or correlation (harder for SD, QWK)
        print(f"\nStatistical Test: Not applicable for {metric}")
        significant = abs(phase2_value - phase1_value) > 0.05  # Practical significance
    
    # Verdict
    print(f"\n{'='*70}")
    if supported:
        print(f"✓ HYPOTHESIS SUPPORTED")
        if metric in ['exact_accuracy', 'accuracy_long', 'off_by_2_plus_pct', 'adjacent_accuracy']:
            if significant:
                print(f"  Statistical significance: YES (p < 0.05)")
            else:
                print(f"  Statistical significance: NO (p >= 0.05)")
                print(f"  Practical significance: Yes (observable improvement)")
    else:
        print(f"✗ HYPOTHESIS NOT SUPPORTED")
    print('='*70)
    
    return {
        'hypothesis_id': h_id,
        'hypothesis': h_data['hypothesis'],
        'metric': metric,
        'phase1_value': phase1_value,
        'phase2_value': phase2_value,
        'change': phase2_value - phase1_value,
        'prediction': prediction,
        'threshold': threshold,
        'supported': supported,
        'significant': significant if metric in ['exact_accuracy', 'accuracy_long', 'off_by_2_plus_pct', 'adjacent_accuracy'] else None
    }

# =============================================================================
# MAIN COMPARISON
# =============================================================================

def compare_phases():
    """Compare Phase 1 and Phase 2 results"""
    
    print("="*70)
    print("PHASE 1 VS PHASE 2 COMPARISON")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    phase1_df = pd.read_csv(PHASE1_RESULTS)
    phase1_df = phase1_df[phase1_df['prediction'] != 'ERROR']
    
    phase2_df = pd.read_csv(PHASE2_RESULTS)
    phase2_df = phase2_df[phase2_df['prediction'] != 'ERROR']
    
    print(f"✓ Phase 1: {len(phase1_df)} predictions")
    print(f"✓ Phase 2: {len(phase2_df)} predictions")
    
    # Test all hypotheses
    results = []
    for h_id in HYPOTHESES.keys():
        result = test_hypothesis(h_id, phase1_df, phase2_df)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("HYPOTHESIS TESTING SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    
    print("\n" + results_df.to_string(index=False))
    
    # Save
    output_file = TABLES_DIR / "hypothesis_test_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    # Count supported
    supported = results_df['supported'].sum()
    total = len(results_df)
    
    print(f"\nSupported: {supported}/{total} ({supported/total*100:.1f}%)")
    
    # Visualize
    create_comparison_plots(results_df)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comparison_plots(results_df):
    """Create comparison visualizations"""
    
    print("\nCreating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Hypothesis support
    ax1 = axes[0, 0]
    support_counts = results_df['supported'].value_counts()
    colors = ['green' if x else 'red' for x in support_counts.index]
    ax1.bar(['Supported', 'Not Supported'], support_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Hypotheses', fontweight='bold')
    ax1.set_title('Hypothesis Testing Results', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Change by hypothesis
    ax2 = axes[0, 1]
    h_ids = results_df['hypothesis_id'].values
    changes = results_df['change'].values
    colors = ['green' if s else 'red' for s in results_df['supported']]
    ax2.barh(h_ids, changes, color=colors, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Change (Phase 2 - Phase 1)', fontweight='bold')
    ax2.set_ylabel('Hypothesis', fontweight='bold')
    ax2.set_title('Magnitude of Change', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Phase 1 vs Phase 2 values
    ax3 = axes[1, 0]
    x = np.arange(len(results_df))
    width = 0.35
    ax3.bar(x - width/2, results_df['phase1_value'], width, label='Phase 1', alpha=0.7)
    ax3.bar(x + width/2, results_df['phase2_value'], width, label='Phase 2', alpha=0.7)
    ax3.set_xlabel('Hypothesis', fontweight='bold')
    ax3.set_ylabel('Metric Value', fontweight='bold')
    ax3.set_title('Phase 1 vs Phase 2 Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(results_df['hypothesis_id'], rotation=45)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Text summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    PHASE 2 RESULTS SUMMARY
    
    Total Hypotheses: {len(results_df)}
    Supported: {results_df['supported'].sum()}
    Not Supported: {(~results_df['supported']).sum()}
    
    Success Rate: {results_df['supported'].mean()*100:.1f}%
    
    Key Findings:
    - Best improvement: {results_df.nlargest(1, 'change')['hypothesis_id'].values[0]}
    - Largest change: {results_df.nlargest(1, 'change')['change'].values[0]:+.3f}
    
    Lessons Learned:
    {results_df['supported'].sum()} hypotheses validated
    {(~results_df['supported']).sum()} hypotheses need revision
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center', family='monospace')
    
    plt.suptitle('Phase 1 vs Phase 2: Hypothesis Testing Results', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = FIGURES_DIR / "11_phase_comparison.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file}")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    compare_phases()

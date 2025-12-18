# scripts/09_analyze_results.py
"""
Results Analysis for ECM3401 Project
Calculates robustness metrics and statistical tests

Analyses:
1. Overall robustness (SD across paraphrases)
2. Stratified by length (short/medium/long)
3. Model comparison (GPT vs Phi-3)
4. Accuracy per strategy
5. Statistical significance tests

Run: python scripts/09_analyze_results.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json

from config import RESULTS_DIR, TABLES_DIR, CEFR_LEVELS

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_FILE = RESULTS_DIR / "full_experiment_results.csv"
SUMMARY_FILE = TABLES_DIR / "analysis_summary.csv"
STATS_FILE = TABLES_DIR / "statistical_tests.json"
ROBUSTNESS_FILE = TABLES_DIR / "robustness_metrics.csv"

ROBUSTNESS_THRESHOLD_GOOD = 3.0  # SD < 3% = deployment-ready
ROBUSTNESS_THRESHOLD_MODERATE = 5.0  # SD < 5% = acceptable

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_accuracy(predictions, true_labels):
    """Calculate accuracy"""
    correct = (predictions == true_labels).sum()
    total = len(predictions)
    return (correct / total) * 100 if total > 0 else 0

def calculate_robustness(group_df):
    """
    Calculate robustness as standard deviation across paraphrases
    
    For each essay, calculate SD of predictions across 3 paraphrases
    Then average across all essays
    """
    robustness_scores = []
    
    # Group by essay
    for essay_id, essay_group in group_df.groupby('essay_id'):
        # Should have 3 predictions (one per variant)
        predictions = essay_group['prediction'].values
        
        if len(predictions) < 2:
            continue
        
        # Convert predictions to numeric (A2=1, B1=2, B2=3, C1=4, C2=5)
        pred_numeric = []
        for p in predictions:
            if p == 'A2': pred_numeric.append(1)
            elif p == 'B1': pred_numeric.append(2)
            elif p == 'B2': pred_numeric.append(3)
            elif p == 'C1': pred_numeric.append(4)
            elif p == 'C2': pred_numeric.append(5)
            else: pred_numeric.append(np.nan)
        
        # Skip if any invalid predictions
        if any(np.isnan(pred_numeric)):
            continue
        
        # Calculate SD (in CEFR levels, not %)
        sd = np.std(pred_numeric, ddof=1) if len(pred_numeric) > 1 else 0
        robustness_scores.append(sd)
    
    if len(robustness_scores) == 0:
        return np.nan, np.nan, 0
    
    mean_sd = np.mean(robustness_scores)
    std_sd = np.std(robustness_scores, ddof=1) if len(robustness_scores) > 1 else 0
    n_essays = len(robustness_scores)
    
    return mean_sd, std_sd, n_essays

def assess_robustness(sd_value):
    """Assess robustness category"""
    if pd.isna(sd_value):
        return "Unknown"
    elif sd_value < ROBUSTNESS_THRESHOLD_GOOD:
        return "Deployment-Ready"
    elif sd_value < ROBUSTNESS_THRESHOLD_MODERATE:
        return "Acceptable"
    else:
        return "Not Robust"

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_results():
    """Run complete analysis"""
    
    print("="*70)
    print("RESULTS ANALYSIS - ECM3401 PROJECT")
    print("="*70)
    
    # Load results
    print("\nLoading results...")
    if not RESULTS_FILE.exists():
        print(f"❌ Results file not found: {RESULTS_FILE}")
        print("Run script 08 first!")
        return
    
    df = pd.read_csv(RESULTS_FILE)
    print(f"✓ Loaded {len(df)} predictions")
    
    # Basic validation
    print(f"\nData validation:")
    print(f"  Unique essays: {df['essay_id'].nunique()}")
    print(f"  Models: {df['model'].unique()}")
    print(f"  Strategies: {df['strategy'].unique()}")
    print(f"  Variants per strategy: {df.groupby('strategy')['variant'].nunique().values}")
    
    # Remove errors
    errors = df[df['prediction'] == 'ERROR']
    if len(errors) > 0:
        print(f"\n⚠️ Removing {len(errors)} ERROR predictions")
        df = df[df['prediction'] != 'ERROR']
    
    # ==========================================================================
    # OVERALL ANALYSIS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("OVERALL ANALYSIS")
    print("="*70)
    
    overall_results = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        print(f"\n{model.upper()}:")
        
        for strategy in df['strategy'].unique():
            strategy_df = model_df[model_df['strategy'] == strategy]
            
            # Accuracy
            accuracy = calculate_accuracy(
                strategy_df['prediction'], 
                strategy_df['true_label']
            )
            
            # Robustness (SD across paraphrases)
            mean_sd, std_sd, n_essays = calculate_robustness(strategy_df)
            
            # Assessment
            assessment = assess_robustness(mean_sd)
            
            print(f"  {strategy}:")
            print(f"    Accuracy: {accuracy:.1f}%")
            print(f"    Robustness SD: {mean_sd:.3f} CEFR levels")
            print(f"    Assessment: {assessment}")
            
            overall_results.append({
                'model': model,
                'strategy': strategy,
                'accuracy_pct': accuracy,
                'robustness_sd': mean_sd,
                'robustness_std': std_sd,
                'n_essays': n_essays,
                'assessment': assessment,
                'length_category': 'overall'
            })
    
    # ==========================================================================
    # STRATIFIED BY LENGTH
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STRATIFIED ANALYSIS BY LENGTH")
    print("="*70)
    
    stratified_results = []
    
    for length_cat in ['short', 'medium', 'long']:
        length_df = df[df['length_category'] == length_cat]
        
        if len(length_df) == 0:
            continue
        
        print(f"\n{length_cat.upper()} ESSAYS ({len(length_df[length_df['model']=='gpt-4o-mini'])//9} essays):")
        
        for model in df['model'].unique():
            model_df = length_df[length_df['model'] == model]
            
            print(f"\n  {model}:")
            
            for strategy in df['strategy'].unique():
                strategy_df = model_df[model_df['strategy'] == strategy]
                
                if len(strategy_df) == 0:
                    continue
                
                # Accuracy
                accuracy = calculate_accuracy(
                    strategy_df['prediction'],
                    strategy_df['true_label']
                )
                
                # Robustness
                mean_sd, std_sd, n_essays = calculate_robustness(strategy_df)
                
                # Assessment
                assessment = assess_robustness(mean_sd)
                
                print(f"    {strategy}: Acc={accuracy:.1f}%, SD={mean_sd:.3f}, {assessment}")
                
                stratified_results.append({
                    'model': model,
                    'strategy': strategy,
                    'accuracy_pct': accuracy,
                    'robustness_sd': mean_sd,
                    'robustness_std': std_sd,
                    'n_essays': n_essays,
                    'assessment': assessment,
                    'length_category': length_cat
                })
    
    # ==========================================================================
    # MODEL COMPARISON
    # ==========================================================================
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    model_comparison = []
    
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        
        gpt_df = strategy_df[strategy_df['model'] == 'gpt-4o-mini']
        phi3_df = strategy_df[strategy_df['model'] == 'phi-3-mini']
        
        # GPT stats
        gpt_acc = calculate_accuracy(gpt_df['prediction'], gpt_df['true_label'])
        gpt_sd, _, _ = calculate_robustness(gpt_df)
        
        # Phi-3 stats
        phi3_acc = calculate_accuracy(phi3_df['prediction'], phi3_df['true_label'])
        phi3_sd, _, _ = calculate_robustness(phi3_df)
        
        print(f"\n{strategy}:")
        print(f"  GPT-4o-mini:  Acc={gpt_acc:.1f}%, SD={gpt_sd:.3f}")
        print(f"  Phi-3-Mini:   Acc={phi3_acc:.1f}%, SD={phi3_sd:.3f}")
        print(f"  Difference:   Acc={gpt_acc-phi3_acc:+.1f}pp, SD={gpt_sd-phi3_sd:+.3f}")
        
        model_comparison.append({
            'strategy': strategy,
            'gpt_accuracy': gpt_acc,
            'gpt_robustness': gpt_sd,
            'phi3_accuracy': phi3_acc,
            'phi3_robustness': phi3_sd,
            'acc_difference': gpt_acc - phi3_acc,
            'robustness_difference': gpt_sd - phi3_sd
        })
    
    # ==========================================================================
    # STATISTICAL TESTS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)
    
    statistical_tests = {}
    
    # Test 1: Does strategy affect robustness? (ANOVA)
    print("\n1. Effect of Strategy on Robustness (ANOVA):")
    
    strategy_groups = []
    strategy_labels = []
    
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        
        # Calculate per-essay SDs
        essay_sds = []
        for essay_id, essay_group in strategy_df.groupby('essay_id'):
            predictions = essay_group['prediction'].values
            pred_numeric = [
                {'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}.get(p, np.nan)
                for p in predictions
            ]
            if not any(np.isnan(pred_numeric)) and len(pred_numeric) > 1:
                essay_sds.append(np.std(pred_numeric, ddof=1))
        
        if len(essay_sds) > 0:
            strategy_groups.append(essay_sds)
            strategy_labels.append(strategy)
    
    if len(strategy_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*strategy_groups)
        print(f"  F-statistic: {f_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✓ Significant difference between strategies (p < 0.05)")
        else:
            print(f"  ✗ No significant difference (p >= 0.05)")
        
        statistical_tests['strategy_anova'] = {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
    
    # Test 2: GPT vs Phi-3 robustness (t-test)
    print("\n2. GPT vs Phi-3 Robustness (t-test):")
    
    gpt_sds = []
    phi3_sds = []
    
    for essay_id in df['essay_id'].unique():
        essay_df = df[df['essay_id'] == essay_id]
        
        # GPT
        gpt_essay = essay_df[essay_df['model'] == 'gpt-4o-mini']
        gpt_preds = [
            {'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}.get(p, np.nan)
            for p in gpt_essay['prediction'].values
        ]
        if not any(np.isnan(gpt_preds)) and len(gpt_preds) > 1:
            gpt_sds.append(np.std(gpt_preds, ddof=1))
        
        # Phi-3
        phi3_essay = essay_df[essay_df['model'] == 'phi-3-mini']
        phi3_preds = [
            {'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}.get(p, np.nan)
            for p in phi3_essay['prediction'].values
        ]
        if not any(np.isnan(phi3_preds)) and len(phi3_preds) > 1:
            phi3_sds.append(np.std(phi3_preds, ddof=1))
    
    if len(gpt_sds) > 0 and len(phi3_sds) > 0:
        t_stat, p_value = stats.ttest_ind(gpt_sds, phi3_sds)
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✓ Significant difference between models (p < 0.05)")
        else:
            print(f"  ✗ No significant difference (p >= 0.05)")
        
        statistical_tests['model_ttest'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Combine overall + stratified
    all_results = overall_results + stratified_results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(ROBUSTNESS_FILE, index=False)
    print(f"✓ Robustness metrics saved: {ROBUSTNESS_FILE}")
    
    # Model comparison
    comparison_df = pd.DataFrame(model_comparison)
    comparison_file = TABLES_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"✓ Model comparison saved: {comparison_file}")
    
    # Statistical tests
    with open(STATS_FILE, 'w') as f:
        json.dump(statistical_tests, f, indent=2)
    print(f"✓ Statistical tests saved: {STATS_FILE}")
    
    # Summary table
    summary = []
    for model in ['gpt-4o-mini', 'phi-3-mini']:
        for strategy in df['strategy'].unique():
            row = results_df[
                (results_df['model'] == model) & 
                (results_df['strategy'] == strategy) &
                (results_df['length_category'] == 'overall')
            ]
            if len(row) > 0:
                summary.append({
                    'Model': model,
                    'Strategy': strategy,
                    'Accuracy (%)': f"{row.iloc[0]['accuracy_pct']:.1f}",
                    'Robustness (SD)': f"{row.iloc[0]['robustness_sd']:.3f}",
                    'Assessment': row.iloc[0]['assessment']
                })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"✓ Summary table saved: {SUMMARY_FILE}")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # Most robust strategy
    overall_df = results_df[results_df['length_category'] == 'overall']
    most_robust = overall_df.nsmallest(1, 'robustness_sd')
    
    if len(most_robust) > 0:
        print(f"\nMost Robust:")
        print(f"  {most_robust.iloc[0]['model']} + {most_robust.iloc[0]['strategy']}")
        print(f"  SD = {most_robust.iloc[0]['robustness_sd']:.3f}")
        print(f"  Assessment: {most_robust.iloc[0]['assessment']}")
    
    # Least robust strategy
    least_robust = overall_df.nlargest(1, 'robustness_sd')
    
    if len(least_robust) > 0:
        print(f"\nLeast Robust:")
        print(f"  {least_robust.iloc[0]['model']} + {least_robust.iloc[0]['strategy']}")
        print(f"  SD = {least_robust.iloc[0]['robustness_sd']:.3f}")
        print(f"  Assessment: {least_robust.iloc[0]['assessment']}")
    
    # Deployment readiness
    deployment_ready = overall_df[overall_df['robustness_sd'] < ROBUSTNESS_THRESHOLD_GOOD]
    print(f"\nDeployment-Ready Configurations:")
    print(f"  {len(deployment_ready)} out of {len(overall_df)} (SD < {ROBUSTNESS_THRESHOLD_GOOD})")
    
    for _, row in deployment_ready.iterrows():
        print(f"  ✓ {row['model']} + {row['strategy']}: SD={row['robustness_sd']:.3f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    print("\nGenerated files:")
    print(f"  1. {ROBUSTNESS_FILE}")
    print(f"  2. {comparison_file}")
    print(f"  3. {STATS_FILE}")
    print(f"  4. {SUMMARY_FILE}")
    
    print("\nNext step: python scripts/10_create_plots.py")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    analyze_results()
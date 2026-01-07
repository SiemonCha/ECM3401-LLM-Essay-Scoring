# scripts/13_advanced_metrics.py
"""
Advanced Metrics and Actionable Insights for ECM3401 Project

Computes sophisticated metrics:
1. Quadratic Weighted Kappa (QWK) - Ordinal classification metric
2. Adjacent Accuracy - Within-1-level correctness
3. Off-by-2+ Errors - Serious mistakes
4. Per-strategy effect sizes
5. What works and what doesn't

Run: python scripts/13_advanced_metrics.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR, FIGURES_DIR, TABLES_DIR, CEFR_LEVELS

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_FILE = RESULTS_DIR / "full_experiment_results.csv"

plt.style.use('seaborn-v0_8-paper')
FIGURE_DPI = 300

# Level mapping for ordinal calculations
LEVEL_TO_NUM = {'A2': 0, 'B1': 1, 'B2': 2, 'C1': 3, 'C2': 4}
NUM_TO_LEVEL = {v: k for k, v in LEVEL_TO_NUM.items()}

# =============================================================================
# METRIC 1: QUADRATIC WEIGHTED KAPPA
# =============================================================================

def calculate_qwk(y_true, y_pred):
    """
    Calculate Quadratic Weighted Kappa
    
    QWK measures agreement between raters for ordinal data
    - 1.0 = perfect agreement
    - 0.0 = random agreement
    - negative = worse than random
    
    For CEFR: penalizes off-by-2 errors more than off-by-1
    """
    # Convert to numeric
    y_true_num = [LEVEL_TO_NUM[y] for y in y_true]
    y_pred_num = [LEVEL_TO_NUM[y] for y in y_pred]
    
    # Calculate QWK
    qwk = cohen_kappa_score(y_true_num, y_pred_num, weights='quadratic')
    
    return qwk

# =============================================================================
# METRIC 2: ADJACENT ACCURACY
# =============================================================================

def calculate_adjacent_accuracy(y_true, y_pred):
    """
    Adjacent Accuracy: % of predictions within 1 level
    
    Example: If true=B1, accept A2, B1, or B2
    
    This is more lenient than exact accuracy
    Important for CEFR where adjacent levels overlap
    """
    y_true_num = np.array([LEVEL_TO_NUM[y] for y in y_true])
    y_pred_num = np.array([LEVEL_TO_NUM[y] for y in y_pred])
    
    # Within 1 level?
    diff = np.abs(y_true_num - y_pred_num)
    adjacent_correct = (diff <= 1).sum()
    
    adjacent_acc = (adjacent_correct / len(y_true)) * 100
    
    return adjacent_acc

# =============================================================================
# METRIC 3: ERROR SEVERITY ANALYSIS
# =============================================================================

def analyze_error_severity(y_true, y_pred):
    """
    Break down errors by severity:
    - Exact match
    - Off by 1 (acceptable)
    - Off by 2 (bad)
    - Off by 3+ (very bad)
    """
    y_true_num = np.array([LEVEL_TO_NUM[y] for y in y_true])
    y_pred_num = np.array([LEVEL_TO_NUM[y] for y in y_pred])
    
    diff = np.abs(y_true_num - y_pred_num)
    
    exact = (diff == 0).sum()
    off_by_1 = (diff == 1).sum()
    off_by_2 = (diff == 2).sum()
    off_by_3_plus = (diff >= 3).sum()
    
    total = len(y_true)
    
    return {
        'exact': (exact / total) * 100,
        'off_by_1': (off_by_1 / total) * 100,
        'off_by_2': (off_by_2 / total) * 100,
        'off_by_3_plus': (off_by_3_plus / total) * 100,
        'exact_count': exact,
        'off_by_1_count': off_by_1,
        'off_by_2_count': off_by_2,
        'off_by_3_plus_count': off_by_3_plus
    }

# =============================================================================
# ANALYSIS: WHAT MAKES ROBUSTNESS BETTER?
# =============================================================================

def analyze_robustness_factors(df):
    """
    Identify factors that IMPROVE robustness
    """
    print("\n" + "="*70)
    print("WHAT BOOSTS ROBUSTNESS? (Lower SD = Better)")
    print("="*70)
    
    results = []
    
    # Factor 1: Prompt Strategy
    print("\n1. PROMPT STRATEGY EFFECT:")
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        
        # Calculate per-essay SD
        essay_sds = []
        for essay_id, essay_group in strategy_df.groupby('essay_id'):
            preds_num = [LEVEL_TO_NUM[p] for p in essay_group['prediction']]
            if len(preds_num) > 1:
                essay_sds.append(np.std(preds_num, ddof=1))
        
        mean_sd = np.mean(essay_sds)
        print(f"   {strategy}: Mean SD = {mean_sd:.3f}")
        
        results.append({
            'factor': 'strategy',
            'value': strategy,
            'mean_sd': mean_sd,
            'recommendation': 'Lower is better'
        })
    
    best_strategy = min(results, key=lambda x: x['mean_sd'])
    print(f"\n   ✓ BEST: {best_strategy['value']} (SD = {best_strategy['mean_sd']:.3f})")
    print(f"   → ACTIONABLE: Use {best_strategy['value']} prompts for maximum robustness")
    
    # Factor 2: Essay Length
    print("\n2. ESSAY LENGTH EFFECT:")
    for length_cat in ['short', 'medium', 'long']:
        length_df = df[df['length_category'] == length_cat]
        
        if len(length_df) == 0:
            continue
        
        essay_sds = []
        for essay_id, essay_group in length_df.groupby('essay_id'):
            preds_num = [LEVEL_TO_NUM[p] for p in essay_group['prediction']]
            if len(preds_num) > 1:
                essay_sds.append(np.std(preds_num, ddof=1))
        
        mean_sd = np.mean(essay_sds) if essay_sds else np.nan
        print(f"   {length_cat}: Mean SD = {mean_sd:.3f}")
        
        results.append({
            'factor': 'length',
            'value': length_cat,
            'mean_sd': mean_sd,
            'recommendation': 'Lower is better'
        })
    
    best_length = min([r for r in results if r['factor'] == 'length'], 
                     key=lambda x: x['mean_sd'])
    print(f"\n   ✓ BEST: {best_length['value']} essays (SD = {best_length['mean_sd']:.3f})")
    print(f"   → ACTIONABLE: Robustness is highest for {best_length['value']}-length texts")
    
    # Factor 3: Model Choice
    print("\n3. MODEL EFFECT:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        essay_sds = []
        for essay_id, essay_group in model_df.groupby('essay_id'):
            preds_num = [LEVEL_TO_NUM[p] for p in essay_group['prediction']]
            if len(preds_num) > 1:
                essay_sds.append(np.std(preds_num, ddof=1))
        
        mean_sd = np.mean(essay_sds)
        print(f"   {model}: Mean SD = {mean_sd:.3f}")
        
        results.append({
            'factor': 'model',
            'value': model,
            'mean_sd': mean_sd,
            'recommendation': 'Lower is better'
        })
    
    best_model = min([r for r in results if r['factor'] == 'model'], 
                    key=lambda x: x['mean_sd'])
    print(f"\n   ✓ BEST: {best_model['value']} (SD = {best_model['mean_sd']:.3f})")
    print(f"   → ACTIONABLE: Use {best_model['value']} for best robustness")
    
    return pd.DataFrame(results)

# =============================================================================
# ANALYSIS: WHAT MAKES ACCURACY BETTER?
# =============================================================================

def analyze_accuracy_factors(df):
    """
    Identify factors that IMPROVE accuracy
    """
    print("\n" + "="*70)
    print("WHAT BOOSTS ACCURACY? (Higher = Better)")
    print("="*70)
    
    results = []
    
    # Factor 1: Prompt Strategy
    print("\n1. PROMPT STRATEGY EFFECT:")
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        acc = (strategy_df['true_label'] == strategy_df['prediction']).mean() * 100
        
        # QWK
        qwk = calculate_qwk(strategy_df['true_label'], strategy_df['prediction'])
        
        # Adjacent accuracy
        adj_acc = calculate_adjacent_accuracy(strategy_df['true_label'], 
                                              strategy_df['prediction'])
        
        print(f"   {strategy}:")
        print(f"      Exact Acc: {acc:.1f}%")
        print(f"      Adjacent Acc: {adj_acc:.1f}%")
        print(f"      QWK: {qwk:.3f}")
        
        results.append({
            'factor': 'strategy',
            'value': strategy,
            'exact_acc': acc,
            'adjacent_acc': adj_acc,
            'qwk': qwk
        })
    
    best_strategy = max(results, key=lambda x: x['exact_acc'])
    print(f"\n   ✓ BEST: {best_strategy['value']}")
    print(f"      Exact: {best_strategy['exact_acc']:.1f}%")
    print(f"      Adjacent: {best_strategy['adjacent_acc']:.1f}%")
    print(f"      QWK: {best_strategy['qwk']:.3f}")
    print(f"   → ACTIONABLE: Use {best_strategy['value']} for highest accuracy")
    
    # Factor 2: Essay Length
    print("\n2. ESSAY LENGTH EFFECT:")
    for length_cat in ['short', 'medium', 'long']:
        length_df = df[df['length_category'] == length_cat]
        
        if len(length_df) == 0:
            continue
        
        acc = (length_df['true_label'] == length_df['prediction']).mean() * 100
        qwk = calculate_qwk(length_df['true_label'], length_df['prediction'])
        adj_acc = calculate_adjacent_accuracy(length_df['true_label'], 
                                              length_df['prediction'])
        
        print(f"   {length_cat}:")
        print(f"      Exact Acc: {acc:.1f}%")
        print(f"      Adjacent Acc: {adj_acc:.1f}%")
        print(f"      QWK: {qwk:.3f}")
        
        results.append({
            'factor': 'length',
            'value': length_cat,
            'exact_acc': acc,
            'adjacent_acc': adj_acc,
            'qwk': qwk
        })
    
    best_length = max([r for r in results if r['factor'] == 'length'], 
                     key=lambda x: x['exact_acc'])
    print(f"\n   ✓ BEST: {best_length['value']} essays")
    print(f"      Exact: {best_length['exact_acc']:.1f}%")
    print(f"      Adjacent: {best_length['adjacent_acc']:.1f}%")
    print(f"      QWK: {best_length['qwk']:.3f}")
    print(f"   → ACTIONABLE: Accuracy highest for {best_length['value']}-length texts")
    
    # Factor 3: Model
    print("\n3. MODEL EFFECT:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        acc = (model_df['true_label'] == model_df['prediction']).mean() * 100
        qwk = calculate_qwk(model_df['true_label'], model_df['prediction'])
        adj_acc = calculate_adjacent_accuracy(model_df['true_label'], 
                                              model_df['prediction'])
        
        print(f"   {model}:")
        print(f"      Exact Acc: {acc:.1f}%")
        print(f"      Adjacent Acc: {adj_acc:.1f}%")
        print(f"      QWK: {qwk:.3f}")
        
        results.append({
            'factor': 'model',
            'value': model,
            'exact_acc': acc,
            'adjacent_acc': adj_acc,
            'qwk': qwk
        })
    
    best_model = max([r for r in results if r['factor'] == 'model'], 
                    key=lambda x: x['exact_acc'])
    print(f"\n   ✓ BEST: {best_model['value']}")
    print(f"      Exact: {best_model['exact_acc']:.1f}%")
    print(f"      Adjacent: {best_model['adjacent_acc']:.1f}%")
    print(f"      QWK: {best_model['qwk']:.3f}")
    print(f"   → ACTIONABLE: Use {best_model['value']} for best accuracy")
    
    return pd.DataFrame(results)

# =============================================================================
# ANALYSIS: WHAT TO AVOID (BAD COMBINATIONS)
# =============================================================================

def identify_bad_combinations(df):
    """
    Identify combinations to AVOID
    """
    print("\n" + "="*70)
    print("WHAT TO AVOID? (Bad Combinations)")
    print("="*70)
    
    bad_combos = []
    
    # Test all combinations
    for model in df['model'].unique():
        for strategy in df['strategy'].unique():
            for length_cat in ['short', 'medium', 'long', 'overall']:
                
                if length_cat == 'overall':
                    subset = df[(df['model'] == model) & (df['strategy'] == strategy)]
                else:
                    subset = df[(df['model'] == model) & 
                               (df['strategy'] == strategy) & 
                               (df['length_category'] == length_cat)]
                
                if len(subset) == 0:
                    continue
                
                # Calculate metrics
                acc = (subset['true_label'] == subset['prediction']).mean() * 100
                qwk = calculate_qwk(subset['true_label'], subset['prediction'])
                
                # Calculate robustness
                essay_sds = []
                for essay_id, essay_group in subset.groupby('essay_id'):
                    preds_num = [LEVEL_TO_NUM[p] for p in essay_group['prediction']]
                    if len(preds_num) > 1:
                        essay_sds.append(np.std(preds_num, ddof=1))
                
                mean_sd = np.mean(essay_sds) if essay_sds else np.nan
                
                # Flag if bad
                is_bad = False
                reasons = []
                
                if acc < 20:  # Worse than random
                    is_bad = True
                    reasons.append(f"accuracy={acc:.1f}% (worse than random)")
                
                if qwk < 0:  # Negative agreement
                    is_bad = True
                    reasons.append(f"QWK={qwk:.3f} (negative)")
                
                if mean_sd > 1.0:  # High variance
                    is_bad = True
                    reasons.append(f"SD={mean_sd:.3f} (high variance)")
                
                if is_bad:
                    bad_combos.append({
                        'model': model,
                        'strategy': strategy,
                        'length': length_cat,
                        'accuracy': acc,
                        'qwk': qwk,
                        'robustness_sd': mean_sd,
                        'reasons': ', '.join(reasons)
                    })
    
    if len(bad_combos) > 0:
        print("\n⚠️ COMBINATIONS TO AVOID:")
        for combo in sorted(bad_combos, key=lambda x: x['accuracy']):
            print(f"\n   ❌ {combo['model']} + {combo['strategy']} on {combo['length']} essays")
            print(f"      Accuracy: {combo['accuracy']:.1f}%")
            print(f"      QWK: {combo['qwk']:.3f}")
            print(f"      Robustness SD: {combo['robustness_sd']:.3f}")
            print(f"      Why bad: {combo['reasons']}")
    else:
        print("\n✓ No combinations are critically bad!")
        print("   All meet minimum acceptable thresholds")
    
    return pd.DataFrame(bad_combos) if bad_combos else None

# =============================================================================
# CREATE COMPREHENSIVE METRICS TABLE
# =============================================================================

def create_comprehensive_metrics_table(df):
    """
    Create table with ALL metrics for ALL combinations
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE METRICS TABLE")
    print("="*70)
    
    results = []
    
    for model in df['model'].unique():
        for strategy in df['strategy'].unique():
            subset = df[(df['model'] == model) & (df['strategy'] == strategy)]
            
            # Basic metrics
            exact_acc = (subset['true_label'] == subset['prediction']).mean() * 100
            
            # Advanced metrics
            qwk = calculate_qwk(subset['true_label'], subset['prediction'])
            adj_acc = calculate_adjacent_accuracy(subset['true_label'], 
                                                  subset['prediction'])
            
            # Error severity
            severity = analyze_error_severity(subset['true_label'], 
                                              subset['prediction'])
            
            # Robustness
            essay_sds = []
            for essay_id, essay_group in subset.groupby('essay_id'):
                preds_num = [LEVEL_TO_NUM[p] for p in essay_group['prediction']]
                if len(preds_num) > 1:
                    essay_sds.append(np.std(preds_num, ddof=1))
            
            mean_sd = np.mean(essay_sds)
            
            results.append({
                'model': model,
                'strategy': strategy,
                'exact_accuracy': exact_acc,
                'adjacent_accuracy': adj_acc,
                'qwk': qwk,
                'robustness_sd': mean_sd,
                'exact_match_pct': severity['exact'],
                'off_by_1_pct': severity['off_by_1'],
                'off_by_2_pct': severity['off_by_2'],
                'off_by_3plus_pct': severity['off_by_3_plus']
            })
    
    results_df = pd.DataFrame(results)
    
    # Print table
    print("\n" + results_df.to_string(index=False))
    
    # Save
    output_file = TABLES_DIR / "comprehensive_metrics.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    return results_df

# =============================================================================
# VISUALIZE QWK AND ADJACENT ACCURACY
# =============================================================================

def plot_advanced_metrics(metrics_df):
    """
    Visualize QWK and Adjacent Accuracy
    """
    print("\n" + "="*70)
    print("CREATING ADVANCED METRICS PLOTS")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: QWK comparison
    ax1 = axes[0]
    pivot_qwk = metrics_df.pivot(index='strategy', columns='model', values='qwk')
    pivot_qwk.plot(kind='bar', ax=ax1, width=0.7)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random')
    ax1.axhline(0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Moderate')
    ax1.axhline(0.7, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good')
    ax1.set_xlabel('Strategy', fontweight='bold')
    ax1.set_ylabel('Quadratic Weighted Kappa', fontweight='bold')
    ax1.set_title('QWK: Agreement for Ordinal Classification', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Adjacent accuracy
    ax2 = axes[1]
    pivot_adj = metrics_df.pivot(index='strategy', columns='model', values='adjacent_accuracy')
    pivot_adj.plot(kind='bar', ax=ax2, width=0.7)
    ax2.axhline(100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect')
    ax2.axhline(80, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Good')
    ax2.set_xlabel('Strategy', fontweight='bold')
    ax2.set_ylabel('Adjacent Accuracy (%)', fontweight='bold')
    ax2.set_title('Adjacent Accuracy: Within-1-Level Correctness', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Advanced Performance Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = FIGURES_DIR / f"10_advanced_metrics.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file}")

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_advanced_analysis():
    """
    Run complete advanced metrics analysis
    """
    print("="*70)
    print("ADVANCED METRICS & ACTIONABLE INSIGHTS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(RESULTS_FILE)
    df = df[df['prediction'] != 'ERROR']
    print(f"✓ Loaded {len(df)} predictions")
    
    # Run analyses
    robustness_factors = analyze_robustness_factors(df)
    accuracy_factors = analyze_accuracy_factors(df)
    bad_combos = identify_bad_combinations(df)
    metrics_df = create_comprehensive_metrics_table(df)
    plot_advanced_metrics(metrics_df)
    
    # Save factors
    robustness_factors.to_csv(TABLES_DIR / "robustness_factors.csv", index=False)
    accuracy_factors.to_csv(TABLES_DIR / "accuracy_factors.csv", index=False)
    if bad_combos is not None:
        bad_combos.to_csv(TABLES_DIR / "bad_combinations.csv", index=False)
    
    # Summary
    print("\n" + "="*70)
    print("KEY ACTIONABLE INSIGHTS")
    print("="*70)
    
    print("\n📈 TO BOOST ROBUSTNESS:")
    best_robust = robustness_factors.nsmallest(1, 'mean_sd').iloc[0]
    print(f"   → Use {best_robust['value']} ({best_robust['factor']})")
    print(f"   → Achieves SD = {best_robust['mean_sd']:.3f}")
    
    print("\n📈 TO BOOST ACCURACY:")
    best_acc = accuracy_factors.nlargest(1, 'exact_acc').iloc[0]
    print(f"   → Use {best_acc['value']} ({best_acc['factor']})")
    print(f"   → Achieves {best_acc['exact_acc']:.1f}% exact, {best_acc['adjacent_acc']:.1f}% adjacent")
    print(f"   → QWK = {best_acc['qwk']:.3f}")
    
    if bad_combos is not None and len(bad_combos) > 0:
        print("\n⚠️ TO AVOID:")
        worst = bad_combos.iloc[0]
        print(f"   → Avoid {worst['model']} + {worst['strategy']} on {worst['length']} essays")
        print(f"   → {worst['reasons']}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    print("\nGenerated Files:")
    print("  - outputs/tables/comprehensive_metrics.csv")
    print("  - outputs/tables/robustness_factors.csv")
    print("  - outputs/tables/accuracy_factors.csv")
    print("  - outputs/tables/bad_combinations.csv (if any)")
    print("  - outputs/figures/10_advanced_metrics.png")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_advanced_analysis()
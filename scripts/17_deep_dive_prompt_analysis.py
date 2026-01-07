# scripts/17_deep_dive_prompt_analysis.py
"""
Deep Dive: Comprehensive Prompt Impact Analysis

Analyzes how EVERY prompt affects EVERY aspect of performance:
1. Per-CEFR-level accuracy
2. Confusion patterns (which levels get mixed up)
3. Length sensitivity (how prompts handle different lengths)
4. Error density sensitivity
5. Model × Strategy interactions
6. Consistency patterns (variant agreement)
7. Failure mode identification
8. Cost-performance optimization

Run: python scripts/17_deep_dive_prompt_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy import stats
from itertools import combinations

from config import RESULTS_DIR, FIGURES_DIR, TABLES_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

LEVEL_TO_NUM = {'A2': 0, 'B1': 1, 'B2': 2, 'C1': 3, 'C2': 4}
NUM_TO_LEVEL = {v: k for k, v in LEVEL_TO_NUM.items()}

plt.style.use('seaborn-v0_8-paper')
FIGURE_DPI = 300

# =============================================================================
# LOAD DATA
# =============================================================================

def load_all_data():
    """Load both phases"""
    phase1 = pd.read_csv(RESULTS_DIR / "phase1_experiment_results.csv")
    phase2 = pd.read_csv(RESULTS_DIR / "phase2_experiment_results.csv")
    
    # Combine
    all_data = pd.concat([phase1, phase2], ignore_index=True)
    
    # Filter errors
    all_data = all_data[all_data['prediction'] != 'ERROR']
    
    print(f"✓ Loaded {len(all_data):,} predictions")
    print(f"  Phase 1: {len(phase1):,}")
    print(f"  Phase 2: {len(phase2):,}")
    
    return all_data

# =============================================================================
# ANALYSIS 1: PER-CEFR-LEVEL ACCURACY
# =============================================================================

def analyze_per_level_accuracy(df):
    """Which prompts work best for EACH CEFR level?"""
    
    print("\n" + "="*70)
    print("ANALYSIS 1: PER-CEFR-LEVEL ACCURACY")
    print("="*70)
    
    results = []
    
    for prompt in df['prompt_name'].unique():
        prompt_data = df[df['prompt_name'] == prompt]
        
        for level in ['A2', 'B1', 'B2', 'C1', 'C2']:
            level_data = prompt_data[prompt_data['true_label'] == level]
            
            if len(level_data) > 0:
                accuracy = (level_data['true_label'] == level_data['prediction']).mean() * 100
                
                results.append({
                    'prompt': prompt,
                    'strategy': prompt.rsplit('_', 1)[0],
                    'variant': prompt.rsplit('_', 1)[1],
                    'level': level,
                    'accuracy': accuracy,
                    'n_samples': len(level_data)
                })
    
    results_df = pd.DataFrame(results)
    
    # Pivot table
    pivot = results_df.pivot_table(
        index='prompt', 
        columns='level', 
        values='accuracy'
    )
    
    print("\nAccuracy by Prompt × CEFR Level:")
    print(pivot.round(1))
    
    # Save
    output_file = TABLES_DIR / "per_level_accuracy.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    # Find best prompt per level
    print("\nBest Prompt for Each Level:")
    for level in ['A2', 'B1', 'B2', 'C1', 'C2']:
        level_results = results_df[results_df['level'] == level]
        best = level_results.nlargest(1, 'accuracy').iloc[0]
        print(f"  {level}: {best['prompt']} ({best['accuracy']:.1f}%)")
    
    # Visualization
    create_per_level_heatmap(pivot)
    
    return results_df

def create_per_level_heatmap(pivot):
    """Heatmap of accuracy by prompt × level"""
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    sns.heatmap(
        pivot, 
        annot=True, 
        fmt='.1f', 
        cmap='RdYlGn', 
        center=50,
        vmin=0, 
        vmax=100,
        cbar_kws={'label': 'Accuracy (%)'},
        ax=ax
    )
    
    ax.set_title('Per-CEFR-Level Accuracy by Prompt', fontweight='bold', fontsize=14)
    ax.set_xlabel('CEFR Level', fontweight='bold')
    ax.set_ylabel('Prompt', fontweight='bold')
    
    plt.tight_layout()
    output_file = FIGURES_DIR / "12_per_level_accuracy.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file}")

# =============================================================================
# ANALYSIS 2: CONFUSION PATTERNS
# =============================================================================

def analyze_confusion_patterns(df):
    """Which levels get confused? How do prompts differ?"""
    
    print("\n" + "="*70)
    print("ANALYSIS 2: CONFUSION PATTERNS")
    print("="*70)
    
    # Get top 3 and bottom 3 prompts by accuracy
    prompt_acc = df.groupby('prompt_name').apply(
        lambda x: (x['true_label'] == x['prediction']).mean()
    ).sort_values()
    
    best_prompts = prompt_acc.tail(3).index.tolist()
    worst_prompts = prompt_acc.head(3).index.tolist()
    
    print(f"\nBest 3 prompts: {best_prompts}")
    print(f"Worst 3 prompts: {worst_prompts}")
    
    # Create confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Best prompts
    for i, prompt in enumerate(best_prompts):
        prompt_data = df[df['prompt_name'] == prompt]
        y_true = [LEVEL_TO_NUM[y] for y in prompt_data['true_label']]
        y_pred = [LEVEL_TO_NUM[y] for y in prompt_data['prediction']]
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt='.1f',
            cmap='Blues',
            xticklabels=['A2', 'B1', 'B2', 'C1', 'C2'],
            yticklabels=['A2', 'B1', 'B2', 'C1', 'C2'],
            ax=axes[0, i],
            cbar=False
        )
        
        acc = prompt_acc[prompt] * 100
        axes[0, i].set_title(f'{prompt}\n(Acc: {acc:.1f}%)', fontweight='bold')
        axes[0, i].set_xlabel('Predicted')
        axes[0, i].set_ylabel('True')
    
    # Worst prompts
    for i, prompt in enumerate(worst_prompts):
        prompt_data = df[df['prompt_name'] == prompt]
        y_true = [LEVEL_TO_NUM[y] for y in prompt_data['true_label']]
        y_pred = [LEVEL_TO_NUM[y] for y in prompt_data['prediction']]
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt='.1f',
            cmap='Reds',
            xticklabels=['A2', 'B1', 'B2', 'C1', 'C2'],
            yticklabels=['A2', 'B1', 'B2', 'C1', 'C2'],
            ax=axes[1, i],
            cbar=False
        )
        
        acc = prompt_acc[prompt] * 100
        axes[1, i].set_title(f'{prompt}\n(Acc: {acc:.1f}%)', fontweight='bold')
        axes[1, i].set_xlabel('Predicted')
        axes[1, i].set_ylabel('True')
    
    plt.suptitle('Confusion Patterns: Best vs Worst Prompts', 
                 fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    
    output_file = FIGURES_DIR / "13_confusion_patterns.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file}")
    
    # Identify most common confusions
    print("\nMost Common Confusions (across all prompts):")
    
    y_true_all = [LEVEL_TO_NUM[y] for y in df['true_label']]
    y_pred_all = [LEVEL_TO_NUM[y] for y in df['prediction']]
    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3, 4])
    
    confusions = []
    for i in range(5):
        for j in range(5):
            if i != j:  # Off-diagonal
                confusions.append({
                    'true': NUM_TO_LEVEL[i],
                    'predicted': NUM_TO_LEVEL[j],
                    'count': cm_all[i, j],
                    'distance': abs(i - j)
                })
    
    confusions_df = pd.DataFrame(confusions).sort_values('count', ascending=False)
    print(confusions_df.head(10).to_string(index=False))
    
    # Save
    output_file = TABLES_DIR / "common_confusions.csv"
    confusions_df.to_csv(output_file, index=False)

# =============================================================================
# ANALYSIS 3: LENGTH SENSITIVITY
# =============================================================================

def analyze_length_sensitivity(df):
    """How does each prompt handle different essay lengths?"""
    
    print("\n" + "="*70)
    print("ANALYSIS 3: LENGTH SENSITIVITY")
    print("="*70)
    
    results = []
    
    for prompt in df['prompt_name'].unique():
        for length_cat in ['short', 'medium', 'long']:
            subset = df[(df['prompt_name'] == prompt) & 
                       (df['length_category'] == length_cat)]
            
            if len(subset) > 0:
                accuracy = (subset['true_label'] == subset['prediction']).mean() * 100
                
                # Robustness (SD across paraphrases)
                essay_sds = []
                for essay_id in subset['essay_id'].unique():
                    essay_preds = subset[subset['essay_id'] == essay_id]['prediction']
                    pred_nums = [LEVEL_TO_NUM[p] for p in essay_preds if p in LEVEL_TO_NUM]
                    if len(pred_nums) > 1:
                        essay_sds.append(np.std(pred_nums, ddof=1))
                
                robustness = np.mean(essay_sds) if essay_sds else np.nan
                
                results.append({
                    'prompt': prompt,
                    'strategy': prompt.rsplit('_', 1)[0],
                    'length_category': length_cat,
                    'accuracy': accuracy,
                    'robustness_sd': robustness,
                    'n_essays': len(subset['essay_id'].unique())
                })
    
    results_df = pd.DataFrame(results)
    
    # Pivot tables
    acc_pivot = results_df.pivot_table(
        index='prompt',
        columns='length_category',
        values='accuracy'
    )[['short', 'medium', 'long']]  # Order columns
    
    rob_pivot = results_df.pivot_table(
        index='prompt',
        columns='length_category',
        values='robustness_sd'
    )[['short', 'medium', 'long']]
    
    print("\nAccuracy by Length:")
    print(acc_pivot.round(1))
    
    print("\nRobustness (SD) by Length:")
    print(rob_pivot.round(3))
    
    # Save
    output_file = TABLES_DIR / "length_sensitivity.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    # Identify length-robust prompts
    print("\nMost Length-Robust Prompts (consistent across lengths):")
    results_df['acc_range'] = results_df.groupby('prompt')['accuracy'].transform(
        lambda x: x.max() - x.min()
    )
    
    best_consistent = results_df.drop_duplicates('prompt').nsmallest(5, 'acc_range')
    for _, row in best_consistent.iterrows():
        print(f"  {row['prompt']}: {row['acc_range']:.1f}% range")
    
    # Visualization
    create_length_sensitivity_plot(results_df)
    
    return results_df

def create_length_sensitivity_plot(results_df):
    """Plot accuracy vs length for each strategy"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    strategies = ['minimal', 'rubric', 'cot']
    length_order = ['short', 'medium', 'long']
    
    for i, strategy in enumerate(strategies):
        strategy_data = results_df[results_df['strategy'] == strategy]
        
        for prompt in strategy_data['prompt'].unique():
            prompt_data = strategy_data[strategy_data['prompt'] == prompt]
            prompt_data = prompt_data.set_index('length_category').loc[length_order]
            
            axes[i].plot(
                length_order,
                prompt_data['accuracy'],
                marker='o',
                label=prompt,
                linewidth=2
            )
        
        axes[i].set_title(f'{strategy.upper()} Strategy', fontweight='bold')
        axes[i].set_xlabel('Essay Length', fontweight='bold')
        axes[i].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 100)
    
    plt.suptitle('Length Sensitivity by Strategy', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    output_file = FIGURES_DIR / "14_length_sensitivity.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file}")

# =============================================================================
# ANALYSIS 4: MODEL × STRATEGY INTERACTIONS
# =============================================================================

def analyze_model_strategy_interactions(df):
    """Do models respond differently to strategies?"""
    
    print("\n" + "="*70)
    print("ANALYSIS 4: MODEL × STRATEGY INTERACTIONS")
    print("="*70)
    
    results = []
    
    for model in df['model'].unique():
        for strategy in ['minimal', 'rubric', 'cot']:
            subset = df[(df['model'] == model) & (df['strategy'] == strategy)]
            
            if len(subset) > 0:
                accuracy = (subset['true_label'] == subset['prediction']).mean() * 100
                
                # Robustness
                essay_sds = []
                for essay_id in subset['essay_id'].unique():
                    essay_preds = subset[subset['essay_id'] == essay_id]['prediction']
                    pred_nums = [LEVEL_TO_NUM[p] for p in essay_preds if p in LEVEL_TO_NUM]
                    if len(pred_nums) > 1:
                        essay_sds.append(np.std(pred_nums, ddof=1))
                
                robustness = np.mean(essay_sds) if essay_sds else np.nan
                
                # QWK
                y_true = [LEVEL_TO_NUM[y] for y in subset['true_label'] if y in LEVEL_TO_NUM]
                y_pred = [LEVEL_TO_NUM[y] for y in subset['prediction'] if y in LEVEL_TO_NUM]
                qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
                
                results.append({
                    'model': model,
                    'strategy': strategy,
                    'accuracy': accuracy,
                    'robustness_sd': robustness,
                    'qwk': qwk,
                    'n_predictions': len(subset)
                })
    
    results_df = pd.DataFrame(results)
    
    print("\nModel × Strategy Performance:")
    print(results_df.to_string(index=False))
    
    # Save
    output_file = TABLES_DIR / "model_strategy_interactions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    # Visualize
    create_interaction_plot(results_df)
    
    return results_df

def create_interaction_plot(results_df):
    """Interaction plot: Model × Strategy"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['accuracy', 'robustness_sd', 'qwk']
    titles = ['Accuracy (%)', 'Robustness (SD)', 'QWK']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        pivot = results_df.pivot_table(
            index='strategy',
            columns='model',
            values=metric
        )
        
        pivot.plot(kind='bar', ax=axes[i], rot=0)
        axes[i].set_title(title, fontweight='bold')
        axes[i].set_xlabel('Strategy', fontweight='bold')
        axes[i].set_ylabel(title, fontweight='bold')
        axes[i].legend(title='Model')
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Model × Strategy Interactions', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    output_file = FIGURES_DIR / "15_model_strategy_interactions.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file}")

# =============================================================================
# ANALYSIS 5: VARIANT CONSISTENCY
# =============================================================================

def analyze_variant_consistency(df):
    """Do variants within strategies agree?"""
    
    print("\n" + "="*70)
    print("ANALYSIS 5: VARIANT CONSISTENCY")
    print("="*70)
    
    results = []
    
    for strategy in ['minimal', 'rubric', 'cot']:
        strategy_data = df[df['strategy'] == strategy]
        prompts = strategy_data['prompt_name'].unique()
        
        # Pairwise agreement
        for prompt1, prompt2 in combinations(prompts, 2):
            p1_data = strategy_data[strategy_data['prompt_name'] == prompt1]
            p2_data = strategy_data[strategy_data['prompt_name'] == prompt2]
            
            # Merge on essay_id
            merged = p1_data.merge(
                p2_data, 
                on='essay_id', 
                suffixes=('_1', '_2')
            )
            
            if len(merged) > 0:
                agreement = (merged['prediction_1'] == merged['prediction_2']).mean() * 100
                
                results.append({
                    'strategy': strategy,
                    'prompt1': prompt1,
                    'prompt2': prompt2,
                    'agreement_pct': agreement,
                    'n_comparisons': len(merged)
                })
    
    results_df = pd.DataFrame(results)
    
    print("\nVariant Agreement (within strategies):")
    for strategy in ['minimal', 'rubric', 'cot']:
        strategy_results = results_df[results_df['strategy'] == strategy]
        mean_agreement = strategy_results['agreement_pct'].mean()
        print(f"  {strategy}: {mean_agreement:.1f}% average agreement")
    
    # Save
    output_file = TABLES_DIR / "variant_consistency.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    return results_df

# =============================================================================
# ANALYSIS 6: COST-PERFORMANCE OPTIMIZATION
# =============================================================================

def analyze_cost_performance(df):
    """Which prompts give best value?"""
    
    print("\n" + "="*70)
    print("ANALYSIS 6: COST-PERFORMANCE OPTIMIZATION")
    print("="*70)
    
    # GPT pricing
    GPT_COST_PER_1K_INPUT = 0.150 / 1000
    GPT_COST_PER_1K_OUTPUT = 0.600 / 1000
    
    results = []
    
    for prompt in df['prompt_name'].unique():
        prompt_data = df[df['prompt_name'] == prompt]
        
        # Accuracy
        accuracy = (prompt_data['true_label'] == prompt_data['prediction']).mean() * 100
        
        # Robustness
        essay_sds = []
        for essay_id in prompt_data['essay_id'].unique():
            essay_preds = prompt_data[prompt_data['essay_id'] == essay_id]['prediction']
            pred_nums = [LEVEL_TO_NUM[p] for p in essay_preds if p in LEVEL_TO_NUM]
            if len(pred_nums) > 1:
                essay_sds.append(np.std(pred_nums, ddof=1))
        
        robustness = np.mean(essay_sds) if essay_sds else np.nan
        
        # Estimate cost (prompt length)
        prompt_file = Path(f"prompts/{prompt}.txt")
        if prompt_file.exists():
            prompt_text = prompt_file.read_text()
            prompt_tokens = len(prompt_text.split()) * 1.3  # Rough estimate
        else:
            prompt_tokens = 50  # Default
        
        # Assume 200 word essay = ~260 tokens
        avg_essay_tokens = 260
        total_input = prompt_tokens + avg_essay_tokens
        output_tokens = 5  # Just CEFR level
        
        cost_per_prediction = (total_input * GPT_COST_PER_1K_INPUT + 
                              output_tokens * GPT_COST_PER_1K_OUTPUT)
        
        # Value metrics
        accuracy_per_dollar = accuracy / cost_per_prediction if cost_per_prediction > 0 else 0
        robustness_per_dollar = (1 / robustness) / cost_per_prediction if robustness > 0 and cost_per_prediction > 0 else 0
        
        results.append({
            'prompt': prompt,
            'strategy': prompt.rsplit('_', 1)[0],
            'accuracy': accuracy,
            'robustness_sd': robustness,
            'cost_per_prediction': cost_per_prediction,
            'accuracy_per_dollar': accuracy_per_dollar,
            'robustness_per_dollar': robustness_per_dollar
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy_per_dollar', ascending=False)
    
    print("\nCost-Performance Rankings:")
    print(results_df[['prompt', 'accuracy', 'cost_per_prediction', 'accuracy_per_dollar']].head(10).to_string(index=False))
    
    # Save
    output_file = TABLES_DIR / "cost_performance.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    return results_df

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all deep dive analyses"""
    
    print("="*70)
    print("DEEP DIVE: COMPREHENSIVE PROMPT IMPACT ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_all_data()
    
    # Run analyses
    per_level = analyze_per_level_accuracy(df)
    confusion = analyze_confusion_patterns(df)
    length_sens = analyze_length_sensitivity(df)
    interactions = analyze_model_strategy_interactions(df)
    consistency = analyze_variant_consistency(df)
    cost_perf = analyze_cost_performance(df)
    
    print("\n" + "="*70)
    print("✓ DEEP DIVE ANALYSIS COMPLETE!")
    print("="*70)
    
    print("\nGenerated:")
    print("  📊 Figures:")
    print("     - 12_per_level_accuracy.png")
    print("     - 13_confusion_patterns.png")
    print("     - 14_length_sensitivity.png")
    print("     - 15_model_strategy_interactions.png")
    print()
    print("  📋 Tables:")
    print("     - per_level_accuracy.csv")
    print("     - common_confusions.csv")
    print("     - length_sensitivity.csv")
    print("     - model_strategy_interactions.csv")
    print("     - variant_consistency.csv")
    print("     - cost_performance.csv")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS FOR THESIS")
    print("="*70)
    
    print("\n1. Best prompts per CEFR level")
    print("2. Most common confusion patterns")
    print("3. Length sensitivity differences")
    print("4. Model-strategy interaction effects")
    print("5. Within-strategy consistency")
    print("6. Cost-performance optimization")
    
    print("\nUse these for Discussion section 5.5-5.8!")

if __name__ == "__main__":
    main()

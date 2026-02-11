#!/usr/bin/env python3
"""
SUPERVISOR REVISIONS SCRIPT
Adds four analyses requested in supervisor feedback:

  1. Krippendorff's Alpha  - inter-variant agreement (replaces simple SD)
  2. Continuous word count - Pearson r using raw token count, not 3 categories
  3. TTR + Lexical Density - text quality features (needs essay text in sample_100.csv)
  4. Parse Tree Depth      - constituency parse depth per sentence (needs stanza)
  5. Merged CEFR figures   - accuracy line + error bars (robustness SD) per level

Usage:
  python supervisor_revisions.py            # runs on both phases
  python supervisor_revisions.py --phase 1  # phase 1 only
  python supervisor_revisions.py --phase 2  # phase 2 only

Outputs (all in figures/ and tables/):
  supervisor_krippendorff_alpha.csv
  supervisor_wc_scatter_phase{N}.png
  supervisor_merged_cefr_phase{N}.png
  supervisor_text_features.png             (only if essay text present)

Time: ~1 minute
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from simple_config import *

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

LEVEL_MAP = {'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}

# =============================================================================
# ARGS
# =============================================================================

def get_phases():
    if '--phase' in sys.argv:
        idx = sys.argv.index('--phase')
        p = int(sys.argv[idx + 1])
        return [p]
    return [1, 2]

# =============================================================================
# HELPERS
# =============================================================================

def load_results(phase):
    f = PHASE1_RESULTS if phase == 1 else PHASE2_RESULTS
    df = pd.read_csv(f)
    df = df[df['prediction'] != 'ERROR'].copy()
    df['pred_num'] = df['prediction'].map(LEVEL_MAP)
    df['true_num'] = df['true_label'].map(LEVEL_MAP)
    return df

# =============================================================================
# 1. KRIPPENDORFF'S ALPHA
# Treats each prompt variant as a separate "annotator" for the same essay.
# Uses ordinal distance function appropriate for CEFR (ordered categories).
# Reference: Krippendorff (2004), Content Analysis, 2nd ed.
# =============================================================================

def ordinal_distance(v1, v2, levels=5):
    """Ordinal metric: (|v1-v2| choose 2) / ((n-1) choose 2)"""
    diff = abs(v1 - v2)
    return (diff * (diff - 1)) if diff > 1 else diff ** 2

def krippendorffs_alpha(reliability_data, metric='ordinal'):
    """
    reliability_data: 2D array [annotators x items]
    Cells with NaN = missing.
    ordinal metric used (CEFR is ordered not interval).
    """
    data = np.array(reliability_data, dtype=float)
    n_annotators, n_items = data.shape

    # Observed disagreement (Do)
    Do = 0.0
    n_pairs = 0
    for item in range(n_items):
        col = data[:, item]
        valid = col[~np.isnan(col)]
        m = len(valid)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                Do += ordinal_distance(valid[i], valid[j])
                n_pairs += 1

    if n_pairs == 0:
        return np.nan

    Do = Do / n_pairs

    # Expected disagreement (De) — based on marginal distribution
    all_values = data.flatten()
    all_values = all_values[~np.isnan(all_values)]
    n = len(all_values)

    De = 0.0
    de_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            De += ordinal_distance(all_values[i], all_values[j])
            de_pairs += 1

    if de_pairs == 0 or De == 0:
        return 1.0  # perfect agreement

    De = De / de_pairs

    alpha = 1.0 - (Do / De)
    return alpha


def run_krippendorff(phases):
    print("\n" + "="*60)
    print("1. KRIPPENDORFF'S ALPHA (Inter-Variant Agreement)")
    print("="*60)

    rows = []

    for phase in phases:
        df = load_results(phase)
        variants = ['v1','v2','v3'] if phase == 1 else ['v4','v5','v6']

        for model in df['model'].unique():
            for strategy in df['strategy'].unique():
                subset = df[
                    (df['model'] == model) &
                    (df['strategy'] == strategy)
                ]

                # Build reliability matrix: rows=variants, cols=essays
                essay_ids = subset['essay_id'].unique()
                matrix = []
                for v in variants:
                    v_preds = subset[subset['variant'] == v].set_index('essay_id')['pred_num']
                    row = [v_preds.get(eid, np.nan) for eid in essay_ids]
                    matrix.append(row)

                alpha = krippendorffs_alpha(matrix)

                # Interpretation (Landis & Koch 1977)
                if alpha < 0:
                    interp = 'Poor (< chance)'
                elif alpha < 0.20:
                    interp = 'Slight'
                elif alpha < 0.40:
                    interp = 'Fair'
                elif alpha < 0.60:
                    interp = 'Moderate'
                elif alpha < 0.80:
                    interp = 'Substantial'
                else:
                    interp = 'Almost Perfect'

                rows.append({
                    'phase': phase,
                    'model': model,
                    'strategy': strategy,
                    'alpha': round(alpha, 4),
                    'interpretation': interp
                })

                print(f"  Phase {phase} | {model:14s} | {strategy:8s} | α={alpha:.4f} ({interp})")

    alpha_df = pd.DataFrame(rows)
    out = TABLES_DIR / "supervisor_krippendorff_alpha.csv"
    alpha_df.to_csv(out, index=False)
    print(f"\n✓ Saved: {out}")
    return alpha_df


# =============================================================================
# 2. CONTINUOUS WORD COUNT CORRELATIONS
# Previous code correlated 3 category means — statistically weak (n=3 dots).
# This computes per-essay Pearson r (n=100) — correct approach.
# =============================================================================

def run_continuous_wc(phases):
    print("\n" + "="*60)
    print("2. CONTINUOUS WORD COUNT CORRELATIONS")
    print("="*60)

    all_rows = []

    for phase in phases:
        df = load_results(phase)

        # Per-essay: mean accuracy and mean SD across variants
        level_map_acc = {'A2':1,'B1':2,'B2':3,'C1':4,'C2':5}
        essay_stats = []

        for essay_id in df['essay_id'].unique():
            e = df[df['essay_id'] == essay_id]
            wc = e['word_count'].iloc[0]

            for model in df['model'].unique():
                m = e[e['model'] == model]
                if len(m) == 0:
                    continue

                acc = (m['prediction'] == m['true_label']).mean()

                # SD per strategy, then average
                sds = []
                for strat in m['strategy'].unique():
                    s = m[m['strategy'] == strat]['pred_num'].dropna()
                    if len(s) >= 2:
                        sds.append(s.std(ddof=1))
                mean_sd = np.mean(sds) if sds else np.nan

                essay_stats.append({
                    'phase': phase,
                    'model': model,
                    'essay_id': essay_id,
                    'word_count': wc,
                    'accuracy': acc,
                    'mean_sd': mean_sd
                })

        stats_df = pd.DataFrame(essay_stats)

        for model in stats_df['model'].unique():
            m = stats_df[stats_df['model'] == model].dropna()

            r_acc, p_acc = stats.pearsonr(m['word_count'], m['accuracy'])
            r_sd, p_sd   = stats.pearsonr(m['word_count'], m['mean_sd'])

            print(f"  Phase {phase} | {model:14s}")
            print(f"    word_count vs accuracy  : r={r_acc:.3f}, p={p_acc:.4f} {'*' if p_acc<0.05 else 'ns'}")
            print(f"    word_count vs robustness: r={r_sd:.3f},  p={p_sd:.4f} {'*' if p_sd<0.05 else 'ns'}")

            all_rows.append({
                'phase': phase, 'model': model,
                'r_wc_accuracy': round(r_acc, 3), 'p_wc_accuracy': round(p_acc, 4),
                'r_wc_robustness': round(r_sd, 3), 'p_wc_robustness': round(p_sd, 4)
            })

        # --- Scatter plot ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        colors = {'gpt-4o-mini': '#1f77b4', 'phi-3-mini': '#ff7f0e'}

        for model in stats_df['model'].unique():
            m = stats_df[stats_df['model'] == model].dropna()
            c = colors.get(model, 'gray')

            ax1.scatter(m['word_count'], m['accuracy'] * 100,
                        alpha=0.45, color=c, label=model, s=30)
            m_clean = m.dropna(subset=['word_count','accuracy'])
            z = np.polyfit(m_clean['word_count'], m_clean['accuracy'] * 100, 1)
            xline = np.linspace(m_clean['word_count'].min(), m_clean['word_count'].max(), 100)
            yline = np.clip(np.polyval(z, xline), 0, 100)
            ax1.plot(xline, yline, color=c, linewidth=2)

            ax2.scatter(m['word_count'], m['mean_sd'],
                        alpha=0.45, color=c, label=model, s=30)
            m_clean2 = m.dropna(subset=['word_count','mean_sd'])
            z2 = np.polyfit(m_clean2['word_count'], m_clean2['mean_sd'], 1)
            xline2 = np.linspace(m_clean2['word_count'].min(), m_clean2['word_count'].max(), 100)
            ax2.plot(xline2, np.polyval(z2, xline2), color=c, linewidth=2)

        ax1.set_xlabel('Word Count (continuous)')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Word Count vs Accuracy')
        ax1.legend()

        ax2.set_xlabel('Word Count (continuous)')
        ax2.set_ylabel('Robustness (SD)')
        ax2.set_title('Word Count vs Robustness')
        ax2.legend()

        plt.suptitle(f'Phase {phase}: Continuous Word Count Effect (n=100 essays)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = FIGURES_DIR / f"supervisor_wc_scatter_phase{phase}.png"
        plt.savefig(out)
        plt.close()
        print(f"  ✓ Saved: {out}")

    corr_df = pd.DataFrame(all_rows)
    out_csv = TABLES_DIR / "supervisor_wc_correlations.csv"
    corr_df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved: {out_csv}")
    return corr_df


# =============================================================================
# 3. TTR + LEXICAL DENSITY
# Requires essay text in sample_100.csv column 'text'.
# Skipped gracefully if text absent or spaCy not installed.
# =============================================================================

def run_text_features(phases):
    print("\n" + "="*60)
    print("3. TTR + LEXICAL DENSITY")
    print("="*60)

    # Check essay text available
    sample = pd.read_csv(SAMPLE_FILE)
    text_col = None
    for col in ['text', 'essay_text', 'essay']:
        if col in sample.columns:
            text_col = col
            break

    if text_col is None:
        print("  ⚠ No essay text column found in sample_100.csv")
        print("  Expected column named: 'text', 'essay_text', or 'essay'")
        print("  Skipping TTR/lexical density analysis.")
        return None

    # Check spaCy
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("  ⚠ spaCy model not found. Run:")
            print("     python -m spacy download en_core_web_sm")
            return None
    except ImportError:
        print("  ⚠ spaCy not installed. Run:")
        print("     pip install spacy --break-system-packages")
        print("     python -m spacy download en_core_web_sm")
        return None

    print("  Computing TTR and lexical density for 100 essays...")
    CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}

    features = []
    for _, row in sample.iterrows():
        doc = nlp(row[text_col][:5000])  # cap at 5000 chars to stay fast
        tokens = [t for t in doc if not t.is_space and not t.is_punct]

        if len(tokens) == 0:
            continue

        ttr = len(set(t.lemma_.lower() for t in tokens)) / len(tokens)
        ld  = sum(1 for t in tokens if t.pos_ in CONTENT_POS) / len(tokens)

        features.append({
            'essay_id': row.get('public_essay_id', row.name),
            'word_count': row['word_count'],
            'ttr': ttr,
            'lexical_density': ld,
            'true_label': row['cefr_mapped']
        })

    feat_df = pd.DataFrame(features)
    print(f"  ✓ Features computed for {len(feat_df)} essays")

    # Merge with per-essay accuracy from both phases
    for phase in phases:
        df = load_results(phase)
        gpt = df[df['model'] == 'gpt-4o-mini']

        essay_acc = (gpt.groupby('essay_id')
                       .apply(lambda x: (x['prediction'] == x['true_label']).mean())
                       .reset_index(name='accuracy'))

        merged = feat_df.merge(essay_acc, left_on='essay_id', right_on='essay_id', how='inner')
        if len(merged) == 0:
            print(f"  ⚠ Could not merge features with Phase {phase} results (ID mismatch?)")
            continue

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for col_idx, feat in enumerate(['ttr', 'lexical_density', 'word_count']):
            r_acc, p_acc = stats.pearsonr(merged[feat], merged['accuracy'])

            # Top row: scatter vs accuracy
            ax = axes[0, col_idx]
            ax.scatter(merged[feat], merged['accuracy'] * 100, alpha=0.5, s=25, color='steelblue')
            z = np.polyfit(merged[feat], merged['accuracy'] * 100, 1)
            xl = np.linspace(merged[feat].min(), merged[feat].max(), 100)
            ax.plot(xl, np.polyval(z, xl), color='red', linewidth=1.5)
            ax.set_xlabel(feat.replace('_', ' ').title())
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'r={r_acc:.3f} ({"*" if p_acc < 0.05 else "ns"})')

            # Bottom row: box per CEFR level
            ax2 = axes[1, col_idx]
            order = ['A2', 'B1', 'B2', 'C1', 'C2']
            data_by_level = [merged[merged['true_label'] == lvl][feat].dropna().values for lvl in order]
            ax2.boxplot(data_by_level, labels=order)
            ax2.set_xlabel('CEFR Level')
            ax2.set_ylabel(feat.replace('_', ' ').title())
            ax2.set_title(f'{feat.replace("_"," ").title()} by CEFR Level')

        plt.suptitle(f'Phase {phase}: Text Features — TTR, Lexical Density, Word Count',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = FIGURES_DIR / f"supervisor_text_features_phase{phase}.png"
        plt.savefig(out)
        plt.close()
        print(f"  ✓ Saved: {out}")

        feat_csv = TABLES_DIR / f"supervisor_text_features_phase{phase}.csv"
        merged.to_csv(feat_csv, index=False)
        print(f"  ✓ Saved: {feat_csv}")

    return feat_df


# =============================================================================
# 4. PARSE TREE DEPTH
# Requires essay text in sample_100.csv and stanza installed.
# pip install stanza --break-system-packages
# python -c "import stanza; stanza.download('en')"
# =============================================================================

def _node_depth(node):
    """Recursively compute depth of a stanza constituency tree node."""
    if not node.children:
        return 0
    return 1 + max(_node_depth(child) for child in node.children)


def run_parse_tree_depth(phases):
    print("\n" + "="*60)
    print("4. PARSE TREE DEPTH (CONSTITUENCY PARSING)")
    print("="*60)

    # --- Check essay text ---
    sample = pd.read_csv(SAMPLE_FILE)
    text_col = None
    for col in ['text', 'essay_text', 'essay']:
        if col in sample.columns:
            text_col = col
            break

    if text_col is None:
        print("  ⚠ No essay text column in sample_100.csv — skipping parse tree depth.")
        print("  Expected column named: 'text', 'essay_text', or 'essay'")
        return None

    # --- Check stanza ---
    try:
        import stanza
    except ImportError:
        print("  ⚠ stanza not installed. Run:")
        print("     pip install stanza --break-system-packages")
        print("     python -c \"import stanza; stanza.download('en')\"")
        return None

    # --- Load pipeline (tokenize + constituency only, skip NER/dep for speed) ---
    print("  Loading stanza constituency pipeline...")
    try:
        nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,constituency',
            use_gpu=False,      # CPU is fine for 100 essays
            verbose=False
        )
    except Exception as e:
        print(f"  ⚠ Failed to load stanza pipeline: {e}")
        print("  Make sure you ran: python -c \"import stanza; stanza.download('en')\"")
        return None

    # --- Compute mean parse tree depth per essay ---
    print(f"  Parsing {len(sample)} essays (may take 3-8 minutes)...")
    records = []
    for idx, row in sample.iterrows():
        essay_id = row.get('public_essay_id', str(row.name))
        text = str(row[text_col])[:3000]   # cap to keep it fast

        try:
            doc = nlp(text)
            depths = []
            for sent in doc.sentences:
                if sent.constituency is not None:
                    d = _node_depth(sent.constituency)
                    depths.append(d)

            mean_depth = np.mean(depths) if depths else np.nan
            max_depth  = np.max(depths)  if depths else np.nan
            n_sents    = len(depths)

        except Exception:
            mean_depth = np.nan
            max_depth  = np.nan
            n_sents    = 0

        records.append({
            'essay_id':   essay_id,
            'true_label': row['cefr_mapped'],
            'word_count': row['word_count'],
            'mean_tree_depth': mean_depth,
            'max_tree_depth':  max_depth,
            'n_sentences':     n_sents,
        })

        if (idx + 1) % 20 == 0:
            print(f"    {idx+1}/{len(sample)} essays done...")

    depth_df = pd.DataFrame(records).dropna(subset=['mean_tree_depth'])
    print(f"  ✓ Tree depths computed for {len(depth_df)} essays")

    # Save raw depths
    raw_csv = TABLES_DIR / "supervisor_parse_tree_depth.csv"
    depth_df.to_csv(raw_csv, index=False)
    print(f"  ✓ Saved: {raw_csv}")

    # --- Merge with per-essay accuracy and robustness per phase ---
    corr_rows = []

    for phase in phases:
        df = load_results(phase)
        variants = ['v1','v2','v3'] if phase == 1 else ['v4','v5','v6']

        for model in ['gpt-4o-mini', 'phi-3-mini']:
            mdf = df[df['model'] == model]

            # Per-essay accuracy (mean across all variants)
            acc = (mdf.groupby('essay_id')
                      .apply(lambda x: (x['prediction'] == x['true_label']).mean() * 100)
                      .reset_index(name='accuracy'))

            # Per-essay robustness (SD of predictions converted to ordinal)
            def essay_sd(g):
                nums = g['prediction'].map(LEVEL_MAP).dropna()
                return nums.std(ddof=1) if len(nums) >= 2 else np.nan

            rob = (mdf.groupby('essay_id')
                      .apply(essay_sd)
                      .reset_index(name='robustness_sd'))

            merged = (depth_df
                      .merge(acc, on='essay_id', how='inner')
                      .merge(rob, on='essay_id', how='inner'))

            if len(merged) < 5:
                print(f"  ⚠ Too few merged rows for Phase {phase} {model} — skipping.")
                continue

            for feat in ['mean_tree_depth', 'max_tree_depth']:
                r_acc, p_acc = stats.pearsonr(merged[feat], merged['accuracy'])
                rob_sub = merged[[feat, 'robustness_sd']].dropna()
                r_rob, p_rob = stats.pearsonr(
                    rob_sub[feat], rob_sub['robustness_sd']
                ) if len(rob_sub) >= 5 else (np.nan, np.nan)

                corr_rows.append({
                    'phase': phase, 'model': model, 'feature': feat,
                    'r_accuracy': round(r_acc, 4), 'p_accuracy': round(p_acc, 4),
                    'r_robustness': round(r_rob, 4) if not np.isnan(r_rob) else np.nan,
                    'p_robustness': round(p_rob, 4) if not np.isnan(p_rob) else np.nan,
                    'n': len(merged)
                })

            # --- Plot: 2x2 scatter for this phase × model ---
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            plot_pairs = [
                ('mean_tree_depth', 'accuracy',       'Mean Parse Tree Depth', 'Accuracy (%)',       axes[0,0]),
                ('mean_tree_depth', 'robustness_sd',  'Mean Parse Tree Depth', 'Robustness (SD)',    axes[0,1]),
                ('max_tree_depth',  'accuracy',       'Max Parse Tree Depth',  'Accuracy (%)',       axes[1,0]),
                ('max_tree_depth',  'robustness_sd',  'Max Parse Tree Depth',  'Robustness (SD)',    axes[1,1]),
            ]

            color = 'steelblue' if model == 'gpt-4o-mini' else '#ff7f0e'

            for xfeat, yfeat, xlabel, ylabel, ax in plot_pairs:
                sub = merged[[xfeat, yfeat]].dropna()
                if len(sub) < 5:
                    ax.set_visible(False)
                    continue
                r, p = stats.pearsonr(sub[xfeat], sub[yfeat])
                sig  = '*' if p < 0.05 else 'ns'

                ax.scatter(sub[xfeat], sub[yfeat], alpha=0.5, s=30, color=color)
                z  = np.polyfit(sub[xfeat], sub[yfeat], 1)
                xl = np.linspace(sub[xfeat].min(), sub[xfeat].max(), 100)
                yl = np.polyval(z, xl)
                # Clip accuracy to [0, 100]
                if yfeat == 'accuracy':
                    yl = np.clip(yl, 0, 100)
                ax.plot(xl, yl, color='red', linewidth=1.5)
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.set_title(f'r = {r:.3f} ({sig}, p={p:.3f})', fontsize=10)

            plt.suptitle(
                f'Phase {phase}: {model} — Parse Tree Depth vs Accuracy & Robustness',
                fontsize=12, fontweight='bold'
            )
            plt.tight_layout()
            out = FIGURES_DIR / f"supervisor_parse_tree_phase{phase}_{model.replace('-','_')}.png"
            plt.savefig(out)
            plt.close()
            print(f"  ✓ Saved: {out}")

    # --- CEFR-level boxplot for mean tree depth ---
    if len(depth_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        order = ['A2', 'B1', 'B2', 'C1', 'C2']
        data_by_level = [
            depth_df[depth_df['true_label'] == lvl]['mean_tree_depth'].dropna().values
            for lvl in order
        ]
        bp = ax.boxplot(data_by_level, labels=order, patch_artist=True)
        colors = ['#a8d5e2', '#90be6d', '#f9c74f', '#f8961e', '#f94144']
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
        ax.set_xlabel('CEFR Level', fontsize=11)
        ax.set_ylabel('Mean Parse Tree Depth', fontsize=11)
        ax.set_title('Parse Tree Depth by CEFR Level', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        out = FIGURES_DIR / "supervisor_parse_tree_by_cefr.png"
        plt.savefig(out)
        plt.close()
        print(f"  ✓ Saved: {out}")

    # --- Save correlation table ---
    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        corr_csv = TABLES_DIR / "supervisor_parse_tree_correlations.csv"
        corr_df.to_csv(corr_csv, index=False)
        print(f"  ✓ Saved: {corr_csv}")
        print("\n  Correlation summary:")
        print(corr_df.to_string(index=False))

    return depth_df


# =============================================================================
# 5. MERGED CEFR FIGURES
# Single plot per phase per model:
#   - Line = accuracy per CEFR level
#   - Error bars = SD (robustness) across variants
# Supervisor said: "combine 5a/5b into one"
# =============================================================================

def run_merged_cefr(phases):
    print("\n" + "="*60)
    print("4. MERGED CEFR FIGURES (accuracy + robustness error bars)")
    print("="*60)

    for phase in phases:
        df = load_results(phase)
        variants = ['v1','v2','v3'] if phase == 1 else ['v4','v5','v6']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax_idx, model in enumerate(['gpt-4o-mini', 'phi-3-mini']):
            mdf = df[df['model'] == model]
            ax = axes[ax_idx]

            acc_per_level = []
            sd_per_level  = []

            for level in CEFR_LEVELS:
                ldf = mdf[mdf['true_label'] == level]

                # Mean accuracy across all variants
                acc = (ldf['prediction'] == ldf['true_label']).mean() * 100
                acc_per_level.append(acc)

                # SD: accuracy per variant (in percentage points) - visible on plot
                variant_accs = []
                for v in variants:
                    vdf = ldf[ldf['variant'] == v]
                    if len(vdf) > 0:
                        variant_accs.append((vdf['prediction'] == vdf['true_label']).mean() * 100)
                sd_per_level.append(np.std(variant_accs, ddof=1) if len(variant_accs) >= 2 else 0.0)

            x = np.arange(len(CEFR_LEVELS))

            ax.errorbar(x, acc_per_level, yerr=sd_per_level,
                        fmt='o-', capsize=5, capthick=1.5,
                        linewidth=2, markersize=7,
                        color='steelblue', ecolor='tomato',
                        label='Accuracy ± SD (Robustness)')

            # Annotate points
            for i, (acc, sd) in enumerate(zip(acc_per_level, sd_per_level)):
                ax.annotate(f'{acc:.0f}%', (x[i], acc),
                            textcoords='offset points', xytext=(0, 9),
                            ha='center', fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(CEFR_LEVELS)
            ax.set_xlabel('CEFR Level')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{model}')
            ax.set_ylim(-5, 110)
            ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle(
            f'Phase {phase}: Accuracy by CEFR Level (error bars = robustness SD across variants)',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        out = FIGURES_DIR / f"supervisor_merged_cefr_phase{phase}.png"
        plt.savefig(out)
        plt.close()
        print(f"  ✓ Saved: {out}")

    # 2x2 grid: both models × both phases
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    models = ['gpt-4o-mini', 'phi-3-mini']
    for row, model in enumerate(models):
        for col, phase in enumerate(phases if len(phases) == 2 else [1, 2]):
            df = load_results(phase)
            mdf = df[df['model'] == model]
            variants = ['v1','v2','v3'] if phase == 1 else ['v4','v5','v6']
            ax = axes[row][col]

            acc_per_level, sd_per_level = [], []
            for level in CEFR_LEVELS:
                ldf = mdf[mdf['true_label'] == level]
                acc = (ldf['prediction'] == ldf['true_label']).mean() * 100
                acc_per_level.append(acc)
                variant_accs = []
                for v in variants:
                    vdf = ldf[ldf['variant'] == v]
                    if len(vdf) > 0:
                        variant_accs.append((vdf['prediction'] == vdf['true_label']).mean() * 100)
                sd_per_level.append(np.std(variant_accs, ddof=1) if len(variant_accs) >= 2 else 0.0)

            x = np.arange(len(CEFR_LEVELS))
            color = 'steelblue' if model == 'gpt-4o-mini' else '#ff7f0e'
            ax.errorbar(x, acc_per_level, yerr=sd_per_level,
                        fmt='o-', capsize=5, capthick=1.5,
                        linewidth=2, markersize=7,
                        color=color, ecolor='tomato')
            for i, acc in enumerate(acc_per_level):
                ax.annotate(f'{acc:.0f}%', (x[i], acc),
                            textcoords='offset points', xytext=(0, 9),
                            ha='center', fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(CEFR_LEVELS)
            ax.set_xlabel('CEFR Level')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'Phase {phase}: {model}')
            ax.set_ylim(-5, 110)
            ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Phase 1 vs Phase 2 — Accuracy by CEFR Level (error bars = variant SD)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = FIGURES_DIR / "supervisor_merged_cefr_both_phases.png"
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved: {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    phases = get_phases()

    print("="*60)
    print("SUPERVISOR REVISIONS ANALYSIS")
    print(f"Phases: {phases}")
    print("="*60)

    # Check files exist
    for p in phases:
        f = PHASE1_RESULTS if p == 1 else PHASE2_RESULTS
        if not f.exists():
            print(f"\n❌ Missing: {f}")
            print(f"   Run: python run_experiment.py --phase {p}")
            sys.exit(1)

    alpha_df  = run_krippendorff(phases)
    corr_df   = run_continuous_wc(phases)
    feat_df   = run_text_features(phases)
    depth_df  = run_parse_tree_depth(phases)
    run_merged_cefr(phases)

    print("\n" + "="*60)
    print("✓ ALL DONE")
    print("="*60)
    print("\nNew files in figures/ and tables/:")
    print("  supervisor_krippendorff_alpha.csv")
    print("  supervisor_wc_correlations.csv")
    print("  supervisor_wc_scatter_phase{1,2}.png")
    print("  supervisor_parse_tree_depth.csv")
    print("  supervisor_parse_tree_correlations.csv")
    print("  supervisor_parse_tree_phase{1,2}_{model}.png")
    print("  supervisor_parse_tree_by_cefr.png")
    print("  supervisor_merged_cefr_phase{1,2}.png")
    print("  supervisor_merged_cefr_both_phases.png")
    print("  supervisor_text_features_phase{1,2}.png  (if essay text present)")
    print("\nPlug these figures into your thesis to replace the split panels.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
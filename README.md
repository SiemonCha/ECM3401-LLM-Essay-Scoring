# Measuring Semantic Robustness in LLM-Based Essay Scoring

**A Paraphrase Sensitivity Analysis of CEFR Classification Across Prompt Variations**

ECM3401 Individual Project · University of Exeter · 2025/26
Sansiri Charoenpong · Supervisor: Dr Rodrigo Souza Wilkens

> **What this study does:** measures whether LLM-based CEFR essay classifiers produce the same prediction when the scoring prompt is reworded into a semantically equivalent paraphrase, and tests whether targeted prompt modifications change the patterns observed.

📄 **Full dissertation will be available here after assessment (July 2026)**

---

## What This Study Examines

The study takes 100 stratified learner essays (20 per CEFR level, A2–C2) and asks two LLMs (GPT-4o-mini, Phi-3-mini) to classify each essay under 18 different prompts. The 18 prompts come from a 3 × 6 grid: three prompting strategies (Minimal, Rubric, Chain-of-Thought) crossed with six prompt variants (v1–v3 are paraphrases of each other; v4–v6 add hypothesis-driven modifications targeting patterns observed in Phase 1).

This produces 3,600 predictions in total. The key question is whether predictions stay consistent across the v1–v3 paraphrases (Phase 1) and how the v4–v6 modifications change those predictions (Phase 2).

### How the Measurements Work

- **Robustness** is measured as the per-essay standard deviation of predicted CEFR levels across prompt variants, after mapping A2–C2 to an ordinal 1–5 scale. A working threshold of SD < 0.5 corresponds to roughly half a CEFR level — the smallest difference that produces a distinct placement decision.
- **Inter-variant agreement** is measured with Krippendorff's α using an ordinal distance function, treating the three variants as independent raters.
- **Accuracy** is the exact-match rate against the human-annotated CEFR labels in the corpus, with adjacent (±1 level) accuracy reported alongside.
- **Cost** is the mean API spend per essay in USD.

---

## Overview

This repository contains the code, prompts, and analysis scripts for the study. The pipeline runs deterministic inference (temperature = 0, fixed seed) on each essay × model × strategy × variant combination, then computes the four metrics above and a set of confound and bias analyses on the resulting predictions.

### Findings

- **GPT-4o-mini** stays within the working SD threshold (mean SD = 0.192, Krippendorff's α = 0.54–0.86 across variant pairs).
- **Phi-3-mini** does not (mean SD = 0.513, α ranging from −0.59 to 0.45).
- **Both models** classify B1 essays correctly 85% of the time but reach 0% accuracy at C1 and C2; 90% of B2 essays are classified as B1.
- **Phase 2 prompt modifications** redistribute predictions: B2 accuracy moves from 10% to 24%, A2 from 70% to 82%, while B1 drops from 85% to 63% and C1/C2 remain at 0%.
- **Single-token prompt changes** correspond to large accuracy shifts: replacing informal rubric wording with formal CEFR terminology (v5) corresponds to GPT-4o-mini rubric accuracy moving from 31% to 6%; including the `≠` symbol (v6) corresponds to Phi-3-mini producing 0% valid outputs for that variant.

---

## Repository Structure

```
ECM3401-LLM-Essay-Scoring/
│
├── README.md
├── .gitignore
├── requirements.txt
├── simple_config.py                # Configuration (API keys, paths, model settings)
├── setup.py                        # Dataset download and stratified sampling
├── run_experiment.py               # Inference pipeline (both phases)
├── analyze.py                      # Per-phase analysis (accuracy, SD, cost)
├── comprehensive_analysis.py       # Confusion matrices, severity, confound analysis
├── compare_phases.py               # Phase 1 vs Phase 2 comparison
│
├── prompts/                        # All 18 prompt templates
│   ├── minimal_v1.txt ... minimal_v6.txt
│   ├── rubric_v1.txt  ... rubric_v6.txt
│   └── cot_v1.txt     ... cot_v6.txt
│
├── data/
│   ├── processed/
│   │   └── sample_100.csv          # Stratified sample (100 essays, 20 per level)
│   └── results/
│       ├── phase1_results.csv      # 1,800 Phase 1 predictions
│       └── phase2_results.csv      # 1,800 Phase 2 predictions
│
└── figures/                        # Generated plots used in dissertation
    ├── phase1_*.png
    ├── phase2_*.png
    └── analysis_*.png
```

### What's *Not* in This Repo (and How to Get It)

To keep the repository at a reasonable size and avoid committing licensed or sensitive material, the following are **excluded via `.gitignore`** and need to be regenerated locally:

| Excluded | Why excluded | How to get it |
|---|---|---|
| `models/` | Phi-3-mini weights (~7.6 GB) | Auto-downloaded by `setup.py` from Hugging Face on first run |
| `data/raw/` | Write & Improve corpus (Cambridge licence) | Auto-downloaded by `setup.py`; users must accept the corpus licence |
| `tables/` | Generated analysis outputs (~MB of CSVs/reports) | Recreated by running `analyze.py` and `comprehensive_analysis.py` |
| `.env` | API keys (must never be committed) | Create locally — see Setup below |
| `__pycache__/`, `*.egg-info/` | Python build artefacts | Auto-generated when scripts run |

The repository is therefore self-contained for **inspection** (code, prompts, the 100-essay sample, the 3,600 raw predictions, and the figures used in the dissertation) and self-bootstrapping for **reproduction** (`setup.py` downloads what's missing).

---

## Experimental Design

| Component | Phase 1: Baseline | Phase 2: Intervention |
|---|---|---|
| **Models** | GPT-4o-mini & Phi-3-mini | GPT-4o-mini & Phi-3-mini |
| **Strategies** | Minimal · Rubric · CoT | Minimal · Rubric · CoT |
| **Variants** | v1–v3 (semantically equivalent paraphrases) | v4–v6 (hypothesis-driven modifications) |
| **Essays** | 100 (20 per CEFR level) | 100 (20 per CEFR level) |
| **Predictions** | 1,800 | 1,800 (total: 3,600) |

Models, strategies, and the essay set are held constant across phases; only the prompt variants differ.

**Inference configuration:** Temperature = 0 (deterministic decoding), max tokens = 50, fixed seed = 42.

### Prompting Strategies

Each strategy provides a different amount and type of guidance to the model:

- **Minimal** — a single instruction sentence (e.g., "Classify this essay's CEFR level: A2, B1, B2, C1, or C2"). No scaffolding; tests what the model produces from its pre-trained knowledge of CEFR alone.
- **Rubric-based** — embeds a one-sentence descriptor for each of the five target levels (A2–C2), paraphrased from the writing self-assessment grid in the Council of Europe's *Common European Framework of Reference*. Tests whether explicit assessment criteria change the prediction.
- **Chain-of-Thought** — instructs the model to reason through evaluation dimensions before committing to a label. Phase 1 uses a lightweight version (named dimensions only); Phase 2 uses a five-stage version (morphosyntax → lexis → discourse → diagnostics → decision).

### Phase 1 Variants (paraphrases)

v1, v2, and v3 vary three controlled axes while preserving the classification task:
1. The instruction verb (*classify* / *determine* / *assess*)
2. The noun phrase referring to the target (*CEFR level* / *CEFR proficiency demonstrated* / *CEFR rating*)
3. The output instruction (*Respond with ONLY…* / *Answer with ONLY…* / *Provide ONLY…*)

### Phase 2 Variants (modifications)

v4–v6 add content beyond paraphrasing:

- **v4** adds explicit B2/C1/C2 diagnostic markers ("B2 requires complex subordination AND hypotheticals, not just fewer errors than B1")
- **v5** adds a length-independence instruction ("Essay length is NOT a proficiency indicator")
- **v6** combines v4 and v5 with an ordinal decision rule ("If uncertain between adjacent levels, prefer the higher classification when diagnostic features are present")

The Phase 2 prompt set was committed to the repository before any Phase 2 predictions were collected, to keep the modifications independent of the Phase 1 results.

---

## Reproduction

### Prerequisites

- Python 3.11+
- ~10 GB free disk space (Phi-3-mini weights + corpus)
- OpenAI API key (for GPT-4o-mini)
- Internet connection (for first-run downloads)

### 1. Clone and install

```bash
git clone https://github.com/SiemonCha/ECM3401-LLM-Essay-Scoring.git
cd ECM3401-LLM-Essay-Scoring
pip install -r requirements.txt
```

### 2. Configure API key

Create a `.env` file in the project root (it is gitignored, so it stays local):

```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 3. Download data and models

`setup.py` handles both the Write & Improve corpus and the Phi-3-mini weights:

```bash
python setup.py
```

This will:
- Download the Write & Improve corpus into `data/raw/` (requires accepting the Cambridge licence)
- Download `microsoft/Phi-3-mini-4k-instruct` from Hugging Face into `models/` (~7.6 GB)
- Regenerate the stratified 100-essay sample at `data/processed/sample_100.csv` using `seed = 42`

### 4. Run the inference pipeline

```bash
# Phase 1: Baseline (v1–v3) — generates data/results/phase1_results.csv
python run_experiment.py --phase 1

# Phase 2: Modifications (v4–v6) — generates data/results/phase2_results.csv
python run_experiment.py --phase 2
```

Each phase requires roughly 3–4 hours. Predictions are saved incrementally every 10 essays, so the pipeline can resume after interruption.

### 5. Recreate the analysis tables and figures

```bash
python analyze.py --phase 1                # → tables/phase1_metrics.csv
python analyze.py --phase 2                # → tables/phase2_metrics.csv
python comprehensive_analysis.py           # → tables/comprehensive_analysis_report.md
                                           #   + figures/analysis_*.png
python compare_phases.py                   # → tables/phase_comparison.csv
                                           #   + figures/phase_*.png
```

The `tables/` and `figures/` directories will be populated. (`tables/` is gitignored; `figures/` is committed because the figures are used directly in the dissertation.)

### Skipping the inference step

If you only want to reproduce the analysis (not the 3,600 LLM calls), the prediction CSVs are already in the repository at `data/results/`. Step 4 can be skipped — `analyze.py` and the other analysis scripts read directly from those files.

---

## Evaluation Metrics

| Metric | What it measures | How to read it |
|---|---|---|
| **SD** (Standard Deviation) | Per-essay variation in predicted CEFR level across prompt variants, on the A2=1 … C2=5 ordinal scale | Lower = more consistent across paraphrases. SD = 0.5 corresponds to half a CEFR level on average |
| **Krippendorff's α** | Inter-variant agreement with an ordinal distance function | α = 1 perfect agreement; α = 0 chance; α < 0 systematic disagreement |
| **Accuracy** | Exact match between predicted and human-annotated CEFR label | Reported as exact and adjacent (±1 level) |
| **Cost** | Mean API cost per essay in USD | GPT-4o-mini ≈ $0.0004/essay; Phi-3-mini = $0 marginal |

---

## Results Summary

### Phase 1: Baseline Performance

| Model · Strategy | Accuracy | SD | SD < 0.5 |
|---|---|---|---|
| GPT-4o-mini · Minimal | 33.7% | 0.185 | ✓ |
| GPT-4o-mini · Rubric | 34.7% | 0.185 | ✓ |
| GPT-4o-mini · CoT | 30.7% | 0.208 | ✓ |
| Phi-3-mini · Minimal | 24.0% | 0.433 | ✗ |
| Phi-3-mini · Rubric | 23.3% | 0.335 | ✗ |
| Phi-3-mini · CoT | 26.0% | 0.771 | ✗ |

### Per-Level Accuracy (GPT-4o-mini)

| CEFR Level | Phase 1 | Phase 2 |
|---|---|---|
| A2 | 70.0% | 82.2% |
| B1 | 85.0% | 63.2% |
| B2 | 10.0% | 23.6% |
| C1 | 0.0% | 0.0% |
| C2 | 0.0% | 0.0% |

### Single-Token Prompt Changes

- Rubric v5 (formal CEFR terminology added): GPT-4o-mini accuracy 31% → 6%
- Rubric v6 (`≠` symbol included): Phi-3-mini 0% valid outputs
- Verb choice (*score* vs *classify*): systematic shift toward higher CEFR predictions for *score*

---

## Dataset

**Write & Improve Corpus** (Bryant et al., 2023)
Cambridge English Write & Improve platform · 23,216 learner essays · CEFR-annotated

Stratified sample: 100 essays (20 per level: A2, B1, B2, C1, C2), fixed seed = 42. Sampling is stratified rather than proportional so that per-level analysis remains interpretable at C-levels, where the corpus distribution is sparse (C2 = 0.6%).

The Write & Improve corpus is publicly available for research purposes from the Cambridge English Write & Improve project. The raw corpus is not redistributed in this repository; `setup.py` downloads it directly from the source.

---

## Dissertation

The full dissertation will be uploaded to this repository after assessment and moderation are complete (expected July 2026). It contains the complete literature review, methodology, results with all figures, discussion, and 18 prompt templates in the appendix.

---

## Citation

If you use this work or build on it, please cite:

```bibtex
@thesis{charoenpong2026,
  title   = {Measuring Semantic Robustness in LLM-Based Essay Scoring:
             A Paraphrase Sensitivity Analysis of CEFR Classification
             Across Prompt Variations},
  author  = {Charoenpong, Sansiri},
  year    = {2026},
  school  = {University of Exeter},
  type    = {BSc Dissertation},
  department = {Computer Science}
}
```

---

## License

This project is submitted as part of ECM3401 Individual Project at the University of Exeter. Code and analysis scripts are available for academic and research purposes. The Write & Improve corpus is subject to its own license from Cambridge English. Phi-3-mini is licensed under Microsoft's MIT-style licence; refer to the model card on Hugging Face.

# Measuring Semantic Robustness in LLM-Based Essay Scoring

**A Paraphrase Sensitivity Analysis of CEFR Classification Across Prompt Variations**

ECM3401 Individual Project · University of Exeter · 2025/26  
Sansiri Charoenpong · Supervisor: Dr Rodrigo Souza Wilkens

> **Research Gap:** While many studies report LLM accuracy for essay scoring, very few — if any — measure whether predictions stay consistent when the scoring prompt is reworded. This study addresses that gap.

📄 **Full dissertation will be available here after assessment (July 2026)**

---

## Novel Contributions

1. **Robustness measurement framework for LLM-based AES** — proposes SD < 0.5 across paraphrased prompts as a deployment-readiness threshold, with Krippendorff's α for inter-variant reliability. To our knowledge, this is among the first studies to quantify this property for essay scoring.
2. **Systematic B1 bias discovery** — 85% accuracy on B1 essays vs 0% on C1/C2; 90% of B2 essays misclassified as B1, exposing a fundamental limitation for advanced learner assessment
3. **Prompt brittleness as a deployment risk** — single-word changes cause 5× accuracy degradation; special characters collapse smaller models entirely
4. **Cost-robustness framework** — GPT-4o-mini achieves SD = 0.192 at $0.0004/essay, economically superior to open-source alternatives until 2.5M+ essays/year

---

## Overview

This repository contains the code, prompts, and analysis scripts for a study measuring whether LLM-based CEFR essay classifiers produce consistent predictions when scoring prompts are paraphrased. The study generated 3,600 predictions across two models (GPT-4o-mini, Phi-3-mini), three prompting strategies (Minimal, Rubric, Chain-of-Thought), and six prompt variants applied to 100 stratified essays from the Write & Improve corpus.

### Key Findings

- **GPT-4o-mini** achieves deployment-ready robustness (SD = 0.192, Krippendorff's α = 0.54–0.86)
- **Phi-3-mini** fails all robustness thresholds (SD = 0.513, α as low as −0.59)
- **Systematic B1 bias**: 85% accuracy on B1 essays, 0% on C1/C2, 90% of B2 essays misclassified as B1
- **Prompt brittleness**: single-word changes caused 5× accuracy degradation; special characters collapsed Phi-3-mini entirely
- **Phase 2 interventions** partially reduced B1 bias (B2 accuracy: 10% → 24%) but introduced catastrophic rubric failures

---

## Repository Structure

```
ECM3401-LLM-Essay-Scoring/
│
├── README.md
├── simple_config.py                # Configuration (API keys, paths, model settings)
├── setup.py                        # Dataset download and stratified sampling
├── run_experiment.py               # Inference pipeline (both phases)
├── analyze.py                      # Basic analysis (accuracy, SD, cost)
├── comprehensive_analysis.py       # Deep analysis (confusion, severity, confounds)
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
├── tables/                         # Analysis outputs (CSV + reports)
│   ├── phase1_metrics.csv
│   ├── phase2_metrics.csv
│   ├── phase_comparison.csv
│   └── comprehensive_analysis_report.md
│
└── figures/                        # All generated plots
    ├── phase1_*.png
    ├── phase2_*.png
    └── analysis_*.png
```

---

## Experimental Design

|  | Phase 1: Baseline | Phase 2: Intervention |
|---|---|---|
| **Models** | GPT-4o-mini & Phi-3-mini | Same |
| **Strategies** | Minimal · Rubric · CoT | Same |
| **Variants** | v1–v3 (true paraphrases) | v4–v6 (targeted modifications) |
| **Essays** | 100 (20 per CEFR level) | Same |
| **Predictions** | 1,800 | 1,800 (total: 3,600) |

**Critical configuration:** Temperature = 0 (deterministic), max tokens = 50, fixed seed = 42.

### Prompting Strategies

- **Minimal**: Single instruction sentence — establishes lower-bound baseline
- **Rubric-based**: CEFR descriptor summaries per level with explicit feature lists
- **Chain-of-Thought**: Multi-step reasoning (morphosyntax → lexis → discourse → diagnostics → decision)

### Phase 2 Interventions

- **v4**: B1 bias correction — explicit B2/C1/C2 diagnostic markers
- **v5**: Length bias control — instruction that length ≠ proficiency
- **v6**: Decision rule — prefer higher level when features indicate

---

## Reproduction

### Prerequisites

- Python 3.11+
- OpenAI API key (for GPT-4o-mini)
- Hugging Face Transformers (for Phi-3-mini local inference)

### Setup

```bash
git clone https://github.com/SiemonCha/ECM3401-LLM-Essay-Scoring.git
cd ECM3401-LLM-Essay-Scoring
pip install -r requirements.txt
python setup.py
```

### Run Experiments

```bash
# Phase 1: Baseline (v1–v3)
python run_experiment.py --phase 1

# Phase 2: Interventions (v4–v6)
python run_experiment.py --phase 2
```

### Analysis

```bash
python analyze.py --phase 1
python analyze.py --phase 2
python comprehensive_analysis.py
python compare_phases.py
```

---

## Evaluation Metrics

| Metric | Measures | Interpretation |
|---|---|---|
| **SD** (Standard Deviation) | Prediction variance across prompt variants | Lower = more stable; SD < 0.5 = deployment-ready |
| **Krippendorff's α** | Inter-variant agreement (ordinal) | α ≥ 0.67 tentative; α ≥ 0.80 definitive |
| **Accuracy** | Exact CEFR level match | Higher = better classification |
| **Cost** | API cost per essay (USD) | GPT-4o-mini: ~$0.0004/essay; Phi-3-mini: $0 |

---

## Results Summary

### Phase 1: Baseline Performance

| Model / Strategy | Accuracy | SD | Status |
|---|---|---|---|
| GPT-4o-mini · Minimal | 33.7% | 0.185 | ✓ Pass |
| GPT-4o-mini · Rubric | 34.7% | 0.185 | ✓ Pass |
| GPT-4o-mini · CoT | 30.7% | 0.208 | ✓ Pass |
| Phi-3-mini · Minimal | 24.0% | 0.433 | ✗ Fail |
| Phi-3-mini · Rubric | 23.3% | 0.335 | ✗ Fail |
| Phi-3-mini · CoT | 26.0% | 0.771 | ✗ Fail |

### Systematic B1 Bias (GPT-4o-mini)

| CEFR Level | Phase 1 Accuracy | Phase 2 Accuracy |
|---|---|---|
| A2 | 70.0% | 82.2% |
| B1 | 85.0% | 63.2% |
| B2 | 10.0% | 23.6% |
| C1 | 0.0% | 0.0% |
| C2 | 0.0% | 0.0% |

### Prompt Brittleness

- Formal CEFR terminology (v5): accuracy 30% → 6% (5× degradation)
- Special character `≠` (v6): Phi-3-mini accuracy → 0% (complete failure)
- Verb phrasing ("score" vs "classify"): systematic upward grading bias

---

## Dataset

**Write & Improve Corpus** (Bryant et al., 2023)  
Cambridge English Write & Improve platform · 23,216 learner essays · CEFR-annotated

Stratified sample: 100 essays (20 per level: A2, B1, B2, C1, C2), fixed seed = 42.

The Write & Improve corpus is publicly available for research purposes from the Cambridge English Write & Improve project.

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

This project is submitted as part of ECM3401 Individual Project at the University of Exeter. Code and analysis scripts are available for academic and research purposes. The Write & Improve corpus is subject to its own license from Cambridge English.

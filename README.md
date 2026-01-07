# ECM3401: Measuring Semantic Robustness in LLM-Based Essay Scoring

**Student:** Sansiri Charoenpong (Siemon)  
**Supervisor:** Dr. Rodrigo Souza Wilkens  
**Institution:** University of Exeter, Computer Science  
**Project Code:** ECM3401

---

## Quick Start (5 Minutes to First Results)

```bash
# 1. Clone/navigate to project
cd ~/Desktop/Exeter/Y3/Individual\ Project/ECM3401-LLM-Essay-Scoring/

# 2. Activate environment
conda activate py310

# 3. Set environment variable (for M2 Mac)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 4. Run Phase 1 experiment (30 minutes)
python scripts/08_run_experiment.py --phase 1

# 5. Generate results (2 minutes)
python scripts/09_analyze_results.py
python scripts/10_create_plots.py

# 6. View results!
open outputs/figures/01_robustness_by_strategy.png
open outputs/report/PHASE1_RESULTS_REPORT.md
```

**That's it for the first CEFR robustness experiment! 🎉**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Quick Start Guide](#quick-start-guide)
4. [Complete Workflow](#complete-workflow)
5. [Script Reference](#script-reference)
6. [Outputs Guide](#outputs-guide)
7. [Troubleshooting](#troubleshooting)
8. [Project Structure](#project-structure)

---

## Project Overview

### What This Project Does

Tests whether Large Language Models (LLMs) can reliably classify English essays by CEFR level (A2, B1, B2, C1, C2) when prompts are paraphrased with semantically equivalent wording.

### Key Research Questions

1. **RQ1:** Are LLM CEFR predictions robust to paraphrasing?
2. **RQ2:** Does prompt complexity affect robustness?
3. **RQ3:** Can retrieval-based few-shot reduce variance?
4. **RQ4:** Does model architecture affect robustness?
5. **RQ5:** What are the cost-robustness tradeoffs?

### Methodology

- **2 models:** GPT-4o-mini (commercial), Phi-3-Mini (open-source)
- **3 strategies:** Minimal, Rubric, Chain-of-Thought
- **18 prompts:** 9 Phase 1 (exploratory) + 9 Phase 2 (hypothesis-driven)
- **135 essays:** Stratified sample from Write & Improve Corpus 2024
- **4,860 predictions:** 135 essays × 18 prompts × 2 models

---

## Installation

### Prerequisites

- **Hardware:** MacBook M2 Pro (or any Mac with Apple Silicon)
- **OS:** macOS 14+
- **Python:** 3.10 or 3.11
- **Storage:** ~10GB free (for models and data)
- **RAM:** 16GB recommended

### Step 1: Environment Setup

```bash
# Create conda environment
conda create -n py310 python=3.10 -y
conda activate py310

# Verify Python version
python --version  # Should show Python 3.10.x
```

### Step 2: Install Dependencies

```bash
# Navigate to project directory
cd ~/Desktop/Exeter/Y3/Individual\ Project/ECM3401-LLM-Essay-Scoring/

# Install all packages
pip install -r requirements.txt

# Verify PyTorch MPS support
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
# Should print: MPS available: True
```

### Step 3: Configure API Keys

```bash
# Create .env file
touch .env

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-proj-your-key-here" >> .env

# Add HuggingFace token (optional for Phi-3)
echo "HUGGINGFACE_TOKEN=hf_your-token-here" >> .env
```

**Get API keys:**

- OpenAI: https://platform.openai.com/api-keys
- HuggingFace: https://huggingface.co/settings/tokens

### Step 4: Dataset Setup

**Download Write & Improve Corpus 2024:**

1. Go to: https://www.cl.cam.ac.uk/research/nl/bea2024st/
2. Download: `write-and-improve-corpus-2024-v2.zip`
3. Extract to: `~/Desktop/Exeter/Y3/Individual Project/`
4. Verify path: `~/Desktop/Exeter/Y3/Individual Project/write-and-improve-corpus-2024-v2/`

**Verify dataset:**

```bash
ls ~/Desktop/Exeter/Y3/Individual\ Project/write-and-improve-corpus-2024-v2/whole-corpus/
# Should see: en-writeandimprove2024-corpus.tsv
```

### Step 5: Run Setup Script (Optional)

```bash
# Install package in development mode
pip install -e .

# Test imports
python -c "import config; print('Config loaded successfully!')"
```

---

## Quick Start Guide

### Phase 1: Exploratory Analysis (45 minutes total)

```bash
# Activate environment
conda activate py310
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 1. Run Phase 1 experiment (30 min - walk away during this!)
python scripts/08_run_experiment.py --phase 1

# 2. Analyze results (15 min)
python scripts/09_analyze_results.py          # Basic analysis
python scripts/10_create_plots.py             # Create figures
python scripts/13_advanced_metrics.py         # Advanced metrics
python scripts/14_actionable_insights_report.py  # Insights report

# 3. View outputs
open outputs/figures/01_robustness_by_strategy.png
open outputs/figures/05_length_effects.png
open outputs/report/PHASE1_RESULTS_REPORT.md
open outputs/report/ACTIONABLE_INSIGHTS.md
```

**Phase 1 Complete! Now we have:**

- 10 publication-quality figures
- Robustness analysis (SD per strategy)
- Accuracy analysis (% correct per strategy)
- Length effects analysis (short vs medium vs long)
- Model comparison (GPT vs Phi-3)
- Actionable insights report

### Phase 2: Hypothesis Testing (40 minutes total)

```bash
# 1. Generate Phase 2 prompts (2 min)
python scripts/15_generate_phase2_prompts.py

# Verify prompts created
ls prompts/*.txt  # Should show 18 files (v1-v6 for each strategy)

# 2. Run Phase 2 experiment (30 min - walk away!)
python scripts/08_run_experiment.py --phase 2

# 3. Compare phases and test hypotheses (5 min)
python scripts/16_compare_phases_and_test_hypotheses.py

# 4. View results
open outputs/figures/11_phase_comparison.png
cat outputs/tables/hypothesis_test_results.csv
```

**Phase 2 Complete! Now we have:**

- 9 hypothesis test results (H1-H9)
- Statistical significance tests (p-values)
- Phase 1 vs Phase 2 comparison
- Validated predictions

### Deep Dive: Comprehensive Analysis (10 minutes)

```bash
# 1. Run deep dive analysis (5 min)
python scripts/17_deep_dive_prompt_analysis.py

# 2. Generate final comprehensive report (1 min)
python scripts/18_generate_final_report.py

# 3. Read final report (this is your thesis outline!)
open outputs/report/FINAL_COMPREHENSIVE_REPORT.md
```

**Deep Dive Complete! Now we have:**

- Per-CEFR-level accuracy analysis
- Confusion pattern analysis
- Length sensitivity analysis
- Model × Strategy interactions
- Cost-performance optimization
- Final comprehensive thesis-ready report

---

## Complete Workflow (All Steps)

### Option A: Run Everything Sequentially (1.5 hours)

```bash
# Activate environment
conda activate py310
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Phase 1: Run + Analyze
python scripts/08_run_experiment.py --phase 1           # 30 min
python scripts/09_analyze_results.py                    # 2 min
python scripts/10_create_plots.py                       # 3 min
python scripts/13_advanced_metrics.py                   # 5 min
python scripts/14_actionable_insights_report.py         # 1 min

# Phase 2: Generate + Run + Compare
python scripts/15_generate_phase2_prompts.py            # 2 min
python scripts/08_run_experiment.py --phase 2           # 30 min
python scripts/16_compare_phases_and_test_hypotheses.py # 5 min

# Deep Dive: Analyze + Report
python scripts/17_deep_dive_prompt_analysis.py          # 5 min
python scripts/18_generate_final_report.py              # 1 min

# Done! Review outputs
open outputs/report/FINAL_COMPREHENSIVE_REPORT.md
```

### Option B: Run Experiments Only (Skip Analysis)

```bash
# Just run experiments (1 hour total)
python scripts/08_run_experiment.py --phase 1  # 30 min
python scripts/15_generate_phase2_prompts.py   # 2 min
python scripts/08_run_experiment.py --phase 2  # 30 min

# Analyze later when ready
```

### Option C: Skip Experiments (Use Existing Results)

```bash
# If you already ran experiments, just generate reports
python scripts/09_analyze_results.py
python scripts/10_create_plots.py
python scripts/13_advanced_metrics.py
python scripts/16_compare_phases_and_test_hypotheses.py
python scripts/17_deep_dive_prompt_analysis.py
python scripts/18_generate_final_report.py

# Done in ~20 minutes!
```

---

## Script Reference

### Setup Scripts (Run Once)

| Script                        | Purpose                     | Runtime |
| ----------------------------- | --------------------------- | ------- |
| `01_explore_dataset.py`       | Explore dataset structure   | 1 min   |
| `02_create_phase1_sample.py`  | Create 135-essay sample     | 1 min   |
| `03_test_gpt.py`              | Test GPT-4o-mini connection | 30 sec  |
| `04_download_phi3.py`         | Download Phi-3-Mini model   | 10 min  |
| `05_test_phi3.py`             | Test Phi-3-Mini inference   | 2 min   |
| `06_test_prompts.py`          | Test all prompts            | 5 min   |
| `07_analyze_essay_lengths.py` | Analyze length distribution | 1 min   |

### Experiment Runner (Core)

| Script                 | Purpose                    | Runtime |
| ---------------------- | -------------------------- | ------- |
| `08_run_experiment.py` | **Run Phase 1 or Phase 2** | 30 min  |

**Usage:**

```bash
# Run Phase 1
python scripts/08_run_experiment.py --phase 1

# Run Phase 2
python scripts/08_run_experiment.py --phase 2
```

### Phase 1 Analysis Scripts

| Script                             | Purpose                        | Runtime |
| ---------------------------------- | ------------------------------ | ------- |
| `09_analyze_results.py`            | Robustness + accuracy analysis | 2 min   |
| `10_create_plots.py`               | Generate 6 main plots          | 3 min   |
| `11_additional_analysis.py`        | Confusion matrices             | 2 min   |
| `12_generate_phase1_report.py`     | Phase 1 standalone report      | 1 min   |
| `13_advanced_metrics.py`           | QWK, adjacent accuracy         | 5 min   |
| `14_actionable_insights_report.py` | What works/doesn't             | 1 min   |

### Phase 2 Scripts

| Script                                     | Purpose                              | Runtime |
| ------------------------------------------ | ------------------------------------ | ------- |
| `15_generate_phase2_prompts.py`            | Generate 9 hypothesis-driven prompts | 2 min   |
| `16_compare_phases_and_test_hypotheses.py` | Compare phases, test H1-H9           | 5 min   |

### Deep Dive Scripts

| Script                            | Purpose                | Runtime |
| --------------------------------- | ---------------------- | ------- |
| `17_deep_dive_prompt_analysis.py` | Comprehensive analysis | 5 min   |
| `18_generate_final_report.py`     | Final thesis report    | 1 min   |

---

## Outputs Guide

### Figures (15 total)

```
outputs/figures/
  ├── 01_robustness_by_strategy.png       # Minimal best (SD=0.163)
  ├── 02_accuracy_by_strategy.png         # Rubric best (35.6%)
  ├── 03_robustness_by_model.png          # GPT vs Phi-3
  ├── 04_accuracy_by_model.png            # GPT outperforms
  ├── 05_length_effects.png               # Short 72.6% vs Long 6.4%!
  ├── 06_length_by_strategy.png           # Strategy × Length interaction
  ├── 07_model_comparison.png             # Model comparison dashboard
  ├── 08_prompt_variant_comparison.png    # Variant agreement
  ├── 09_combined_dashboard.png           # Complete overview
  ├── 10_advanced_metrics.png             # QWK, adjacent accuracy
  ├── 11_phase_comparison.png             # Phase 1 vs Phase 2
  ├── 12_per_level_accuracy.png           # Heatmap by CEFR level
  ├── 13_confusion_patterns.png           # Best vs worst confusion
  ├── 14_length_sensitivity.png           # Length handling by prompt
  └── 15_model_strategy_interactions.png  # Model × Strategy effects
```

**Key Figures for Thesis:**

- **Figure 01:** Shows minimal prompts most robust
- **Figure 05:** Shows dramatic length effect (key finding!)
- **Figure 11:** Shows hypothesis validation
- **Figure 12:** Shows per-level heterogeneity

### Tables (20+ total)

```
outputs/tables/
  ├── robustness_summary.csv              # SD by strategy/prompt
  ├── accuracy_summary.csv                # Accuracy by strategy/prompt
  ├── comprehensive_metrics.csv           # All metrics combined
  ├── hypothesis_test_results.csv         # H1-H9 with p-values
  ├── per_level_accuracy.csv              # Accuracy per CEFR level
  ├── common_confusions.csv               # Top misclassifications
  ├── length_sensitivity.csv              # Performance by length
  ├── model_strategy_interactions.csv     # Model × Strategy
  ├── variant_consistency.csv             # Variant agreement rates
  └── cost_performance.csv                # ROI analysis
```

### Reports (3 total)

```
outputs/report/
  ├── PHASE1_RESULTS_REPORT.md            # Phase 1 standalone
  ├── ACTIONABLE_INSIGHTS.md              # What works/doesn't
  └── FINAL_COMPREHENSIVE_REPORT.md       # Master thesis report
```

**Must Read:** `FINAL_COMPREHENSIVE_REPORT.md` - This is literally your thesis Results + Discussion outline!

---

## Troubleshooting

### Problem: ModuleNotFoundError: No module named 'config'

**Solution:**

```bash
# Make sure you're in project root
cd ~/Desktop/Exeter/Y3/Individual\ Project/ECM3401-LLM-Essay-Scoring/

# Install in development mode
pip install -e .

# OR run with module syntax
python -m scripts.09_analyze_results
```

### Problem: MPS not available

**Solution:**

```bash
# Check PyTorch installation
pip install torch torchvision torchaudio

# Verify MPS
python -c "import torch; print(torch.backends.mps.is_available())"

# If still false, check macOS version (needs 12.3+)
sw_vers
```

### Problem: OpenAI API key not found

**Solution:**

```bash
# Check .env file exists
ls -la .env

# Check contents
cat .env

# Should see: OPENAI_API_KEY=sk-proj-...

# If missing, create it:
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Problem: Dataset not found

**Solution:**

```bash
# Verify dataset path in config.py
python -c "from config import CORPUS_FILE; print(CORPUS_FILE)"

# Should print: .../write-and-improve-corpus-2024-v2/whole-corpus/en-writeandimprove2024-corpus.tsv

# If wrong, update DATASET_ROOT in config.py
```

### Problem: Phi-3 inference very slow

**Solution:**

```bash
# Set environment variable for MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Add to ~/.zshrc for permanent fix
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc
source ~/.zshrc

# Consider using only GPT-4o-mini if Phi-3 too slow
# Edit config.py: MODELS = ['gpt-4o-mini']  # Remove 'phi-3-mini'
```

### Problem: FileNotFoundError during analysis

**Solution:**

```bash
# Check experiment ran successfully
ls data/results/phase1_experiment_results.csv

# If missing, re-run experiment
python scripts/08_run_experiment.py --phase 1

# Check for errors in log
cat data/results/phase1_experiment_log.txt
```

---

## Project Structure

```
ECM3401-LLM-Essay-Scoring/
│
├── README.md                      # This file!
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── config.py                      # Configuration file
├── .env                          # API keys (not in git)
├── .gitignore
│
├── data/
│   ├── processed/
│   │   ├── phase1_sample_100.csv       # 135 essays (stratified)
│   │   └── phase1_essay_ids.csv
│   └── results/
│       ├── phase1_experiment_results.csv  # 2,430 predictions
│       ├── phase1_experiment_log.txt
│       ├── phase2_experiment_results.csv  # 2,430 predictions
│       └── phase2_experiment_log.txt
│
├── prompts/
│   ├── minimal_v1.txt             # Phase 1 minimal prompts
│   ├── minimal_v2.txt
│   ├── minimal_v3.txt
│   ├── minimal_v4.txt             # Phase 2 minimal prompts
│   ├── minimal_v5.txt
│   ├── minimal_v6.txt
│   ├── rubric_v1-v6.txt          # Rubric prompts
│   └── cot_v1-v6.txt             # Chain-of-Thought prompts
│
├── scripts/
│   ├── 01_explore_dataset.py
│   ├── 02_create_phase1_sample.py
│   ├── 03_test_gpt.py
│   ├── 04_download_phi3.py
│   ├── 05_test_phi3.py
│   ├── 06_test_prompts.py
│   ├── 07_analyze_essay_lengths.py
│   ├── 08_run_experiment.py      # ⭐ Main experiment runner
│   ├── 09_analyze_results.py
│   ├── 10_create_plots.py
│   ├── 11_additional_analysis.py
│   ├── 12_generate_phase1_report.py
│   ├── 13_advanced_metrics.py
│   ├── 14_actionable_insights_report.py
│   ├── 15_generate_phase2_prompts.py
│   ├── 16_compare_phases_and_test_hypotheses.py
│   ├── 17_deep_dive_prompt_analysis.py
│   └── 18_generate_final_report.py
│
├── outputs/
│   ├── figures/                  # 15 publication-quality figures
│   ├── tables/                   # 20+ CSV tables
│   └── report/                   # 3 markdown reports
│
└── models/
    └── phi3_cache/               # Phi-3-Mini (~8GB)
```

---

## For Thesis Writing

### Using This Project in Thesis

**Methods Chapter 4:**

```markdown
4.1 Dataset
→ Reference: data/processed/phase1_sample_100.csv
→ 135 essays, stratified by CEFR level

4.2 Phase 1 Design
→ Reference: prompts/minimal_v1-v3.txt, rubric_v1-v3.txt, cot_v1-v3.txt
→ 9 prompts, 3 paraphrases each

4.3 Phase 2 Design
→ Reference: scripts/15_generate_phase2_prompts.py
→ Hypothesis-driven design, 9 controlled variants

4.4 Evaluation Metrics
→ Reference: scripts/13_advanced_metrics.py
→ Robustness (SD), Accuracy, QWK, Adjacent Accuracy
```

**Results Chapter 5:**

```markdown
5.1 Robustness Analysis
→ Use: outputs/figures/01_robustness_by_strategy.png
→ Report: Minimal SD=0.163 (best)

5.2 Accuracy Analysis
→ Use: outputs/figures/02_accuracy_by_strategy.png
→ Report: Rubric 35.6% (best)

5.3 Length Effects
→ Use: outputs/figures/05_length_effects.png
→ Report: Short 72.6% vs Long 6.4% (11.3× gap!)

... (continue for all 10 sections)
```

**Discussion Chapter 6:**

```markdown
6.1 What Worked
→ Use: outputs/report/ACTIONABLE_INSIGHTS.md

6.2 What Didn't Work
→ Use: outputs/report/ACTIONABLE_INSIGHTS.md

6.3 How to Improve
→ Use: outputs/report/ACTIONABLE_INSIGHTS.md

... (continue for all 8 sections)
```

**Use the Final Report as Your Outline:**

- Open: `outputs/report/FINAL_COMPREHENSIVE_REPORT.md`
- This is literally your Results + Discussion outline
- Copy structure, expand with details from figures/tables

---

## Expected Results

### Phase 1 Key Findings

- **Universal robustness:** All prompts SD < 0.3 (deployment threshold < 3.0)
- **Simpler is better:** Minimal prompts most robust (SD=0.163)
- **Length matters:** Short essays 72.6% vs Long 6.4% (major finding!)
- **Model gap:** GPT-4o-mini outperforms Phi-3-Mini by 10pp

### Phase 2 Key Findings

- **H1 supported:** Ultra-simple prompts improve robustness (p<0.05)
- **H2 supported:** Length-aware prompts help long essays (p<0.05)
- **7/9 hypotheses supported** (78% validation rate)

### Deep Dive Insights

- **Per-level heterogeneity:** 2.5× performance gap (A2: 45% vs C2: 18%)
- **Confusion patterns:** Best prompts 85% errors adjacent, worst 62%
- **Cost optimization:** minimal_v1 best accuracy-per-dollar
- **Deployment strategy:** Low stakes 90% automation, high stakes 0%

---

## Next Steps After Running

1. **Review all 15 figures** - These go directly in your thesis
2. **Read FINAL_COMPREHENSIVE_REPORT.md** - This is your Results + Discussion outline
3. **Read ACTIONABLE_INSIGHTS.md** - This tells you what to discuss
4. **Start thesis writing** - Use figures, tables, and reports directly
5. **Meet with supervisor** - Show them the comprehensive report

---

## Contact

**Student:** Sansiri Charoenpong (Siemon)  
**Email:** sc1332@exeter.ac.uk  
**Supervisor:** Dr. Rodrigo Souza Wilkens  
**Meeting:** Wednesdays, 3:00 PM

---

## License

This project is for academic purposes as part of the ECM3401 Individual Project module at the University of Exeter.

---

## Acknowledgments

- Write & Improve Corpus team (Cambridge Assessment)
- Dr. Rodrigo Souza Wilkens (supervisor)
- Anthropic (Claude API)
- OpenAI (GPT-4o-mini API)
- Microsoft (Phi-3-Mini model)

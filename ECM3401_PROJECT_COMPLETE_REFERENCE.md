# ECM3401 Complete Project Reference
**Generated:** December 11, 2025  
**Purpose:** Complete documentation for continuing in new conversations

---

## TABLE OF CONTENTS
1. [Project Overview](#project-overview)
2. [Dataset Structure](#dataset-structure)
3. [Project File Structure](#project-file-structure)
4. [Configuration Files](#configuration-files)
5. [Scripts Created](#scripts-created)
6. [Environment Setup](#environment-setup)
7. [Key Decisions Made](#key-decisions-made)
8. [Commands Reference](#commands-reference)
9. [Current Status](#current-status)
10. [Next Steps](#next-steps)

---

## PROJECT OVERVIEW

**Title:** Measuring Semantic Robustness in LLM-Based Essay Scoring  
**Student:** Sansiri Charoenpong (Siemon)  
**Supervisor:** Dr. Rodrigo Souza Wilkens  
**Institution:** University of Exeter, Computer Science  
**Meeting Schedule:** Wednesdays, 3:00 PM  
**Project Code:** ECM3401

**Research Questions:**
1. RQ1: Are LLM CEFR predictions robust to paraphrasing?
2. RQ2: Does prompt complexity affect robustness?
3. RQ3: Can retrieval-based few-shot reduce variance?
4. RQ4: Does model architecture affect robustness?
5. RQ5: Cost-robustness tradeoff?

**Methodology:**
- 2 models (commercial + open-source)
- 4 prompting strategies × 3 paraphrases = 12 prompt variants
- 100 essays (Phase 1)
- Total predictions: 2,400
- Target: SD < 3% for deployment readiness

---

## DATASET STRUCTURE

### Dataset Location
```
/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/write-and-improve-corpus-2024-v2/
```

### Complete Folder Structure

```
write-and-improve-corpus-2024-v2/
│
├── README
│
├── multigec-2025-files/
│   ├── en-writeandimprove2024-orig-dev.md
│   ├── en-writeandimprove2024-orig-dev.tmp
│   ├── en-writeandimprove2024-orig-test.md
│   ├── en-writeandimprove2024-orig-test.tmp
│   ├── en-writeandimprove2024-orig-train.md
│   ├── en-writeandimprove2024-orig-train.tmp
│   ├── en-writeandimprove2024-ref1-dev.md
│   ├── en-writeandimprove2024-ref1-dev.tmp
│   ├── en-writeandimprove2024-ref1-dev.m2
│   ├── en-writeandimprove2024-ref1-train.md
│   ├── en-writeandimprove2024-ref1-train.tmp
│   │
│   └── local_eval/
│       ├── cleanconll.py
│       ├── download_spacy_english_model.py
│       ├── download_spacy_udpipe_models.py
│       ├── gleu_errant_evaluation.py
│       ├── README.md
│       ├── udpipe2_client_orignal.py
│       ├── utils_transform_markdown_to_one_essay_per_line.py
│       │
│       └── ref/
│           ├── en-writeandimprove2024-orig-dev.md
│           ├── en-writeandimprove2024-orig-dev.tmp
│           ├── en-writeandimprove2024-ref1-dev.m2
│           ├── en-writeandimprove2024-ref1-dev.tmp
│           └── en-writeandimprove2024-ref1-dev.md
│
├── user-prompt-final-versions/
│   ├── en-writeandimprove2024-final-versions-dev-essays.conll
│   ├── en-writeandimprove2024-final-versions-dev-essays.m2
│   ├── en-writeandimprove2024-final-versions-dev-sentences.conll
│   ├── en-writeandimprove2024-final-versions-dev-sentences.corr
│   ├── en-writeandimprove2024-final-versions-dev-sentences.ids
│   ├── en-writeandimprove2024-final-versions-dev-sentences.m2
│   ├── en-writeandimprove2024-final-versions-dev-sentences.orig
│   ├── en-writeandimprove2024-final-versions-m2-essay-info
│   ├── en-writeandimprove2024-final-versions-m2-sentence-info
│   ├── en-writeandimprove2024-final-versions-orig-dev
│   ├── en-writeandimprove2024-final-versions-orig-dev.tmp
│   ├── en-writeandimprove2024-final-versions-orig-test
│   ├── en-writeandimprove2024-final-versions-orig-test.tmp
│   ├── en-writeandimprove2024-final-versions-orig-train
│   ├── en-writeandimprove2024-final-versions-orig-train.tmp
│   ├── en-writeandimprove2024-final-versions-ref1-dev
│   ├── en-writeandimprove2024-final-versions-ref1-dev.tmp
│   ├── en-writeandimprove2024-final-versions-ref1-train
│   ├── en-writeandimprove2024-final-versions-ref1-train.tmp
│   ├── en-writeandimprove2024-final-versions-test-sentences.ids
│   ├── en-writeandimprove2024-final-versions-test-sentences.orig
│   ├── en-writeandimprove2024-final-versions-train-essays.conll
│   ├── en-writeandimprove2024-final-versions-train-essays.m2
│   ├── en-writeandimprove2024-final-versions-train-sentences.conll
│   ├── en-writeandimprove2024-final-versions-train-sentences.corr
│   ├── en-writeandimprove2024-final-versions-train-sentences.ids
│   ├── en-writeandimprove2024-final-versions-train-sentences.m2
│   └── en-writeandimprove2024-final-versions-train-sentences.orig
│
├── user-prompt-first-versions/
│   ├── en-writeandimprove2024-first-versions-dev-essays.conll
│   ├── en-writeandimprove2024-first-versions-dev-essays.m2
│   ├── en-writeandimprove2024-first-versions-dev-sentences.conll
│   ├── en-writeandimprove2024-first-versions-dev-sentences.corr
│   ├── en-writeandimprove2024-first-versions-dev-sentences.ids
│   ├── en-writeandimprove2024-first-versions-dev-sentences.m2
│   ├── en-writeandimprove2024-first-versions-dev-sentences.orig
│   ├── en-writeandimprove2024-first-versions-m2-essay-info
│   ├── en-writeandimprove2024-first-versions-m2-sentence-info
│   ├── en-writeandimprove2024-first-versions-orig-dev
│   ├── en-writeandimprove2024-first-versions-orig-dev.tmp
│   ├── en-writeandimprove2024-first-versions-orig-test
│   ├── en-writeandimprove2024-first-versions-orig-test.tmp
│   ├── en-writeandimprove2024-first-versions-orig-train
│   ├── en-writeandimprove2024-first-versions-orig-train.tmp
│   ├── en-writeandimprove2024-first-versions-ref1-dev
│   ├── en-writeandimprove2024-first-versions-ref1-dev.tmp
│   ├── en-writeandimprove2024-first-versions-ref1-train
│   ├── en-writeandimprove2024-first-versions-test-sentences.ids
│   ├── en-writeandimprove2024-first-versions-test-sentences.orig
│   ├── en-writeandimprove2024-first-versions-train-essays.conll
│   ├── en-writeandimprove2024-first-versions-train-essays.m2
│   ├── en-writeandimprove2024-first-versions-train-sentences.conll
│   ├── en-writeandimprove2024-first-versions-train-sentences.corr
│   ├── en-writeandimprove2024-first-versions-train-sentences.ids
│   ├── en-writeandimprove2024-first-versions-train-sentences.m2
│   └── en-writeandimprove2024-first-versions-train-sentences.orig
│
└── whole-corpus/ ★ PRIMARY DATA SOURCE ★
    ├── en-writeandimprove2024-corpus.tsv (23,216 essays)
    └── en-writeandimprove2024-prompts-info
```

### Corpus TSV Structure

**File:** `whole-corpus/en-writeandimprove2024-corpus.tsv`  
**Rows:** 23,216 total essays  
**Usable:** 4,546 essays (final versions, human-rated, train/dev splits)

**Columns (15 total):**
1. `public_essay_id` - Unique identifier
2. `created_epoch` - Timestamp (epoch)
3. `created_timestamp` - Human-readable date
4. `public_prompt_id` - Which writing prompt
5. `public_user_id` - Which user wrote it
6. `user_prompt` - User+prompt combined ID
7. `essay_version_num` - Version number (1st, 2nd, 3rd...)
8. `is_first_version` - Boolean
9. `is_final_version` - Boolean ★ FILTER ON THIS
10. `language` - L1 background (22 languages)
11. `text` - ★ THE ACTUAL ESSAY TEXT ★
12. `wi_suspecttokens` - Detected errors
13. `automarker_cefr_level` - Auto-marker prediction
14. `humannotator_cefr_level` - ★ GROUND TRUTH LABEL ★
15. `split` - train/dev/test/NA

### CEFR Distribution (Raw)

**From 4,546 usable essays:**
```
A1       1
A1+      9
A2     182
A2+    790
B1     878
B1+    929
B2     793
B2+    500
C1     346
C1+     91
C2      27
```

**After mapping (combine + levels):**
```
A2:    972 essays (182 + 790)
B1:  1,807 essays (878 + 929)
B2:  1,293 essays (793 + 500)
C1:    437 essays (346 + 91)
C2:     27 essays
```

### L1 Language Distribution (Top 10)

```
Spanish       834
Portuguese    596
Arabic        396
Vietnamese    356
Japanese      326
Italian       253
Polish        198
French        192
Chinese       174
Romanian      166
(12 more languages with fewer samples)
```

---

## PROJECT FILE STRUCTURE

```
/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/ECM3401-LLM-Essay-Scoring/
│
├── config.py ★ MAIN CONFIGURATION
├── requirements.txt ★ PYTHON 3.10 COMPATIBLE
├── setup.py
├── .env ★ API KEYS (NOT IN GIT)
├── .gitignore
├── README.md
├── ECM3401_PROJECT_COMPLETE_REFERENCE.md ★ THIS FILE
│
├── data/
│   ├── raw/ (empty - external dataset)
│   ├── processed/
│   │   ├── phase1_sample_100.csv ★ CREATED
│   │   └── phase1_essay_ids.csv ★ CREATED
│   └── results/ (empty - will hold predictions)
│
├── scripts/
│   ├── 01_explore_dataset.py ★ WORKING
│   ├── 02_create_phase1_sample.py ★ WORKING
│   ├── 03_test_gpt.py ★ WORKING
│   ├── 04_download_llama.py ★ CREATED
│   └── 05_test_llama.py ★ CREATED
│
├── src/ (future utility functions)
│   └── __init__.py
│
├── prompts/ (to be created)
│   ├── minimal_v1.txt
│   ├── minimal_v2.txt
│   ├── minimal_v3.txt
│   ├── rubric_v1.txt
│   ├── rubric_v2.txt
│   ├── rubric_v3.txt
│   ├── cot_v1.txt
│   ├── cot_v2.txt
│   └── cot_v3.txt
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── report/
│
├── notebooks/ (optional Jupyter notebooks)
│   └── exploration.ipynb
│
└── models/
    └── llama_cache/ (Llama model downloaded here, ~8GB)
```

---

## KEY DECISIONS MADE

### Model Choices

**Commercial Model:** GPT-4o-mini (NOT GPT-5-nano)
- **Why:** GPT-5-nano doesn't support temperature=0
- **Temperature=0 critical:** Ensures deterministic outputs for robustness measurement
- **Cost:** $0.12 for 2,400 predictions

**Open-Source Model:** Meta-Llama-3-8B-Instruct or Llama-3.1-8B-Instruct
- **Why:** Runs on M2 Pro MPS, no external GPU needed
- **Version note:** Using 3.1 or 3, not 3.3 (availability/compatibility issues)
- **Cost:** $0 (runs locally)

### CEFR Level Handling

**Decision:** Combine intermediate levels (A2+, B1+, etc.) with base levels

**Mapping:**
```python
CEFR_MAPPING = {
    'A1': 'A2', 'A1+': 'A2',
    'A2': 'A2', 'A2+': 'A2',
    'B1': 'B1', 'B1+': 'B1',
    'B2': 'B2', 'B2+': 'B2',
    'C1': 'C1', 'C1+': 'C1',
    'C2': 'C2', 'C2+': 'C2'
}
```

**Rationale:**
- Increases sample sizes significantly (A2: 182→972, B1: 878→1807)
- Research focuses on robustness, not fine-grained CEFR distinctions
- 5-level classification cleaner for analysis

### Temperature Setting

**Decision:** temperature=0.0 for BOTH models

**Why:**
- Deterministic outputs
- Variance measurements = pure paraphrase effect
- No random sampling noise

### Python Version

**Decision:** Stay with Python 3.10

**Why:**
- py310 environment already working
- Some packages (bitsandbytes 0.43+, numpy 1.25+) require Python 3.11
- Python 3.10-compatible versions exist and work fine
- Avoid breaking changes mid-project

---

## ENVIRONMENT SETUP

### Python Environment

**Environment:** `py310`  
**Python Version:** 3.10.x  
**Created with:** conda

```bash
# Activate
conda activate py310

# Verify
python --version  # Should show Python 3.10.x
```

### Hardware

**Machine:** M2 Pro MacBook  
**GPU:** Apple Silicon (MPS)  
**Memory:** 16GB+ unified memory  
**Storage:** ~10GB free needed for Llama

### GPU Status

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")  # True
print(f"MPS built: {torch.backends.mps.is_built()}")  # True
```

---

## CONFIGURATION FILES

### requirements.txt (Python 3.10 Compatible)

```txt
# Core Data Science
pandas==2.0.3
numpy==1.24.4
scipy==1.11.4
scikit-learn==1.3.2

# PyTorch (MPS Support for Apple Silicon)
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2

# LLM APIs
openai==2.11.0
httpx==0.28.1
httpcore==1.0.9

# Transformers Ecosystem
transformers==4.44.2
accelerate==0.34.2
bitsandbytes==0.42.0
sentence-transformers==2.7.0
tokenizers==0.19.1
safetensors==0.4.5
huggingface-hub==0.25.2

# Vector Search
faiss-cpu==1.8.0.post1

# OpenAI Utilities
tiktoken==0.7.0

# Text Processing
nltk==3.8.1
textstat==0.7.3

# Visualization
matplotlib==3.8.4
seaborn==0.13.2

# Utilities
tqdm==4.66.5
python-dotenv==1.0.1
pyyaml==6.0.2
requests==2.32.3
pydantic==2.9.2

# Jupyter (Optional)
jupyter==1.0.0
jupyterlab==4.0.13
ipykernel==6.29.5
```

### config.py Key Settings

```python
# Dataset paths
DATASET_ROOT = Path('/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/write-and-improve-corpus-2024-v2')
CORPUS_FILE = DATASET_ROOT / "whole-corpus" / "en-writeandimprove2024-corpus.tsv"

# Experiment settings
RANDOM_SEED = 42
PHASE1_SAMPLE_SIZE = 100
ESSAYS_PER_LEVEL = 20
CEFR_LEVELS = ['A2', 'B1', 'B2', 'C1', 'C2']

# GPT-4o-mini
GPT_MODEL = "gpt-4o-mini"
GPT_TEMPERATURE = 0.0
GPT_MAX_TOKENS = 50

# Llama
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # or Llama-3.1
LLAMA_DEVICE = "mps"  # Apple Silicon GPU
LLAMA_TEMPERATURE = 0.0
LLAMA_MAX_TOKENS = 50
LLAMA_QUANTIZATION = "4bit"
```

### .env File

```bash
OPENAI_API_KEY=sk-proj-your-key-here
HUGGINGFACE_TOKEN=hf_your-token-here
```

---

## COMMANDS REFERENCE

### Environment

```bash
# Activate
conda activate py310

# Install packages
pip install -r requirements.txt

# Verify
python --version
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Running Scripts

```bash
# Explore dataset
python -m scripts.01_explore_dataset

# Create sample
python -m scripts.02_create_phase1_sample

# Test GPT
python ./scripts/03_test_gpt.py

# Download Llama (one-time, ~20 min)
python ./scripts/04_download_llama.py

# Test Llama
python ./scripts/05_test_llama.py
```

---

## CURRENT STATUS

### ✅ Completed

- [x] Dataset acquired (Write & Improve 2024)
- [x] Dataset explored (23,216 essays, 4,546 usable)
- [x] Phase 1 sample created (100 essays, stratified)
- [x] Python environment configured (py310)
- [x] Requirements.txt finalized (Python 3.10 compatible)
- [x] Config.py complete
- [x] CEFR mapping implemented
- [x] GPT-4o-mini tested and working
- [x] Scripts 01-05 created
- [x] .env file configured
- [x] Project structure established

### ⏳ In Progress

- [ ] Install Python 3.10 compatible requirements
- [ ] Download Llama model (~20 min)
- [ ] Test Llama on M2 Pro
- [ ] Verify both models work

### 📅 Next Steps (This Week)

- [ ] Complete Llama setup
- [ ] Wednesday supervisor meeting
- [ ] Write remaining prompt variants (rubric, CoT)
- [ ] Run 10-essay pilot test
- [ ] Begin Phase 1 validation

---

## TROUBLESHOOTING NOTES

### Common Errors Encountered

**Error 1:** `No module named 'config'`  
**Solution:** Run with `python -m scripts.scriptname` OR `python ./scripts/scriptname.py`

**Error 2:** `bitsandbytes==0.45.0 not found`  
**Solution:** Python 3.10 max version is 0.42.0 (updated in requirements.txt)

**Error 3:** `rope_scaling dictionary error`  
**Solution:** transformers version too old, upgraded to 4.44.2

**Error 4:** `max_tokens not supported (GPT-5)`  
**Solution:** GPT-5 uses `max_completion_tokens` instead

**Error 5:** `temperature not supported (GPT-5-nano)`  
**Solution:** GPT-5-nano only supports temperature=1, switched to GPT-4o-mini

**Error 6:** `MPS not available`  
**Solution:** Reinstall PyTorch: `pip install torch torchvision torchaudio`

**Error 7:** `Llama-3.3-8B not found`  
**Solution:** Use Meta-Llama-3-8B-Instruct or Llama-3.1-8B-Instruct instead

### Version Compatibility Matrix (Python 3.10)

| Package | Max Version | Reason |
|---------|-------------|--------|
| bitsandbytes | 0.42.0 | 0.43+ requires Python 3.11 |
| torch | 2.1.2 | Stable MPS support |
| transformers | 4.44.2 | Supports Llama 3/3.1 |
| numpy | 1.24.4 | 1.25+ requires Python 3.11 |
| pandas | 2.0.3 | Stable version |

---

## NEXT STEPS - IMMEDIATE (30 MIN)

```bash
# 1. Install requirements (5 min)
conda activate py310
pip install -r requirements.txt

# 2. Verify installation (1 min)
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# 3. Download Llama (20 min - can walk away)
python ./scripts/04_download_llama.py

# 4. Test Llama (3 min)
python ./scripts/05_test_llama.py

# 5. Confirm both models work
python ./scripts/03_test_gpt.py
```

---

## HOW TO USE THIS DOCUMENT IN A NEW CHAT

**Simply upload this file or paste its content and say:**

> "I'm continuing my ECM3401 project. Here's my complete reference document. We were setting up Llama on M2 Pro. Where did we leave off?"

Or:

> "Reference: ECM3401_PROJECT_COMPLETE_REFERENCE.md  
> Question: How do I run the Llama inference test?"

**I'll have ALL context immediately!**

---

## IMPORTANT FILE PATHS

```bash
# Dataset
/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/write-and-improve-corpus-2024-v2/

# Project
/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/ECM3401-LLM-Essay-Scoring/

# Main corpus file
/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/write-and-improve-corpus-2024-v2/whole-corpus/en-writeandimprove2024-corpus.tsv

# Phase 1 sample
/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/ECM3401-LLM-Essay-Scoring/data/processed/phase1_sample_100.csv

# Config
/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/ECM3401-LLM-Essay-Scoring/config.py
```

---

**END OF REFERENCE DOCUMENT**  
**Generated:** December 11, 2025  
**For:** ECM3401 Individual Project  
**Student:** Sansiri Charoenpong (Siemon)

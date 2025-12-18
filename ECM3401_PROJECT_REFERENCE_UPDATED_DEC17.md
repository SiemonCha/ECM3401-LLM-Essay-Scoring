# ECM3401 Complete Project Reference - UPDATED
**Generated:** December 17, 2025  
**Last Updated:** After supervisor approval meeting  
**Purpose:** Complete documentation with ALL decisions and current state

---

## TABLE OF CONTENTS
1. [Project Overview](#project-overview)
2. [Final Experimental Design](#final-experimental-design)
3. [Dataset & Sample](#dataset-sample)
4. [Models Configuration](#models-configuration)
5. [Prompting Strategies](#prompting-strategies)
6. [File Locations](#file-locations)
7. [Key Decisions Timeline](#key-decisions-timeline)
8. [Current Status](#current-status)
9. [Next Steps](#next-steps)

---

## PROJECT OVERVIEW

**Title:** Measuring Semantic Robustness in LLM-Based Essay Scoring  
**Student:** Sansiri Charoenpong (Siemon)  
**Supervisor:** Dr. Rodrigo Souza Wilkens  
**Institution:** University of Exeter, Computer Science  
**Meeting Schedule:** Wednesdays, 3:00 PM  
**Project Code:** ECM3401  
**Hardware:** MacBook M2 Pro, 16GB RAM

**Core Research Question:**
> "Are LLM predictions for CEFR essay classification robust to semantically equivalent prompt paraphrases?"

**Updated Research Questions (After removing few-shot):**
1. **RQ1:** Are LLM CEFR predictions robust to paraphrasing?
2. **RQ2:** Does prompt complexity (minimal vs rubric vs CoT) affect robustness?
3. **RQ3:** Do commercial models (GPT) differ from open-source (Phi-3) in robustness?
4. **RQ4:** How does essay length affect robustness patterns?
5. **RQ5:** What are deployment recommendations based on robustness thresholds?

---

## FINAL EXPERIMENTAL DESIGN

### Sample Size: 135 Essays ✓

**Reasoning:**
- C2 essays are extremely rare (only 27 available in entire corpus)
- Used ALL 27 C2 essays + sampled 27 from each other level
- Perfectly balanced across CEFR levels
- Stratified random sampling with seed=42 for reproducibility

### Predictions: 2,430 Total

**Calculation:**
```
135 essays
×   9 prompts (3 strategies × 3 paraphrases)
×   2 models (GPT-4o-mini + Phi-3-Mini)
─────────────────────────────────────────
= 2,430 predictions
```

### Expected Runtime: ~21 Hours

```
GPT-4o-mini: 135 × 9 = 1,215 predictions × ~1 sec = 20 minutes
Phi-3-Mini:  135 × 9 = 1,215 predictions × ~30 sec = 10 hours
Total: ~10.5 hours per model × 2 = 21 hours
```

**Schedule:** Thursday 6pm → Friday 3pm (overnight run)

### Cost: $0.12

```
GPT-4o-mini: $0.12 (API costs)
Phi-3-Mini:  $0.00 (runs locally)
Total:       $0.12
```

---

## DATASET & SAMPLE

### Source Dataset

**Name:** Write & Improve Corpus 2024  
**Location:** `/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/write-and-improve-corpus-2024-v2/`  
**Main File:** `whole-corpus/en-writeandimprove2024-corpus.tsv`

**Total Essays:** 23,216 total  
**Usable Essays:** 4,546 (final versions, human-rated, train/dev splits)

### CEFR Distribution (Full Corpus After Mapping)

```
A2:    972 essays (includes A1, A1+, A2, A2+)
B1:  1,807 essays (includes B1, B1+)
B2:  1,293 essays (includes B2, B2+)
C1:    437 essays (includes C1, C1+)
C2:     27 essays (includes C2, C2+)
Total: 4,546 usable essays
```

**Key Finding:** C2 is extremely rare (0.6% of corpus) - realistic learner distribution!

### Phase 1 Sample (Final)

**Sample File:** `data/processed/phase1_sample_100.csv` (filename kept but contains 135)  
**Essay IDs:** `data/processed/phase1_essay_ids.csv`  
**With Lengths:** `data/processed/phase1_sample_with_lengths.csv`

**Sample Composition:**
```
Total: 135 essays
Per CEFR level: 27 essays each (A2, B1, B2, C1, C2)
Random seed: 42 (reproducible)
```

**Length Distribution:**
```
Overall:
- Min: 32 words
- Max: 430 words
- Mean: 184.5 words
- Median: 191 words

By Category:
- Short (<100 words): 32 essays
- Medium (100-200 words): 39 essays  
- Long (200+ words): 64 essays

By CEFR Level (Mean Length):
- A2:  60.4 words
- B1: 140.9 words
- B2: 200.6 words
- C1: 253.5 words
- C2: 267.3 words
```

**Key Pattern:** Strong correlation between CEFR level and essay length (validates stratification!)

### Length × CEFR Crosstab

```
Length Category | A2  B1  B2  C1  C2
----------------+-------------------
Short (<100)    | 26   6   0   0   0
Medium (100-200)|  1  18  15   4   1
Long (200+)     |  0   3  12  23  26
```

**Analysis Implication:**
- Short essays: Mostly A2/B1 (test robustness for lower proficiency)
- Medium essays: ALL 5 levels represented ✓ (best for comprehensive comparison)
- Long essays: Mostly B2/C1/C2 (test robustness for higher proficiency)

---

## MODELS CONFIGURATION

### Model 1: GPT-4o-mini (Commercial)

**Provider:** OpenAI API  
**Model String:** `gpt-4o-mini`  
**Temperature:** 0.0 (deterministic - CRITICAL!)  
**Max Tokens:** 10 (forces brevity)  
**Speed:** ~1 second per essay  
**Cost:** $0.12 for 1,215 predictions

**Why GPT-4o-mini (not GPT-5-nano):**
- GPT-5-nano doesn't support temperature=0
- Deterministic outputs essential for robustness measurement
- Any variance must be from paraphrase, not sampling

### Model 2: Phi-3-Mini (Open-Source)

**Provider:** Microsoft via HuggingFace  
**Model String:** `microsoft/Phi-3-mini-4k-instruct`  
**Size:** 3.8B parameters  
**Device:** MPS (Apple Silicon GPU)  
**Temperature:** 0.0 (deterministic)  
**Max Tokens:** 10  
**Speed:** ~30 seconds per essay on M2 Pro  
**Cost:** $0 (runs locally)

**Why Phi-3-Mini (not Llama-3-8B):**
- Llama-3-8B: 720 seconds/essay (480 hours total) - IMPRACTICAL ❌
- Phi-3-Mini: 30 seconds/essay (10 hours total) - PRACTICAL ✓
- 24x faster than Llama!
- High quality (Microsoft's best small model)
- No HuggingFace token needed (not gated)

**MPS Fallback:** `PYTORCH_ENABLE_MPS_FALLBACK=1` (for unsupported ops)

### Why NOT Other Options:

**Ollama (considered but rejected):**
- Adds dependency complexity
- PyTorch approach more standard
- Already working with Phi-3

**Llama-3-8B (rejected):**
- 720s per essay too slow
- 480 hours = 20 days runtime
- Even with Ollama (12s) still slower than Phi-3

**Qwen2.5-7B (considered but rejected):**
- 100-200s per essay
- Better quality but still too slow
- Phi-3 good enough for robustness research

---

## PROMPTING STRATEGIES

### Strategy 1: Minimal (Baseline)

**Purpose:** Direct classification with no guidance  
**Files:** `prompts/minimal_v1.txt`, `minimal_v2.txt`, `minimal_v3.txt`

**Example (v1):**
```
Classify this essay's CEFR level (A2, B1, B2, C1, or C2).

{essay_text}

Respond with ONLY the CEFR level (e.g., "B1"). Do not provide explanation.

CEFR Level:
```

**Paraphrases:**
- v1: "Classify this essay's CEFR level..."
- v2: "Determine the CEFR proficiency level..."
- v3: "What is the CEFR rating..."

### Strategy 2: Rubric-Guided (Structured)

**Purpose:** Provides explicit CEFR descriptors  
**Files:** `prompts/rubric_v1.txt`, `rubric_v2.txt`, `rubric_v3.txt`

**Structure:**
1. Instruction
2. A2 descriptor (Elementary)
3. B1 descriptor (Intermediate)
4. B2 descriptor (Upper-Intermediate)
5. C1 descriptor (Advanced)
6. C2 descriptor (Proficient)
7. Essay text
8. Output instruction (level only)

**Paraphrases:**
- v1: "using the following criteria..."
- v2: "according to these descriptors..."
- v3: "using the CEFR framework below..."

### Strategy 3: Chain-of-Thought (Reasoning)

**Purpose:** Encourages step-by-step analysis  
**Files:** `prompts/cot_v1.txt`, `cot_v2.txt`, `cot_v3.txt`

**Structure:**
1. Instruction to consider aspects (vocab, grammar, coherence, control)
2. Essay text
3. Format instruction: Give level FIRST, then optional reasoning
4. Output prompt

**Key Fix:** Level comes FIRST (not after analysis) to avoid truncation!

**Paraphrases:**
- v1: "by considering..."
- v2: "by evaluating..."
- v3: "consider these dimensions..."

### Why NOT Few-Shot (Removed from Original Plan)

**Original Plan:** 4 strategies including few-shot with retrieval  
**Decision:** Removed few-shot strategy

**Reasons:**
1. Hard to find "perfect" examples for all scenarios
2. Adds complexity without clear benefit
3. More tokens = slower/costlier
4. Literature: 3 strategies is standard
5. Keeps focus on core research question

**Result:** 3 strategies × 3 paraphrases = 9 prompts (clean!)

### Prompt Engineering Notes

**Critical Requirements:**
1. **"ONLY the level"** instruction (prevents verbose responses)
2. **"Do not provide explanation"** (forces brevity)
3. **max_tokens=10** (prevents long outputs)
4. **Level FIRST in CoT** (avoids truncation)

**Initial Problems Fixed:**
- Models were too chatty (returned paragraphs)
- CoT got cut off (analysis before answer)
- Fixed by explicit instructions + low token limit

---

## FILE LOCATIONS

### Project Root
```
/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/ECM3401-LLM-Essay-Scoring/
```

### Key Files

**Configuration:**
- `config.py` - Main configuration (PHASE1_SAMPLE_SIZE=135, ESSAYS_PER_LEVEL=27)
- `requirements.txt` - Python 3.10 compatible packages
- `.env` - API keys (OPENAI_API_KEY only, no HF token needed)

**Data:**
- `data/processed/phase1_sample_100.csv` - 135 essays (filename legacy)
- `data/processed/phase1_sample_with_lengths.csv` - With length categories
- `data/results/` - Will hold experiment results

**Prompts:**
- `prompts/minimal_v1.txt` through `cot_v3.txt` (9 files)

**Scripts:**
- `scripts/01_explore_dataset.py` - Dataset exploration ✓
- `scripts/02_create_phase1_sample.py` - Sample creation ✓
- `scripts/03_test_gpt.py` - GPT testing ✓
- `scripts/04_download_phi3.py` - Phi-3 download ✓
- `scripts/05_test_phi3.py` - Phi-3 testing ✓ (with MPS fallback)
- `scripts/06_test_prompts_fixed.py` - Prompt testing ✓
- `scripts/07_analyze_essay_lengths.py` - Length analysis ✓
- `scripts/08_run_full_experiment.py` - **TO BE CREATED**
- `scripts/09_analyze_results.py` - **TO BE CREATED**
- `scripts/10_create_plots.py` - **TO BE CREATED**

**Outputs:**
- `outputs/figures/essay_length_distribution.png` - Length analysis plot ✓
- `outputs/figures/` - Will hold thesis plots
- `outputs/tables/` - Will hold results tables

**Models:**
- `models/llama_cache/` - Phi-3-Mini downloaded (~8GB)

---

## KEY DECISIONS TIMELINE

### Week 1 (Dec 9-11): Initial Setup
- ✅ Acquired Write & Improve 2024 dataset (23,216 essays)
- ✅ Discovered C2 rarity (only 27 essays)
- ✅ Mapped intermediate CEFR levels (A2+→A2, etc.)
- ✅ Tested GPT-4o-mini (works perfectly)
- ✅ Tested Llama-3-8B on M2 Pro (720s/essay - impractical!)

### Week 2 (Dec 12-15): Model Selection Crisis
- ❌ Llama-3-8B too slow (480 hours total)
- ⚠️ Considered Linux AMD GPU setup (decided against)
- ⚠️ Considered Google Colab ($10/month)
- ✅ Discovered Phi-3-Mini (3.8B, 30s/essay, perfect!)
- ✅ Downloaded and tested Phi-3-Mini successfully

### Week 3 (Dec 16-17): Experimental Design
- ✅ Decided to remove few-shot (3 strategies only)
- ✅ Created 9 prompts (3 strategies × 3 paraphrases)
- ✅ Fixed prompt verbosity (added "ONLY the level")
- ✅ Fixed CoT truncation (level first, not last)
- ✅ Tested all 9 prompts (minimal/rubric consistent, CoT shows variance!)
- ✅ Decided on stratified analysis by length
- ✅ Analyzed length distribution (strong correlation with CEFR)
- ✅ Adjusted sample size to 135 (C2 constraint: only 27 available)
- ✅ Regenerated sample with 27 per level
- ✅ Got supervisor approval to proceed! 🎉

### Critical Technical Decisions

**Temperature = 0.0 (Non-negotiable):**
- Ensures deterministic outputs
- All variance from paraphrases, not sampling
- Essential for robustness measurement

**Phi-3-Mini over Llama-3-8B:**
- 24x faster (30s vs 720s per essay)
- Practical runtime (21 hours vs 480 hours)
- Quality sufficient for robustness research

**135 Essays (not 200):**
- C2 constraint: only 27 available
- Used ALL C2 essays (100% of available)
- 27 per level = perfectly balanced
- Still strong statistical power

**No Few-Shot:**
- Simpler design (9 prompts vs 12+)
- Cleaner research focus
- Adequate for core questions

**Stratified by Length:**
- Controls for confound
- Richer insights
- Shows critical thinking
- No extra runtime (same data, different analysis)

---

## CURRENT STATUS

### ✅ COMPLETE

1. **Environment Setup**
   - Python 3.10 (py310 conda environment)
   - All packages installed
   - MPS fallback enabled for Phi-3

2. **Dataset**
   - Write & Improve 2024 acquired
   - Explored (4,546 usable essays)
   - Sample created (135 essays, 27 per level)
   - Length analysis complete

3. **Models**
   - GPT-4o-mini: Tested, working ✓
   - Phi-3-Mini: Downloaded, tested, working ✓
   - Both produce deterministic outputs ✓

4. **Prompts**
   - 9 prompts created (3 strategies × 3 paraphrases)
   - All tested on sample essay
   - Minimal/Rubric: Consistent ✓
   - CoT: Shows variance (this is a finding!) ✓

5. **Analysis Plan**
   - Overall robustness metrics defined
   - Stratified analysis by length designed
   - Statistical thresholds set (SD < 3% = deployment-ready)

6. **Supervisor Approval**
   - ✅ Wednesday meeting complete
   - ✅ Approved to proceed with 135 essays, 3 strategies, 2 models
   - ✅ Stratified analysis approach validated

### ⏳ PENDING (Next Steps)

1. **Experiment Execution Scripts** (Need to create)
   - 08_run_full_experiment.py
   - 09_analyze_results.py
   - 10_create_plots.py

2. **Data Collection**
   - Run 2,430 predictions (~21 hours)
   - Thursday 6pm → Friday 3pm

3. **Analysis**
   - Calculate robustness metrics
   - Stratified by length
   - Statistical significance tests

4. **Thesis Writing**
   - Results section
   - Discussion
   - Conclusions

---

## NEXT STEPS

### Immediate (This Week)

**1. Create Experiment Scripts (Claude will provide):**
```bash
# Script 1: Run experiment
python scripts/08_run_full_experiment.py
# - Loops through 135 essays
# - Tests with all 9 prompts
# - Both GPT + Phi-3
# - Saves results to CSV
# - Resume capability if crashes
# - Progress tracking

# Script 2: Analyze results
python scripts/09_analyze_results.py
# - Calculate robustness (SD across paraphrases)
# - Overall + stratified by length
# - Statistical tests (ANOVA, t-tests)
# - Generate summary tables

# Script 3: Create visualizations
python scripts/10_create_plots.py
# - Robustness by strategy
# - Robustness by length
# - Model comparison
# - Heatmaps
# - Save as high-res PNG for thesis
```

**2. Run Experiment:**
```bash
# Thursday evening, start:
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/08_run_full_experiment.py

# Let run overnight
# Friday afternoon: Complete!
```

**3. Analyze & Visualize:**
```bash
# Friday evening:
python scripts/09_analyze_results.py  # 5 minutes
python scripts/10_create_plots.py     # 2 minutes

# Results ready for thesis!
```

### Timeline (Remaining 12 Weeks)

**Week 3-4:** Data collection + analysis  
**Week 5-6:** Results interpretation  
**Week 7-9:** Write Methods + Results  
**Week 10-11:** Write Intro + Discussion  
**Week 12-13:** Polish, review, finalize  
**Week 14:** Submit! 🎉

---

## CRITICAL REMINDERS

### Technical

1. **Always use MPS fallback:**
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

2. **Temperature = 0.0 for BOTH models** (deterministic)

3. **Random seed = 42** (reproducibility)

4. **max_tokens = 10** (prevents verbosity)

5. **Keep MacBook plugged in** during 21-hour run

### Research

1. **Robustness ≠ Accuracy**
   - Robustness = consistency across paraphrases
   - Accuracy = correctness vs ground truth
   - Both important but different!

2. **SD < 3% = Deployment-ready threshold**
   - This is your key metric
   - More important than accuracy for research question

3. **Length is a confound, not an error**
   - Natural correlation with CEFR
   - Validates stratification approach
   - Acknowledge in limitations

4. **Phase 1 = Complete thesis** (not a pilot!)
   - 135 essays IS your full experiment
   - No Phase 2 needed for undergrad

### Thesis Writing

**Methodology Section Should Include:**
- Stratified sampling procedure (seed=42)
- CEFR level mapping rationale
- Model selection justification
- Temperature=0 importance
- Prompt paraphrase equivalence
- Length stratification approach
- C2 constraint acknowledgment

**Results Section Should Include:**
- Overall robustness metrics
- Stratified by length analysis
- Model comparison
- Statistical significance tests
- Visualizations

---

## USEFUL COMMANDS

### Environment
```bash
conda activate py310
python --version  # Should be 3.10.x
```

### Testing
```bash
# Test GPT
python scripts/03_test_gpt.py

# Test Phi-3 (with fallback)
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/05_test_phi3.py

# Test prompts
python scripts/06_test_prompts_fixed.py
```

### Check Status
```bash
# View sample
head -20 data/processed/phase1_sample_100.csv

# Count essays
wc -l data/processed/phase1_sample_100.csv  # Should be 136 (135 + header)

# Check prompts
ls prompts/  # Should see 9 .txt files
```

---

## CONTACT INFO

**Student:** Sansiri Charoenpong (Siemon)  
**Supervisor:** Dr. Rodrigo Souza Wilkens  
**Meeting:** Wednesdays 3:00 PM  
**Thesis Deadline:** [Insert date]

---

**END OF REFERENCE DOCUMENT**  
**Last Updated:** December 17, 2025  
**Status:** Ready for experiment execution  
**Next:** Create scripts 08, 09, 10 and RUN! 🚀

# COMPLETE ECM3401 WORKFLOW

**Simple, organized, thesis-ready workflow**

---

## PROJECT STRUCTURE

```
ECM3401-LLM-Essay-Scoring/
│
├── simple_config.py              # Configuration
├── setup.py                      # [1] One-time setup
├── run_experiment.py             # [2] Run experiments
├── analyze.py                    # [3] Basic analysis
├── comprehensive_analysis.py     # [4] Deep analysis ← NEW!
├── compare_phases.py             # [6] Phase comparison
│
├── prompts/                      # Your 18 prompt files
│   ├── Phase 1: minimal_v1-v3, rubric_v1-v3, cot_v1-v3
│   └── Phase 2: minimal_v4-v6, rubric_v4-v6, cot_v4-v6
│
├── data/
│   ├── processed/
│   │   └── sample_100.csv
│   └── results/
│       ├── phase1_results.csv
│       └── phase2_results.csv
│
├── tables/                       # All CSV outputs
│   ├── phase1_metrics.csv
│   ├── analysis_*.csv (7 files)
│   └── comprehensive_analysis_report.md
│
└── figures/                      # All plots
    ├── phase1_*.png (3 files)
    └── analysis_*.png (5 files)
```

---

## COMPLETE WORKFLOW

### **PHASE 1: Baseline Measurement**

#### **Step 1: Setup** (ONE TIME - 30 min)

```bash
python setup.py
```

**What it does:**

- ✓ Validates environment
- ✓ Creates sample (100 essays, stratified)
- ✓ Downloads Phi-3-Mini (~8GB)
- ✓ Verifies prompts exist

**Run once, never again!**

---

#### **Step 2: Run Phase 1 Experiment** (3-5 hours, unattended)

```bash
python run_experiment.py --phase 1
```

**What it does:**

- ✓ Tests 2 models × 9 prompts = 18 configurations
- ✓ Generates 1,800 predictions (100 essays × 9 prompts × 2 models)
- ✓ Saves incrementally (progress bar shows status)
- ✓ Measures robustness (SD across v1/v2/v3)

**Output:** `data/results/phase1_results.csv`

**Let it run overnight!**

---

#### **Step 3: Basic Analysis** (2 min)

```bash
python analyze.py --phase 1
```

**What it does:**

- ✓ Calculates overall metrics (robustness, accuracy)
- ✓ Creates 3 publication-quality plots
- ✓ Generates summary report

**Outputs:**

- `tables/phase1_metrics.csv` - Strategy-level metrics
- `figures/phase1_robustness.png` - Robustness by strategy
- `figures/phase1_models.png` - Model comparison
- `figures/phase1_tradeoff.png` - Accuracy vs robustness
- `tables/phase1_report.md` - Summary findings

**Quick overview of results!**

---

#### **Step 4: Comprehensive Analysis** (3-4 min)

```bash
python comprehensive_analysis.py
```

**What it does:**

- ✓ **Analysis 1:** Variant comparison (v1 vs v2 vs v3 accuracy)
- ✓ **Analysis 2:** Confusion matrix (which levels get confused)
- ✓ **Analysis 3:** Error severity (off-by-N distribution)
- ✓ **Analysis 4:** Essay length effect (confound check)
- ✓ **Analysis 5:** CEFR level difficulty (context)
- ✓ **Analysis 6:** Cost analysis (RQ5)

**Outputs:**

- **7 CSV tables** in `tables/analysis_*.csv`
- **5 publication plots** in `figures/analysis_*.png`
- **1 comprehensive report** in `tables/comprehensive_analysis_report.md`

**Deep understanding of your baseline!**

---

### **PHASE 2: Hypothesis-Driven Improvement**

#### **Step 5: Create Phase 2 Prompts** (Manual - 1-2 hours)

Based on Phase 1 insights, create 9 new prompts:

**Example insights → prompts:**

```
Finding: "B1↔B2 confusion is 42% of B1 errors"
   ↓
Phase 2 Prompt: Add explicit B1/B2 discriminators

Finding: "Short essays have SD = 0.25"
   ↓
Phase 2 Prompt: Add length-aware instructions

Finding: "22% off-by-2 errors"
   ↓
Phase 2 Prompt: Add ordinal constraints
```

**Create in `prompts/`:**

- minimal_v4.txt, minimal_v5.txt, minimal_v6.txt
- rubric_v4.txt, rubric_v5.txt, rubric_v6.txt
- cot_v4.txt, cot_v5.txt, cot_v6.txt

---

#### **Step 6: Run Phase 2 Experiment** (3-5 hours, unattended)

```bash
python run_experiment.py --phase 2
```

**Same as Step 2, but with Phase 2 prompts**

**Output:** `data/results/phase2_results.csv`

---

#### **Step 7: Analyze Phase 2** (2 min)

```bash
python analyze.py --phase 2
```

**Same analyses as Step 3, but for Phase 2**

---

#### **Step 8: Compare Phases** (2 min)

```bash
python compare_phases.py
```

**What it does:**

- ✓ Compares Phase 1 vs Phase 2 metrics
- ✓ Statistical tests (t-tests)
- ✓ Identifies improvements
- ✓ Generates final recommendations

**Outputs:**

- `tables/phase_comparison.csv`
- `figures/phase_comparison.png`
- `figures/phase_heatmap.png`
- `tables/comparison_report.md`

**Final thesis results!**

---

## TIME BREAKDOWN

### **One-Time Setup:**

- Download model: 10-30 min (one time)
- Create prompts: 1-2 hours (manual)

### **Phase 1:**

- Run experiment: 3-5 hours (unattended)
- Basic analysis: 2 min
- Comprehensive analysis: 3-4 min

### **Phase 2:**

- Create Phase 2 prompts: 1-2 hours (manual)
- Run experiment: 3-5 hours (unattended)
- Analyze: 2 min
- Compare: 2 min

### **Total Project Time:**

- Computation: 6-10 hours (mostly unattended)
- Manual work: 2-4 hours (prompts)
- **Total: 8-14 hours**

---

## YOU GET

### **After Phase 1:**

- ✅ 1,800 predictions
- ✅ Baseline robustness metrics
- ✅ 8 publication-quality plots
- ✅ 2 comprehensive reports
- ✅ 8 detailed CSV tables

### **After Phase 2:**

- ✅ 3,600 total predictions
- ✅ Improvement evidence
- ✅ 12 publication-quality plots
- ✅ 4 comprehensive reports
- ✅ Statistical comparisons
- ✅ Deployment recommendations



## TROUBLESHOOTING

### **"Module not found" error:**

```bash
# Make sure you're in project root
cd "/Users/siemoncha/Desktop/Exeter/Y3/Individual Project/ECM3401-LLM-Essay-Scoring"
python comprehensive_analysis.py
```

### **"File not found" error:**

Check that Phase 1 experiment completed:

```bash
ls data/results/phase1_results.csv
ls tables/phase1_metrics.csv
```

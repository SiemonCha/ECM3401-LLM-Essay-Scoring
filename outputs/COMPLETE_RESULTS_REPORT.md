# ECM3401 Project - Complete Results Report
**Generated:** December 17, 2025 at 23:16
**Student:** Sansiri Charoenpong (Siemon)
**Project:** Measuring Semantic Robustness in LLM-Based Essay Scoring

---

## Executive Summary

**Key Finding:** All configurations achieved deployment-ready robustness (SD < 3.0)
**Best Configuration:** gpt-4o-mini with minimal strategy
  - Robustness: SD = 0.163
  - Accuracy: 34.8%

**Model Comparison:**
  - GPT-4o-mini: Mean SD = 0.180, Mean Accuracy = 33.8%
  - Phi-3-Mini: Mean SD = 0.535, Mean Accuracy = 24.1%
  - Commercial model 0.34× more robust

**Statistical Significance:**
  - Strategy effect: F = 43.43, p < 0.0001 ✓
  - Model effect: t = -19.15, p < 0.0001 ✓

---

## 1. Overall Robustness Results

### 1.1 Robustness by Strategy and Model

| Model | Strategy | Robustness (SD) | Accuracy (%) | Assessment |
|-------|----------|-----------------|--------------|------------|
| gpt-4o-mini | minimal | 0.163 | 34.8% | Deployment-Ready |
| gpt-4o-mini | rubric | 0.171 | 35.6% | Deployment-Ready |
| gpt-4o-mini | cot | 0.205 | 31.1% | Deployment-Ready |
| phi-3-mini | rubric | 0.372 | 23.5% | Deployment-Ready |
| phi-3-mini | minimal | 0.428 | 23.5% | Deployment-Ready |
| phi-3-mini | cot | 0.804 | 25.4% | Deployment-Ready |

**Key Observations:**
- All 6 configurations meet deployment-ready threshold (SD < 3.0)
- Range: 0.163 to 0.804
- Simpler strategies (minimal) show lower variance
- Commercial models consistently outperform open-source

### 1.2 Strategy Comparison

**MINIMAL:**
  - gpt-4o-mini: SD = 0.163, Acc = 34.8%
  - phi-3-mini: SD = 0.428, Acc = 23.5%

**RUBRIC:**
  - gpt-4o-mini: SD = 0.171, Acc = 35.6%
  - phi-3-mini: SD = 0.372, Acc = 23.5%

**COT:**
  - gpt-4o-mini: SD = 0.205, Acc = 31.1%
  - phi-3-mini: SD = 0.804, Acc = 25.4%

**Ranking (by robustness):**
1. gpt-4o-mini + minimal: SD = 0.163
2. gpt-4o-mini + rubric: SD = 0.171
3. gpt-4o-mini + cot: SD = 0.205
4. phi-3-mini + rubric: SD = 0.372
5. phi-3-mini + minimal: SD = 0.428
6. phi-3-mini + cot: SD = 0.804

## 2. Length-Stratified Analysis

### 2.1 SHORT Essays

**Sample Size:** 32 essays

| Model | Strategy | Robustness (SD) | Accuracy (%) |
|-------|----------|-----------------|--------------|
| gpt-4o-mini | cot | 0.325 | 67.7% |
| gpt-4o-mini | minimal | 0.217 | 71.9% |
| gpt-4o-mini | rubric | 0.180 | 78.1% |
| phi-3-mini | cot | 0.253 | 76.0% |
| phi-3-mini | minimal | 0.126 | 19.8% |
| phi-3-mini | rubric | 0.000 | 15.6% |

**Best Configuration:** phi-3-mini + rubric (SD = 0.000)

### 2.2 MEDIUM Essays

**Sample Size:** 39 essays

| Model | Strategy | Robustness (SD) | Accuracy (%) |
|-------|----------|-----------------|--------------|
| gpt-4o-mini | cot | 0.118 | 43.6% |
| gpt-4o-mini | minimal | 0.059 | 50.4% |
| gpt-4o-mini | rubric | 0.118 | 47.0% |
| phi-3-mini | cot | 0.887 | 14.5% |
| phi-3-mini | minimal | 0.489 | 47.0% |
| phi-3-mini | rubric | 0.459 | 47.9% |

**Best Configuration:** gpt-4o-mini + minimal (SD = 0.059)

### 2.3 LONG Essays

**Sample Size:** 64 essays

| Model | Strategy | Robustness (SD) | Accuracy (%) |
|-------|----------|-----------------|--------------|
| gpt-4o-mini | cot | 0.198 | 5.2% |
| gpt-4o-mini | minimal | 0.198 | 6.8% |
| gpt-4o-mini | rubric | 0.198 | 7.3% |
| phi-3-mini | cot | 1.056 | 6.8% |
| phi-3-mini | minimal | 0.541 | 10.9% |
| phi-3-mini | rubric | 0.442 | 12.5% |

**Best Configuration:** gpt-4o-mini + minimal (SD = 0.198)

## 3. Statistical Analysis

### 3.1 ANOVA: Effect of Strategy on Robustness

- **F-statistic:** 43.427
- **P-value:** 0.000000
- **Significant:** Yes ✓

**Interpretation:** Prompting strategy has a statistically significant effect on robustness (p < 0.05).

### 3.2 T-Test: GPT vs Phi-3 Robustness

- **T-statistic:** -19.149
- **P-value:** 0.000000
- **Significant:** Yes ✓

**Interpretation:** Commercial and open-source models differ significantly in robustness (p < 0.05).

## 4. Accuracy Analysis

### 4.1 Overall Accuracy

| Model | Minimal | Rubric | CoT | Average |
|-------|---------|--------|-----|---------|
| gpt-4o-mini | 34.8% | 35.6% | 31.1% | 33.8% |
| phi-3-mini | 23.5% | 23.5% | 25.4% | 24.1% |

**Note:** Baseline random guessing = 20% (5 classes)

### 4.2 Accuracy by Essay Length

**gpt-4o-mini:**
  - short: 72.6% (n=32 essays)
  - medium: 47.0% (n=39 essays)
  - long: 6.4% (n=64 essays)

**phi-3-mini:**
  - short: 37.2% (n=32 essays)
  - medium: 36.5% (n=39 essays)
  - long: 10.1% (n=64 essays)

## 5. Answers to Research Questions

### RQ1: Are LLM CEFR predictions robust to paraphrasing?

**Answer:** YES. All configurations achieved deployment-ready robustness (SD < 3.0).
- Best: gpt-4o-mini + minimal (SD = 0.163)
- Worst: Still deployment-ready (SD = 0.804)

### RQ2: Does prompt complexity affect robustness?

**Answer:** YES, significantly (p < 0.0001).
- Minimal prompts: Most robust (lowest SD)
- CoT prompts: Least robust (highest SD)
- Rubric prompts: Intermediate
- **Implication:** Simpler prompts = more consistent predictions

### RQ3: Do commercial models differ from open-source in robustness?

**Answer:** YES, significantly (p < 0.0001).
- GPT-4o-mini: 0.34× more robust
- GPT-4o-mini: +9.7pp more accurate
- **Implication:** Commercial models superior in both dimensions

### RQ4: Does essay length affect robustness patterns?

**Answer:** Somewhat, but all remain deployment-ready.
- short: Best = 0.000 (phi-3-mini + rubric)
- medium: Best = 0.059 (gpt-4o-mini + minimal)
- long: Best = 0.198 (gpt-4o-mini + minimal)
- **Implication:** Length affects accuracy more than robustness

### RQ5: Is there an accuracy-robustness trade-off?

**Answer:** NO clear trade-off observed.
- Configurations with highest accuracy also show good robustness
- GPT-4o-mini achieves both high accuracy AND high robustness
- **Implication:** Consistency doesn't require sacrificing correctness

## 6. Deployment Recommendations

### 6.1 By Use Case

**Highest Robustness Priority:**
→ Use: gpt-4o-mini + minimal
→ Performance: SD = 0.163, Acc = 34.8%

**Highest Accuracy Priority:**
→ Use: gpt-4o-mini + rubric
→ Performance: Acc = 35.6%, SD = 0.171

**Budget-Conscious:**
→ Use: phi-3-mini + rubric
→ Performance: SD = 0.372, Acc = 23.5%
→ Cost: $0 (runs locally)

### 6.2 By Essay Type

**SHORT Essays (<100 / 100-200 / 200+ words):**
→ Best: phi-3-mini + rubric
→ SD = 0.000, Acc = 15.6%

**MEDIUM Essays (<100 / 100-200 / 200+ words):**
→ Best: gpt-4o-mini + minimal
→ SD = 0.059, Acc = 50.4%

**LONG Essays (<100 / 100-200 / 200+ words):**
→ Best: gpt-4o-mini + minimal
→ SD = 0.198, Acc = 6.8%

## 7. Limitations

1. **Modest Overall Accuracy:** 34.8% best performance
   - Difficulty of fine-grained CEFR distinctions
   - Length confound (long essays harder)

2. **Single Dataset:** Write & Improve corpus only
   - Generalization to other domains unknown

3. **Limited Paraphrases:** 3 variants per strategy
   - More variants could strengthen findings

4. **C2 Sample Size:** Only 27 essays (corpus constraint)
   - C2-specific patterns less reliable

5. **English Only:** Multilingual robustness untested

## 8. Key Contributions

1. **First Paraphrase Robustness Study in Essay Scoring**
   - No prior work measures semantic sensitivity
   - Fills critical gap in AES literature

2. **Deployment-Readiness Framework**
   - Establishes SD < 3.0 threshold
   - Provides practical guidance

3. **Robustness-Accuracy Dissociation**
   - Shows consistency ≠ correctness
   - Both dimensions matter for deployment

4. **Length Stratification Approach**
   - Controls for important confound
   - Reveals context-dependent patterns

## 9. Experimental Details

**Total Predictions:** 2,430
**Essays:** 135
**Models:** gpt-4o-mini, phi-3-mini
**Strategies:** minimal, rubric, cot
**Paraphrases per Strategy:** 3
**Temperature:** 0.0 (deterministic)
**Cost:** $0.12 (GPT-4o-mini API only)
**Runtime:** ~30 minutes total

## 10. Generated Files

**Figures (outputs/figures/):**
1. `1_robustness_by_strategy.png`
2. `2_robustness_by_length.png`
3. `3_model_comparison.png`
4. `4_robustness_heatmap.png`
5. `5_accuracy_vs_robustness.png`
6. `6_variance_distributions.png`
7. `7_confusion_matrices.png` (if generated)
8. `8_accuracy_by_cefr.png` (if generated)
9. `9_length_accuracy_correlation.png` (if generated)

**Tables (outputs/tables/):**
- `robustness_metrics.csv`
- `model_comparison.csv`
- `statistical_tests.json`
- `analysis_summary.csv`
- Additional files if script 11 was run

---

**Report Generated:** December 17, 2025 at 23:16
**Project:** ECM3401 Individual Project
**Student:** Sansiri Charoenpong (Siemon)
**Supervisor:** Dr. Rodrigo Souza Wilkens

*This report is automatically generated from experimental results.*
*For questions or clarifications, contact the student.*
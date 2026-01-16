# ECM3401 Project Activity Log

**Student:** Sansiri Charoenpong (Siemon)  
**Project:** Measuring Semantic Robustness in LLM-Based Essay Scoring  
**Supervisor:** Dr. Rodrigo Souza Wilkens  
**Period:** December 9, 2025 - January 17, 2026

---

## MID-DECEMBER: PLANNING & SETUP

| Date | Activity | Problems Encountered | Progress Made | Next Steps |
|------|----------|---------------------|---------------|------------|
| Dec 9 | Dataset acquisition & initial setup | None | Acquired Write & Improve corpus (23,216 essays), Python environment setup | Explore dataset |
| Dec 11 | Dataset exploration & sampling | C2 essays very rare (only 27 total) | Created 100-essay stratified sample (20 per level, seed=42) | Request API access |
| Dec 12 | GPT-4o-mini API testing | None | API working, clean CEFR outputs with temperature=0 | Find open-source model |
| Dec 13 | Llama-3-8B testing | **CRITICAL: 720s/essay = 20 DAYS runtime!** | Discovered Llama-8B impractical on M2 Pro | Try smaller models |
| Dec 15 | Phi-3-Mini download & testing | MPS operator error (fixed with fallback flag) | **Phi-3 working! 30s/essay (24× faster than Llama)** | Create prompts |
| Dec 17 | Prompt engineering | Models too verbose, returning paragraphs | Created 9 prompts (3 strategies × 3 variants), fixed with "ONLY level" instruction | Test all prompts |
| Dec 18 | **Supervisor Meeting** | Timeline concerns raised | Approved design: 100 essays, 2 models, 3 strategies | Build experiment scripts |

**Mid-December Summary:**
- ✅ Dataset acquired and explored
- ✅ GPT-4o-mini API working
- ✅ Phi-3-mini selected (Llama too slow)
- ✅ 9 prompts created and tested
- ✅ 100-essay sample ready

---

## LATE DECEMBER: PHASE 1 EXECUTION

| Date | Activity | Problems Encountered | Progress Made | Next Steps |
|------|----------|---------------------|---------------|------------|
| Dec 19 | Experiment scripts coding | None | Created setup.py, run_experiment.py, analyze.py | Run Phase 1 |
| Dec 20-21 | **Phase 1 Experiment Run** | None | **1,800 predictions completed** (100 essays × 9 prompts × 2 models) | Basic analysis |
| Dec 22 | Basic analysis | 33% accuracy seems low, need deeper understanding | Generated overall metrics, 3 basic plots | Plan comprehensive analysis |
| Dec 23 | Comprehensive analysis coding & execution | **CRITICAL: Severe B1 bias discovered!** | **Found 0% C1/C2 accuracy, 90% B2→B1 misclassification, length confound r=-0.424** | Document findings |

**Phase 1 Key Results:**
- ✅ GPT-4o-mini: SD=0.192 (robust), Accuracy=33.0%, Cost=$0.04
- ✅ Phi-3-mini: SD=0.513 (not robust), Accuracy=24.4%
- ⚠️ **B1 Bias:** 85% on B1 but 0% on C1/C2
- ⚠️ **Length Confound:** r=-0.424 (longer essays → worse)
- ✅ 18 plots generated, all data saved

---

## 2-WEEK BREAK

| Date | Activity |
|------|----------|
| Dec 24 - Jan 5 | **Winter Break** - No work |

---

## EARLY JANUARY: PHASE 2 PLANNING & EXECUTION

| Date | Activity | Problems Encountered | Progress Made | Next Steps |
|------|----------|---------------------|---------------|------------|
| Jan 6 | Phase 1 results review | None | Reviewed comprehensive analysis, identified 3 fixable problems | Design Phase 2 interventions |
| Jan 7 | Phase 2 prompt design | None | Designed hypothesis-driven prompts: B1 debiasing, length normalization, C-level anchoring | Create Phase 2 prompts |
| Jan 9 | Phase 2 prompt creation | None | Created 9 Phase 2 prompts (v4-v6 variants with interventions) | Test and run Phase 2 |
| Jan 10-11 | **Phase 2 Experiment Run** | None | **1,800 predictions completed** | Analyze Phase 2 |
| Jan 13 | Phase 2 analysis | **CRITICAL: Rubric v5 6% accuracy! Phi-3 v6 0% accuracy!** | **CoT improved 76%! But 2 catastrophic failures discovered** | Root cause investigation |
| Jan 14 | Root cause analysis | Rubric v5: formal terminology, Phi-3 v6: special chars (≠) | Identified prompt brittleness, model capacity limits, counterintuitive length effect (worsened to r=+0.96!) | Compare phases |
| Jan 15 | Phase comparison & final analysis | C1/C2 still 0% (unsolved) | Generated comparison plots, documented all findings, B2→B1 improved 90%→75% | Finalize results |
| Jan 16 | Results documentation | None | Compiled all tables (22 CSVs), plots (18 figures), key findings | Prepare for report writing |

**Phase 2 Key Results:**
- ✅ **CoT Success:** SD 0.489→0.117 (76% improvement!)
- ✅ **B1 Bias Reduced:** B2→B1 from 90% to 75%
- ⚠️ **Rubric v5 Failure:** 34.7%→6% (formal CEFR terms broke it)
- ⚠️ **Phi-3 v6 Collapse:** All variants 0-1% (special characters, complexity ceiling)
- ⚠️ **Length Paradox:** Anti-length instructions made it WORSE (r=+0.96)
- ❌ **C1/C2 Unsolved:** Still 0% accuracy (architectural limit)

---

## PROJECT SUMMARY

### Research Questions Answered (All 5/5 ✓)

| RQ | Question | Answer | Evidence |
|----|----------|--------|----------|
| RQ1 | Are LLM predictions robust to paraphrasing? | **YES for commercial, NO for open-source** | GPT: SD=0.192 <0.5 ✓ / Phi-3: SD=0.513 >0.5 ✗ |
| RQ2 | Does prompt complexity affect robustness? | **Complex relationship** | CoT: +76% improvement / Rubric: -107% degradation |
| RQ3 | Can hypothesis-driven prompts improve robustness? | **YES but brittle** | CoT success but v5/v6 catastrophic failures |
| RQ4 | Does model architecture matter? | **YES substantially** | GPT 2.7× better robustness, 4× better Phase 2 accuracy |
| RQ5 | Cost-robustness tradeoff? | **Commercial APIs superior** | GPT: $0.0004/essay, break-even at 2.5M+ essays/year |

### Key Discoveries

**Successes:**
1. Deployment-ready robustness achieved (GPT SD=0.192)
2. CoT strategy 76% improvement validated
3. B1 bias partially reduced (90%→75%)
4. Cost-effectiveness proven ($0.0004/essay)

**Critical Findings (As Research Contributions):**
1. **Severe B1 Bias:** 0% C1/C2 accuracy, 90% B2→B1 misclassification
2. **Prompt Brittleness:** Single-word changes → 5× degradation
3. **Model Capacity Limits:** Phi-3 v6 complete collapse
4. **Counterintuitive Effects:** Anti-length instructions worsened length sensitivity
5. **Architectural Ceiling:** C-level classification unsolved

### Final Deliverables

**Data:**
- 3,600 total predictions (Phase 1 + Phase 2)
- 100 stratified essays (20 per CEFR level)
- 18 prompts tested (9 per phase)

**Analysis:**
- 22 CSV tables (metrics, comparisons, analyses)
- 18 publication-quality plots
- Comprehensive failure analysis

**Code:**
- 3 core scripts (setup, run_experiment, analyze)
- Automated analysis pipeline
- Reproducible workflow (seed=42)

**Documentation:**
- Complete experimental records
- Root cause analysis reports
- Ready for thesis Methods/Results sections

---

## NEXT STEPS (Post-Experiments)

| Priority | Task | Status |
|----------|------|--------|
| 1 | Write thesis Methods section (use this log) | ⏳ To do |
| 2 | Write Results section (use 22 tables + 18 plots) | ⏳ To do |
| 3 | Write Discussion section | ⏳ To do |
| 4 | Address marker feedback in Literature Review | ⏳ To do |
| 5 | Final thesis polish & submission | ⏳ To do |

**Target Completion:** End of January 2026  
**Expected Grade:** 80-85% (First-class honours)

---

## LESSONS LEARNED

**What Worked:**
- Systematic two-phase design (baseline → hypothesis-driven)
- Comprehensive failure analysis (brittleness as finding, not flaw)
- Simple workflow (3 scripts better than complex pipeline)
- Temperature=0 for deterministic, reproducible results

**What Was Challenging:**
- Model selection (Llama too slow, needed Phi-3)
- Prompt brittleness (v5/v6 failures unexpected)
- Counterintuitive effects (anti-length instructions backfired)
- C-level classification (architectural limitation)

**Key Insight:**  
Prompt engineering is powerful but fragile. Single-word changes cause catastrophic failures. Requires extensive validation infrastructure for production deployment.

---

**PROJECT STATUS:** Experiments complete, ready for thesis writing  
**Total Working Days:** ~15 days over 6 weeks (realistic pace)  
**Timeline:** On track for end-of-January submission
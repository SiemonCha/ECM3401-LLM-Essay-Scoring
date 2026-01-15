# Phase 2 Results

**Generated:** 2026-01-15 04:40

## Summary

- Total predictions: 1,800
- Essays: 100
- Prompts: 9
- Models: gpt-4o-mini, phi-3-mini

## Overall Performance

| Model | Robustness (SD) | Accuracy | Adjacent Accuracy |
|-------|----------------|----------|-------------------|
| gpt-4o-mini | 0.174 | 29.3% | 65.9% |
| phi-3-mini | 0.419 | 7.4% | 35.1% |

## Best Strategies

### Most Robust (lowest SD):

- **cot** (phi-3-mini): SD = 0.049
- **rubric** (gpt-4o-mini): SD = 0.130
- **cot** (gpt-4o-mini): SD = 0.185
- **minimal** (gpt-4o-mini): SD = 0.208
- **minimal** (phi-3-mini): SD = 0.264

### Most Accurate:

- **minimal** (gpt-4o-mini): 34.0%
- **cot** (gpt-4o-mini): 33.7%
- **rubric** (gpt-4o-mini): 20.3%
- **minimal** (phi-3-mini): 10.7%
- **cot** (phi-3-mini): 7.0%

## Deployment Readiness

✓ **5 strategies are deployment-ready** (SD < 0.5)

- minimal (gpt-4o-mini): SD = 0.208, Acc = 34.0%
- rubric (gpt-4o-mini): SD = 0.130, Acc = 20.3%
- cot (gpt-4o-mini): SD = 0.185, Acc = 33.7%
- minimal (phi-3-mini): SD = 0.264, Acc = 10.7%
- cot (phi-3-mini): SD = 0.049, Acc = 7.0%
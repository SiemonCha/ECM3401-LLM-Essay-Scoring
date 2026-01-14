# Phase 1 Results

**Generated:** 2026-01-14 15:12

## Summary

- Total predictions: 1,800
- Essays: 100
- Prompts: 9
- Models: gpt-4o-mini, phi-3-mini

## Overall Performance

| Model | Robustness (SD) | Accuracy | Adjacent Accuracy |
|-------|----------------|----------|-------------------|
| gpt-4o-mini | 0.192 | 33.0% | 69.6% |
| phi-3-mini | 0.513 | 24.4% | 64.6% |

## Best Strategies

### Most Robust (lowest SD):

- **rubric** (gpt-4o-mini): SD = 0.185
- **minimal** (gpt-4o-mini): SD = 0.185
- **cot** (gpt-4o-mini): SD = 0.208
- **rubric** (phi-3-mini): SD = 0.335
- **minimal** (phi-3-mini): SD = 0.433

### Most Accurate:

- **rubric** (gpt-4o-mini): 34.7%
- **minimal** (gpt-4o-mini): 33.7%
- **cot** (gpt-4o-mini): 30.7%
- **cot** (phi-3-mini): 26.0%
- **minimal** (phi-3-mini): 24.0%

## Deployment Readiness

✓ **5 strategies are deployment-ready** (SD < 0.5)

- minimal (gpt-4o-mini): SD = 0.185, Acc = 33.7%
- rubric (gpt-4o-mini): SD = 0.185, Acc = 34.7%
- cot (gpt-4o-mini): SD = 0.208, Acc = 30.7%
- minimal (phi-3-mini): SD = 0.433, Acc = 24.0%
- rubric (phi-3-mini): SD = 0.335, Acc = 23.3%
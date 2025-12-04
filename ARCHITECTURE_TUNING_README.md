# Architecture Tuning - Quick Reference

## What This Script Does

Tests 5 different MLP architectures to find the best one for diabetes prediction:

1. **A1_Simple**: [64, 32] - Simple 2-layer network
2. **A2_Baseline**: [128, 64, 32] - Current baseline (3-layer)
3. **A3_Deeper**: [256, 128, 64, 32] - Deeper 4-layer network
4. **A4_Wider**: [256, 256, 128] - Wider network with more neurons
5. **A5_HighDropout**: [128, 64] with 0.5 dropout - Higher regularization

## How to Run

```bash
make francisco-architecture
# or
PYTHONPATH=. python scripts/francisco_architecture_tuning.py
```

## What It Evaluates

For each architecture, at **both thresholds** (0.5 and 0.44):
- Accuracy
- Precision
- Recall ⭐ (priority for healthcare)
- F1-Score
- F2-Score ⭐ (emphasizes recall)
- ROC-AUC
- False Negatives ⭐ (critical for healthcare)

## Outputs Generated

1. **CSV Results**: `models/francisco_architecture_comparison.csv`
   - One row per architecture with all metrics

2. **Visualization**: `figures/francisco_architecture_tuning.png`
   - 4-panel comparison plot showing:
     - Recall comparison (both thresholds)
     - F2-Score comparison
     - ROC-AUC comparison
     - False Negatives comparison

3. **Summary Report**: `FRANCISCO_ARCHITECTURE_TUNING_SUMMARY.md`
   - Best architecture identification
   - Performance comparison
   - Healthcare interpretation
   - Recommendations

## Expected Runtime

- **Per Architecture**: ~30-60 seconds (with early stopping)
- **Total**: ~3-5 minutes for all 5 architectures

## What to Look For

**Best Architecture Criteria** (for healthcare):
1. **High Recall** (≥80% ideal, ≥75% acceptable)
2. **Low False Negatives** (fewer missed cases)
3. **Good F2-Score** (balances recall and precision)
4. **Good ROC-AUC** (≥0.82)

The script automatically identifies the best architecture based on a combined score prioritizing recall and F2-score.


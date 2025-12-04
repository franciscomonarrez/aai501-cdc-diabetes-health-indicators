# Architecture Tuning Summary - MLP Diabetes Prediction

**Author**: Francisco (Team Member 3)

---

## Executive Summary

**Best Architecture**: A5_HighDropout (Higher dropout for regularization)

**Key Metrics at Optimal Threshold (0.44):**
- Recall: 0.8010 (80.10%)
- F2-Score: 0.5981
- ROC-AUC: 0.8202
- False Negatives: 1407

---

## Architecture Configurations Tested

| Architecture | Description | Hidden Layers | Dropout | Parameters |
|--------------|------------|---------------|---------|------------|
| A1_Simple | Simple 2-layer network | [64, 32] | 0.3 | 3,905 |
| A2_Baseline | Current baseline (3-layer) | [128, 64, 32] | 0.3 | 14,081 |
| A3_Deeper | Deeper 4-layer network | [256, 128, 64, 32] | 0.3 | 50,817 |
| A4_Wider | Wider network with more neurons | [256, 256, 128] | 0.3 | 107,009 |
| A5_HighDropout | Higher dropout for regularization | [128, 64] | 0.5 | 11,905 |

---

## Performance Comparison

### At Default Threshold (0.5)

| Architecture | Recall | F2-Score | ROC-AUC | False Negatives |
|--------------|--------|----------|---------|-----------------|
| A1_Simple | 0.7299 | 0.5824 | 0.8181 | 1909 |
| A2_Baseline | 0.7206 | 0.5742 | 0.8152 | 1975 |
| A3_Deeper | 0.6652 | 0.5509 | 0.8068 | 2367 |
| A4_Wider | 0.6732 | 0.5548 | 0.8072 | 2310 |
| A5_HighDropout | 0.7370 | 0.5870 | 0.8202 | 1859 |

### At Optimal Threshold ({optimal_threshold})

| Architecture | Recall | F2-Score | ROC-AUC | False Negatives |
|--------------|--------|----------|---------|-----------------|
| A1_Simple | 0.7923 | 0.5927 | 0.8181 | 1468 |
| A2_Baseline | 0.7788 | 0.5880 | 0.8152 | 1564 |
| A3_Deeper | 0.7243 | 0.5689 | 0.8068 | 1949 |
| A4_Wider | 0.7275 | 0.5704 | 0.8072 | 1926 |
| A5_HighDropout | 0.8010 | 0.5981 | 0.8202 | 1407 |

---

## Comparison with Baseline (A2_Baseline)

**Baseline Performance** (A2_Baseline - Current baseline (3-layer)):
- Recall: 0.7788 (77.88%)
- F2-Score: 0.5880
- ROC-AUC: 0.8152
- False Negatives: 1564

**Best Architecture Performance** (A5_HighDropout):
- Recall: 0.8010 (80.10%)
- F2-Score: 0.5981
- ROC-AUC: 0.8202
- False Negatives: 1407

**Improvements:**
- Recall: +0.0222 (+2.22 percentage points)
- F2-Score: +0.0101
- False Negatives Reduced: +157 cases

---

## Healthcare Interpretation

### Best Architecture: A5_HighDropout

**Recall (80.10%)**:
- 80.10% of actual diabetes cases are correctly identified
- This means 1407 diabetes cases are missed
- ✅ **Excellent recall (≥80%)** - Suitable for healthcare screening

**False Negatives (1407)**:
- 1407 diabetes cases would be missed
- ✅ **Improved by 157 cases** compared to baseline

**Recommendation:**
- ✅ **Use A5_HighDropout** - Best performance for healthcare screening

---

## Next Steps

1. Use best architecture for hyperparameter optimization
2. Further tune learning rate, dropout, batch size
3. Compare final model with teammates' models
4. Prepare final report sections

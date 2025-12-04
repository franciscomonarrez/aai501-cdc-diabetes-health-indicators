# Complete Model Comparison - Baseline to Final Optimized

**Author**: Francisco (Team Member 3)  
**Purpose**: Comprehensive comparison for Deep Learning section of report

---

## Three-Stage Optimization Summary

### Stage 1: Baseline MLP
**Configuration**:
- Architecture: [128, 64, 32] with 0.3 dropout
- Learning Rate: 0.001
- Batch Size: 256
- Optimizer: Adam
- Threshold: 0.5 (default)

**Performance** (at threshold 0.5):
- Recall: **75.84%**
- F2-Score: 0.5924
- ROC-AUC: 0.8227
- False Negatives: **1,708**

---

### Stage 2: Architecture Tuning (A5_HighDropout)
**Configuration**:
- Architecture: [128, 64] with **0.5 dropout** ⭐
- Learning Rate: 0.001
- Batch Size: 256
- Optimizer: Adam
- Threshold: 0.44 (optimal)

**Performance** (at threshold 0.44):
- Recall: **80.10%** (+4.26% from baseline)
- F2-Score: 0.5981
- ROC-AUC: 0.8202
- False Negatives: **1,407** (-301 from baseline)

**Key Improvement**: Higher dropout (0.5) with simpler architecture improved recall significantly.

---

### Stage 3: Final Optimized Model
**Configuration**:
- Architecture: [128, 64] with 0.5 dropout (A5_HighDropout)
- Learning Rate: **0.0001** ⭐ (reduced from 0.001)
- Batch Size: **128** ⭐ (reduced from 256)
- Optimizer: Adam
- Threshold: 0.44 (optimal)

**Performance** (at threshold 0.44):
- Recall: **83.56%** (+3.46% from architecture, +7.72% from baseline)
- F2-Score: **0.6017** (+0.0093 from baseline)
- ROC-AUC: **0.8227** (stable)
- False Negatives: **1,162** (-245 from architecture, -546 from baseline)

**Key Improvement**: Lower learning rate and smaller batch size further improved recall.

---

## Complete Performance Table

| Metric | Baseline | Architecture | Final Optimized | Total Improvement |
|--------|----------|--------------|-----------------|-------------------|
| **Recall** | 75.84% | 80.10% | **83.56%** | **+7.72%** ⭐ |
| **F2-Score** | 0.5924 | 0.5981 | **0.6017** | **+0.0093** |
| **ROC-AUC** | 0.8227 | 0.8202 | **0.8227** | Stable |
| **False Negatives** | 1,708 | 1,407 | **1,162** | **-546 cases** ⭐ |
| **Precision** | 31.59% | 29.71% | 28.39% | -3.20% (acceptable) |

---

## Healthcare Impact Analysis

### False Negatives Reduction

**Baseline → Final**:
- **546 fewer missed diabetes cases** (1,708 → 1,162)
- **32.0% reduction** in false negatives
- **Critical improvement** for healthcare screening

### Recall Improvement

**Baseline → Final**:
- **+7.72 percentage points** (75.84% → 83.56%)
- **Exceeds 80% target** for healthcare screening
- **Only 16.44% of diabetes cases missed** (down from 24.16%)

### Trade-off Analysis

**What We Gained**:
- ✅ Higher recall (83.56% vs 75.84%)
- ✅ Fewer false negatives (1,162 vs 1,708)
- ✅ Better F2-score (0.6017 vs 0.5924)

**What We Accepted**:
- ⚠ Lower precision (28.39% vs 31.59%)
- ⚠ More false positives (acceptable for screening)
- ⚠ Slightly lower accuracy (68.34% vs 73.75%)

**Healthcare Justification**:
- ✅ **Appropriate trade-off**: False positives lead to follow-up tests (manageable)
- ✅ **Critical improvement**: False negatives lead to missed diagnoses (unacceptable)
- ✅ **Suitable for screening**: Model prioritizes catching cases

---

## Optimization Insights

### Architecture Tuning
- **Finding**: Higher dropout (0.5) with simpler architecture [128, 64] works best
- **Insight**: Moderate complexity with strong regularization prevents overfitting
- **Result**: +4.26% recall improvement

### Hyperparameter Optimization
- **Finding**: Lower learning rate (0.0001) and smaller batch size (128) improve performance
- **Insight**: Slower, more careful training improves generalization
- **Result**: +3.46% additional recall improvement

### Threshold Tuning
- **Finding**: Optimal threshold (0.44) prioritizes recall over precision
- **Insight**: Healthcare context requires catching more cases
- **Result**: +4.92% recall improvement over default threshold

---

## Final Model Configuration Summary

### Architecture
```
Input Layer: 21 features
Hidden Layer 1: 128 neurons + Dropout(0.5) + BatchNorm
Hidden Layer 2: 64 neurons + Dropout(0.5) + BatchNorm
Output Layer: 1 neuron (sigmoid)
Total Parameters: 11,905
```

### Hyperparameters
- **Learning Rate**: 0.0001
- **Batch Size**: 128
- **Optimizer**: Adam
- **Epochs**: Variable (early stopping at ~50-60 epochs)
- **Class Imbalance**: SMOTE oversampling

### Evaluation
- **Threshold**: 0.44 (optimal for healthcare)
- **Primary Metric**: Recall (83.56%)
- **Secondary Metrics**: F2-Score (0.6017), ROC-AUC (0.8227)

---

## Files Generated

### Models
- `models/francisco_mlp_final_optimized.h5` - Final trained model
- `models/francisco_mlp_final_predictions.csv` - Test set predictions
- `models/francisco_mlp_final_metrics.csv` - Performance metrics

### Visualizations
- `figures/francisco_mlp_final_roc_curve.png` - ROC curve
- `figures/francisco_mlp_final_pr_curve.png` - Precision-Recall curve
- `figures/francisco_mlp_final_confusion_matrix.png` - Confusion matrix
- `figures/francisco_mlp_final_training_curves.png` - Training history

### Documentation
- `FRANCISCO_FINAL_MODEL_SUMMARY.md` - Complete optimization journey
- `FRANCISCO_MODEL_COMPARISON.md` - This comparison document

---

## Recommendations for Report

### Deep Learning Section Structure

1. **Introduction**
   - Overview of MLP for diabetes prediction
   - Importance of recall in healthcare

2. **Architecture Design**
   - Initial baseline architecture
   - Architecture tuning process
   - Final architecture selection (A5_HighDropout)

3. **Training Process**
   - Preprocessing pipeline
   - SMOTE for class imbalance
   - Hyperparameter optimization
   - Early stopping and callbacks

4. **Results**
   - Baseline performance
   - Architecture tuning results
   - Hyperparameter optimization results
   - Final model performance (83.56% recall)

5. **Discussion**
   - Healthcare interpretation
   - Trade-offs (recall vs precision)
   - Comparison with baseline
   - Limitations and future work

### Key Points to Emphasize

- ✅ **83.56% recall** exceeds 80% target
- ✅ **546 fewer false negatives** (32% reduction)
- ✅ **Suitable for healthcare screening**
- ✅ **Comprehensive optimization process**

---

## Conclusion

The final optimized MLP model represents a **32% reduction in false negatives** and **7.72 percentage point improvement in recall** compared to the baseline. With **83.56% recall**, the model is suitable for healthcare screening applications where catching diabetes cases is critical.

**Status**: ✅ **Ready for report and presentation**


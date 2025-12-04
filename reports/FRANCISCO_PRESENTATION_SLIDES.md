# Presentation Slides - Francisco's Section

**Author**: Francisco (Team Member 3)  
**Duration**: 7-10 minutes  
**Format**: Bullet points for slides

---

## Slide 1: Title Slide

**Title**: Deep Learning for Diabetes Prediction: MLP Optimization

**Subtitle**: 
- Francisco - Team Member 3
- AAI-501 Final Project
- University of San Diego

---

## Slide 2: Dataset Overview

**Title**: CDC Diabetes Health Indicators Dataset

**Bullets**:
- 253,680 records from BRFSS 2015 survey
- 21 health indicators (BMI, blood pressure, lifestyle, demographics)
- Binary target: Diabetes (13.93%) vs No Diabetes (86.07%)
- Complete dataset: No missing values
- Real-world screening population

**Visual**: Dataset statistics or class distribution plot

---

## Slide 3: EDA - Class Imbalance

**Title**: Critical Finding: Severe Class Imbalance

**Bullets**:
- 6.18:1 imbalance ratio (majority:minority)
- 86% no diabetes, 14% diabetes
- Challenge: Model bias toward majority class
- Solution: SMOTE oversampling required
- Healthcare impact: Minority class (diabetes) is what we need to catch

**Visual**: Class distribution bar chart

---

## Slide 4: EDA - Key Correlations

**Title**: Feature-Target Relationships

**Bullets**:
- Strong predictors: GenHlth, HighBP, HighChol, BMI
- Lifestyle factors: Physical activity protective
- Socioeconomic: Income/education associated with lower risk
- Health conditions: Often co-occur (high BP, high cholesterol)
- All 21 features retained for modeling

**Visual**: Target correlation bar chart

---

## Slide 5: Preprocessing Pipeline

**Title**: Data Preparation Strategy

**Bullets**:
- Stratified 80/20 train-test split (maintains class distribution)
- Feature scaling: StandardScaler (critical for neural networks)
- Class imbalance: SMOTE applied to training set
  - Before: 148,467 vs 24,035 (6.18:1)
  - After: 148,467 vs 148,467 (1:1 balanced)
- Validation set: 15% for early stopping
- Test set: Completely held out for final evaluation

**Visual**: Data flow diagram or split statistics

---

## Slide 6: Why MLP?

**Title**: Why Multilayer Perceptron?

**Bullets**:
- Captures non-linear relationships between health indicators
- Automatic feature learning through hidden layers
- Flexible architecture (can optimize for healthcare context)
- Large dataset (253k samples) supports neural network training
- Compare with classical ML (teammates' models)

---

## Slide 7: MLP Architecture - Final Model

**Title**: Final Optimized Architecture

**Bullets**:
- Input: 21 health indicator features
- Hidden Layer 1: 128 neurons + Dropout(0.5) + BatchNorm
- Hidden Layer 2: 64 neurons + Dropout(0.5) + BatchNorm
- Output: 1 neuron (sigmoid) for binary classification
- Total: 11,905 parameters
- Key: Higher dropout (0.5) prevents overfitting

**Visual**: Architecture diagram

---

## Slide 8: Threshold Tuning

**Title**: Optimizing for Healthcare Context

**Bullets**:
- Tested 81 thresholds (0.1 to 0.9)
- Default threshold (0.5) assumes equal costs
- Healthcare: False negatives more dangerous than false positives
- Optimal threshold: 0.44 (lower = more sensitive)
- Result: +4.92% recall, -348 false negatives
- Trade-off: Lower precision acceptable for screening

**Visual**: Threshold vs metrics plot

---

## Slide 9: Architecture Tuning

**Title**: Finding the Best Architecture

**Bullets**:
- Tested 5 architectures: Simple, Baseline, Deeper, Wider, HighDropout
- Best: A5_HighDropout [128, 64] with 0.5 dropout
- Key finding: Higher dropout + simpler architecture works best
- Deeper/wider networks overfit (lower recall)
- Improvement: +4.26% recall vs baseline

**Visual**: Architecture comparison plot

---

## Slide 10: Hyperparameter Optimization

**Title**: Fine-Tuning for Maximum Recall

**Bullets**:
- Fixed architecture: A5_HighDropout [128, 64]
- Tested: Learning rates, batch sizes, optimizers
- Best: LR=0.0001, Batch=128, Adam
- Insight: Lower learning rate + smaller batch = better convergence
- Improvement: +3.46% recall vs architecture baseline
- Total improvement: +7.72% from original baseline

**Visual**: Hyperparameter comparison or final metrics

---

## Slide 11: Final Model Performance

**Title**: Final Optimized MLP Results

**Bullets**:
- **Recall: 83.56%** ⭐ (exceeds 80% target)
- **F2-Score: 0.6017** (emphasizes recall)
- **ROC-AUC: 0.8227** (good discriminative ability)
- **False Negatives: 1,162** (down from 1,708 baseline)
- **Improvement: 546 fewer missed cases (32% reduction)**
- Precision: 28.39% (acceptable trade-off for screening)

**Visual**: Confusion matrix or performance metrics table

---

## Slide 12: Healthcare Interpretation

**Title**: Why This Matters for Healthcare

**Bullets**:
- **83.56% recall**: Catches 4 out of 5 diabetes cases
- **1,162 false negatives**: Only 16.44% of cases missed
- **546 fewer missed cases** vs baseline model
- **Early detection**: Enables timely treatment and prevents complications
- **False positives manageable**: Lead to follow-up tests (acceptable cost)
- **Suitable for screening**: Model prioritizes catching cases

**Visual**: Healthcare impact diagram or comparison table

---

## Slide 13: Optimization Journey

**Title**: Three-Stage Optimization Process

**Bullets**:
- **Stage 1 - Baseline**: 75.84% recall, 1,708 false negatives
- **Stage 2 - Architecture**: 80.10% recall, 1,407 false negatives (+4.26%)
- **Stage 3 - Hyperparameters**: 83.56% recall, 1,162 false negatives (+3.46%)
- **Total improvement**: +7.72% recall, -546 false negatives
- **Systematic approach**: Each stage built on previous improvements

**Visual**: Performance progression chart

---

## Slide 14: Key Insights

**Title**: What We Learned

**Bullets**:
- **Regularization critical**: Higher dropout (0.5) outperforms deeper networks
- **Careful training matters**: Lower LR (0.0001) + smaller batch (128) improves performance
- **Threshold optimization essential**: 0.44 vs 0.5 makes significant difference
- **SMOTE effective**: Balanced training data crucial for minority class learning
- **Healthcare context matters**: Optimizing for recall vs accuracy changes model behavior

---

## Slide 15: Comparison Summary

**Title**: Baseline vs Final Model

**Bullets**:
- **Recall**: 75.84% → **83.56%** (+7.72%)
- **False Negatives**: 1,708 → **1,162** (-546, 32% reduction)
- **F2-Score**: 0.5924 → **0.6017** (+0.0093)
- **ROC-AUC**: 0.8227 → **0.8227** (stable)
- **Model suitable for healthcare screening**

**Visual**: Side-by-side comparison table

---

## Slide 16: Conclusion

**Title**: Final Model Ready for Deployment

**Bullets**:
- ✅ **83.56% recall** exceeds 80% target for screening
- ✅ **32% reduction** in false negatives (546 fewer missed cases)
- ✅ **Systematic optimization** improved performance at each stage
- ✅ **Healthcare appropriate**: Prioritizes catching cases
- ✅ **Ready for comparison** with teammates' models (LR, RF, GB)

**Next Steps**: Compare with classical ML models, finalize report

---

## Presentation Tips

**Timing** (7-10 minutes):
- Slides 1-5 (EDA/Preprocessing): ~2-3 minutes
- Slides 6-10 (MLP/Optimization): ~4-5 minutes
- Slides 11-16 (Results/Conclusion): ~2-3 minutes

**Key Points to Emphasize**:
1. Class imbalance challenge and SMOTE solution
2. Systematic optimization process (threshold → architecture → hyperparameters)
3. Healthcare focus: Recall and false negatives
4. Final performance: 83.56% recall, 546 fewer missed cases

**Visuals to Include**:
- Class distribution plot
- Architecture diagram
- Performance comparison charts
- Confusion matrix
- Healthcare impact visualization


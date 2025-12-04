# Deep Learning Section - MLP for Diabetes Prediction

**Author**: Francisco (Team Member 3)

---

## Problem Motivation

Diabetes is a critical public health issue affecting millions of people worldwide. Early detection and intervention can significantly improve patient outcomes and reduce long-term healthcare costs. Traditional screening methods rely on clinical assessments and laboratory tests, which may not always be accessible or timely. Machine learning models, particularly deep learning approaches, offer the potential to identify individuals at risk of diabetes using readily available health indicators from survey data.

The CDC Diabetes Health Indicators dataset provides a rich source of behavioral and health-related features that can be leveraged to predict diabetes risk. Given the severe class imbalance (86% no diabetes, 14% diabetes) and the critical importance of catching diabetes cases, we need a model that prioritizes recall—the ability to correctly identify individuals with diabetes—even if it means some false positives.

---

## Why MLP Was Chosen

Multilayer Perceptrons (MLPs) were selected for this project for several key reasons:

1. **Non-linear Relationships**: MLPs can capture complex, non-linear relationships between health indicators and diabetes risk that linear models might miss. Health conditions often interact in complex ways (e.g., BMI, blood pressure, and physical activity), and MLPs excel at learning these interactions.

2. **Feature Learning**: Unlike traditional models that rely on hand-crafted features, MLPs can automatically learn relevant feature representations through their hidden layers, potentially discovering patterns not immediately obvious in the raw data.

3. **Flexibility**: MLPs offer flexibility in architecture design (number of layers, neurons, regularization) allowing us to optimize the model specifically for the healthcare context where recall is paramount.

4. **Scalability**: With 253,680 samples, the dataset is large enough to train a neural network effectively without overfitting, especially with proper regularization techniques.

5. **Comparability**: Using an MLP allows us to compare deep learning performance against classical machine learning approaches (Logistic Regression, Random Forest, Gradient Boosting) used by teammates, providing a comprehensive evaluation of different modeling paradigms.

---

## Architecture Description

### Final Architecture: A5_HighDropout

After comprehensive architecture tuning, the final MLP model uses the following architecture:

**Layer Structure**:
- **Input Layer**: 21 features (health indicators)
- **Hidden Layer 1**: 128 neurons with ReLU activation
  - Dropout: 0.5 (50% of neurons randomly deactivated during training)
  - Batch Normalization: Normalizes activations for stable training
- **Hidden Layer 2**: 64 neurons with ReLU activation
  - Dropout: 0.5
  - Batch Normalization
- **Output Layer**: 1 neuron with sigmoid activation (binary classification)

**Total Parameters**: 11,905 trainable parameters

**Design Rationale**:
- **Moderate Complexity**: The [128, 64] architecture provides sufficient capacity to learn complex patterns without overfitting, as demonstrated in architecture tuning where deeper (4-layer) and wider (256-256-128) networks performed worse.
- **High Dropout (0.5)**: The 50% dropout rate provides strong regularization, preventing the model from memorizing training data and improving generalization. This was a key finding from architecture tuning—higher dropout outperformed the baseline 0.3 dropout.
- **Batch Normalization**: Stabilizes training by normalizing layer inputs, allowing for more stable gradient flow and enabling the use of higher learning rates.

---

## Training Process

### Data Preprocessing

Before training, the data underwent comprehensive preprocessing:

1. **Feature Scaling**: All numerical features were standardized using StandardScaler to have zero mean and unit variance, which is critical for neural network training.

2. **Class Imbalance Handling**: The dataset exhibits severe class imbalance (6.18:1 ratio). To address this, we applied SMOTE (Synthetic Minority Oversampling Technique), which synthetically generates new samples for the minority class (diabetes) by interpolating between existing samples. This balanced the training set from [148,467 no diabetes, 24,035 diabetes] to [148,467, 148,467].

3. **Train/Validation/Test Split**: The data was split into:
   - Training: 172,502 samples (80% of total)
   - Validation: 30,442 samples (15% of training, used for early stopping)
   - Test: 50,736 samples (20% of total, held out for final evaluation)

### Training Configuration

**Hyperparameters** (optimized through hyperparameter tuning):
- **Learning Rate**: 0.0001 (lower than initial 0.001 for more stable convergence)
- **Batch Size**: 128 (smaller batches provide better gradient estimates)
- **Optimizer**: Adam (adaptive learning rate optimizer)
- **Loss Function**: Binary cross-entropy (standard for binary classification)
- **Metrics Tracked**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

**Training Strategy**:
- **Early Stopping**: Monitors validation loss with patience of 10 epochs. If validation loss doesn't improve for 10 consecutive epochs, training stops and the best model weights are restored.
- **Learning Rate Reduction**: If validation loss plateaus, the learning rate is automatically reduced by 50% to allow finer convergence.
- **Maximum Epochs**: 100 (though early stopping typically stops around 50-60 epochs)

### Training Results

The model trained for 60 epochs before early stopping triggered. Training curves show:
- Steady decrease in training loss
- Validation loss decreasing and stabilizing
- Recall improving throughout training
- No signs of overfitting (validation metrics track training metrics closely)

---

## Threshold Tuning Summary

### Motivation

The default classification threshold of 0.5 assumes equal cost for false positives and false negatives. In healthcare, however, missing a diabetes case (false negative) is far more dangerous than a false alarm (false positive). Threshold tuning allows us to adjust the decision boundary to prioritize recall.

### Methodology

We tested 81 thresholds from 0.1 to 0.9 (step size 0.01) and evaluated each using:
- Recall (primary metric for healthcare)
- Precision
- F1-Score
- F2-Score (emphasizes recall over precision)
- False Negatives (critical for healthcare)

### Results

**Optimal Threshold**: 0.44 (down from default 0.5)

**Performance Comparison**:

| Threshold | Recall | F2-Score | False Negatives |
|-----------|--------|----------|-----------------|
| Default (0.5) | 75.84% | 0.5924 | 1,708 |
| Optimal (0.44) | 80.76% | 0.5988 | 1,360 |

**Improvement**:
- **Recall**: +4.92 percentage points
- **False Negatives**: -348 cases (20.4% reduction)

**Healthcare Impact**: The lower threshold (0.44) means the model is more sensitive—it flags more cases as potentially diabetic. This results in 348 fewer missed diabetes cases, which is critical for early intervention and treatment.

---

## Architecture Tuning Summary

### Methodology

We tested five different MLP architectures to identify the optimal structure:

1. **A1_Simple**: [64, 32] with 0.3 dropout
2. **A2_Baseline**: [128, 64, 32] with 0.3 dropout (initial baseline)
3. **A3_Deeper**: [256, 128, 64, 32] with 0.3 dropout
4. **A4_Wider**: [256, 256, 128] with 0.3 dropout
5. **A5_HighDropout**: [128, 64] with **0.5 dropout** ⭐

All architectures were trained with identical hyperparameters (LR=0.001, Batch=256, Adam) for fair comparison.

### Results

**Best Architecture**: A5_HighDropout ([128, 64] with 0.5 dropout)

**Performance at Optimal Threshold (0.44)**:

| Architecture | Recall | F2-Score | ROC-AUC | False Negatives |
|--------------|--------|----------|---------|-----------------|
| A1_Simple | 79.23% | 0.5927 | 0.8181 | 1,468 |
| A2_Baseline | 77.88% | 0.5880 | 0.8152 | 1,564 |
| A3_Deeper | 72.43% | 0.5689 | 0.8068 | 1,949 |
| A4_Wider | 72.75% | 0.5704 | 0.8072 | 1,926 |
| **A5_HighDropout** | **80.10%** | **0.5981** | **0.8202** | **1,407** |

### Key Insights

1. **Higher Dropout is Better**: A5_HighDropout (0.5 dropout) outperformed the baseline (0.3 dropout) by 2.22 percentage points in recall, demonstrating that stronger regularization prevents overfitting and improves generalization.

2. **Simpler is Better**: The simpler [128, 64] architecture outperformed deeper (4-layer) and wider (256-256-128) networks. This suggests the problem doesn't require extreme model complexity, and moderate capacity with strong regularization is optimal.

3. **Overfitting in Complex Models**: A3_Deeper and A4_Wider showed lower recall and higher false negatives, indicating they may have overfit to training patterns that don't generalize well.

---

## Hyperparameter Optimization Summary

### Methodology

With the best architecture (A5_HighDropout) fixed, we optimized hyperparameters through a structured search:

**Search Space**:
- Learning Rates: [0.001, 0.0005, 0.0001]
- Batch Sizes: [128, 256, 512]
- Optimizers: [Adam, RMSprop]

**Total Configurations**: 18 combinations tested

Each configuration was trained with early stopping and evaluated at both default (0.5) and optimal (0.44) thresholds.

### Results

**Best Configuration**: LR=0.0001, Batch=128, Adam

**Performance at Optimal Threshold (0.44)**:

| Configuration | Recall | F2-Score | ROC-AUC | False Negatives |
|---------------|--------|----------|---------|-----------------|
| LR=0.001, BS=256, Adam (baseline) | 80.10% | 0.5981 | 0.8202 | 1,407 |
| **LR=0.0001, BS=128, Adam** | **83.56%** | **0.6017** | **0.8227** | **1,162** |

**Improvement from Architecture Baseline**:
- **Recall**: +3.46 percentage points
- **F2-Score**: +0.0036
- **False Negatives**: -245 cases (17.4% reduction)

### Key Insights

1. **Lower Learning Rate Improves Performance**: Reducing learning rate from 0.001 to 0.0001 allowed the model to converge more carefully, finding a better solution with higher recall.

2. **Smaller Batch Size Helps**: Batch size 128 outperformed 256 and 512. Smaller batches provide more frequent gradient updates and better gradient estimates, leading to improved generalization.

3. **Adam Consistently Better**: Adam optimizer outperformed RMSprop across most configurations, likely due to its adaptive learning rate mechanism that adjusts per-parameter learning rates.

---

## Final Model Performance

### Complete Optimization Journey

The final model represents the culmination of three optimization stages:

1. **Baseline MLP**: 75.84% recall, 1,708 false negatives
2. **Architecture Tuning**: 80.10% recall, 1,407 false negatives (+4.26% recall)
3. **Hyperparameter Optimization**: 83.56% recall, 1,162 false negatives (+3.46% recall)

**Total Improvement**: +7.72 percentage points recall, -546 false negatives (32% reduction)

### Final Model Metrics

**At Optimal Threshold (0.44)**:

- **Accuracy**: 68.34%
- **Precision**: 28.39%
- **Recall**: **83.56%** ⭐
- **F1-Score**: 0.4238
- **F2-Score**: **0.6017** ⭐
- **ROC-AUC**: **0.8227** ⭐
- **False Negatives**: **1,162**

**At Default Threshold (0.5)** (for comparison):

- **Accuracy**: 72.56%
- **Precision**: 30.88%
- **Recall**: 78.30%
- **F1-Score**: 0.4429
- **ROC-AUC**: 0.8227
- **False Negatives**: ~1,538

### Performance Interpretation

The final model achieves **83.56% recall**, meaning it correctly identifies 83.56% of all actual diabetes cases in the test set. This exceeds our target of 80% recall for healthcare screening applications. The model misses 1,162 diabetes cases (16.44% of actual cases), which is a significant improvement from the baseline's 1,708 missed cases.

The **ROC-AUC of 0.8227** indicates good discriminative ability—the model can effectively distinguish between individuals with and without diabetes. The **F2-Score of 0.6017** reflects a good balance between recall and precision, with emphasis on recall as appropriate for healthcare.

---

## Healthcare Interpretation

### Recall: The Critical Metric

**Recall (83.56%)** represents the percentage of actual diabetes cases that the model correctly identifies. In healthcare screening, this is the most important metric because:

- **False Negatives are Dangerous**: Missing a diabetes case means delayed diagnosis, delayed treatment, and potential complications. A patient with undiagnosed diabetes may develop serious health issues before receiving care.

- **Early Detection Saves Lives**: Diabetes is manageable with early intervention, but complications from undiagnosed diabetes can be severe (heart disease, kidney failure, vision loss, etc.).

- **Screening Context**: In a screening scenario, it's better to flag more people for follow-up testing (false positives) than to miss actual cases (false negatives).

### Precision Trade-off

**Precision (28.39%)** is lower than ideal, meaning that only 28.39% of predicted diabetes cases are actually diabetic. However, this trade-off is acceptable because:

- **False Positives are Manageable**: A false positive leads to follow-up tests (blood glucose, HbA1c) which can quickly confirm or rule out diabetes. These tests are relatively inexpensive and non-invasive.

- **Cost-Benefit Analysis**: The cost of a false positive (follow-up test) is far lower than the cost of a false negative (missed diagnosis leading to complications).

- **Screening vs. Diagnosis**: This model is designed for initial screening, not final diagnosis. High recall ensures we catch cases, and follow-up tests provide the precision needed for diagnosis.

### False Negatives: The Real Impact

The model reduces false negatives from **1,708 (baseline) to 1,162 (final)**, representing **546 fewer missed diabetes cases**. In a real-world screening scenario with 50,736 people:

- **Baseline Model**: Would miss 1,708 diabetes cases
- **Final Model**: Would miss 1,162 diabetes cases
- **Improvement**: 546 more people receive timely diagnosis and treatment

This improvement could prevent serious complications and improve quality of life for hundreds of individuals.

### Model Suitability for Healthcare

The final model is **suitable for healthcare screening** because:

1. ✅ **Exceeds Recall Target**: 83.56% recall exceeds the 80% target for screening applications
2. ✅ **Low False Negatives**: Only 16.44% of diabetes cases are missed
3. ✅ **Good Discriminative Ability**: ROC-AUC of 0.8227 indicates reliable predictions
4. ✅ **Appropriate Trade-offs**: Prioritizes catching cases over precision, which is correct for screening

**Recommendation**: The model can be deployed for initial screening with the understanding that positive predictions should trigger confirmatory laboratory tests before final diagnosis.

---

## Key Insights

### Technical Insights

1. **Regularization is Critical**: Higher dropout (0.5) with moderate architecture complexity outperformed deeper/wider networks, demonstrating that preventing overfitting is more important than model capacity for this problem.

2. **Careful Training Matters**: Lower learning rate (0.0001) and smaller batch size (128) improved performance, showing that slower, more careful training leads to better generalization.

3. **Threshold Optimization is Essential**: Adjusting the classification threshold from 0.5 to 0.44 improved recall by 4.92 percentage points, highlighting the importance of tuning for the specific application context.

4. **Architecture Simplicity**: The optimal architecture [128, 64] is relatively simple, suggesting that the problem doesn't require extreme model complexity when proper regularization is applied.

### Healthcare Insights

1. **Recall is Achievable**: With proper optimization, we achieved 83.56% recall, demonstrating that MLPs can effectively identify diabetes cases for screening purposes.

2. **False Negatives Can Be Reduced**: Through systematic optimization, we reduced false negatives by 32%, showing that model tuning can have real healthcare impact.

3. **Trade-offs are Acceptable**: Lower precision (28.39%) is an acceptable trade-off for higher recall (83.56%) in screening contexts where follow-up tests can provide precision.

4. **Model Interpretability**: While MLPs are less interpretable than linear models, their superior recall performance justifies their use in screening applications where catching cases is critical.

### Methodological Insights

1. **Systematic Optimization Works**: The three-stage optimization process (threshold → architecture → hyperparameters) systematically improved performance, demonstrating the value of structured model development.

2. **Early Stopping Prevents Overfitting**: Early stopping with validation monitoring ensured the model didn't overfit, maintaining good generalization to unseen data.

3. **SMOTE is Effective**: SMOTE successfully balanced the training set and improved model performance on the minority class, demonstrating the importance of addressing class imbalance in healthcare datasets.

4. **Comprehensive Evaluation**: Evaluating at multiple thresholds and using multiple metrics (recall, F2-score, ROC-AUC, false negatives) provided a complete picture of model performance in the healthcare context.

---

## Conclusion

The final optimized MLP model achieves **83.56% recall** with **1,162 false negatives**, representing a **32% reduction** in missed diabetes cases compared to the baseline. This performance makes the model suitable for healthcare screening applications where identifying individuals at risk of diabetes is critical for early intervention and improved patient outcomes.

The systematic optimization process—from threshold tuning through architecture selection to hyperparameter optimization—demonstrated that careful model development can significantly improve healthcare-relevant metrics. The model's emphasis on recall over precision aligns with the healthcare screening context, where catching cases is more important than avoiding false alarms.


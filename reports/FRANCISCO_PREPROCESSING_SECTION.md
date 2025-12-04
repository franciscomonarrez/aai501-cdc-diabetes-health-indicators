# Preprocessing Section - Data Preparation for MLP Training

**Author**: Francisco (Team Member 3)

---

## Dataset Loading

The CDC Diabetes Health Indicators dataset was obtained from the UCI Machine Learning Repository and contains 253,680 records with 22 features (21 predictors + 1 target variable). The dataset is based on the Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey, which collects health-related data from U.S. adults.

**Dataset Characteristics**:
- **Total Samples**: 253,680
- **Features**: 21 health indicators
- **Target Variable**: `Diabetes_binary` (0 = no diabetes, 1 = prediabetes/diabetes)
- **Data Types**: All features are numerical (float64)
- **Missing Values**: None (complete dataset)

**Features Include**:
- Demographics: Age, Sex, Education, Income
- Health Conditions: HighBP, HighChol, Stroke, HeartDiseaseorAttack
- Lifestyle: BMI, Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump
- Healthcare Access: AnyHealthcare, NoDocbcCost
- Health Status: GenHlth, MentHlth, PhysHlth, DiffWalk

The dataset was loaded using pandas and validated to ensure the target column exists and contains valid values.

---

## Class Imbalance Description

### Imbalance Analysis

The dataset exhibits **severe class imbalance**, which is a critical issue for machine learning models:

**Class Distribution**:
- **No Diabetes (0)**: 218,334 samples (86.07%)
- **Diabetes (1)**: 35,346 samples (13.93%)
- **Imbalance Ratio**: 6.18:1 (majority class is 6.18 times larger)

### Why This Matters

Class imbalance poses several challenges:

1. **Model Bias**: Without addressing imbalance, models tend to predict the majority class (no diabetes) most of the time, achieving high accuracy but poor recall for the minority class (diabetes).

2. **Healthcare Impact**: In our healthcare context, the minority class (diabetes) is the one we most need to identify. A model that predicts "no diabetes" for everyone would achieve 86% accuracy but would miss 100% of actual diabetes cases—completely useless for screening.

3. **Evaluation Misleading**: High overall accuracy can be misleading when classes are imbalanced. A model with 86% accuracy might be predicting "no diabetes" for all cases, which is unacceptable for our use case.

4. **Training Instability**: Imbalanced datasets can lead to unstable training, especially for neural networks, where the model may not see enough examples of the minority class to learn meaningful patterns.

### Visual Evidence

The class distribution plot (see `figures/francisco_class_distribution.png`) clearly shows the severe imbalance:
- A large blue bar representing 218,334 "no diabetes" cases
- A much smaller red bar representing 35,346 "diabetes" cases
- The pie chart shows 86% vs 14% split

This visual representation makes it immediately clear why class imbalance handling is necessary.

---

## SMOTE Explanation

### What is SMOTE?

SMOTE (Synthetic Minority Oversampling Technique) is an oversampling method that addresses class imbalance by creating synthetic samples of the minority class. Unlike simple oversampling (which duplicates existing samples), SMOTE generates new samples by interpolating between existing minority class samples.

### How SMOTE Works

1. **For each minority class sample**:
   - Find its k nearest neighbors (also from the minority class)
   - Randomly select one of these neighbors
   - Create a new synthetic sample at a random point along the line segment connecting the original sample and the selected neighbor

2. **Result**: The minority class is expanded with realistic, synthetic samples that maintain the feature distribution characteristics of the original minority class.

### Why SMOTE for This Project

We chose SMOTE over alternative approaches for several reasons:

1. **Preserves Feature Relationships**: SMOTE creates samples that respect the feature space structure, unlike random oversampling which just duplicates samples.

2. **No Information Loss**: Unlike undersampling (which removes majority class samples), SMOTE doesn't discard any information from the original dataset.

3. **Effective for Neural Networks**: SMOTE works well with deep learning models, providing balanced training data that helps the model learn minority class patterns.

4. **Proven in Healthcare**: SMOTE has been successfully used in medical/healthcare machine learning applications with imbalanced datasets.

### SMOTE Application

**Before SMOTE**:
- Training set: [148,467 no diabetes, 24,035 diabetes]
- Imbalance ratio: 6.18:1

**After SMOTE**:
- Balanced training set: [148,467, 148,467]
- Perfect 1:1 balance

**Process**:
- SMOTE generated 124,432 synthetic diabetes samples
- These synthetic samples are interpolations between real diabetes cases
- The balanced dataset allows the model to see equal numbers of both classes during training

### Alternative Considered: Class Weights

We also considered using class weights (assigning higher weight to minority class samples in the loss function) instead of SMOTE. However, SMOTE was chosen because:

1. **More Effective**: SMOTE provides actual balanced data, while class weights only adjust the loss function
2. **Better for Neural Networks**: Balanced training data often leads to better convergence and performance
3. **Explicit Oversampling**: Makes the class balance handling transparent and interpretable

---

## Train/Validation/Test Split Method

### Split Strategy

The dataset was split using a **stratified train-test split** to ensure class distribution is maintained across splits:

**Split Proportions**:
- **Training Set**: 80% (202,944 samples)
- **Test Set**: 20% (50,736 samples)

**Stratification**: The split was stratified by the target variable (`Diabetes_binary`) to ensure that both training and test sets maintain the same class distribution (86% no diabetes, 14% diabetes).

### Validation Set Creation

From the training set, a validation set was created for model monitoring during training:

- **Final Training**: 85% of training set (172,502 samples)
- **Validation Set**: 15% of training set (30,442 samples)

**Purpose of Validation Set**:
- **Early Stopping**: Monitor validation loss to stop training when the model stops improving
- **Hyperparameter Tuning**: Evaluate different configurations without touching the test set
- **Overfitting Detection**: Compare training vs validation metrics to detect overfitting

### Why This Split Method?

1. **Stratification Maintains Distribution**: Ensures both splits have the same class imbalance, making evaluation fair and representative.

2. **Test Set Isolation**: The test set (20%) is completely held out and only used for final evaluation, providing an unbiased estimate of model performance.

3. **Validation for Tuning**: The validation set (15% of training) allows us to tune hyperparameters and select models without contaminating the test set.

4. **Standard Practice**: 80/20 train-test split with additional validation split is standard practice in machine learning and ensures robust evaluation.

### Data Flow

```
Original Dataset (253,680 samples)
    ↓
Train/Test Split (stratified)
    ├── Training (202,944) → Used for model training
    └── Test (50,736) → Held out for final evaluation
    
Training Set (202,944)
    ↓
Train/Val Split (stratified)
    ├── Final Training (172,502) → Used for training
    └── Validation (30,442) → Used for early stopping
    
Final Training (172,502)
    ↓
SMOTE Application
    ↓
Balanced Training (296,934) → [148,467, 148,467]
```

### Preprocessing Pipeline

**Step-by-Step Process**:

1. **Load Dataset**: Read CSV file into pandas DataFrame
2. **Validate Data**: Check for missing values, verify target column exists
3. **Split Features and Target**: Separate predictors (X) from target (y)
4. **Train-Test Split**: 80/20 stratified split
5. **Create Preprocessor**: Fit StandardScaler on training data
6. **Transform Data**: Scale both training and test sets
7. **Train-Val Split**: 85/15 stratified split from training
8. **Apply SMOTE**: Balance training set (only on training, not validation or test)
9. **Ready for Training**: Balanced, scaled training data with validation and test sets prepared

### Important Notes

- **SMOTE Only on Training**: SMOTE is applied only to the training set, never to validation or test sets. This ensures that validation and test performance reflect real-world conditions with natural class imbalance.

- **Scaling Before SMOTE**: Features are scaled before SMOTE application to ensure distance calculations (for finding nearest neighbors) are meaningful across different feature scales.

- **Reproducibility**: All splits use `random_state=42` to ensure reproducibility across runs.

---

## Feature Scaling

### StandardScaler

All numerical features were standardized using StandardScaler, which transforms features to have:
- **Mean**: 0
- **Standard Deviation**: 1

**Formula**: `z = (x - μ) / σ`

Where:
- `x` = original feature value
- `μ` = mean of the feature
- `σ` = standard deviation of the feature
- `z` = standardized value

### Why Scaling is Critical

1. **Neural Network Requirement**: Neural networks are sensitive to feature scales. Features with larger ranges can dominate the learning process, while features with smaller ranges may be ignored.

2. **Gradient Stability**: Standardized features lead to more stable gradients during backpropagation, enabling more effective training.

3. **SMOTE Effectiveness**: SMOTE relies on distance calculations (Euclidean distance) to find nearest neighbors. Without scaling, features with larger ranges (e.g., Age: 1-13) would dominate distance calculations over features with smaller ranges (e.g., binary features: 0-1).

4. **Convergence Speed**: Scaled features typically lead to faster convergence during training.

### Scaling Process

1. **Fit on Training**: StandardScaler is fitted only on training data to compute mean and standard deviation
2. **Transform Training**: Apply scaling to training data
3. **Transform Test**: Apply the same scaling (using training statistics) to test data

**Important**: Test data is scaled using training statistics (mean and std) to prevent data leakage. Using test set statistics would be cheating and give overly optimistic performance estimates.

---

## Data Quality Checks

### Missing Values

**Result**: No missing values found in the dataset.

The dataset is complete, which simplifies preprocessing. No imputation was necessary.

### Data Types

**Result**: All 22 columns are float64 (numerical).

This is expected since the dataset contains:
- Binary features (0/1) encoded as floats
- Categorical features (Age, Education, Income) encoded as numerical codes
- Continuous features (BMI, GenHlth, MentHlth, PhysHlth)

No categorical encoding (one-hot encoding) was needed since all features are already numerical.

### Outlier Detection

While not explicitly handled in preprocessing, the dataset contains some extreme values:
- **BMI**: Range from 12 to 98 (some values may be outliers or data entry errors)
- **MentHlth/PhysHlth**: Range from 0 to 30 (number of days with poor health)

These extreme values are preserved in the dataset as they may represent legitimate cases (e.g., very high BMI indicating severe obesity, which is a diabetes risk factor). StandardScaler helps normalize these values, and the model can learn to handle them appropriately.

---

## Preprocessing Summary

The preprocessing pipeline transforms the raw CDC Diabetes Health Indicators dataset into a format suitable for MLP training:

1. ✅ **Dataset Loaded**: 253,680 samples, 21 features, no missing values
2. ✅ **Stratified Splits**: 80/20 train-test, then 85/15 train-val (maintains class distribution)
3. ✅ **Feature Scaling**: StandardScaler applied (fit on training, transform all sets)
4. ✅ **Class Imbalance Addressed**: SMOTE applied to training set (6.18:1 → 1:1)
5. ✅ **Data Quality**: Validated for completeness and consistency

**Final Preprocessed Data**:
- **Training**: 296,934 balanced samples (after SMOTE)
- **Validation**: 30,442 samples (natural imbalance)
- **Test**: 50,736 samples (natural imbalance, held out)

This preprocessing ensures the model trains on balanced data while being evaluated on realistic, imbalanced test data that reflects real-world screening conditions.


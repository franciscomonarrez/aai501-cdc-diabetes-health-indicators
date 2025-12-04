# Exploratory Data Analysis (EDA) Section

**Author**: Francisco (Team Member 3)

---

## Dataset Overview

The CDC Diabetes Health Indicators dataset contains **253,680 records** from the Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey. Each record represents one adult respondent and includes 21 health-related features plus a binary target variable indicating diabetes status.

**Dataset Characteristics**:
- **Total Samples**: 253,680
- **Features**: 21 health indicators
- **Target Variable**: `Diabetes_binary` (0 = no diabetes, 1 = prediabetes/diabetes)
- **Data Completeness**: 100% (no missing values)
- **Data Types**: All numerical (float64)

---

## Class Distribution Analysis

### Overall Distribution

The dataset exhibits **severe class imbalance**, which is a critical finding for model development:

- **No Diabetes (0)**: 218,334 samples (**86.07%**)
- **Diabetes (1)**: 35,346 samples (**13.93%**)
- **Imbalance Ratio**: **6.18:1** (majority class is 6.18 times larger than minority class)

### Visual Representation

The class distribution plot (see `figures/francisco_class_distribution.png`) clearly illustrates this imbalance:
- A large bar representing 218,334 "no diabetes" cases (86.07%)
- A much smaller bar representing 35,346 "diabetes" cases (13.93%)
- The pie chart shows the dramatic 86% vs 14% split

### Implications

This severe imbalance has several important implications:

1. **Model Development**: Without addressing imbalance, models will be biased toward predicting the majority class, achieving high accuracy but poor recall for diabetes cases.

2. **Evaluation Metrics**: Overall accuracy becomes misleading. A model that predicts "no diabetes" for everyone would achieve 86% accuracy but would be useless for screening.

3. **Healthcare Context**: The minority class (diabetes) is the one we most need to identify, making this imbalance particularly problematic.

4. **Preprocessing Requirement**: This imbalance necessitates class imbalance handling techniques (SMOTE or class weights) during model training.

---

## Feature Distributions

### Key Features Analyzed

The feature distribution analysis (see `figures/francisco_feature_distributions.png`) reveals important patterns in the health indicators:

**BMI (Body Mass Index)**:
- Range: 12 to 98
- Distribution: Right-skewed with most values between 20-35
- Insight: Higher BMI is associated with diabetes risk, and the distribution shows many individuals in the overweight/obese range

**Age**:
- Range: 1 to 13 (encoded categories)
- Distribution: Relatively uniform across age groups
- Insight: Age is a known diabetes risk factor, and the dataset includes representation across age groups

**General Health (GenHlth)**:
- Range: 1 to 5 (1=excellent, 5=poor)
- Distribution: Skewed toward lower values (better health)
- Insight: Self-reported health status may correlate with diabetes risk

**Mental Health Days (MentHlth)**:
- Range: 0 to 30 (days in past month)
- Distribution: Highly right-skewed with most values at 0
- Insight: Most respondents report no mental health issues, but some report many days

**Physical Health Days (PhysHlth)**:
- Range: 0 to 30 (days in past month)
- Distribution: Similar to mental health, highly right-skewed
- Insight: Physical health issues may be more common than mental health issues in this population

**Income and Education**:
- Both encoded as categorical variables
- Distributions show representation across socioeconomic levels
- Insight: Socioeconomic factors may influence diabetes risk and healthcare access

### Distribution Patterns

Most features show:
- **Right-skewed distributions**: Many health-related features (BMI, health days) have long tails toward higher values
- **Binary-like distributions**: Many features are effectively binary (0/1) despite being encoded as floats
- **Categorical encoding**: Age, Education, Income are encoded as numerical categories

These distribution patterns inform preprocessing decisions (scaling, handling of extreme values) and model design (activation functions, loss functions).

---

## Correlation Analysis

### Correlation Matrix

The correlation heatmap (see `figures/francisco_correlation_heatmap.png`) reveals relationships between features and the target variable.

### Key Correlations with Target

The target correlation plot (see `figures/francisco_target_correlations.png`) shows which features are most strongly associated with diabetes:

**Strongest Positive Correlations** (higher values associated with diabetes):
- **GenHlth** (General Health): Poorer self-reported health correlates with diabetes
- **PhysHlth** (Physical Health Days): More days with poor physical health correlates with diabetes
- **HighBP** (High Blood Pressure): Strong correlation with diabetes
- **HighChol** (High Cholesterol): Strong correlation with diabetes
- **BMI**: Higher BMI correlates with diabetes risk
- **DiffWalk** (Difficulty Walking): Mobility issues correlate with diabetes

**Strongest Negative Correlations** (higher values associated with no diabetes):
- **PhysActivity** (Physical Activity): More physical activity correlates with lower diabetes risk
- **Income**: Higher income correlates with lower diabetes risk
- **Education**: Higher education correlates with lower diabetes risk

### Feature-Feature Correlations

The correlation matrix reveals several interesting feature relationships:

1. **Health Conditions Cluster**: HighBP, HighChol, HeartDiseaseorAttack, and Stroke show positive correlations with each other, indicating that these conditions often co-occur.

2. **Lifestyle Factors**: PhysActivity, Fruits, and Veggies show positive correlations, suggesting that individuals who engage in one healthy behavior tend to engage in others.

3. **Socioeconomic Factors**: Income and Education are positively correlated, and both are negatively correlated with diabetes risk, suggesting socioeconomic status influences health outcomes.

4. **Health Status Indicators**: GenHlth, MentHlth, and PhysHlth are correlated, indicating that different aspects of health status are related.

### Implications for Modeling

1. **Feature Redundancy**: Some features are highly correlated, which could lead to multicollinearity. However, neural networks can handle correlated features better than linear models.

2. **Important Predictors**: Features with strong correlations to the target (GenHlth, HighBP, HighChol, BMI) are likely to be important predictors in the model.

3. **Feature Engineering**: The correlations suggest that composite features (e.g., combining health conditions) might be informative, though we kept features separate for interpretability.

---

## Class Imbalance Insights

### Detailed Imbalance Analysis

The class imbalance is not just a statistical issue—it reflects real-world prevalence:

- **Real-world Prevalence**: The 13.93% diabetes rate in the dataset is consistent with U.S. diabetes prevalence estimates, making this a realistic representation of screening populations.

- **Screening Context**: In a screening scenario, we expect most people to not have diabetes, so this imbalance is expected and realistic.

- **Model Challenge**: The imbalance makes it difficult for models to learn minority class patterns, as they see far more examples of the majority class during training.

### Impact on Model Development

1. **Evaluation Strategy**: We must use metrics that account for imbalance (recall, F2-score, ROC-AUC) rather than relying solely on accuracy.

2. **Preprocessing Requirement**: SMOTE or class weights are essential to prevent the model from simply predicting the majority class.

3. **Threshold Tuning**: The optimal classification threshold (0.44) is lower than default (0.5) to prioritize recall and catch more diabetes cases.

4. **Healthcare Priority**: The imbalance reinforces the importance of recall—we must catch the minority class (diabetes) cases, even if it means more false positives.

---

## Summary Statistics

### Descriptive Statistics

The dataset summary statistics (see `reports/francisco_eda_summary.txt`) provide key insights:

**Key Statistics**:
- **Mean BMI**: 28.38 (overweight range, which is a diabetes risk factor)
- **Mean Age**: 8.03 (encoded, represents middle-aged to older adults)
- **HighBP Prevalence**: 42.9% of respondents
- **HighChol Prevalence**: 42.4% of respondents
- **Physical Activity**: 75.7% report engaging in physical activity
- **Healthcare Access**: 95.1% report having healthcare access

### Health Profile Insights

1. **High Risk Population**: The high prevalence of high blood pressure (42.9%) and high cholesterol (42.4%) suggests this is a population with multiple cardiovascular risk factors, which often co-occur with diabetes.

2. **Lifestyle Factors**: Most respondents (75.7%) report physical activity, suggesting that even with healthy behaviors, diabetes risk remains, possibly due to other factors (genetics, age, other health conditions).

3. **Healthcare Access**: High healthcare access (95.1%) is positive, but the dataset still shows significant diabetes prevalence, indicating that access alone doesn't prevent diabetes.

4. **BMI Distribution**: Mean BMI of 28.38 is in the overweight range, which is a known diabetes risk factor. The range (12-98) includes extreme values that may represent data entry errors or severe cases.

---

## Data Quality Assessment

### Completeness

- **Missing Values**: None detected across all 253,680 records and 22 columns
- **Data Quality**: The dataset is complete and ready for analysis without imputation

### Consistency

- **Data Types**: All features are consistently numerical (float64)
- **Value Ranges**: All features fall within expected ranges based on their definitions
- **No Obvious Errors**: While some extreme values exist (e.g., BMI=98), these may represent legitimate severe cases rather than errors

### Representativeness

- **Large Sample Size**: 253,680 samples provide sufficient data for training complex models
- **National Representation**: BRFSS is a national survey, so the dataset should be representative of U.S. adults
- **Temporal Context**: 2015 data may not reflect current trends, but provides a solid baseline for model development

---

## Key EDA Findings

### Summary of Insights

1. **Severe Class Imbalance**: 6.18:1 ratio requires class imbalance handling (SMOTE)

2. **Strong Predictors Identified**: GenHlth, HighBP, HighChol, BMI show strong correlations with diabetes

3. **Health Condition Clustering**: Multiple health conditions (high BP, high cholesterol, heart disease) often co-occur

4. **Lifestyle Factors Matter**: Physical activity, diet (fruits/veggies) show protective associations

5. **Socioeconomic Factors**: Income and education are associated with lower diabetes risk

6. **Complete Dataset**: No missing values simplifies preprocessing

7. **Realistic Prevalence**: 13.93% diabetes rate reflects real-world screening populations

### Implications for Modeling

These EDA findings directly informed our modeling approach:

- **Class Imbalance Handling**: SMOTE was essential given the 6.18:1 imbalance
- **Feature Selection**: All 21 features were retained as they all show some relationship to diabetes
- **Evaluation Focus**: Recall and F2-score prioritized over accuracy due to imbalance
- **Threshold Tuning**: Lower threshold (0.44) needed to catch more diabetes cases
- **Model Complexity**: Moderate architecture sufficient given feature relationships

---

## Visualizations Reference

All EDA visualizations are saved in the `figures/` directory:

1. **Class Distribution** (`francisco_class_distribution.png`): Bar chart and pie chart showing 86% vs 14% split
2. **Feature Distributions** (`francisco_feature_distributions.png`): Histograms of key features (BMI, Age, GenHlth, etc.)
3. **Correlation Heatmap** (`francisco_correlation_heatmap.png`): Full correlation matrix showing all feature relationships
4. **Target Correlations** (`francisco_target_correlations.png`): Bar chart showing which features correlate most with diabetes

These visualizations provide clear, interpretable insights that support the modeling decisions and can be included directly in the final report.


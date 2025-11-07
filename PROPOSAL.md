# AAI-501 Final Project — Proposal

## Topic

Predict the likelihood of diabetes using CDC BRFSS health indicators to support screening and prevention.

## Problem, algorithms, system

* Supervised classification on tabular survey data
* Algorithms: Logistic Regression, Random Forest, XGBoost
* System: clean data → EDA → cross-validated training → selection → explainability via feature importance or SHAP

## Course topics

Classification, model evaluation, cross-validation, imbalanced learning, basic hyperparameter search.

## Expected behaviors

Input: health indicators
Output: probability of diabetes
Behavior: calibrated scores, interpretable drivers

## Issues to focus on

Class imbalance, calibration, leakage prevention, sensitivity across demographics.

## Initial references (APA 7)

* UCI Machine Learning Repository. CDC Diabetes Health Indicators. [https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
* Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR.
* Chen, T., & Guestrin, C. (2016). XGBoost. KDD.
* Lundberg, S. M., & Lee, S.-I. (2017). SHAP. NeurIPS.

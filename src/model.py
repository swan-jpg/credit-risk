"""
model.py

Baseline logistic regression and model evaluation
for the credit risk model.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

BASELINE_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]
# ============================================================
# Data Preparation
# ============================================================

def prepare_features(df, target_col, features):
    """
    Separate predictors and target.
    """

    X = df[features].copy()
    y = df[target_col].copy()

    return X, y

# ============================================================
# Sklearn Logistic Regression
# ============================================================

def fit_logistic_model(train_df, target_col, features):
    """
    Fit logistic regression using sklearn.

    Used primarily for prediction and model performance
    evaluation.
    """

    X_train, y_train = prepare_features(train_df, target_col,features)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    return model

# ============================================================
# Statsmodels Logistic Regression
# ============================================================

def fit_logit_inference(train_df, target_col, features):
    """
    Fit logistic regression using statsmodels.

    Used primarily for statistical inference:
    coefficients, p-values, standard errors, and
    confidence intervals.
    """

    X_train, y_train = prepare_features(
        train_df,
        target_col,
        features
    )

    X_train = sm.add_constant(X_train)

    model = sm.Logit(
        y_train,
        X_train
    )

    result = model.fit()

    return result


# ============================================================
# Predictions
# ============================================================

def predict_probabilities(model, df, features):
    """
    Generate predicted probabilities of default
    using a fitted sklearn model.
    """

    X = df[features]
    return model.predict_proba(X)[:, 1]

# ============================================================
# ROC-AUC
# ============================================================

def calculate_roc_auc(y_true, probabilities):
    """
    Calculate ROC-AUC.
    """
    return roc_auc_score(y_true,probabilities)

# ============================================================
# KS Statistic
# ============================================================

def calculate_ks(y_true, probabilities):
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic.

    KS measures the maximum separation between the
    cumulative distributions of predicted probabilities
    for defaults and non-defaults.
    """

    fpr, tpr, thresholds = roc_curve(y_true, probabilities)

    ks = max(tpr - fpr)

    return ks



# ============================================================
# Model Evaluation
# ============================================================

def evaluate_model(y_true, probabilities):
    """
    Wrapper for ROC-AUC and KS statistic
    """

    roc_auc = calculate_roc_auc(y_true,probabilities)    
    ks = calculate_ks(y_true,probabilities)

    return {
        "ROC-AUC": roc_auc,
        "KS": ks
    }

if __name__ == "__main__":
    pass
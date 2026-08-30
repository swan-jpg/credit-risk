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
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

    X_train, y_train = prepare_features(train_df, target_col, features)

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

def summarize_logit_inference(result):
    """
    Gathers statistical informaiton from tted statsmodels
    logistic regression result.

    Returns:
        DataFrame containing coefficients, standard errors,
        p-values, and confidence intervals.
    """

    summary = pd.DataFrame({
        "coefficient": result.params,
        "std_error": result.bse,
        "p_value": result.pvalues
    })

    confidence_intervals = result.conf_int()

    summary["ci_lower"] = confidence_intervals[0]
    summary["ci_upper"] = confidence_intervals[1]

    return summary

def calc_information_criterion(result):
    """
    Calculate AIC and BIC for a fitted statsmodels
    logistic regression model.

    Lower values are better
    """

    return {
        "AIC": float(round(result.aic,2)),
        "BIC": float(round(result.bic,2))
    }

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

    fpr, tpr, _ = roc_curve(y_true, probabilities)

    ks = max(tpr - fpr)

    return ks

# ============================================================
# VIF(Variance Inflation Factor)
# ============================================================
def calculate_vif(df, features):
    """
    Calculates VIF
    """
    X = df[features].copy()
    X = sm.add_constant(X)

    vif = pd.DataFrame()

    vif["feature"] = X.columns
    vif["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    return vif

# ============================================================
# Weight of Evidence (WoE) and Information Value (IV)
# ============================================================
def calculate_woe_iv(
    df,
    feature,
    target,
    bins=10,
    low_cardinality_threshold=10,
    dominant_value_threshold=0.5,
    epsilon=1e-6,
):
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV)
    for a single feature.

    Missing values are placed into a separate "Missing" bin.

    Numeric features are handled in one of two ways:

    - Categorical-style binning (bin on each unique value directly):
        Used when either of the following is true:
          (a) the feature has <= low_cardinality_threshold unique
              non-missing values (e.g. binary flags, small counts), or
          (b) a single value accounts for more than
              dominant_value_threshold of the non-missing
              observations (e.g. delinquency counts that are
              mostly zero).
        Condition (b) matters because quantile binning assumes
        roughly even population across bins. A single dominant
        value collapses all quantile edges onto that value, and
        after duplicates are dropped the whole column falls into
        one bin, producing IV = 0 regardless of how predictive the
        feature actually is.

    - Quantile binning:
        Used for higher-cardinality numeric features with no
        single dominant value.

    Non-numeric features are treated as categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    feature : str
        Feature to analyze.

    target : str
        Binary target variable where 0 = good and 1 = bad.

    bins : int, default=10
        Number of quantile bins for higher-cardinality numeric
        variables with no dominant value.

    low_cardinality_threshold : int, default=10
        Numeric features with at most this many unique non-missing
        values are treated as categorical variables.

    dominant_value_threshold : float, default=0.5
        If a single value accounts for more than this fraction of
        non-missing observations, the feature is treated as
        categorical regardless of its unique value count.

    epsilon : float, default=1e-6
        Small value used to prevent division by zero when calculating
        distributions and WoE.

    Returns
    -------
    iv : float
        Total Information Value for the feature.

    woe_table : pd.DataFrame
        Bin-level statistics including total observations, good,
        bad, distributions, WoE, and IV.

    Notes
    -----
    WoE:

        WoE = ln(distribution_good / distribution_bad)

    Information Value:

        IV = (distribution_good - distribution_bad) * WoE
    """

    # ==============================================================
    # Validate inputs
    # ==============================================================

    required_columns = {feature, target}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(
            f"Missing columns in dataframe: {sorted(missing_columns)}"
        )

    if bins < 1:
        raise ValueError("bins must be >= 1")

    if low_cardinality_threshold < 0:
        raise ValueError(
            "low_cardinality_threshold must be >= 0"
        )

    if not (0 < dominant_value_threshold <= 1):
        raise ValueError(
            "dominant_value_threshold must be in (0, 1]"
        )

    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    # Work only with the feature and target
    data = df[[feature, target]].copy()

    # ==============================================================
    # Validate target
    # ==============================================================

    non_missing_target = data[target].dropna()

    if not non_missing_target.isin([0, 1]).all():
        raise ValueError(
            f"Target '{target}' must contain only 0 and 1 values."
        )

    # Remove rows with missing target
    data = data.dropna(subset=[target])

    # ==============================================================
    # Identify feature type
    # ==============================================================

    is_numeric = pd.api.types.is_numeric_dtype(data[feature])

    missing_mask = data[feature].isna()

    non_missing_feature = data.loc[~missing_mask, feature]
    n_unique = non_missing_feature.nunique()

    # Share of non-missing observations held by the single most
    # common value. Guards against quantile binning collapsing to
    # one bin when a feature is heavily concentrated on one value
    # (e.g. delinquency counts that are mostly zero).
    if len(non_missing_feature) > 0:
        max_value_share = (
            non_missing_feature.value_counts(normalize=True).iloc[0]
        )
    else:
        max_value_share = 0.0

    use_categorical_binning = (
        n_unique <= low_cardinality_threshold
        or max_value_share > dominant_value_threshold
    )

    # ==============================================================
    # Create bins
    # ==============================================================

    if is_numeric:

        if use_categorical_binning:
            # Treat low-cardinality or heavily-concentrated numeric
            # variables as categorical.
            # Example: 0, 1, 2, 3 delinquency counts.
            data["bin"] = data[feature].astype("object")

        else:
            # Quantile-bin higher-cardinality, non-concentrated
            # numeric variables.
            data["bin"] = pd.qcut(
                data[feature],
                q=bins,
                duplicates="drop",
            ).astype(str)

    else:
        # Treat non-numeric variables as categorical.
        data["bin"] = data[feature].astype("object")

    # Missing values receive their own bin
    data.loc[missing_mask, "bin"] = "Missing"

    # Convert bins to strings for cleaner output
    data["bin"] = data["bin"].astype(str)

    # ==============================================================
    # Calculate counts by bin
    # ==============================================================

    grouped = (
        data.groupby("bin", observed=False)[target]
        .agg(
            total="count",
            bad="sum",
        )
    )

    grouped["good"] = grouped["total"] - grouped["bad"]

    # ==============================================================
    # Calculate distributions
    # ==============================================================

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    if total_good == 0 or total_bad == 0:
        raise ValueError(
            "WoE/IV cannot be calculated because the target "
            "contains only one class after removing missing "
            "target values."
        )

    grouped["dist_good"] = (
        grouped["good"] / total_good
    ).clip(lower=epsilon)

    grouped["dist_bad"] = (
        grouped["bad"] / total_bad
    ).clip(lower=epsilon)

    # ==============================================================
    # Calculate WoE and IV
    # ==============================================================

    grouped["woe"] = np.log(
        grouped["dist_good"] / grouped["dist_bad"]
    )

    grouped["iv"] = (
        grouped["dist_good"] - grouped["dist_bad"]
    ) * grouped["woe"]

    # Total Information Value
    iv = grouped["iv"].sum()

    # ==============================================================
    # Return results
    # ==============================================================

    woe_table = grouped.reset_index()

    return iv, woe_table


def calculate_iv_all_features(
    df,
    target,
    features,
    bins=10,
    low_cardinality_threshold=10,
    dominant_value_threshold=0.5,
):
    """
    Calculate Information Value for multiple features.

    Returns
    -------
    pd.DataFrame
        Features ranked from highest to lowest IV.
    """

    results = []

    for feature in features:
        iv, _ = calculate_woe_iv(
            df=df,
            feature=feature,
            target=target,
            bins=bins,
            low_cardinality_threshold=low_cardinality_threshold,
            dominant_value_threshold=dominant_value_threshold,
        )
        results.append({
            "feature": feature,
            "IV": iv
        })

    iv_table = pd.DataFrame(results)

    return iv_table.sort_values(
        "IV",
        ascending=False
    ).reset_index(drop=True)

# ============================================================
# Model Evaluation
# ============================================================

def evaluate_model(y_true, probabilities):
    """
    Wrapper for ROC-AUC and KS statistic
    """

    roc_auc = calculate_roc_auc(y_true, probabilities)    
    ks = calculate_ks(y_true, probabilities)

    return {
        "ROC-AUC": round(float(roc_auc), 5),
        "KS": round(float(ks), 5)
    }

if __name__ == "__main__":
    pass
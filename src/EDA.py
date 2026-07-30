
"""
exploratory_analysis.py

EDA for the Give Me Some Credit dataset (SeriousDlqin2yrs).
No data is modified.

Findings from this module should be recorded (see `eda_findings` dict
at the bottom) to directly justify decisions in preprocessing.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pprint

TARGET = "SeriousDlqin2yrs"


# ---------------------------------------------------------------------
# 1. Understand the Prediction Target
# ---------------------------------------------------------------------

def target_balance(df: pd.DataFrame, target_col: str = TARGET) -> pd.DataFrame:
    """
    Summarizes the distribution of the target variable

    Returns class counts, percentage distribution, and imbalance ratio
    to assess target imbalance and help determine appropriate evaluation
    metrics and modeling strategies.
    """

    class_counts = df[target_col].value_counts()
    percentages = (df[target_col].value_counts(normalize=True) * 100).round(2)

    summary = pd.DataFrame({"count": class_counts, "percent": percentages})
    summary["imbalance_ratio"] = (class_counts / class_counts.min()).round(2)

    print("\n==================== Target Distribution ====================")
    print(summary)
    
    return summary
# ---------------------------------------------------------------------
# 2. Understand Each Feature Individually
# ---------------------------------------------------------------------

def feature_distribution(df: pd.DataFrame, col: str) -> dict:
    """
    Distribution, skew, missingness, range, etc. for a single feature.
    - Flags candidates for log-transform, imputation, or investigation
       of impossible values (e.g. age = 0, negative income).

    Returns a dict of findings rather than just printing, so results
    can be collected across all features.
    """
   
    if not pd.api.types.is_numeric_dtype(df[col]): 
        print(f"Skipping {col}, non-numeric data values")
        return {       
            "feature": col,
            "dtype": df[col].dtype
        }

    mean_value = df[col].mean() 
    median_value = df[col].median() 
    min_value = df[col].min()
    max_value = df[col].max()

    findings = {
        "feature": col,
        "dtype": df[col].dtype,
        "missing_count": df[col].isnull().sum(),
        "missing_percent": ((df[col].isnull().sum() / len(df[col]))*100).round(2),
        "n_unique": df[col].nunique(), 
        "mean": mean_value,
        "median": median_value, 
        "mean_median_gap": round(mean_value - median_value,2), 
        "skew": df[col].skew(), 
        "kurtosis": df[col].kurtosis(),
        "min": min,
        "max": max,
        "range": round(max_value - min_value,2,),
        "std": df[col].std()
        }
    
    print(f"\n==================== {col} Findings ====================")
    pprint(findings)

    return findings


def scan_all_features(df: pd.DataFrame, exclude: list = None) -> pd.DataFrame:
    """
    Runs feature_distribution() across every column and returns a summary
    table
    """
    if exclude is None:
        exclude = [TARGET]
    summaries = []

    for col in df.columns:
        if col not in exclude:
            summaries.append(feature_distribution(df, col))

    return pd.DataFrame(summaries)

def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Flags known dataset-specific issues per feature: 

    Impossible values: values that cannot exist(eg. negative age)

    Sentinal Values: Values from old systems that cannot actually 
    exist. 269 values for this dataset accross 3 columns. 

    Implausible value thresholds: Values that theoretically 
    could exist, but are so outliery that they cannot 
    exist in large quantities reasonably
    """
    findings = {}

    # --- impossible values ---
    findings["age_zero_or_negative"] = int((df["age"] <= 0).sum())

    negative_check_cols = ["DebtRatio", "MonthlyIncome", "RevolvingUtilizationOfUnsecuredLines"]
    # print(df[negative_check_cols].dtypes)

    for col in negative_check_cols:
        findings[f"{col}_negative"] = int((df[col] < 0).sum())
        

    # --- sentinel/encoding artifacts ---
    past_due_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for col in past_due_cols:
        findings[f"{col}_sentinel_96_98"] = int(df[col].isin([96, 98]).sum())

    # confirms that the sentinal values are all from the same rows
    mask = (
        df["NumberOfTime30-59DaysPastDueNotWorse"].isin([96, 98]) &
        df["NumberOfTime60-89DaysPastDueNotWorse"].isin([96, 98]) &
        df["NumberOfTimes90DaysLate"].isin([96, 98])
    )

    sentinal_values = df.loc[mask]
    # print(sentinal_values)
    # print(mask.sum())

    
    # --- implausible thresholds ---
    findings["revolving_util_above_2"] = int((df["RevolvingUtilizationOfUnsecuredLines"] > 2).sum())
    findings["debt_ratio_above_10"] = int((df["DebtRatio"] > 10).sum())

    print("\n==================== Data Quality Checks ====================")
    pprint(findings)

    return findings

# ---------------------------------------------------------------------
# 3. Examine Relationships with the Target
# ---------------------------------------------------------------------

def feature_vs_target(df: pd.DataFrame, col: str, target_col: str = TARGET, bins: int = 10):
    """
    Examines the relationship between a numerical feature and the target variable.

    Bins the feature into quantiles and calculates the average target rate
    within each bin to identify trends and potential predictive patterns.
    """
    feature = df[col] 
    binned_values = pd.qcut(feature, q = bins) 
    grouped = df.groupby(binned_values)[target_col].mean()

    print(grouped)
    return grouped

def rank_features_by_signal(df: pd.DataFrame, target_col: str = TARGET, method="pearson") -> pd.Series:
    """
    Takes pearson correlation(point biserial for binary output) 
    of numeric value columns with target and returns them sorted 
    with the strongest correlated variables. 
    """
    correlations = df.corr(method = method, numeric_only=True)[target_col].drop(target_col)
    sorted_correlations = correlations.sort_values(ascending=False, key = abs).round(3)

    print(sorted_correlations)
    return sorted_correlations

# ---------------------------------------------------------------------
# 4. Examine Relationships Between Predictors
# ---------------------------------------------------------------------


def correlation_matrix(df: pd.DataFrame, method="pearson"):
    """
    Computes the Pearson correlation matrix for all numeric features.
    """

    corr = df.corr(method=method, numeric_only=True)

    # testing; remove later
    print("\n==================== Correlation Matrix ====================")
    print(corr.round(2))

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()

    # plt.savefig("correlation_heatmap.png")
    plt.show()

    return corr

# ---------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------

def summarize_findings(df: pd.DataFrame, target_col: str = TARGET) -> dict:
    """
    Pulls together outputs from the functions above into 
    one structured findings dict and is the actual 
    deliverable of this module. 
    """


    pass




if __name__ == "__main__": 
    from pathlib import Path
    from data_loader import load_data
    DATA_PATH = Path("data") / "raw" / "cs-training.csv"
    df = load_data(DATA_PATH) 

    # target_balance(df, TARGET)
    # feature_distribution(df,"RevolvingUtilizationOfUnsecuredLines")
    # scan_all_features(df)
    # check_data_quality(df)
    # feature_vs_target(df, "RevolvingUtilizationOfUnsecuredLines")
    # rank_features_by_signal(df, TARGET)

    # correlation_matrix(df) 


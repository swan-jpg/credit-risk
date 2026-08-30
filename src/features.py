"""
feature.py

Feature engineering and candidate feature screening for the
credit risk model.

Takes cleaned train/test DataFrames (post-preprocessing) and
produces candidate features
"""

import pandas as pd
import numpy as np

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

CANDIDATE_FEATURES = [
    "AnyPastDue",
    "SevereDelinquency",
    "PastDueSeverity",
    "HighUtilization_75",
    "HighUtilization_90",
    "MaxedOutUtilization",
    "HighUtilization_x_AnyPastDue",
    "SevereDelinquency_x_HighUtilization",
    "Utilization_x_TotalPastDue",
    "DebtRatio_x_LogIncome",
    "LogMonthlyIncome",
]

DELINQUENCY_FEATURES = [
    "AnyPastDue",
    "SevereDelinquency",
    "PastDueSeverity",
]

UTILIZATION_FEATURES = [
    "HighUtilization_50",
    "HighUtilization_75",
    "HighUtilization_90",
    "MaxedOutUtilization",
]

INTERACTION_FEATURES = [
    "Utilization_x_TotalPastDue",
    "HighUtilization_x_AnyPastDue",
    "SevereDelinquency_x_HighUtilization",
]

def engineer_features(df):
    """
    Add candidate engineered features to a cleaned DataFrame.

    Does not mutate the input DataFrame.
    Returns a new DataFrame.
    """
    df = df.copy()

    # ============================================================
    # Original variables
    # ============================================================

    util = "RevolvingUtilizationOfUnsecuredLines"
    age = "age"
    income = "MonthlyIncome"

    late_30_59 = "NumberOfTime30-59DaysPastDueNotWorse"
    late_60_89 = "NumberOfTime60-89DaysPastDueNotWorse"
    late_90 = "NumberOfTimes90DaysLate"

    open_credit = "NumberOfOpenCreditLinesAndLoans"
    real_estate = "NumberRealEstateLoansOrLines"
    dependents = "NumberOfDependents"

    debt_ratio = "DebtRatio"


    # ============================================================
    # 1. Aggregate delinquency features
    # ============================================================

    df["TotalPastDue"] = df[late_30_59] + df[late_60_89] + df[late_90]

    df["AnyPastDue"] = (df["TotalPastDue"] > 0).astype(int)

    df["Any30Plus"] = ((df[late_30_59] > 0) | (df[late_60_89] > 0) | (df[late_90] > 0)).astype(int)

    df["Any60Plus"] = ((df[late_60_89] > 0) | (df[late_90] > 0)).astype(int)

    df["Any90Plus"] = (df[late_90] > 0).astype(int)

    df["MultipleDelinquencies"] = (df["TotalPastDue"] >= 2).astype(int)

    # ============================================================
    # 2. Delinquency severity
    # ============================================================

    df["PastDueSeverity"] = (1 * df[late_30_59]+ 2 * df[late_60_89]+ 3 * df[late_90])


    # ============================================================
    # 3. Continuous transformations
    # ============================================================

    # Income is highly right-skewed
    df["LogMonthlyIncome"] = np.log1p(
        df[income].clip(lower=0))

    # Share of credit portfolio represented by real-estate loans
    df["RealEstateLoanShare"] = (
        df[real_estate] /
        (df[open_credit] + 1))


    # ============================================================
    # 4. Utilization thresholds
    # ============================================================

    df["HighUtilization_50"] = (df[util] > 0.50).astype(int)

    df["HighUtilization_75"] = (df[util] > 0.75).astype(int)

    df["HighUtilization_90"] = (df[util] > 0.90).astype(int)

    df["MaxedOutUtilization"] = (df[util] > 1.00).astype(int)

    # ============================================================
    # 5. Delinquency thresholds
    # ============================================================

    df["MultiplePastDue_2Plus"] = (
        df["TotalPastDue"] >= 2
    ).astype(int)

    df["MultiplePastDue_3Plus"] = (
        df["TotalPastDue"] >= 3
    ).astype(int)

    df["SevereDelinquency"] = (
        df[late_90] >= 1
    ).astype(int)

    df["RepeatedSevereDelinquency"] = (
        df[late_90] >= 2
    ).astype(int)


    # ============================================================
    # 6. Age thresholds
    # ============================================================

    df["YoungBorrower"] = (
        df[age] < 30
    ).astype(int)

    df["OlderBorrower"] = (
        df[age] >= 65
    ).astype(int)


    # ============================================================
    # 7. Credit-line thresholds
    # ============================================================

    df["ManyOpenCreditLines"] = (
        df[open_credit] >= 10
    ).astype(int)

    df["HighRealEstateLoans"] = (
        df[real_estate] >= 3
    ).astype(int)


    # ============================================================
    # 8. Interaction features
    # ============================================================

    # Utilization × delinquency frequency
    df["Utilization_x_TotalPastDue"] = df[util] * df["TotalPastDue"]

    # Debt burden × income
    df["DebtRatio_x_LogIncome"] = df[debt_ratio] * df["LogMonthlyIncome"]

    # Age × utilization
    df["Age_x_Utilization"] = df[age] * df[util]

    # High utilization + previous delinquency
    df["HighUtilization_x_AnyPastDue"] = df["HighUtilization_75"] * df["AnyPastDue"]

    # Severe delinquency + high utilization
    df["SevereDelinquency_x_HighUtilization"] = df["SevereDelinquency"] * df["HighUtilization_75"]

    # ============================================================
    # 9. Binned features
    # ============================================================

    df["AgeBin"] = pd.cut(
        df[age],
        bins=[0, 30, 40, 50, 60, 70, np.inf],
        labels=False,
        include_lowest=True
    )

    df["UtilizationBin"] = pd.cut(
        df[util],
        bins=[-np.inf, 0.10, 0.30, 0.50, 0.75, 1.00, np.inf],
        labels=False
    )

    df["DebtRatioBin"] = pd.cut(
        df[debt_ratio],
        bins=[-np.inf, 0.10, 0.30, 0.50, 1.00, 2.00, np.inf],
        labels=False
    )

    # Estimated dollar debt burden (DebtRatio * MonthlyIncome)
    df["EstimatedDebt"] = df[debt_ratio] * df[income]

    # Income available per household dependent
    df["IncomePerDependent"] = df[income] / (df[dependents] + 1)

    return df


def create_train_test_features(train_df, test_df):
    """
    Apply feature engineering consistently to train and test data.
    """

    train_features = engineer_features(train_df)
    test_features = engineer_features(test_df)

    return train_features, test_features

if __name__ == "__main__":
    from pathlib import Path
    from data_loader import load_data
    DATA_PATH = Path("data") / "raw" / "cs-training.csv"
    df = load_data(DATA_PATH) 

    # ============================================================
    # Temporary Analysis
    # ============================================================

    engineer_features(df)
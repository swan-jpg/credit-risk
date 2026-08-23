"""
feature.py

Feature engineering and candidate feature screening for the
credit risk model.

Takes cleaned train/test DataFrames (post-preprocessing) and
produces candidate features
"""

import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Add candidate engineered features to a cleaned DataFrame.

    Does not mutate the input DataFrame.
    Returns a new DataFrame.
    """
    df = df.copy()

    # ============================================================
    # 1. Combine / aggregate existing variables
    # ============================================================

    # Candidate aggregate features here.
    #
    # Example:
    df["TotalPastDue"] = df[
        [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate"
        ]
    ].sum(axis=1)


    # ============================================================
    # 2. Ratio / continuous transformations
    # ============================================================

    # Add candidate transformations here.
    #
    # Example:
    # df["Some_Log_Feature"] = np.log1p(df["Some_Feature"])


    # ============================================================
    # 3. Interaction features
    # ============================================================

    # Add candidate interactions here.
    #
    # Example:
    # df["Age_Income"] = df["age"] * df["MonthlyIncome"]


    # ============================================================
    # 4. Threshold / indicator features
    # ============================================================

    # Add candidate binary indicators here.
    #
    # Example:
    # df["High_Utilization"] = (
    #     df["RevolvingUtilizationOfUnsecuredLines"] > 1
    # ).astype(int)


    # ============================================================
    # 5. Binned features
    # ============================================================

    # Add candidate bins here when EDA suggests a nonlinear
    # relationship with default risk.
    #
    # Example:
    # df["age_bin"] = pd.cut(
    #     df["age"],
    #     bins=[0, 30, 40, 50, 60, 100],
    #     labels=False
    # )


    return df


def create_train_test_features(train_df, test_df):
    """
    Apply feature engineering consistently to train and test data.

    Returns:
        train_features: engineered training DataFrame
        test_features: engineered test DataFrame
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
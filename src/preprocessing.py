'''
preprocessing.py

Preprocessing functions for Give Me Some 
Credit Dataset from Kaggle.

Decions are based off EDA.py's findings:
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "SeriousDlqin2yrs"

def split_data(df, test_size=0.2, random_state=67):
    '''
    Splits dataset into training and testing data.
    '''

    train_df, test_df = train_test_split(df, test_size=0.2, 
                                         stratify=df[TARGET],
                                         random_state=random_state)

    return train_df, test_df

def remove_invalid_ages(df):
    '''
    Removes any invalid ages(less than 0) 
    from the dataframe
    ''' 
    return df[df["age"] > 0].copy()

def handle_sentinel_values(df, cols=None):
    '''
    Flags and drops sentinel values (96/98 codes) found
    from EDA (269 rows).
    '''
    if cols is None:
        cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate"
        ]
    mask = df[cols].isin([96, 98])
    df["has_sentinel"] = mask.any(axis=1)
    df = df[df["has_sentinel"] == 0].copy()
    return df

def handle_debt_ratio_outlier(train_df, test_df):
    '''
    Handle extreme outliers in DebtRatio column

    There are enormous outliers in the "DebtRatio" column that
    are primarily caused by ~0 monthly income values. 

    Creates a "ZeroIncome" column in train_df and test_df that 
    whether or not MonthlyIncome is equal 0. 
    For the values that have zero income, we impute the debt 
    ratio value using median impute on valid train_df values 
    and then apply to both train_df and test_df
    '''
    train_df["ZeroIncome"] = (train_df["MonthlyIncome"] <= 0).astype(int)
    test_df["ZeroIncome"] = (test_df["MonthlyIncome"] <= 0).astype(int)

    train_mask = train_df["MonthlyIncome"] <= 0
    test_mask = test_df["MonthlyIncome"] <= 0

    train_df.loc[train_mask, "DebtRatio"] = np.nan
    test_df.loc[test_mask, "DebtRatio"] = np.nan 

    valid_train = train_df["MonthlyIncome"] > 0
    debt_ratio_impute = train_df.loc[valid_train, "DebtRatio"].median()

    train_df["DebtRatio"] = train_df["DebtRatio"].fillna(debt_ratio_impute)
    test_df["DebtRatio"] = test_df["DebtRatio"].fillna(debt_ratio_impute)

    return train_df, test_df

def impute_missing(train_df, test_df, col, strategy="median"):
    pass

if __name__ == "__main__":
    from pathlib import Path
    from data_loader import load_data

    DATA_PATH = Path("data") / "raw" / "cs-training.csv"
    df = load_data(DATA_PATH)

    # ---------------------------------------------------------
    # 1. Split Data
    # ---------------------------------------------------------
    train_df, test_df = split_data(df)
    print("TEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARY")
    print(
    train_df.loc[
        train_df["DebtRatio"] > 100,
        ["DebtRatio", "MonthlyIncome"]
    ].describe())
    print("TEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARY")
    print(
    train_df.loc[
        train_df["DebtRatio"] > 100,
        ["DebtRatio", "MonthlyIncome"]
    ].head(20))
    print("TEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARY")
    print(train_df.loc[
        (train_df["DebtRatio"] > 100) &
        (train_df["MonthlyIncome"].isna()),
        ["DebtRatio", "MonthlyIncome"]
    ].describe())
    print("TEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARY")
    print(train_df.loc[
        (train_df["DebtRatio"] > 100) &
        (train_df["MonthlyIncome"] > 0),
        ["DebtRatio", "MonthlyIncome"]
    ].describe())
    print("TEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARY")
    print(train_df.loc[
        (train_df["DebtRatio"] > 100) &
        (train_df["MonthlyIncome"] > 0),
        "MonthlyIncome"
    ].value_counts().sort_index())
    print("TEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARY")
    print(train_df.loc[
        (train_df["DebtRatio"] > 100) &
        (train_df["MonthlyIncome"] > 1),
        ["DebtRatio", "MonthlyIncome"]
    ].sort_values("DebtRatio", ascending=False))
    print("TEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARYTEMPORARY")
    # ---------------------------------------------------------
    # 2. Remove Invalid Ages
    # ---------------------------------------------------------
    train_df = remove_invalid_ages(train_df)
    test_df = remove_invalid_ages(test_df)

    # ---------------------------------------------------------
    # 3. Remove Sentinel Values
    # ---------------------------------------------------------
    train_df = handle_sentinel_values(train_df)
    test_df = handle_sentinel_values(test_df)

    # ---------------------------------------------------------
    # 4. Inspect DebtRatio before handling
    # ---------------------------------------------------------
    print("\nDEBT RATIO BEFORE HANDLING")

    print(
        train_df["DebtRatio"].quantile(
            [0.90, 0.95, 0.99, 0.995, 0.999]
        )
    )

    print("Train max:", train_df["DebtRatio"].max())
    print("Test max:", test_df["DebtRatio"].max())

    # ---------------------------------------------------------
    # 5. Apply DebtRatio outlier function
    # ---------------------------------------------------------
    train_df, test_df = handle_debt_ratio_outlier(
        train_df,
        test_df
    )

    # ---------------------------------------------------------
    # 4. Inspect DebtRatio after handling
    # ---------------------------------------------------------
    print("\nDEBT RATIO AFTER HANDLING")

    print(
        train_df["DebtRatio"].quantile(
            [0.90, 0.95, 0.99, 0.995, 0.999]
        )
    )

    print("Train max:", train_df["DebtRatio"].max())
    print("Test max:", test_df["DebtRatio"].max())

    # ---------------------------------------------------------
    # 5. Verify zero-income rows were imputed
    # ---------------------------------------------------------
    train_zero_income = train_df["ZeroIncome"] == 1
    test_zero_income = test_df["ZeroIncome"] == 1

    print("\nZERO-INCOME DEBT RATIO VALUES")

    print(
        "Train:",
        train_df.loc[
            train_zero_income,
            "DebtRatio"
        ].unique()
    )

    print(
        "Test:",
        test_df.loc[
            test_zero_income,
            "DebtRatio"
        ].unique()
    )

    # ---------------------------------------------------------
    # 6. Check for remaining missing values
    # ---------------------------------------------------------
    print("\nMISSING DEBT RATIO VALUES")

    print(
        "Train:",
        train_df["DebtRatio"].isna().sum()
    )

    print(
        "Test:",
        test_df["DebtRatio"].isna().sum()
    )

    # ---------------------------------------------------------
    # 7. Inspect remaining maximum
    # ---------------------------------------------------------
    print("\nREMAINING MAXIMUM DEBT RATIO")

    print(
        train_df.loc[
            train_df["DebtRatio"] == train_df["DebtRatio"].max(),
            ["MonthlyIncome", "DebtRatio"]
        ]
    )

    
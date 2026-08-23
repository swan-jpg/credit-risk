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

    train_df, test_df = train_test_split(df, test_size=test_size, 
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

    Creates a "LowIncome" column in train_df and test_df that 
    whether or not MonthlyIncome is 0, 1, or NA. 
    For the values in LowIncome, we impute the debt 
    ratio value using median impute on valid train_df values 
    and then apply to both train_df and test_df
    '''
    train_df["LowIncome"] = ((train_df["MonthlyIncome"] <= 1) | train_df["MonthlyIncome"].isna()).astype(int)
    test_df["LowIncome"] = ((test_df["MonthlyIncome"] <= 1) | test_df["MonthlyIncome"].isna()).astype(int)

    train_mask = (train_df["MonthlyIncome"] <= 1) | (train_df["MonthlyIncome"].isna())
    test_mask = (test_df["MonthlyIncome"] <= 1) | (test_df["MonthlyIncome"].isna())

    train_df.loc[train_mask, "DebtRatio"] = np.nan
    test_df.loc[test_mask, "DebtRatio"] = np.nan 

    valid_train = train_df["MonthlyIncome"] > 1
    debt_ratio_impute = train_df.loc[valid_train, "DebtRatio"].median()

    train_df["DebtRatio"] = train_df["DebtRatio"].fillna(debt_ratio_impute)
    test_df["DebtRatio"] = test_df["DebtRatio"].fillna(debt_ratio_impute)

    # Cap extreme remaining values
    train_df["DebtRatio"] = train_df["DebtRatio"].clip(upper=10)
    test_df["DebtRatio"] = test_df["DebtRatio"].clip(upper=10)

    return train_df, test_df

def handle_revolving_utilization_outlier(train_df, test_df):
    '''
    Handle extreme outliers in RevolvingUtilizationOfUnsecuredLines.

    Revolving utilization contains data with a highly skewed upper tail
    with extreme values reaching 50,708(which is obviously impossible)
    Values will be capped at 1.5 in order to preserve influence of higher valued 
    datapoints while also limiting the influence of extreme outliers.
    '''
    train_df["RevolvingUtilizationOfUnsecuredLines"] = train_df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=1.5)
    test_df["RevolvingUtilizationOfUnsecuredLines"] = test_df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=1.5)

    return train_df, test_df

def impute_missing(train_df, test_df, col):
    '''
    Imputes missing values using median imputation.

    Takes the median value of the training dataset and 
    applies that to NA values in the training and 
    the testing dataset. 
        
    '''
    median_value = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_value)
    test_df[col] = test_df[col].fillna(median_value)

    return train_df, test_df

if __name__ == "__main__":
    from pathlib import Path
    from data_loader import load_data

    DATA_PATH = Path("data") / "raw" / "cs-training.csv"
    df = load_data(DATA_PATH)

    # ---------------------------------------------------------
    # 1. Split Data
    # ---------------------------------------------------------
    train_df, test_df = split_data(df)

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
    # 6. Inspect DebtRatio after handling
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
    # 7. Verify low-income rows were imputed
    # ---------------------------------------------------------
    train_low_income = train_df["LowIncome"] == 1
    test_low_income = test_df["LowIncome"] == 1

    print("\nLOW-INCOME DEBT RATIO VALUES")

    print(
        "Train:",
        train_df.loc[
            train_low_income,
            "DebtRatio"
        ].unique()
    )

    print(
        "Test:",
        test_df.loc[
            test_low_income,
            "DebtRatio"
        ].unique()
    )

    # ---------------------------------------------------------
    # 8. Check for remaining missing values
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
    # 9. RevolvingUtilization before handling
    # ---------------------------------------------------------
    print("\nREVOLVING UTILIZATION BEFORE HANDLING")

    print(
        train_df["RevolvingUtilizationOfUnsecuredLines"].quantile(
            [0.90, 0.95, 0.99, 0.995, 0.999]
        )
    )

    print(
        "Train max:",
        train_df["RevolvingUtilizationOfUnsecuredLines"].max()
    )

    print(
        "Test max:",
        test_df["RevolvingUtilizationOfUnsecuredLines"].max()
    )

    # ---------------------------------------------------------
    # 10. Apply RevolvingUtilization outlier function
    # ---------------------------------------------------------
    train_df, test_df = handle_revolving_utilization_outlier(
        train_df,
        test_df
    )

    # ---------------------------------------------------------
    # 11. RevolvingUtilization after handling
    # ---------------------------------------------------------
    print("\nREVOLVING UTILIZATION AFTER HANDLING")

    print(
        train_df["RevolvingUtilizationOfUnsecuredLines"].quantile(
            [0.90, 0.95, 0.99, 0.995, 0.999]
        )
    )

    print(
        "Train max:",
        train_df["RevolvingUtilizationOfUnsecuredLines"].max()
    )

    print(
        "Test max:",
        test_df["RevolvingUtilizationOfUnsecuredLines"].max()
    )
    # ---------------------------------------------------------
    # 12. Impute missing values in MonthlyIncome and NumberOfDependents
    # ---------------------------------------------------------
    print("\nIMPUTE NA VALUES IN MONTHLYINCOME AND NUMBEROFDEPENDENTS")
    train_df, test_df = impute_missing(train_df=train_df, 
                                       test_df=test_df, 
                                       col="MonthlyIncome")
    train_df, test_df = impute_missing(train_df=train_df, 
                                       test_df=test_df, 
                                       col="NumberOfDependents")

    # ---------------------------------------------------------
    # 13. Verify missing values were imputed correctly
    # ---------------------------------------------------------
    print("\nVERIFY MISSING VALUES WERE IMPUTED:")
    print("\nMISSING VALUES AFTER IMPUTATION")

    print("Train:")
    print(
        train_df[["MonthlyIncome", "NumberOfDependents"]].isna().sum()
    )

    print("Test:")
    print(
        test_df[["MonthlyIncome", "NumberOfDependents"]].isna().sum()
    )

    print("MonthlyIncome median:", train_df["MonthlyIncome"].median())
    print("NumberOfDependents median:", train_df["NumberOfDependents"].median())
    print("MonthlyIncome mean:", train_df["MonthlyIncome"].mean())
    print("NumberOfDependents mean:", train_df["NumberOfDependents"].mean())

import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df 

def data_summary(df): 
    print("\n==================== Shape ====================")
    print(f"rows: {df.shape[0]}, columns: {df.shape[1]}")

    print("\n==================== Sample Rows ====================")
    print(df.head(3)) 

    summary = pd.DataFrame({"dtype": df.dtypes,
                            "n_unique": df.nunique(), 
                            "missing_count": df.isnull().sum(),
                            "missing_percent": ((df.isnull().sum() / len(df)) * 100).round(2)
                            }) 
    
    print("\n==================== Summary ====================")
    print(summary) 
    return summary   
                           
def missing_values(df): 
    missing = pd.DataFrame({"missing_value_count": df.isnull().sum(),
                            "missing_value_percent": ((df.isnull().sum() / len(df))*100).round(2)
    })

    missing_mask = missing["missing_value_percent"] > 0
    filtered_missing = missing[missing_mask]
    print("\n==================== Missing Values ====================")
    print(filtered_missing["missing_value_percent"])
    return filtered_missing
    
def summary_statistics(df):
    stats = df.describe().T
    stats["skew"] = df.skew(numeric_only=True).round(5)
    stats["kurtosis"] = df.kurtosis(numeric_only=True).round(5)


    print("\n==================== Summary Statistics ====================")
    print(stats)
    return stats

def duplicates(df): 
    duplicate = (df.duplicated())
    # print(duplicate.head(5))

    duplicate_count = duplicate.sum()
    print("\n==================== Duplicate Amount ====================")
    print(f"Duplicate Rows: {duplicate_count}")
    return duplicate_count


def outlier_check(df):
    checks = {
        "age_zero_or_negative": (df["age"] <= 10).sum(),
        "revolving_util_above_2": (df["RevolvingUtilizationOfUnsecuredLines"] > 2).sum(),
        "debt_ratio_above_10": (df["DebtRatio"] > 10).sum(),
        "past_due_96_98_sentinel": (
            (df["NumberOfTime30-59DaysPastDueNotWorse"] >= 96) |
            (df["NumberOfTime60-89DaysPastDueNotWorse"] >= 96) |
            (df["NumberOfTimes90DaysLate"] >= 96)
        ).sum(),
    }

    result = pd.Series(checks, name="flagged_count").to_frame()
    print("\n==================== Outlier / Sanity Checks ====================")
    print(result)

    return result

    
if __name__  == "__main__":
    from pathlib import Path 
    DATA_PATH = Path("data") / "raw" / "cs-training.csv"
    df = load_data(DATA_PATH) 

    # data_summary(df)
    # summary_statistics(df)
    # duplicates(df)
    # missing_values(df)
    outlier_check(df)
   

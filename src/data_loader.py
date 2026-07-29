import pandas as pd

def load_data(path, index_column=0):
    df = pd.read_csv(path, index_col=index_column, na_values=["NA"]) 

    df.columns = df.columns.str.strip()

    df["MonthlyIncome"] = pd.to_numeric(
        df["MonthlyIncome"].str.strip(),
        errors="coerce"
    )

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
    
if __name__  == "__main__":
    from pathlib import Path 
    DATA_PATH = Path("data") / "raw" / "cs-training.csv"
    df = load_data(DATA_PATH) 

    # print(df.columns)

    # data_summary(df)
    # summary_statistics(df)
    # duplicates(df)
    # missing_values(df)


   

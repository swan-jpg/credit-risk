"""
validation.py

Model validation utilities for the credit risk model:
discrimination, calibration, and stability diagnostics.
"""

import numpy as np
import pandas as pd
from scipy import stats

def calculate_gini(auc):
    """
    Gini coefficient from ROC-AUC.
    Gini = 2*AUC - 1
    """
    return round(2 * auc - 1, 5)

# ============================================================
# Shared decile binning
# ============================================================

def bin_by_decile(scores, n_bins=10):
    """
    Assign each score to a bin (0 = lowest risk decile,
    n_bins - 1 = highest, 10 bins total).

    Used for the binning scheme for both calibration
    (Hosmer-Lemeshow, calibration plot) so that both diagnostics
    are computed on identical groups.

    Returns
    -------
    pd.Series of integer bin labels
    """
    scores = pd.Series(scores)
    return pd.qcut(scores,q=n_bins,labels = False,duplicates="drop")

# ============================================================
# Calibration: Hosmer-Lemeshow test
# ============================================================

def hosmer_lemeshow_test(y_true, y_prob, n_bins=10):
    """
    Hosmer-Lemeshow goodness-of-fit test for calibration.

    Groups observations into deciles of predicted probability
    (via bin_by_decile) and compares observed vs. expected event
    counts in each group using a chi-square statistic.

    A low p-value (e.g. < 0.05) suggests the model's predicted
    probabilities are miscalibrated relative to observed outcomes.

    Returns
    -------
    dict with 'chi_square', 'p_value', 'degrees_of_freedom',
    and 'table' (per-decile observed/expected breakdown).
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_prob = pd.Series(y_prob).reset_index(drop=True)

    bins = bin_by_decile(y_prob, n_bins=n_bins)

    df = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob,
        "bin": bins
    })

    grouped = df.groupby("bin", observed=True).agg(
        total=("y_true", "count"),
        observed_events=("y_true", "sum"),
        expected_events=("y_prob", "sum")
    )

    grouped["observed_nonevents"] = grouped["total"] - grouped["observed_events"]
    grouped["expected_nonevents"] = grouped["total"] - grouped["expected_events"]

    chi_square = (
        ((grouped["observed_events"] - grouped["expected_events"]) ** 2
         / grouped["expected_events"])
        + ((grouped["observed_nonevents"] - grouped["expected_nonevents"]) ** 2
           / grouped["expected_nonevents"])
    ).sum()

    degrees_of_freedom = len(grouped) - 2
    p_value = 1 - stats.chi2.cdf(chi_square, degrees_of_freedom)

    return {
        "chi_square": round(float(chi_square), 4),
        "p_value": round(float(p_value), 4),
        "degrees_of_freedom": int(degrees_of_freedom),
        "table": grouped.reset_index()
    }


# ============================================================
# Calibration: decile table for plotting
# ============================================================

def calibration_table(y_true, y_prob, n_bins=10):
    """
    Build a decile table of mean predicted probability vs.
    observed default rate, using the same binning as
    hosmer_lemeshow_test so the plot and test agree on groups.

    Returns
    -------
    pd.DataFrame with columns: bin, mean_predicted, observed_rate, total
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_prob = pd.Series(y_prob).reset_index(drop=True)

    bins = bin_by_decile(y_prob, n_bins=n_bins)

    df = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob,
        "bin": bins
    })

    table = df.groupby("bin", observed=True).agg(
        mean_predicted=("y_prob", "mean"),
        observed_rate=("y_true", "mean"),
        total=("y_true", "count"),
    ).reset_index()

    return table


# ============================================================
# Stability: Population Stability Index (PSI)
# ============================================================

def calculate_psi(expected, actual, n_bins=10, epsilon=1e-6):
    """
    Calculate Population Stability Index between two distributions
    of the same variable (e.g. training scores vs. test/recent scores,
    or training feature values vs. test/recent feature values).

    Bin edges are defined by the quantiles of `expected` and reused
    for `actual`, so both distributions are compared on identical
    bins.

    Interpretation (standard industry thresholds):
        PSI < 0.10            no significant shift
        0.10 <= PSI < 0.25     moderate shift, investigate
        PSI >= 0.25            significant shift, model may need review

    Returns
    -------
    dict with 'psi' (float) and 'table' (per-bin breakdown).
    """
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    bin_edges = np.quantile(
        expected,
        np.linspace(0, 1, n_bins + 1)
    )
    bin_edges = np.unique(bin_edges)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_bins = pd.cut(expected, bins=bin_edges)
    actual_bins = pd.cut(actual, bins=bin_edges)

    expected_dist = (
        expected_bins.value_counts(normalize=True, sort=False)
        .clip(lower=epsilon)
    )
    actual_dist = (
        actual_bins.value_counts(normalize=True, sort=False)
        .clip(lower=epsilon)
    )

    psi_table = pd.DataFrame({
        "expected_pct": expected_dist,
        "actual_pct": actual_dist,
    })

    psi_table["psi"] = (
        (psi_table["actual_pct"] - psi_table["expected_pct"])
        * np.log(psi_table["actual_pct"] / psi_table["expected_pct"])
    )

    total_psi = psi_table["psi"].sum()

    return {
        "psi": round(float(total_psi), 5),
        "table": psi_table.reset_index().rename(columns={"index": "bin"}),
    }


if __name__ == "__main__":
    pass






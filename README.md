# Credit Risk Model: Development & Validation

A logistic regression credit risk model built on the Give Me Some Credit dataset,
with focus on model validation rather than performance maximization.

## Why Logistic Regression

This project prioritizes interpretability and validation defensibility over
raw predictive performance. Logistic regression allows every modeling decision
(feature selection/engineering, coefficient signs, calibration) to be explained
and defended, which is the core skill here rather than squeezing out marginal 
AUC gains with an exotic model.

## Project Structure

```text
credit-risk-model/
├── data/
│   ├── processed/          # Cleaned and transformed data ready for modeling
│   └── raw/
│       └── cs-training.csv # Original training dataset
├── notebooks/
│   ├── 01_modeling.ipynb   # Model development, tuning, and experimentation
│   └── 02_validation.ipynb # Validation checks and performance evaluation
├── outputs/
│   └── figures/
│       ├── calibration_plot.png
│       └── correlation_matrix.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Loads data
│   ├── EDA.py              # Exploratory Data Analysis scripts
│   ├── features.py         # Feature engineering
│   ├── model.py            # Model training and prediction 
│   ├── preprocessing.py    # Data cleaning 
│   └── validation.py       # Evaluation metrics
├── .gitignore
├── README.md
└── requirements.txt        # Project dependencies
```

## Model Development

- **Preprocessing**: sentinel row handling, invalid-age removal, DebtRatio
  outlier treatment (flag-plus-null), utilization capping, median imputation
- **Feature engineering**: 12 candidate features screened individually against
  baseline; only `AnyPastDue` survived on statistical significance + VIF +
  interpretability grounds
- **Final model**: baseline features + `AnyPastDue`

## Key Results

| Metric | Train | Test |
|---|---|---|
| ROC-AUC | 0.855 | 0.857 |
| KS | 0.552 | 0.561 |
| Gini | — | 0.714 |

AIC improved by ~995 points over baseline with the addition of a single
engineered feature. All VIFs under 2.4.

## Validation Summary

Independent validation (`02_validation.ipynb`) reproduces the final model from
raw data and assesses:

- **Discrimination**: strong (AUC 0.857, KS 0.561), negligible train/test gap
- **Calibration**: Hosmer-Lemeshow test rejects perfect calibration (p = 0.0001),
  but the practical miscalibration is small (~1–2pp), consistent with the
  test's high power at this sample size
- **Stability**: PSI near zero across score and key features (genuine time-based stability test would require historical data)

**Recommendation: Approve with Conditions.** Suitable for rank-ordering
decisions (accept/reject, risk tiering) as-is; recalibration recommended
before use in applications that need exact probability values (pricing,
reserving).

## Out of Scope

- Benchmarking against a challenger model(exploring models beyond logistic regression)
- Unit tests for `validation.py`
- Time-based stability testing (requires historical vintages, unavailable
  with this static dataset)

## Running the Project

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_modeling.ipynb
```
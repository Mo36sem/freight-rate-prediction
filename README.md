# Freight Rate Prediction

Machine Learning Engineer take-home assessment for predicting freight rates from structured load and market features.

The solution uses chronological validation, feature engineering, and gradient-boosted trees to predict `posted_rate` for unseen freight loads.

## Project Overview

The labeled development dataset contains freight-load information including:

- Origin and destination
- Geographic coordinates
- Distance
- Equipment type
- Load weight
- Market indicators
- Quote signals
- Date
- Posted freight rate

The goal is to train a regression model and generate predictions for the 12,000 unseen loads in `validation.csv`.

A second fixed December 2025 scenario is also predicted and visualized using the provided scoring script.

## Validation Strategy

Because the task represents future freight-rate prediction, a chronological holdout was used instead of a shuffled random split.

- **Training:** January 1, 2025 – August 31, 2025
- **Validation:** September 1, 2025 – October 31, 2025
- **Training rows:** 38,477
- **Validation rows:** 9,523

This setup better represents the real inference setting, where historical observations are used to predict later freight loads.

## Data Quality and Preprocessing

The main preprocessing steps include:

- Correcting invalid negative weight values
- Imputing missing weight using equipment-specific training medians
- Imputing missing `market_index` using month-based training medians
- Using global training medians as fallback values
- Encoding categorical variables with one-hot encoding
- Deriving temporal features from the load date

During model validation, all learned preprocessing statistics are calculated from the training portion only to prevent data leakage.

## Feature Engineering

Several domain-inspired features were evaluated:

- `estimated_base_cost`
- `market_adjusted_cost`
- `weight_distance_k`
- `adjusted_rpm`
- `haversine_distance`
- `route_circuity`
- `month`
- `day_of_week`
- `quarter`
- `is_weekend`

Feature groups were evaluated using chronological validation performance rather than being selected solely from correlation statistics.

## Model Experiments

The following regression models were evaluated sequentially:

1. Dummy Regressor
2. Linear Regression
3. Ridge Regression
4. Decision Tree Regressor
5. Random Forest Regressor
6. XGBoost Regressor

A log transformation of the target was also evaluated because `posted_rate` showed substantial positive skew.

The strongest overall validation performance was achieved using XGBoost with engineered features and a log-transformed target.

## Final Model

The final model is an `XGBRegressor` trained on:

```text
log1p(posted_rate)
```

Predictions are converted back to the original dollar scale using:

```text
expm1(prediction)
```

Final hyperparameters:

```text
n_estimators       = 600
learning_rate      = 0.025
max_depth          = 4
min_child_weight   = 7
subsample          = 0.75
colsample_bytree   = 0.75
reg_alpha          = 0.1
reg_lambda         = 1.5
random_state       = 42
```

### Internal Chronological Validation

| Metric | Training | Validation |
|---|---:|---:|
| MAE | $92.21 | $121.52 |
| RMSE | $567.39 | $637.75 |
| MAPE | 4.15% | 5.32% |
| R² | 0.8523 | 0.8254 |

These are **internal holdout metrics** from the chronological September–October validation period.

The final hidden-validation metrics are calculated by the assessment provider after submission.

## Final Training

After model selection and tuning, the final workflow is retrained using the full labeled development dataset from January through October 2025.

The trained workflow is then used to predict every row in `validation.csv`.

The resulting file is:

```text
validation_predictions.csv
```

with exactly:

```text
load_id,predicted_rate
```

for all 12,000 required loads.

## December 2025 Scenario

The provided December scenario contains 31 daily observations with fixed freight characteristics:

- **Pickup:** Lexington
- **Delivery:** Fort Wayne
- **Distance:** 360 miles
- **Equipment:** Dry Van
- **Weight:** 32,000 lb

Only the date changes.

The December input does not provide `market_index`, `quote_signal`, or geographic coordinates required by the final model.

To use the same final model rather than training a separate December model:

- Route coordinates are derived from the labeled development data
- Missing market features are estimated from recent historical data for the same route and equipment type
- The five most recent Lexington → Fort Wayne Dry Van observations are used for the market-feature fallback

The resulting fallback values are:

```text
market_index = 0.90458
quote_signal = 1.87336
```

No future December market values are used.

The resulting December predictions are approximately:

```text
Minimum: $837.83
Maximum: $840.43
Mean:    $839.10
```

Because the load characteristics are fixed, only date-derived features produce small day-to-day variation.

## Data Availability

The raw development and hidden-validation datasets are intentionally excluded from this public repository to avoid redistributing assessment data.

To reproduce the full workflow, place the assessment-provided files in the `data/` directory using their original filenames:

```text
data/
├── train-test.csv
├── validation.csv
├── validation_predictions_template.csv
└── december-chart-inputs.csv
```

The repository includes the completed `december-chart-inputs.csv` containing the final December predictions used by the scoring script.

Additional information is available in `data/README.md`.

## Repository Structure

```text
freight-rate-prediction/
│
├── .gitignore
├── README.md
├── requirements.txt
├── score.py
├── validation_predictions.csv
│
├── data/
│   ├── README.md
│   └── december-chart-inputs.csv
│
├── notebooks/
│   └── freight_rate_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── scorer_results/
│   └── candidate_december.png
│
└── report/
    └── freight_rate_prediction_report.docx
```

## Installation

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Main dependencies include:

- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn
- XGBoost
- joblib

## Reproducing the Workflow

Before running the training and prediction scripts, place the assessment-provided raw datasets inside the `data/` directory.

Train the final model:

```bash
python src/train.py
```

Generate validation and December predictions:

```bash
python src/predict.py
```

Run the provided scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

A successful run validates:

- All 12,000 validation predictions
- All 31 December predictions

and generates:

```text
scorer_results/candidate_december.png
```

## Notebook

The complete exploration and model-development process is available in:

```text
notebooks/freight_rate_modeling.ipynb
```

It includes:

- Exploratory data analysis
- Data-quality checks
- Feature engineering
- Chronological validation
- Model comparison
- Feature-set experiments
- Target-transformation experiments
- XGBoost tuning
- Final training
- Validation predictions
- December predictions

## Output Files

The main submission outputs are:

```text
validation_predictions.csv
data/december-chart-inputs.csv
scorer_results/candidate_december.png
report/freight_rate_prediction_report.docx
```

## Notes

- Model selection was based only on the internal chronological validation period.
- Hidden validation labels were never used during model development.
- Further manual tuning was intentionally stopped once improvements became marginal to reduce the risk of overfitting model selection to the holdout period.
- December market features are documented estimates derived only from historical development data.

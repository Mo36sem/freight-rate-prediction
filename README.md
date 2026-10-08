\# Freight Rate Prediction



Machine Learning Engineer take-home assessment for predicting freight rates from structured load and market features.



The solution uses chronological validation, feature engineering, and gradient-boosted trees to predict `posted\_rate` for unseen freight loads.



\## Project Overview



The labeled development dataset contains freight-load information including:



\- origin and destination

\- geographic coordinates

\- distance

\- equipment type

\- load weight

\- market indicators

\- quote signals

\- date

\- posted freight rate



The goal is to train a regression model and generate predictions for the 12,000 unseen loads in `validation.csv`.



A second fixed December 2025 scenario is also predicted and visualized using the provided scorer.



\---



\## Validation Strategy



Because the task represents future freight-rate prediction, a chronological holdout was used instead of a shuffled random split.



\- \*\*Training:\*\* January 1, 2025 – August 31, 2025

\- \*\*Validation:\*\* September 1, 2025 – October 31, 2025

\- \*\*Training rows:\*\* 38,477

\- \*\*Validation rows:\*\* 9,523



This setup better represents the real inference setting where historical observations are used to predict later freight loads.



\---



\## Data Quality and Preprocessing



The main preprocessing steps include:



\- correcting invalid negative weight values

\- imputing missing weight using equipment-specific training medians

\- imputing missing `market\_index` using month-based training medians

\- using global training medians as fallback values

\- encoding categorical variables with one-hot encoding

\- deriving temporal features from the load date



All learned preprocessing statistics are calculated from the training data only during model validation to prevent data leakage.



\---



\## Feature Engineering



Several domain-inspired features were evaluated:



\- `estimated\_base\_cost`

\- `market\_adjusted\_cost`

\- `weight\_distance\_k`

\- `adjusted\_rpm`

\- `haversine\_distance`

\- `route\_circuity`

\- `month`

\- `day\_of\_week`

\- `quarter`

\- `is\_weekend`



Feature groups were evaluated using validation performance rather than being selected solely from correlation statistics.



\---



\## Model Experiments



The following regression models were evaluated sequentially:



1\. Dummy Regressor

2\. Linear Regression

3\. Ridge Regression

4\. Decision Tree Regressor

5\. Random Forest Regressor

6\. XGBoost Regressor



A log transformation of the target was also evaluated because `posted\_rate` showed substantial positive skew.



The strongest overall validation performance was achieved using XGBoost with engineered features and a log-transformed target.



\---



\## Final Model



The final model is an `XGBRegressor` trained on:



```text

log1p(posted\_rate)

```



Predictions are converted back to the original dollar scale using:



```text

expm1(prediction)

```



Final hyperparameters:



```text

n\_estimators       = 600

learning\_rate      = 0.025

max\_depth          = 4

min\_child\_weight   = 7

subsample          = 0.75

colsample\_bytree   = 0.75

reg\_alpha          = 0.1

reg\_lambda         = 1.5

random\_state       = 42

```



\### Internal Chronological Validation



| Metric | Training | Validation |

|---|---:|---:|

| MAE | $92.21 | $121.52 |

| RMSE | $567.39 | $637.75 |

| MAPE | 4.15% | 5.32% |

| R² | 0.8523 | 0.8254 |



These are \*\*internal holdout metrics\*\* from the chronological September–October validation period.



The final hidden validation metrics are calculated by the assessment provider after submission.



\---



\## Final Training



After model selection and tuning, the final workflow is retrained using the full labeled development dataset from January through October 2025.



The trained workflow is then used to predict every row in `validation.csv`.



The resulting file is:



```text

validation\_predictions.csv

```



with exactly:



```text

load\_id,predicted\_rate

```



for all 12,000 required loads.



\---



\## December 2025 Scenario



The provided December scenario contains 31 daily observations with fixed freight characteristics:



\- \*\*Pickup:\*\* Lexington

\- \*\*Delivery:\*\* Fort Wayne

\- \*\*Distance:\*\* 360 miles

\- \*\*Equipment:\*\* Dry Van

\- \*\*Weight:\*\* 32,000 lb



Only the date changes.



The December input does not provide `market\_index`, `quote\_signal`, or geographic coordinates required by the final model.



To keep the same final model rather than training a separate December model:



\- route coordinates are derived from the labeled development data

\- missing market features are estimated from recent training history for the same route and equipment type

\- the five most recent Lexington → Fort Wayne Dry Van observations are used for the market-feature fallback



The resulting fallback values are:



```text

market\_index = 0.90458

quote\_signal = 1.87336

```



No future December market values are used.



The resulting December predictions are approximately:



```text

Minimum: $837.83

Maximum: $840.43

Mean:    $839.10

```



Because the load characteristics are fixed, only date-derived features produce small day-to-day variation.



\---



\## Repository Structure



```text

freight-rate-prediction/

│

├── README.md

├── requirements.txt

├── score.py

├── validation\_predictions.csv

│

├── data/

│   ├── train\_test.csv

│   ├── validation.csv

│   ├── validation\_predictions\_template.csv

│   └── december\_chart\_inputs.csv

│

├── notebooks/

│   └── freight\_rate\_modeling.ipynb

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── features.py

│   ├── train.py

│   └── predict.py

│

├── scorer\_results/

│   └── candidate\_december.png

│

└── report/

&#x20;   └── Freight\_Rate\_Prediction\_Report.pdf

```



\---



\## Installation



Create a Python environment and install the dependencies:



```bash

python -m pip install -r requirements.txt

```



Main dependencies include:



\- NumPy

\- pandas

\- Matplotlib

\- seaborn

\- scikit-learn

\- XGBoost



\---



\## Notebook



The complete exploration and model-development process is available in:



```text

notebooks/freight\_rate\_modeling.ipynb

```



It includes:



\- exploratory data analysis

\- data-quality checks

\- feature engineering

\- chronological validation

\- model comparison

\- feature-set experiments

\- target-transformation experiments

\- XGBoost tuning

\- final training

\- validation predictions

\- December predictions



\---



\## Run the Scorer



After generating both prediction files, run:



```bash

python score.py \\

&#x20; --predictions validation\_predictions.csv \\

&#x20; --december-predictions data/december\_chart\_inputs.csv

```



A successful run validates:



\- all 12,000 validation predictions

\- all 31 December predictions



and generates:



```text

scorer\_results/candidate\_december.png

```



\---



\## Output Files



The main submission outputs are:



```text

validation\_predictions.csv

data/december\_chart\_inputs.csv

scorer\_results/candidate\_december.png

report/Freight\_Rate\_Prediction\_Report.pdf

```



\---



\## Notes



\- Model selection was based only on the internal chronological validation period.

\- The hidden validation labels were never used during model development.

\- Further manual tuning was intentionally stopped once improvements became marginal to reduce the risk of overfitting model selection to the holdout period.

\- December market features are documented estimates derived only from historical development data.


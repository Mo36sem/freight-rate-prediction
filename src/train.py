from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

from features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    fit_imputation_stats,
    prepare_features,
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "train-test.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "freight_rate_model.joblib"
)


# --------------------------------------------------
# Final model pipeline
# --------------------------------------------------

def build_model():
    """Build final preprocessing + XGBoost pipeline."""

    numeric_features = [
        feature
        for feature in ALL_FEATURES
        if feature not in CATEGORICAL_FEATURES
    ]

    numeric_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            CATEGORICAL_FEATURES
        )
    ])

    xgb_model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.025,
        max_depth=4,
        min_child_weight=7,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.5,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb_model),
    ])


# --------------------------------------------------
# Training workflow
# --------------------------------------------------

def main():

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_PATH}"
        )

    # Load labeled development data
    df = pd.read_csv(TRAIN_PATH)

    df["date"] = pd.to_datetime(df["date"])

    print(
        f"Loaded development data: "
        f"{df.shape[0]:,} rows, "
        f"{df.shape[1]} columns"
    )

    # Target
    y = df["posted_rate"].copy()

    # Input features
    X = df.drop(
        columns=[
            "posted_rate",
            "load_id",
            "target_log",
        ],
        errors="ignore",
    )

    # Learn final imputation statistics from the
    # complete labeled development dataset.
    imputation_stats = fit_imputation_stats(X)

    X_prepared = prepare_features(
        X,
        imputation_stats,
    )

    # Final model was selected using log target.
    y_log = np.log1p(y)

    model = build_model()

    print("Training final XGBoost model...")

    model.fit(
        X_prepared[ALL_FEATURES],
        y_log,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bundle = {
        "model": model,
        "imputation_stats": imputation_stats,
        "features": ALL_FEATURES,
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    print("Final model trained successfully.")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
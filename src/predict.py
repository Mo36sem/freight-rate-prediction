from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import (
    ALL_FEATURES,
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

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "validation.csv"
)

DECEMBER_PATH = (
    PROJECT_ROOT
    / "data"
    / "december-chart-inputs.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "freight_rate_model.joblib"
)

VALIDATION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "validation_predictions.csv"
)


# --------------------------------------------------
# Hidden validation predictions
# --------------------------------------------------

def predict_validation(model_bundle):
    """
    Predict every load in validation.csv and
    create validation_predictions.csv.
    """

    model = model_bundle["model"]
    imputation_stats = model_bundle["imputation_stats"]

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_PATH}"
        )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    load_ids = validation_df[
        "load_id"
    ].copy()

    X_validation = validation_df.drop(
        columns=["load_id"],
        errors="ignore",
    )

    X_validation = prepare_features(
        X_validation,
        imputation_stats,
    )

    log_predictions = model.predict(
        X_validation[ALL_FEATURES]
    )

    predictions = np.expm1(
        log_predictions
    )

    output = pd.DataFrame({
        "load_id": load_ids,
        "predicted_rate": predictions,
    })

    # --------------------------------------------------
    # Validation checks
    # --------------------------------------------------

    if len(output) != 12_000:
        raise ValueError(
            "Expected 12,000 validation predictions, "
            f"but generated {len(output):,}."
        )

    if output["load_id"].isna().any():
        raise ValueError(
            "Missing load_id values detected."
        )

    if output["load_id"].duplicated().any():
        raise ValueError(
            "Duplicate load_id values detected."
        )

    if output["predicted_rate"].isna().any():
        raise ValueError(
            "Missing validation predictions detected."
        )

    if not np.isfinite(
        output["predicted_rate"]
    ).all():
        raise ValueError(
            "Non-finite validation predictions detected."
        )

    if (
        output["predicted_rate"] <= 0
    ).any():
        raise ValueError(
            "Non-positive validation predictions detected."
        )

    output.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    print(
        "Validation predictions generated successfully."
    )

    print(
        f"Saved to: {VALIDATION_OUTPUT_PATH}"
    )

    print(
        f"Prediction range: "
        f"${predictions.min():,.2f} - "
        f"${predictions.max():,.2f}"
    )

    print(
        f"Mean prediction: "
        f"${predictions.mean():,.2f}"
    )


# --------------------------------------------------
# December reduced-schema adapter
# --------------------------------------------------

def prepare_december_data():
    """
    Add model-required features that are not supplied
    in the fixed December scenario.

    Only historical development data is used.
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_PATH}"
        )

    if not DECEMBER_PATH.exists():
        raise FileNotFoundError(
            f"December file not found: {DECEMBER_PATH}"
        )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    december_df = pd.read_csv(
        DECEMBER_PATH
    )

    train_df["date"] = pd.to_datetime(
        train_df["date"]
    )

    december_df["date"] = pd.to_datetime(
        december_df["date"]
    )

    # --------------------------------------------------
    # Lexington -> Fort Wayne historical route
    # --------------------------------------------------

    route_history = train_df[
        (train_df["pickup"] == "Lexington")
        & (
            train_df["delivery"]
            == "Fort Wayne"
        )
    ].copy()

    if route_history.empty:
        raise ValueError(
            "No historical Lexington -> Fort Wayne "
            "loads were found."
        )

    # --------------------------------------------------
    # Route coordinates
    # --------------------------------------------------

    route_coordinates = (
        route_history[
            [
                "pickup_lat",
                "pickup_lon",
                "delivery_lat",
                "delivery_lon",
            ]
        ]
        .median()
    )

    december_df["pickup_lat"] = (
        route_coordinates["pickup_lat"]
    )

    december_df["pickup_lon"] = (
        route_coordinates["pickup_lon"]
    )

    december_df["delivery_lat"] = (
        route_coordinates["delivery_lat"]
    )

    december_df["delivery_lon"] = (
        route_coordinates["delivery_lon"]
    )

    # --------------------------------------------------
    # Same route + Dry Van historical observations
    # --------------------------------------------------

    route_equipment_history = route_history[
        route_history["equipment"] == "Dry Van"
    ].copy()

    route_equipment_history = (
        route_equipment_history
        .dropna(
            subset=[
                "market_index",
                "quote_signal",
            ]
        )
        .sort_values("date")
    )

    if route_equipment_history.empty:
        raise ValueError(
            "No Dry Van history was found for "
            "Lexington -> Fort Wayne."
        )

    # Five most recent available observations
    recent_history = (
        route_equipment_history
        .tail(5)
    )

    market_index_fallback = (
        recent_history[
            "market_index"
        ].median()
    )

    quote_signal_fallback = (
        recent_history[
            "quote_signal"
        ].median()
    )

    december_df["market_index"] = (
        market_index_fallback
    )

    december_df["quote_signal"] = (
        quote_signal_fallback
    )

    print(
        "December market_index fallback:",
        round(
            market_index_fallback,
            5,
        ),
    )

    print(
        "December quote_signal fallback:",
        round(
            quote_signal_fallback,
            5,
        ),
    )

    return december_df


# --------------------------------------------------
# December predictions
# --------------------------------------------------

def predict_december(model_bundle):

    model = model_bundle["model"]
    imputation_stats = model_bundle["imputation_stats"]

    december_df = prepare_december_data()

    december_prepared = prepare_features(
        december_df,
        imputation_stats,
    )

    log_predictions = model.predict(
        december_prepared[ALL_FEATURES]
    )

    predictions = np.expm1(
        log_predictions
    )

    december_df["predicted_rate"] = predictions

    # Keep exactly the seven columns required
    # by the provided scorer.
    december_output = december_df[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "predicted_rate",
        ]
    ].copy()

    # --------------------------------------------------
    # Sanity checks
    # --------------------------------------------------

    if len(december_output) != 31:
        raise ValueError(
            "Expected 31 December predictions, "
            f"but generated {len(december_output)}."
        )

    if december_output[
        "predicted_rate"
    ].isna().any():
        raise ValueError(
            "Missing December predictions detected."
        )

    if not np.isfinite(
        december_output["predicted_rate"]
    ).all():
        raise ValueError(
            "Non-finite December predictions detected."
        )

    if (
        december_output["predicted_rate"]
        <= 0
    ).any():
        raise ValueError(
            "Non-positive December predictions detected."
        )

    december_output.to_csv(
        DECEMBER_PATH,
        index=False,
    )

    print(
        "December predictions generated successfully."
    )

    print(
        f"Saved to: {DECEMBER_PATH}"
    )

    print(
        f"Prediction range: "
        f"${predictions.min():,.2f} - "
        f"${predictions.max():,.2f}"
    )

    print(
        f"Mean prediction: "
        f"${predictions.mean():,.2f}"
    )


# --------------------------------------------------
# Main prediction workflow
# --------------------------------------------------

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found.\n"
            "Run this first:\n"
            "python src/train.py"
        )

    model_bundle = joblib.load(
        MODEL_PATH
    )

    print("Generating validation predictions...")
    predict_validation(
        model_bundle
    )

    print(
        "\nGenerating December predictions..."
    )

    predict_december(
        model_bundle
    )

    print(
        "\nPrediction workflow completed successfully."
    )


if __name__ == "__main__":
    main()
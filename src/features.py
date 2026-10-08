import numpy as np
import pandas as pd


# --------------------------------------------------
# Final model features
# --------------------------------------------------

BASE_FEATURES = [
    "distance",
    "weight",
    "quote_signal",
    "market_index",
    "equipment",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "month",
    "day_of_week",
    "quarter",
    "is_weekend",
]

ENGINEERED_FEATURES = [
    "estimated_base_cost",
    "market_adjusted_cost",
    "weight_distance_k",
    "adjusted_rpm",
    "haversine_distance",
    "route_circuity",
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

CATEGORICAL_FEATURES = [
    "equipment",
    "pickup",
    "delivery",
]


# --------------------------------------------------
# Geographic features
# --------------------------------------------------

def haversine_np(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance in miles."""

    earth_radius_miles = 3958.8

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_miles * c


# --------------------------------------------------
# Imputation
# --------------------------------------------------

def fit_imputation_stats(data):
    """
    Learn imputation statistics from training data.
    """

    df = data.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Correct invalid negative weights.
    df["weight"] = df["weight"].abs()

    weight_by_equipment = (
        df.groupby("equipment")["weight"]
        .median()
        .to_dict()
    )

    global_weight_median = df["weight"].median()

    market_by_month = (
        df.groupby(df["date"].dt.month)["market_index"]
        .median()
        .to_dict()
    )

    global_market_median = df["market_index"].median()

    return {
        "weight_by_equipment": weight_by_equipment,
        "global_weight_median": global_weight_median,
        "market_by_month": market_by_month,
        "global_market_median": global_market_median,
    }


# --------------------------------------------------
# Feature preparation
# --------------------------------------------------

def prepare_features(data, imputation_stats):
    """
    Clean input data, apply learned imputations,
    and generate final model features.
    """

    df = data.copy()

    df["date"] = pd.to_datetime(df["date"])

    # --------------------------------------------------
    # Weight cleaning / imputation
    # --------------------------------------------------

    df["weight"] = df["weight"].abs()

    missing_weight = df["weight"].isna()

    df.loc[missing_weight, "weight"] = (
        df.loc[missing_weight, "equipment"]
        .map(imputation_stats["weight_by_equipment"])
        .fillna(imputation_stats["global_weight_median"])
    )

    # --------------------------------------------------
    # Market index imputation
    # --------------------------------------------------

    missing_market = df["market_index"].isna()

    monthly_market_values = (
        df.loc[missing_market, "date"]
        .dt.month
        .map(imputation_stats["market_by_month"])
        .fillna(imputation_stats["global_market_median"])
    )

    df.loc[missing_market, "market_index"] = monthly_market_values

    # --------------------------------------------------
    # Temporal features
    # --------------------------------------------------

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter

    df["is_weekend"] = (
        df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    # --------------------------------------------------
    # Cost / interaction features
    # --------------------------------------------------

    df["estimated_base_cost"] = (
        df["distance"] * df["quote_signal"]
    )

    df["market_adjusted_cost"] = (
        df["estimated_base_cost"]
        * df["market_index"]
    )

    df["weight_distance_k"] = (
        df["weight"] * df["distance"]
    ) / 1000

    df["adjusted_rpm"] = (
        df["quote_signal"]
        * df["market_index"]
    )

    # --------------------------------------------------
    # Geographic features
    # --------------------------------------------------

    df["haversine_distance"] = haversine_np(
        df["pickup_lat"],
        df["pickup_lon"],
        df["delivery_lat"],
        df["delivery_lon"],
    )

    df["route_circuity"] = (
        df["distance"]
        / (df["haversine_distance"] + 1e-5)
    )

    return df
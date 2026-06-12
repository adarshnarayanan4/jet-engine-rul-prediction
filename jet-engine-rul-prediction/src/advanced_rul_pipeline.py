"""
Reusable helper functions for the NASA C-MAPSS Remaining Useful Life project.
The main workflow is intended to be run in the Google Colab notebook.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError:
    xgb = None


SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
BASE_FEATURE_COLS = ["time_cycles"] + SETTING_COLS + SENSOR_COLS


def load_cmapss_data(train_path, test_path, rul_path):
    columns = (
        ["unit_number", "time_cycles"] +
        SETTING_COLS +
        SENSOR_COLS
    )

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=columns)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=columns)
    true_rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["true_RUL"])

    return train_df, test_df, true_rul_df


def create_rul_labels(train_df, rul_cap):
    train_df = train_df.copy()
    max_cycles = train_df.groupby("unit_number")["time_cycles"].max()
    train_df["RUL"] = train_df.apply(
        lambda row: max_cycles[row["unit_number"]] - row["time_cycles"],
        axis=1
    )
    train_df["RUL_capped"] = train_df["RUL"].clip(upper=rul_cap)
    return train_df


def calculate_slope(values):
    if len(values) < 2:
        return 0
    x = np.arange(len(values))
    y = np.array(values)
    return np.polyfit(x, y, 1)[0]


def add_engineered_features(df):
    df = df.copy().sort_values(["unit_number", "time_cycles"])

    for sensor in SENSOR_COLS:
        df[f"{sensor}_rolling_mean_5"] = (
            df.groupby("unit_number")[sensor]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"{sensor}_rolling_std_5"] = (
            df.groupby("unit_number")[sensor]
            .rolling(window=5, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )

        df[f"{sensor}_diff"] = df.groupby("unit_number")[sensor].diff()

        df[f"{sensor}_slope_10"] = (
            df.groupby("unit_number")[sensor]
            .rolling(window=10, min_periods=2)
            .apply(calculate_slope, raw=False)
            .reset_index(level=0, drop=True)
        )

        df[f"{sensor}_baseline_deviation"] = (
            df[sensor] -
            df.groupby("unit_number")[sensor].transform(
                lambda x: x.iloc[:min(20, len(x))].mean()
            )
        )

    return df.fillna(0)


def get_tree_feature_cols():
    engineered_feature_cols = []
    for sensor in SENSOR_COLS:
        engineered_feature_cols.extend([
            f"{sensor}_rolling_mean_5",
            f"{sensor}_rolling_std_5",
            f"{sensor}_diff",
            f"{sensor}_slope_10",
            f"{sensor}_baseline_deviation",
        ])
    return BASE_FEATURE_COLS + engineered_feature_cols


def get_tree_models():
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if xgb is not None:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

    return models


def get_risk_category(rul):
    if rul <= 30:
        return "High Risk"
    if rul <= 60:
        return "Moderate Risk"
    return "Low Risk"


def evaluate_predictions(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }

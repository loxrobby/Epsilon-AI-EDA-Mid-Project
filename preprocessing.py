"""
Shared data loading and cleaning for the Social Media vs Productivity EDA project.

Imputation (median): productivity scores, sleep, stress — per project spec.
Additional numeric columns with missing values are median-imputed for analysis-ready data.
IQR winsorization: daily_social_media_time, coffee_consumption_per_day, number_of_notifications.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "social_media_vs_productivity.csv"
OUTPUT_GRAPHS_DIR = BASE_DIR / "output" / "graphs"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "age",
    "gender",
    "job_type",
    "daily_social_media_time",
    "social_platform_preference",
    "number_of_notifications",
    "work_hours_per_day",
    "perceived_productivity_score",
    "actual_productivity_score",
    "stress_level",
    "sleep_hours",
    "screen_time_before_sleep",
    "breaks_during_work",
    "uses_focus_apps",
    "has_digital_wellbeing_enabled",
    "coffee_consumption_per_day",
    "days_feeling_burnout_per_month",
    "weekly_offline_hours",
    "job_satisfaction_score",
)

BOOL_COLS = ("uses_focus_apps", "has_digital_wellbeing_enabled")
SPEC_MEDIAN_COLS = (
    "perceived_productivity_score",
    "actual_productivity_score",
    "sleep_hours",
    "stress_level",
)
IQR_WINSOR_COLS = (
    "daily_social_media_time",
    "coffee_consumption_per_day",
    "number_of_notifications",
)


def _parse_bool_series(s: pd.Series) -> pd.Series:
    def one(x: object) -> bool | float:
        if pd.isna(x):
            return np.nan
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            if x == 1:
                return True
            if x == 0:
                return False
        xs = str(x).strip().lower()
        if xs in ("true", "1", "yes", "t"):
            return True
        if xs in ("false", "0", "no", "f", ""):
            return False
        return np.nan

    return s.map(one)


def validate_columns(df: pd.DataFrame, columns: Iterable[str] | None = None) -> None:
    cols = list(columns) if columns is not None else list(REQUIRED_COLUMNS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")


def load_raw_data(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path if csv_path is not None else DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    validate_columns(df)
    return df


def _median_impute_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        med = out[c].median()
        if pd.isna(med):
            med = 0.0
        out[c] = out[c].fillna(med)
    return out


def _iqr_winsorize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return s.clip(lower=low, upper=high)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: bool parsing, median imputation (scores/sleep/stress + other numerics),
    IQR winsorization on selected columns, engineered features.
    """
    validate_columns(df)
    out = df.copy()

    for c in BOOL_COLS:
        out[c] = _parse_bool_series(out[c])
        mode = out[c].mode(dropna=True)
        fill_b = bool(mode.iloc[0]) if len(mode) else False
        out[c] = out[c].fillna(fill_b).astype(bool)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    # Impute specified columns first (documented); then any remaining numeric NaNs
    ordered_med = list(SPEC_MEDIAN_COLS) + [c for c in numeric_cols if c not in SPEC_MEDIAN_COLS]
    out = _median_impute_numeric(out, ordered_med)

    for c in IQR_WINSOR_COLS:
        if c in out.columns:
            out[c] = _iqr_winsorize(out[c])

    cat_cols = ["gender", "job_type", "social_platform_preference"]
    for c in cat_cols:
        if c in out.columns:
            mode = out[c].mode(dropna=True)
            fill = str(mode.iloc[0]) if len(mode) else "Unknown"
            out[c] = out[c].fillna(fill).astype(str)

    out["perceived_minus_actual"] = (
        out["perceived_productivity_score"] - out["actual_productivity_score"]
    )

    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
    out["age_group"] = pd.cut(
        out["age"].astype(float),
        bins=age_bins,
        labels=age_labels,
        right=True,
        include_lowest=True,
    ).astype(str)

    return out


def load_cleaned_data(csv_path: Path | None = None) -> pd.DataFrame:
    raw = load_raw_data(csv_path)
    return clean_dataframe(raw)

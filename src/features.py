from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


PRACTICE_SESSIONS = ["FP1", "FP2", "FP3"]
CATEGORICAL_COLS = ["Driver", "Team", "GrandPrix"]
COMPOUND_COLS = ["FP1_compound", "FP2_compound", "FP3_compound"]


def _compound_for_best_lap(session_laps: pd.DataFrame) -> str:
    """Return the Compound used on the driver's fastest lap in a session."""
    if session_laps.empty or "Compound" not in session_laps.columns:
        return "UNKNOWN"
    valid = session_laps.dropna(subset=["LapTime_s"])
    if valid.empty:
        return "UNKNOWN"
    return str(valid.loc[valid["LapTime_s"].idxmin(), "Compound"])


def parse_laptime_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Add LapTime_s column by converting the LapTime timedelta string to float seconds."""
    df = df.copy()
    df["LapTime_s"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-driver, per-GP practice stats and weather into one row per driver-weekend.

    Rows where the driver has no qualifying time are dropped (no target available).
    Missing practice sessions produce NaN in the corresponding best/mean/std columns.
    """
    if "LapTime_s" not in df.columns:
        df = parse_laptime_to_seconds(df)

    results: list[dict] = []

    for (year, gp, driver), group in df.groupby(["Year", "GrandPrix", "Driver"]):
        row: dict = {"Year": year, "GrandPrix": gp, "Driver": driver}

        for sess in PRACTICE_SESSIONS:
            sess_laps = group.loc[group["Session"] == sess]
            lap_times = sess_laps["LapTime_s"]
            row[f"{sess}_best"] = lap_times.min() if not lap_times.empty else np.nan
            row[f"{sess}_mean"] = lap_times.mean() if not lap_times.empty else np.nan
            row[f"{sess}_std"] = lap_times.std() if not lap_times.empty else np.nan
            row[f"{sess}_compound"] = _compound_for_best_lap(sess_laps)

        fp1 = row["FP1_best"]
        fp3 = row["FP3_best"]
        row["track_evolution"] = (
            fp1 - fp3 if not (np.isnan(fp1) or np.isnan(fp3)) else np.nan
        )

        q_laps = group.loc[group["Session"] == "Q"]
        row["AirTemp"] = q_laps["AirTemp"].mean() if not q_laps.empty else np.nan
        row["TrackTemp"] = q_laps["TrackTemp"].mean() if not q_laps.empty else np.nan
        row["Humidity"] = q_laps["Humidity"].mean() if not q_laps.empty else np.nan
        row["Rainfall"] = int(q_laps["Rainfall"].any()) if not q_laps.empty else 0
        row["Team"] = group["Team"].iloc[0] if "Team" in group.columns else "Unknown"

        q_times = group.loc[group["Session"] == "Q", "LapTime_s"]
        row["quali_best"] = q_times.min() if not q_times.empty else np.nan

        results.append(row)

    out = pd.DataFrame(results)
    return out.dropna(subset=["quali_best"]).reset_index(drop=True)


def encode_categoricals(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode Driver, Team, and GrandPrix columns.

    Pass `encoders=None` to fit new encoders (training); pass existing encoders
    to transform without refitting (inference / test sets).
    Returns (transformed_df, encoders).
    """
    df = df.copy()
    fit_new = encoders is None
    if fit_new:
        encoders = {}
    for col in CATEGORICAL_COLS + COMPOUND_COLS:
        if col not in df.columns:
            continue
        if fit_new:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            df[f"{col}_enc"] = encoders[col].transform(df[col].astype(str))
    return df, encoders

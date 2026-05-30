"""Retrain all models with compound features and save artefacts to models/.

Usage:
    python src/train.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable when running as a script
_PROJECT_ROOT_STR = str(Path(__file__).parent.parent)
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


from src.features import (
    build_features as _build_features,
    encode_categoricals,
    parse_laptime_to_seconds,
)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    if "LapTime_s" not in raw.columns:
        raw = parse_laptime_to_seconds(raw)
    return _build_features(raw)


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    return encode_categoricals(df)


def train(cfg: dict) -> None:
    raw_csv = PROJECT_ROOT / cfg["data"]["raw_output"]
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    encoder_prefix = cfg["paths"]["encoder_prefix"]
    feature_cols = cfg["model"]["features"]
    target = cfg["model"]["target"]
    test_size = 1.0 - cfg["model"]["train_split"]
    random_state = cfg["model"]["random_forest"]["random_state"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading raw data from %s", raw_csv)
    raw = pd.read_csv(raw_csv)

    log.info("Building features (including tyre compounds)…")
    feat_df = build_features(raw)
    feat_df, encoders = encode(feat_df)

    feat_csv = processed_dir / "features.csv"
    feat_df.to_csv(feat_csv, index=False)
    log.info("Saved features → %s  (%d rows, %d cols)", feat_csv, *feat_df.shape)

    available = [c for c in feature_cols if c in feat_df.columns]
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        log.warning("Dropping missing feature columns: %s", missing)

    X = feat_df[available].fillna(0).values
    y = feat_df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    log.info("Train=%d  Test=%d", len(X_train), len(X_test))

    rf_cfg = cfg["model"]["random_forest"]
    gb_cfg = cfg["model"]["gradient_boosting"]
    xgb_cfg = {k: v for k, v in cfg["model"]["xgboost"].items() if k != "verbosity"}

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=rf_cfg["n_estimators"],
            random_state=rf_cfg["random_state"],
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=gb_cfg["n_estimators"],
            learning_rate=gb_cfg["learning_rate"],
            random_state=gb_cfg["random_state"],
        ),
        "Ridge": Ridge(),
        "XGBoost": XGBRegressor(**xgb_cfg),
    }

    results: dict[str, float] = {}
    trained: dict[str, object] = {}

    for name, m in candidates.items():
        m.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, m.predict(X_test))
        results[name] = mae
        trained[name] = m
        log.info("  %-20s  MAE = %.4f s", name, mae)

    best_name = min(results, key=results.__getitem__)
    log.info("Best model: %s  (MAE = %.4f s)", best_name, results[best_name])

    joblib.dump(trained[best_name], models_dir / "best_model.pkl")
    log.info("Saved best_model.pkl")

    with open(models_dir / "feature_cols.json", "w") as fh:
        json.dump(available, fh)
    log.info("Saved feature_cols.json  (%d features)", len(available))

    encoder_map = {
        "driver": "Driver",
        "team": "Team",
        "grandprix": "GrandPrix",
        "fp1_compound": "FP1_compound",
        "fp2_compound": "FP2_compound",
        "fp3_compound": "FP3_compound",
    }
    for file_suffix, col_key in encoder_map.items():
        if col_key in encoders:
            path = models_dir / f"{encoder_prefix}{file_suffix}.pkl"
            joblib.dump(encoders[col_key], path)
            log.info("Saved %s", path.name)
        else:
            log.warning("No encoder found for %s — skipping", col_key)

    log.info("Done.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train F1 qualifying predictor")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

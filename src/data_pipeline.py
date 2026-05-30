"""Standalone data-collection pipeline.

Fetches FastF1 session laps for all configured seasons and circuits, attaches
weather data, and writes a versioned CSV to data/raw/.

Usage:
    python src/data_pipeline.py [--config config.yaml] [--dry-run]

Output:
    data/raw/all_sessions_YYYYMMDD_HHMMSS.csv  (versioned)
    data/raw/all_sessions.csv                   (latest symlink / copy)
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import fastf1
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# FastF1 helpers
# ---------------------------------------------------------------------------

def _fetch_session(year: int, gp: str, session_name: str) -> pd.DataFrame:
    """Load one FastF1 session and return laps with weather attached.

    Returns an empty DataFrame on any error so the caller can continue.
    """
    try:
        session = fastf1.get_session(year, gp, session_name)
        session.load(telemetry=False, weather=True, messages=False)
        laps = session.laps.pick_quicklaps().copy()

        weather = session.weather_data
        laps["AirTemp"] = weather["AirTemp"].mean()
        laps["TrackTemp"] = weather["TrackTemp"].mean()
        laps["Humidity"] = weather["Humidity"].mean()
        laps["Rainfall"] = weather["Rainfall"].any()

        laps["Year"] = year
        laps["GrandPrix"] = gp
        laps["Session"] = session_name
        log.info("  ✓ %s %s %s — %d laps", year, gp, session_name, len(laps))
        return laps

    except Exception as exc:
        log.warning("  ✗ Skipped %s %s %s — %s", year, gp, session_name, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(cfg: dict, dry_run: bool = False) -> pd.DataFrame | None:
    seasons: list[int] = cfg["data"]["seasons"]
    sessions: list[str] = cfg["data"]["sessions"]
    cache_dir = Path(cfg["data"]["cache_dir"])
    raw_output = Path(cfg["data"]["raw_output"])
    processed_dir = Path(cfg["data"]["processed_dir"])

    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root / cache_dir
    raw_output = project_root / raw_output
    processed_dir = project_root / processed_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_dir))

    if dry_run:
        log.info("[dry-run] Would fetch seasons=%s sessions=%s", seasons, sessions)
        return None

    all_frames: list[pd.DataFrame] = []

    for year in seasons:
        log.info("=== Season %d ===", year)
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            log.error("Could not fetch schedule for %d: %s", year, exc)
            continue

        for _, event in schedule.iterrows():
            gp = event["EventName"]
            log.info("Loading %d — %s", year, gp)
            for sess in sessions:
                frame = _fetch_session(year, gp, sess)
                if not frame.empty:
                    all_frames.append(frame)

    if not all_frames:
        log.error("No data collected — check seasons/circuits in config.yaml")
        return None

    raw_df = pd.concat(all_frames, ignore_index=True)

    # Versioned output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = raw_output.parent / f"all_sessions_{timestamp}.csv"
    raw_df.to_csv(versioned_path, index=False)
    log.info("Saved versioned output: %s (%d rows)", versioned_path, len(raw_df))

    # Canonical latest copy
    shutil.copy2(versioned_path, raw_output)
    log.info("Updated latest: %s", raw_output)

    return raw_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 qualifying data pipeline")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
        help="Path to config.yaml (default: <project_root>/config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without calling FastF1",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log.info("Loading config from %s", args.config)
    cfg = load_config(args.config)
    result = build_dataset(cfg, dry_run=args.dry_run)
    if result is None and not args.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

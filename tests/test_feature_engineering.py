"""Unit tests for src/features.py feature engineering functions."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.features import (
    CATEGORICAL_COLS,
    build_features,
    encode_categoricals,
    parse_laptime_to_seconds,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_laps(
    *,
    include_q: bool = True,
    include_fp1: bool = True,
    include_fp2: bool = True,
    include_fp3: bool = True,
    rainfall: bool = False,
) -> pd.DataFrame:
    """Build a minimal raw-laps DataFrame mimicking all_sessions.csv structure."""
    rows = []
    sessions = []
    if include_fp1:
        sessions.append(("FP1", "0:01:29.000"))
        sessions.append(("FP1", "0:01:30.500"))
        sessions.append(("FP1", "0:01:31.200"))
    if include_fp2:
        sessions.append(("FP2", "0:01:28.500"))
        sessions.append(("FP2", "0:01:29.800"))
    if include_fp3:
        sessions.append(("FP3", "0:01:27.400"))
        sessions.append(("FP3", "0:01:28.100"))
    if include_q:
        sessions.append(("Q", "0:01:26.800"))
        sessions.append(("Q", "0:01:27.900"))

    for sess, lap in sessions:
        rows.append(
            {
                "Year": 2023,
                "GrandPrix": "Bahrain Grand Prix",
                "Driver": "VER",
                "Team": "Red Bull Racing",
                "Session": sess,
                "LapTime": lap,
                "AirTemp": 28.0,
                "TrackTemp": 38.5,
                "Humidity": 55.0,
                "Rainfall": rainfall,
            }
        )
    return pd.DataFrame(rows)


def _make_two_driver_raw(n_gps: int = 2) -> pd.DataFrame:
    """Two drivers across multiple GPs, all sessions present."""
    frames = []
    gps = [f"GP{i}" for i in range(n_gps)]
    for gp in gps:
        for driver, team in [("VER", "Red Bull Racing"), ("HAM", "Mercedes")]:
            for sess, lap in [
                ("FP1", "0:01:30.000"),
                ("FP2", "0:01:29.000"),
                ("FP3", "0:01:28.000"),
                ("Q", "0:01:27.000"),
            ]:
                frames.append(
                    {
                        "Year": 2023,
                        "GrandPrix": gp,
                        "Driver": driver,
                        "Team": team,
                        "Session": sess,
                        "LapTime": lap,
                        "AirTemp": 25.0,
                        "TrackTemp": 35.0,
                        "Humidity": 50.0,
                        "Rainfall": False,
                    }
                )
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# parse_laptime_to_seconds
# ---------------------------------------------------------------------------

class TestParseLaptimeToSeconds:
    def test_adds_column(self):
        df = pd.DataFrame({"LapTime": ["0:01:30.000", "0:01:29.500"]})
        result = parse_laptime_to_seconds(df)
        assert "LapTime_s" in result.columns

    def test_correct_conversion(self):
        df = pd.DataFrame({"LapTime": ["0:01:30.000"]})
        result = parse_laptime_to_seconds(df)
        assert result["LapTime_s"].iloc[0] == pytest.approx(90.0)

    def test_sub_minute_lap(self):
        df = pd.DataFrame({"LapTime": ["0:00:59.123"]})
        result = parse_laptime_to_seconds(df)
        assert result["LapTime_s"].iloc[0] == pytest.approx(59.123)

    def test_nat_becomes_nan(self):
        df = pd.DataFrame({"LapTime": [None, "0:01:30.000"]})
        result = parse_laptime_to_seconds(df)
        assert np.isnan(result["LapTime_s"].iloc[0])
        assert result["LapTime_s"].iloc[1] == pytest.approx(90.0)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"LapTime": ["0:01:30.000"]})
        parse_laptime_to_seconds(df)
        assert "LapTime_s" not in df.columns

    def test_output_shape_unchanged(self):
        df = pd.DataFrame({"LapTime": ["0:01:30.000", "0:01:29.000", "0:01:28.000"]})
        result = parse_laptime_to_seconds(df)
        assert result.shape == (3, 2)  # original col + new LapTime_s


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_output_columns_present(self):
        raw = _make_raw_laps()
        result = build_features(raw)
        expected = {
            "Year", "GrandPrix", "Driver", "Team",
            "FP1_best", "FP1_mean", "FP1_std",
            "FP2_best", "FP2_mean", "FP2_std",
            "FP3_best", "FP3_mean", "FP3_std",
            "track_evolution",
            "AirTemp", "TrackTemp", "Humidity", "Rainfall",
            "quali_best",
        }
        assert expected.issubset(result.columns)

    def test_one_row_per_driver_gp(self):
        raw = _make_two_driver_raw(n_gps=3)
        result = build_features(raw)
        # 2 drivers × 3 GPs = 6 rows
        assert result.shape[0] == 6

    def test_single_driver_single_gp_shape(self):
        raw = _make_raw_laps()
        result = build_features(raw)
        assert result.shape[0] == 1

    def test_fp_best_is_minimum(self):
        raw = _make_raw_laps()
        result = build_features(raw)
        # FP1 laps are 89, 90.5, 91.2 → best = 89
        assert result["FP1_best"].iloc[0] == pytest.approx(89.0)

    def test_fp_mean_correct(self):
        raw = _make_raw_laps()
        result = build_features(raw)
        expected_mean = np.mean([89.0, 90.5, 91.2])
        assert result["FP1_mean"].iloc[0] == pytest.approx(expected_mean)

    def test_track_evolution_is_fp1_minus_fp3(self):
        raw = _make_raw_laps()
        result = build_features(raw)
        fp1_best = result["FP1_best"].iloc[0]
        fp3_best = result["FP3_best"].iloc[0]
        assert result["track_evolution"].iloc[0] == pytest.approx(fp1_best - fp3_best)

    def test_quali_best_is_minimum_q_lap(self):
        raw = _make_raw_laps()
        result = build_features(raw)
        assert result["quali_best"].iloc[0] == pytest.approx(86.8)  # 1:26.800

    def test_driver_with_no_quali_is_dropped(self):
        raw = _make_raw_laps(include_q=False)
        result = build_features(raw)
        assert result.empty

    def test_missing_fp1_produces_nan_not_error(self):
        raw = _make_raw_laps(include_fp1=False)
        result = build_features(raw)
        assert result.shape[0] == 1
        assert np.isnan(result["FP1_best"].iloc[0])
        assert np.isnan(result["FP1_mean"].iloc[0])

    def test_missing_fp1_makes_track_evolution_nan(self):
        raw = _make_raw_laps(include_fp1=False)
        result = build_features(raw)
        assert np.isnan(result["track_evolution"].iloc[0])

    def test_missing_fp3_makes_track_evolution_nan(self):
        raw = _make_raw_laps(include_fp3=False)
        result = build_features(raw)
        assert np.isnan(result["track_evolution"].iloc[0])

    def test_rainfall_flag_captured(self):
        raw_wet = _make_raw_laps(rainfall=True)
        result = build_features(raw_wet)
        assert result["Rainfall"].iloc[0] == 1

    def test_rainfall_flag_dry(self):
        raw_dry = _make_raw_laps(rainfall=False)
        result = build_features(raw_dry)
        assert result["Rainfall"].iloc[0] == 0

    def test_does_not_mutate_input(self):
        raw = _make_raw_laps()
        cols_before = list(raw.columns)
        build_features(raw)
        assert list(raw.columns) == cols_before

    def test_multiple_drivers_same_gp_all_kept(self):
        raw = _make_two_driver_raw(n_gps=1)
        result = build_features(raw)
        assert set(result["Driver"]) == {"VER", "HAM"}

    def test_index_is_reset(self):
        raw = _make_two_driver_raw(n_gps=2)
        result = build_features(raw)
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# encode_categoricals
# ---------------------------------------------------------------------------

class TestEncodeCategoricals:
    def _sample_features_df(self) -> pd.DataFrame:
        raw = _make_two_driver_raw(n_gps=2)
        return build_features(raw)

    def test_enc_columns_added(self):
        df = self._sample_features_df()
        encoded, _ = encode_categoricals(df)
        for col in CATEGORICAL_COLS:
            assert f"{col}_enc" in encoded.columns

    def test_enc_values_are_integers(self):
        df = self._sample_features_df()
        encoded, _ = encode_categoricals(df)
        for col in CATEGORICAL_COLS:
            assert encoded[f"{col}_enc"].dtype in (np.int32, np.int64, int)

    def test_fit_returns_encoders_dict(self):
        df = self._sample_features_df()
        _, encoders = encode_categoricals(df)
        assert set(encoders.keys()) == set(CATEGORICAL_COLS)
        assert all(isinstance(v, LabelEncoder) for v in encoders.values())

    def test_transform_mode_uses_existing_encoders(self):
        df = self._sample_features_df()
        _, encoders = encode_categoricals(df)

        # Same data → same codes
        encoded2, _ = encode_categoricals(df, encoders=encoders)
        encoded1, _ = encode_categoricals(df)
        for col in CATEGORICAL_COLS:
            assert (encoded1[f"{col}_enc"] == encoded2[f"{col}_enc"]).all()

    def test_unseen_label_raises_in_transform_mode(self):
        df = self._sample_features_df()
        _, encoders = encode_categoricals(df)

        # Introduce a driver the encoder has never seen
        bad_df = df.copy()
        bad_df.loc[0, "Driver"] = "GHOST_DRIVER"
        with pytest.raises(ValueError):
            encode_categoricals(bad_df, encoders=encoders)

    def test_does_not_mutate_input(self):
        df = self._sample_features_df()
        cols_before = set(df.columns)
        encode_categoricals(df)
        assert set(df.columns) == cols_before

    def test_driver_enc_range(self):
        df = self._sample_features_df()
        encoded, encoders = encode_categoricals(df)
        n_classes = len(encoders["Driver"].classes_)
        assert encoded["Driver_enc"].between(0, n_classes - 1).all()

    def test_grandprix_enc_range(self):
        df = self._sample_features_df()
        encoded, encoders = encode_categoricals(df)
        n_classes = len(encoders["GrandPrix"].classes_)
        assert encoded["GrandPrix_enc"].between(0, n_classes - 1).all()

"""
Load F1 processed data into BigQuery.

Usage (from project root):
    python pipeline/load_to_bigquery.py

Requires: pip install google-cloud-bigquery google-cloud-bigquery-storage db-dtypes pyarrow
"""

import json
import os

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "..", "gcp_credentials.json")
DATASET_ID = "f1_data"

RAW_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "all_sessions.csv")
FEATURES_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv")

# Columns stored as pandas timedelta strings ("0 days HH:MM:SS.ffffff") in the CSV
TIMEDELTA_COLS = [
    "Time",
    "LapTime",
    "PitOutTime",
    "PitInTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Sector1SessionTime",
    "Sector2SessionTime",
    "Sector3SessionTime",
    "LapStartTime",
]


def get_client():
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    with open(CREDENTIALS_PATH) as f:
        project_id = json.load(f).get("project_id")
    if not project_id:
        raise ValueError("project_id not found in gcp_credentials.json")
    return bigquery.Client(project=project_id, credentials=creds), project_id


def ensure_dataset(client, project_id):
    ref = bigquery.DatasetReference(project_id, DATASET_ID)
    try:
        client.get_dataset(ref)
        print(f"Dataset {DATASET_ID} already exists.")
    except Exception:
        dataset = bigquery.Dataset(ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"Created dataset {DATASET_ID}.")


def convert_timedeltas(df):
    """Convert timedelta string columns to float seconds and drop the originals."""
    for col in TIMEDELTA_COLS:
        if col in df.columns:
            df[f"{col}_seconds"] = pd.to_timedelta(df[col], errors="coerce").dt.total_seconds()
    df = df.drop(columns=[c for c in TIMEDELTA_COLS if c in df.columns])
    return df


def load_table(client, project_id, df, table_name):
    table_id = f"{project_id}.{DATASET_ID}.{table_name}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    table = client.get_table(table_id)
    print(f"  Loaded {table.num_rows:,} rows -> {table_id}")


def main():
    print("Connecting to BigQuery...")
    client, project_id = get_client()
    print(f"  Project: {project_id}")

    ensure_dataset(client, project_id)

    print("\nLoading raw_lap_times...")
    raw_df = pd.read_csv(RAW_CSV)
    raw_df = convert_timedeltas(raw_df)
    load_table(client, project_id, raw_df, "raw_lap_times")

    print("\nLoading qualifying_features...")
    features_df = pd.read_csv(FEATURES_CSV)
    load_table(client, project_id, features_df, "qualifying_features")

    print("\nAll tables loaded successfully.")


if __name__ == "__main__":
    main()

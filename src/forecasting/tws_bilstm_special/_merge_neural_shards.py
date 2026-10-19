from pathlib import Path

import pandas as pd

from candidate_config import (
    LAND_TYPES,
    OUTPUT_CACHE_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
)


def combine(paths):
    frames = [
        pd.read_csv(path)
        for path in paths
        if path.exists() and path.stat().st_size > 0
    ]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def merge_bilstm():
    base_pixel = TABLES_DIR / "hydroclimate_bilstm_special_pixel_metrics.csv"
    base_monthly = TABLES_DIR / "hydroclimate_bilstm_special_monthly_metrics.csv"
    base_manifest = TABLES_DIR / "hydroclimate_bilstm_special_run_manifest.csv"
    base_predictions = (
        PREDICTIONS_DIR / "hydroclimate_bilstm_special_predictions.csv.gz"
    )
    stems = [f"bilstm_worker_{land.lower()}" for land in LAND_TYPES]
    pixel = combine(
        [base_pixel] + [OUTPUT_CACHE_DIR / f"{stem}_pixel.csv" for stem in stems]
    )
    monthly = combine(
        [base_monthly] + [OUTPUT_CACHE_DIR / f"{stem}_monthly.csv" for stem in stems]
    )
    manifest = combine(
        [base_manifest] + [OUTPUT_CACHE_DIR / f"{stem}_manifest.csv" for stem in stems]
    )
    predictions = combine(
        [base_predictions]
        + [OUTPUT_CACHE_DIR / f"{stem}_predictions.csv.gz" for stem in stems]
    )
    keys = ["model", "configuration", "land_type", "variable", "fold", "seed"]
    pixel.drop_duplicates(keys, keep="last").to_csv(base_pixel, index=False)
    monthly.drop_duplicates(keys, keep="last").to_csv(base_monthly, index=False)
    manifest.drop_duplicates(keys, keep="last").to_csv(base_manifest, index=False)
    predictions.drop_duplicates(keys + ["date", "pixel_id"], keep="last").to_csv(
        base_predictions, index=False, compression="gzip"
    )


def merge_nbeats():
    base_pixel = TABLES_DIR / "spei_nbeats_special_pixel_metrics.csv"
    base_monthly = TABLES_DIR / "spei_nbeats_special_monthly_metrics.csv"
    base_manifest = TABLES_DIR / "spei_nbeats_special_run_manifest.csv"
    base_predictions = PREDICTIONS_DIR / "spei_nbeats_special_predictions.csv.gz"
    stems = [f"nbeats_worker_{land.lower()}" for land in LAND_TYPES]
    pixel = combine(
        [base_pixel] + [OUTPUT_CACHE_DIR / f"{stem}_pixel.csv" for stem in stems]
    )
    monthly = combine(
        [base_monthly] + [OUTPUT_CACHE_DIR / f"{stem}_monthly.csv" for stem in stems]
    )
    manifest = combine(
        [base_manifest] + [OUTPUT_CACHE_DIR / f"{stem}_manifest.csv" for stem in stems]
    )
    predictions = combine(
        [base_predictions]
        + [OUTPUT_CACHE_DIR / f"{stem}_predictions.csv.gz" for stem in stems]
    )
    keys = ["model", "configuration", "land_type", "variable", "fold", "seed"]
    pixel.drop_duplicates(keys, keep="last").to_csv(base_pixel, index=False)
    monthly.drop_duplicates(keys, keep="last").to_csv(base_monthly, index=False)
    manifest.drop_duplicates(keys, keep="last").to_csv(base_manifest, index=False)
    predictions.drop_duplicates(keys + ["date", "pixel_id"], keep="last").to_csv(
        base_predictions, index=False, compression="gzip"
    )


if __name__ == "__main__":
    merge_bilstm()
    merge_nbeats()

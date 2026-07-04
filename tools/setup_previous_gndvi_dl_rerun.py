"""Create a separate GNDVI-only reviewer-safe forecasting rerun package.

The copied scripts preserve model architecture/training functions. The copied
configuration is redirected to the completed previous/no-QA GNDVI data tree and
restricted to the GNDVI target so non-GNDVI outputs are not rerun.
"""

from __future__ import annotations

import shutil
from pathlib import Path


SOURCE = Path(
    r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses"
    r"\05_forecasting_raster_reviewer_safe_candidate"
)
DEST = Path(r"C:\Users\LOQ\OneDrive\Desktop\DL_GNDVI_previous_noQA_rerun")
GNDVI_DATA = Path(
    r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses"
    r"\final_previous_GNDVI_with_test_rebuild\data_for_forecasting"
)


def copy_package() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ["scripts"]:
        src = SOURCE / name
        dst = DEST / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    for name in ["tables", "figures", "report", "logs", "models", "predictions", "cache"]:
        (DEST / name).mkdir(exist_ok=True)


def patch_config() -> None:
    path = DEST / "scripts" / "candidate_config.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        r'PACKAGE_ROOT = Path('
        '\n'
        r'    r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses"'
        '\n'
        r'    r"\05_forecasting_raster_reviewer_safe_candidate"'
        '\n'
        r')',
        f'PACKAGE_ROOT = Path(r"{DEST}")',
    )
    text = text.replace(
        r'HISTORY_ROOT = THESIS_ROOT / "Data"',
        f'HISTORY_ROOT = Path(r"{GNDVI_DATA / "Data"}")',
    )
    text = text.replace(
        r'TEST_ROOT = THESIS_ROOT / "Test Data" / "Masked_LandTypes"',
        f'TEST_ROOT = Path(r"{GNDVI_DATA / "Test Data" / "Masked_LandTypes"}")',
    )
    text = text.replace(
        'VARIABLES = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]',
        'VARIABLES = ["GNDVI"]',
    )
    text = text.replace(
        '# raster_pipeline.py imports CACHE_DIR. Point it at the verified read-only cache.\nCACHE_DIR = SOURCE_PACKAGE / "cache"',
        '# GNDVI-only previous/no-QA rerun uses a fresh cache in this package.\nCACHE_DIR = PACKAGE_ROOT / "cache"',
    )
    path.write_text(text, encoding="utf-8")


def patch_xgboost() -> None:
    path = DEST / "scripts" / "03_run_reviewer_safe_xgboost.py"
    text = path.read_text(encoding="utf-8")
    start = text.index("def main():")
    replacement = r'''def main():
    log_path = LOGS_DIR / "reviewer_safe_xgboost_log.txt"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started previous/no-QA GNDVI rerun: {datetime.now().isoformat()}\n")
        pixel_frames = []
        monthly_frames = []
        prediction_frames = []
        manifest_frames = []
        for seed in SEEDS:
            temp_log = LOGS_DIR / f"_xgboost_seed_{seed}.log"
            pixel, monthly, predictions, manifest = run_xgboost_tasks(
                land_types=LAND_TYPES,
                variables=VARIABLES,
                seeds_for_task=lambda model, land, variable, spec, value=seed: [value],
                include_mode_b=True,
                log_path=temp_log,
                include_mode_a=False,
            )
            for frame in [pixel, monthly, predictions, manifest]:
                if "mode" in frame:
                    frame["mode"] = "reviewer_safe_rolling_origin"
            pixel["reuse_status"] = "new_previous_noQA_GNDVI_rerun"
            monthly["reuse_status"] = "new_previous_noQA_GNDVI_rerun"
            predictions["reuse_status"] = "new_previous_noQA_GNDVI_rerun"
            pixel_frames.append(pixel)
            monthly_frames.append(monthly)
            prediction_frames.append(predictions)
            manifest_frames.append(manifest)
            log.write(temp_log.read_text(encoding="utf-8"))
            temp_log.unlink(missing_ok=True)
            save(
                pd.concat(pixel_frames, ignore_index=True),
                pd.concat(monthly_frames, ignore_index=True),
                pd.concat(prediction_frames, ignore_index=True),
                manifest_frames,
            )
            log.write(f"Checkpoint saved after seed {seed}\n")
            log.flush()
        log.write(f"Finished: {datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()
'''
    path.write_text(text[:start] + replacement, encoding="utf-8")


def write_readme() -> None:
    (DEST / "README_previous_noQA_GNDVI_rerun.md").write_text(
        "# Previous/no-QA GNDVI Reviewer-Safe Forecasting Rerun\n\n"
        "This package is a separate GNDVI-only copy of the reviewer-safe forecasting pipeline.\n"
        "It points `HISTORY_ROOT` and `TEST_ROOT` to the completed previous/no-QA GNDVI data tree\n"
        "under `final_previous_GNDVI_with_test_rebuild/data_for_forecasting` and restricts\n"
        "`VARIABLES` to `[\"GNDVI\"]`. Model code, architecture helpers, lookback, folds,\n"
        "seeds, scaling, and preprocessing functions are otherwise inherited from the reviewer-safe\n"
        "pipeline.\n",
        encoding="utf-8",
    )


def main() -> None:
    copy_package()
    patch_config()
    patch_xgboost()
    write_readme()
    shutil.copy2(Path(__file__), DEST / "setup_previous_gndvi_dl_rerun.py")


if __name__ == "__main__":
    main()

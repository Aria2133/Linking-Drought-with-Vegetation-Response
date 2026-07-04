import importlib.util
from pathlib import Path

import pandas as pd


MAIN = Path(__file__).resolve().parent / "gndvi_full_source_audit_old_vs_QA.py"
spec = importlib.util.spec_from_file_location("gndvi_audit", MAIN)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    inv = pd.read_csv(audit.TABLES / "GNDVI_raster_inventory_old_vs_QA.csv")
    pairing = pd.read_csv(audit.TABLES / "GNDVI_pairing_and_grid_alignment_audit.csv")
    if "same_grid" not in pairing.columns:
        pairing["same_grid"] = pairing["same_shape"] & pairing["same_transform"] & pairing["same_CRS"]
    pixel = pd.read_csv(audit.TABLES / "GNDVI_pixel_level_old_vs_QA_comparison.csv")
    monthly = pd.read_csv(audit.TABLES / "GNDVI_landtype_monthly_mean_old_vs_QA.csv")
    monthly["date"] = pd.to_datetime(monthly["date"])
    sim = pd.read_csv(audit.TABLES / "GNDVI_landtype_monthly_similarity_metrics.csv")
    seasonal = pd.read_csv(audit.TABLES / "GNDVI_seasonal_cycle_old_vs_QA.csv")
    inter = pd.read_csv(audit.TABLES / "GNDVI_interindex_correlation_diagnostic.csv")
    ccf_diag = pd.read_csv(audit.TABLES / "GNDVI_CCF_old_vs_QA_diagnostic.csv")
    checklist = audit.failure_mode_checklist(inv, pairing, pixel, monthly, inter, ccf_diag)
    audit.final_report(inv, pairing, pixel, monthly, sim, seasonal, inter, ccf_diag, checklist)
    print("Final checklist/report regenerated")


if __name__ == "__main__":
    main()

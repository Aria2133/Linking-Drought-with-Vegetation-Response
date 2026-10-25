"""Install the rebuilt 2022 MOD17A2H GPP (see src/ecological/gpp/mod17_monthly_from_planetary_computer.py) into a
corrected copy of the land-type rasters and of the forecasting task cache. Nothing in the original data folders is changed.

1. Land-type rasters: the rebuilt California files are masked with the 2021 MCD12Q1 land-type masks exactly as in
   mask_test_period_2022_2024.py (nearest-neighbour mask on the product grid, float32, NaN outside the land type) and
   written as <out>/Masked_LandTypes/<Land>/GPP/GPP_MOD17A2H_CA_5km_2022_MM_<Land>.tif.
2. Forecasting cache: <cache>/<LAND>_GPP.npz is copied with the 12 rows for 2022 of `values` replaced by the masked
   rebuilt values at the cached pixels; all other arrays and all other months are unchanged (pixel selection uses
   months up to 2021 and 2024 only). Other variables' caches are copied unchanged.

Usage: python apply_rebuilt_mod17_2022.py --rebuilt <dir> --masks <CA_LandMasks dir> --cache <task_data dir> --out <dir>
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

LANDS = {"Grasslands": "GRASSLANDS", "NaturalForests": "NATURAL_FORESTS", "Savannas": "SAVANNAS",
         "Shrublands": "SHRUBLANDS"}


def apply_mask(input_path, mask_path, output_path):
    with rasterio.open(input_path) as src, rasterio.open(mask_path) as mask_src:
        data = src.read(1)
        meta = src.meta.copy()
        mask = np.empty(data.shape, dtype=np.uint8)
        reproject(source=rasterio.band(mask_src, 1), destination=mask, src_transform=mask_src.transform,
                  src_crs=mask_src.crs, dst_transform=src.transform, dst_crs=src.crs, resampling=Resampling.nearest)
        masked = np.where(mask == 1, data, np.nan).astype(np.float32)
    meta.update(dtype="float32", nodata=np.nan, compress="LZW")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(masked, 1)
    return masked


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuilt", required=True, type=Path)
    ap.add_argument("--masks", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    masked = {}
    for land in LANDS:
        for m in range(1, 13):
            src = args.rebuilt / f"GPP_MOD17A2H_CA_5km_2022_{m:02d}.tif"
            dst = args.out / "Masked_LandTypes" / land / "GPP" / f"GPP_MOD17A2H_CA_5km_2022_{m:02d}_{land}.tif"
            masked[land, m] = apply_mask(src, args.masks / f"CA_{land}_2021.tif", dst)

    cache_out = args.out / "forecast_cache" / "task_data"
    cache_out.mkdir(parents=True, exist_ok=True)
    report = {}
    for f in sorted(args.cache.glob("*.npz")):
        if not f.name.endswith("_GPP.npz"):
            shutil.copy2(f, cache_out / f.name)
            continue
        land = next(k for k, v in LANDS.items() if f.name == f"{v}_GPP.npz")
        z = dict(np.load(f, allow_pickle=False))
        values = z["values"].copy()
        for m in range(1, 13):
            i = int(np.where(z["dates"] == 202200 + m)[0][0])
            new = masked[land, m][z["rows"], z["cols"]]
            assert np.isfinite(new).all(), f"{land} 2022-{m:02d}: cached pixel outside the land-type mask"
            values[i] = new
        z["values"] = values.astype(z["values"].dtype)
        meta = json.loads(str(z["metadata_json"]))
        meta["mod17_2022_replaced"] = "rebuilt from Planetary Computer MOD17A2H v061 (apply_rebuilt_mod17_2022.py)"
        z["metadata_json"] = np.array(json.dumps(meta))
        np.savez(cache_out / f.name, **z)
        report[land] = {"months_replaced": 12, "pixels": int(values.shape[1])}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

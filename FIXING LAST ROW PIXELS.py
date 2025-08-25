"""
fix_geotransform.py
───────────────────
Correct the pixel size in the affine transform of every GeoTIFF in a folder.

Requires:  rasterio  →  pip install rasterio
"""

import pathlib
import rasterio
from rasterio.transform import Affine

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_DIR   = r"C:\Users\LOQ\OneDrive\Desktop\GOSIF_Tonzi_Clipped CORRECT"      # <<– change me
RECURSIVE   = False       # True  → walk sub‑folders
SUFFIX      = "_fixed"    # outfile = infile + SUFFIX + ".tif"
OVERWRITE   = False       # True  → allow overwriting existing *_fixed.tif
# ─────────────────────────────────────────────────────────────────────────────


def correct_transform(src_path: pathlib.Path) -> None:
    out_path = src_path.with_name(src_path.stem + SUFFIX + src_path.suffix)
    if out_path.exists() and not OVERWRITE:
        print(f"⚠  {out_path.name} exists — skipping")
        return

    with rasterio.open(src_path) as src:
        bounds = src.bounds
        width, height = src.width, src.height

        # Compute the TRUE pixel size from the bounds and shape
        new_dx = (bounds.right  - bounds.left)  / width
        new_dy = (bounds.top    - bounds.bottom) / height    # positive
        new_transform = Affine(new_dx, 0, bounds.left,
                               0, -new_dy, bounds.top)

        # Keep everything else identical
        profile = src.profile.copy()
        profile.update(transform=new_transform)

        data = src.read()           # read ALL bands

    # Write out corrected raster
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    print(f"✓ {src_path.name} → {out_path.name}")


def main():
    root = pathlib.Path(INPUT_DIR)
    pattern = "**/*.tif" if RECURSIVE else "*.tif"
    tifs = sorted(root.glob(pattern))
    if not tifs:
        print("No .tif files found in", root)
        return

    print(f"Fixing {len(tifs)} raster(s)…")
    for tif in tifs:
        try:
            correct_transform(tif)
        except Exception as exc:
            print(f"‼  {tif.name}: {exc}")


if __name__ == "__main__":
    main()

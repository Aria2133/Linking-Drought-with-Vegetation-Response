"""Rebuild monthly MOD17A2H GPP (California, 5 km, kg C m-2 month-1) from Microsoft Planetary Computer, without Earth Engine.

Replicates gee/scripts/modis_mod17a2h_gpp_monthly_5km_2022_2024.js: Terra MOD17A2H v061 'Gpp' x 0.0001, sum of the
8-day composites whose start date falls in the month, nearest-neighbour sampling at the centres of the exported 5 km
EPSG:4326 grid, clipped to California. The grid and the California mask are taken from an existing Earth Engine export
(--template), so outputs align pixel-for-pixel with the 2023-2024 files.

Two choices are not visible in the GEE script and are therefore kept as variants and settled empirically against a
valid Earth Engine year (`validate`): which processing version of each granule (Planetary Computer lists the original
and a 2024 reprocessing for many dates) and whether fill values (> 32700) are masked.

Usage:
  sample   --year Y --template <GEE tif> --out <dir>          read granules remotely, store the sampled raw values (npz)
  validate --year Y --samples <dir> --gee-folder <dir>        compare every variant with the GEE export of year Y
  build    --year Y --samples <dir> --template <GEE tif> --version latest|original --mask-fill yes|no --out <dir>
"""
import argparse
import calendar
import json
import re
import sys
import time
import warnings
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}"  # token for the blob container
CA_BBOX = [-124.6, 32.4, -114.0, 42.1]
PATTERN = "GPP_MOD17A2H_CA_5km_{year}_{month:02d}.tif"
# MODIS sinusoidal grid (sphere R = 6371007.181 m)
R = 6371007.181
X0, Y0, TILE = -20015109.354, 10007554.677, 1111950.5197665
FILL_MIN = 32700
LEVELS = 6  # block sizes 1, 2, 4, 8, 16, 32 native pixels  # valid Gpp range is 0..30000; 32761..32767 are fill/land-cover classes


def post_json(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def search(year):
    items, body = [], {"collections": ["modis-17A2H-061"], "bbox": CA_BBOX,
                       "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z", "limit": 250}
    page = post_json(STAC, body)
    while True:
        items += page["features"]
        nxt = [l for l in page.get("links", []) if l.get("rel") == "next"]
        if not nxt:
            break
        page = post_json(nxt[0]["href"], nxt[0].get("body", body)) if nxt[0].get("method") == "POST" else \
            json.load(urllib.request.urlopen(nxt[0]["href"], timeout=120))
    rows = []
    for it in items:
        m = re.match(r"MOD17A2H\.A(\d{7})\.h(\d\d)v(\d\d)\.061\.(\d{13})$", it["id"])
        if not m:  # skips Aqua (MYD17A2H)
            continue
        start = pd.Timestamp(it["properties"]["start_datetime"][:10])
        if start.year != year:
            continue
        rows.append({"id": it["id"], "start": start, "h": int(m[2]), "v": int(m[3]), "production": m[4],
                     "href": it["assets"]["Gpp_500m"]["href"]})
    df = pd.DataFrame(rows).sort_values(["start", "h", "v", "production"])
    df["version_rank"] = df.groupby(["start", "h", "v"]).production.rank(method="first")
    df["n_versions"] = df.groupby(["start", "h", "v"]).production.transform("size")
    return df


def template_points(template):
    import rasterio

    with rasterio.open(template) as src:
        a = src.read(1)
        tr = src.transform
        meta = src.profile
    rows, cols = np.nonzero(np.isfinite(a))
    lon = tr.c + (cols + 0.5) * tr.a
    lat = tr.f + (rows + 0.5) * tr.e
    x = R * np.radians(lon) * np.cos(np.radians(lat))
    y = R * np.radians(lat)
    h = np.floor((x - X0) / TILE).astype(int)
    v = np.floor((Y0 - y) / TILE).astype(int)
    pix = TILE / 2400
    col = np.floor((x - (X0 + h * TILE)) / pix).astype(int)
    row = np.floor(((Y0 - v * TILE) - y) / pix).astype(int)
    return {"rows": rows, "cols": cols, "h": h, "v": v, "prow": row, "pcol": col, "shape": a.shape, "meta": meta}


def sample(args):
    import rasterio

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pts = template_points(args.template)
    items = search(args.year)
    items.to_csv(out / f"granules_{args.year}.csv", index=False)
    need = set(zip(pts["h"], pts["v"]))
    missing = need - set(zip(items.h, items.v))
    assert not missing, f"tiles without granules: {missing}"
    m = re.match(r"https://([^.]+)\.blob\.core\.windows\.net/([^/]+)/", items.href.iloc[0])
    token = json.load(urllib.request.urlopen(SAS.format(account=m[1], container=m[2]), timeout=60))["token"]
    done = {p.name[:-len(".levels.npy")] for p in out.glob("*.levels.npy")}
    todo = items[items.id.map(lambda i: i not in done) & [(h, v) in need for h, v in zip(items.h, items.v)]]
    if args.all_versions_month:  # every version for this month (version check), latest version only elsewhere
        keep = (todo.start.dt.month == args.all_versions_month) | (todo.version_rank == todo.n_versions)
        todo = todo[keep].assign(first=lambda t: t.start.dt.month != args.all_versions_month)
        todo = todo.sort_values(["first", "start"])
    todo = list(todo.itertuples())

    def fetch(it):
        # one GET per granule (a few MB) is much faster than GDAL range reads over the network
        for attempt in range(6):
            try:
                with urllib.request.urlopen(it.href + "?" + token, timeout=300) as r:
                    data = r.read()
                break
            except OSError as e:  # dropped connections are common on slow links; retry with back-off
                if attempt == 5:
                    print(f"failed {it.id}: {e}", flush=True)
                    return
                time.sleep(10 * (attempt + 1))
        with rasterio.MemoryFile(data) as mem, mem.open() as src:
            band = src.read(1)
        sel = (pts["h"] == it.h) & (pts["v"] == it.v)
        vals = np.full(len(pts["h"]), -1, dtype=np.int32)  # -1 = point not in this tile
        vals[sel] = band[pts["prow"][sel], pts["pcol"][sel]]
        np.save(out / f"{it.id}.npy", vals)
        # Earth Engine reads coarse requests from its image pyramid (block means of unmasked pixels), so store the
        # value of the 2^k x 2^k block containing each point for k = 0..LEVELS-1 (fill values masked)
        g = np.where(band < FILL_MIN, band.astype("float32"), np.nan)
        lev = np.full((LEVELS, len(pts["h"])), -1, dtype="float32")
        for k in range(LEVELS):
            b = 2 ** k
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                blk = np.nanmean(g.reshape(2400 // b, b, 2400 // b, b), axis=(1, 3))
            lev[k, sel] = blk[pts["prow"][sel] // b, pts["pcol"][sel] // b]
        np.save(out / f"{it.id}.levels.npy", lev)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as pool:
        for k, _ in enumerate(pool.map(fetch, todo), 1):
            if k % 20 == 0:
                print(f"{k}/{len(todo)} granules", flush=True)
    print("done", args.year)


def monthly(items, samples_dir, year, version, mask_fill, level=None):
    """Return {month: 1-D array over template points} for one variant (level = pyramid level, None = nearest pixel)."""
    pick = items[items.version_rank == (items.n_versions if version == "latest" else 1)]
    res = {}
    for m in range(1, 13):
        sub = pick[pick.start.dt.month == m]
        total, valid = None, None
        for start, grp in sub.groupby("start"):
            comp = None
            for it in grp.itertuples():
                if level is None:
                    v = np.load(Path(samples_dir) / f"{it.id}.npy").astype("float64")
                else:
                    v = np.load(Path(samples_dir) / f"{it.id}.levels.npy")[level].astype("float64")
                    v = np.where(np.isnan(v), FILL_MIN, v)  # no valid pixel in the block -> treated as fill
                comp = v if comp is None else np.where(v >= 0, v, comp)
            ok = comp >= 0
            if mask_fill or level is not None:
                ok &= comp < FILL_MIN
            g = np.where(ok, comp * 0.0001, 0.0)
            total = g if total is None else total + g
            valid = ok if valid is None else valid | ok
        res[m] = np.where(valid, total, np.nan)
    return res


def load_items(samples_dir, year):
    df = pd.read_csv(Path(samples_dir) / f"granules_{year}.csv", parse_dates=["start"])
    have = {p.name[:-len(".levels.npy")] for p in Path(samples_dir).glob("*.levels.npy")}
    df = df[df.id.isin(have)]
    df["version_rank"] = df.groupby(["start", "h", "v"]).production.rank(method="first")
    df["n_versions"] = df.groupby(["start", "h", "v"]).production.transform("size")
    return df


def validate(args):
    import rasterio

    items = load_items(args.samples, args.year)
    template = Path(args.gee_folder) / PATTERN.format(year=args.year, month=1)
    pts = template_points(template)
    rows = []
    for version in ("latest", "original"):
        for mask_fill in (True, False):
            res = monthly(items, args.samples, args.year, version, mask_fill)
            for m in range(1, 13):
                with rasterio.open(Path(args.gee_folder) / PATTERN.format(year=args.year, month=m)) as src:
                    gee = src.read(1)[pts["rows"], pts["cols"]]
                ours = res[m]
                both = np.isfinite(gee) & np.isfinite(ours)
                d = ours[both] - gee[both]
                rows.append({"version": version, "mask_fill": mask_fill, "month": m, "n_compared": int(both.sum()),
                             "n_nan_mismatch": int((np.isfinite(gee) != np.isfinite(ours)).sum()),
                             "frac_equal_1e-6": float(np.mean(np.abs(d) < 1e-6)), "max_abs_diff": float(np.abs(d).max()),
                             "mean_diff": float(d.mean()), "r": float(np.corrcoef(ours[both], gee[both])[0, 1]),
                             "gee_mean": float(gee[both].mean()), "pc_mean": float(ours[both].mean())})
    rep = pd.DataFrame(rows)
    rep.to_csv(Path(args.samples) / f"validation_vs_gee_{args.year}.csv", index=False)
    summ = rep.groupby(["version", "mask_fill"]).agg(min_frac_equal=("frac_equal_1e-6", "min"),
                                                     mean_frac_equal=("frac_equal_1e-6", "mean"),
                                                     max_abs_diff=("max_abs_diff", "max"), min_r=("r", "min"),
                                                     nan_mismatch=("n_nan_mismatch", "sum"))
    print(summ.to_string())


def build(args):
    import rasterio

    items = load_items(args.samples, args.year)
    items = items[(items.version_rank == items.n_versions) | (args.version == 'original')]
    pts = template_points(args.template)
    res = monthly(items, args.samples, args.year, args.version, args.mask_fill == "yes", args.level)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(pts["meta"], dtype="float64")
    for m, vals in res.items():
        a = np.full(pts["shape"], np.nan)
        a[pts["rows"], pts["cols"]] = vals
        with rasterio.open(out / PATTERN.format(year=args.year, month=m), "w", **meta) as dst:
            dst.write(a, 1)
            dst.update_tags(source="MODIS/061/MOD17A2H via Microsoft Planetary Computer", units="kg C m-2 month-1",
                            version=args.version, mask_fill=args.mask_fill, pyramid_block=str(2 ** args.level),
                            n_composites=str(items[items.start.dt.month == m].start.nunique()))
    print("written", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--year", type=int, required=True)
    s.add_argument("--template", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--all-versions-month", type=int, help="download every processing version for this month (first)")
    v = sub.add_parser("validate")
    v.add_argument("--year", type=int, required=True)
    v.add_argument("--samples", required=True)
    v.add_argument("--gee-folder", required=True)
    b = sub.add_parser("build")
    b.add_argument("--year", type=int, required=True)
    b.add_argument("--samples", required=True)
    b.add_argument("--template", required=True)
    b.add_argument("--version", choices=["latest", "original"], required=True)
    b.add_argument("--mask-fill", choices=["yes", "no"], required=True)
    b.add_argument("--level", type=int, default=3, help="block of 2^level native pixels (3 = 8x8, best match to GEE)")
    b.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"sample": sample, "validate": validate, "build": build}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

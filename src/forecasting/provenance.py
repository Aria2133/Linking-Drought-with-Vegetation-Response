"""Provenance checks for forecasting results (pure pandas; no model code).

The registry `config/forecast_result_registry.csv` names, for every model x target, the saved metric table and
row filter that supply the main common benchmark, plus the sensitivity and superseded experiments that must
never enter it. Downstream aggregators call these checks before building figures, tables or statistics.

Metric convention (unchanged from the published workflow): each saved row is one model x target x land type x
fold x seed run, with R2 and nRMSE computed on all valid pixel-month predictions of that run pooled together
(standardized calendar-month z space). A task cell is the mean over its fold x seed rows. Because every fold of a
cell has the same seeds, this equals the mean over folds of the per-fold seed means (checked). Model and target
summaries average the cells with equal weight per task, so tasks are never weighted by their number of runs.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROLES = {"main", "sensitivity", "superseded"}
MAIN_LOOKBACK = 12
MAIN_PREPROCESSING = {"reviewer_mode_b", "reviewer_mode_b_no_detrend"}
MAIN_SCALING = {"monthwise_z"}
FOLDS = [1, 2, 3, 4]
KEY = ["model", "variable", "land_type", "fold", "seed"]
REQUIRED = KEY + ["lookback", "preprocessing", "scaling", "status", "r2", "normalized_rmse"]
MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
TARGETS = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
LAND_TYPES = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]


class ProvenanceError(ValueError):
    """Raised when results do not satisfy the documented main-benchmark design."""


def _split(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return []
    return [v.strip() for v in str(value).split(";") if v.strip()]


def load_registry(path) -> pd.DataFrame:
    reg = pd.read_csv(path, dtype=str, keep_default_na=False)
    bad = set(reg["role"]) - ROLES
    if bad:
        raise ProvenanceError(f"unknown roles in registry: {bad}")
    if reg["configuration_id"].duplicated().any():
        raise ProvenanceError("duplicate configuration_id in registry")
    return reg


def main_entries(registry: pd.DataFrame) -> pd.DataFrame:
    main = registry[registry["role"] == "main"]
    owner: Dict[tuple, str] = {}
    for _, e in main.iterrows():
        for m in _split(e["models"]):
            for t in _split(e["targets"]):
                if (m, t) in owner:
                    raise ProvenanceError(f"{m} x {t} has two main configurations: {owner[(m, t)]}, {e['configuration_id']}")
                owner[(m, t)] = e["configuration_id"]
    missing = [(m, t) for m in MODELS for t in TARGETS if (m, t) not in owner]
    if missing:
        raise ProvenanceError(f"registry has no main configuration for {missing}")
    return main


def select_rows(table: pd.DataFrame, entry: pd.Series) -> pd.DataFrame:
    """Rows of a saved metric table that belong to one registry entry, with provenance columns attached."""
    missing = [c for c in REQUIRED if c not in table.columns]
    if missing:
        raise ProvenanceError(f"{entry['configuration_id']}: source table lacks required columns {missing}")
    rows = table[table["model"].isin(_split(entry["models"])) & table["variable"].isin(_split(entry["targets"]))]
    if str(entry.get("row_filter", "")).strip():
        rows = rows.query(entry["row_filter"])
    rows = rows.copy()
    rows["configuration_id"] = entry["configuration_id"]
    rows["role"] = entry["role"]
    rows["figure7_label"] = entry.get("figure7_label", "")
    return rows


def validate_entry_rows(rows: pd.DataFrame, entry: pd.Series) -> pd.DataFrame:
    """Design checks for one entry: lookback, preprocessing, scaling, duplicates and complete fold x seed grids."""
    cid = entry["configuration_id"]
    if rows.empty:
        raise ProvenanceError(f"{cid}: no rows selected")
    checks = {
        "lookback": {int(entry["expected_lookback"])},
        "preprocessing": {entry["expected_preprocessing"]},
        "scaling": {entry["expected_scaling"]},
        "status": {"success"},
    }
    for col, allowed in checks.items():
        found = set(rows[col].astype(int) if col == "lookback" else rows[col].astype(str))
        if not found <= allowed:
            raise ProvenanceError(f"{cid}: unexpected {col} values {sorted(found - allowed)}")
    dup = rows.duplicated(KEY, keep=False)
    if dup.any():
        raise ProvenanceError(f"{cid}: duplicate task/fold/seed records:\n{rows.loc[dup, KEY].head()}")
    seeds = {int(s) for s in _split(entry["expected_seeds"])}
    for (m, v, lt), g in rows.groupby(["model", "variable", "land_type"]):
        grid = {(int(f), int(s)) for f, s in zip(g["fold"], g["seed"])}
        expected = {(f, s) for f in FOLDS for s in seeds} if seeds else grid
        if grid != expected:
            raise ProvenanceError(f"{cid}: {m} x {v} x {lt} fold/seed grid {sorted(grid)} != expected {sorted(expected)}")
    combos = {(m, v, lt) for m, v, lt in zip(rows["model"], rows["variable"], rows["land_type"])}
    expected_combos = {(m, v, lt) for m in _split(entry["models"]) for v in _split(entry["targets"]) for lt in LAND_TYPES}
    if combos != expected_combos:
        raise ProvenanceError(f"{cid}: missing tasks {sorted(expected_combos - combos)}")
    return rows


def assert_main_design(rows: pd.DataFrame) -> None:
    """Main-benchmark rows: 12-month lookback, calendar-month z scaling, main roles only, one configuration per task."""
    if (rows["role"] != "main").any():
        bad = rows.loc[rows["role"] != "main", ["configuration_id", "model", "variable"]].drop_duplicates()
        raise ProvenanceError(f"non-main experiments in main-benchmark aggregation:\n{bad}")
    if set(rows["lookback"].astype(int)) != {MAIN_LOOKBACK}:
        raise ProvenanceError(f"main benchmark must use lookback {MAIN_LOOKBACK}; found {sorted(set(rows['lookback']))}")
    if not set(rows["preprocessing"]) <= MAIN_PREPROCESSING or not set(rows["scaling"]) <= MAIN_SCALING:
        raise ProvenanceError("main benchmark mixes preprocessing or scaling variants")
    per_task = rows.groupby(["model", "variable", "land_type"])["configuration_id"].nunique()
    if (per_task > 1).any():
        raise ProvenanceError(f"tasks mixing configurations: {per_task[per_task > 1].index.tolist()}")
    if rows.duplicated(KEY).any():
        raise ProvenanceError("duplicate task/fold/seed records across configurations")


def aggregate_cells(rows: pd.DataFrame) -> pd.DataFrame:
    """Task cells: mean and SD (ddof=1) over fold x seed rows; verifies the fold-balanced equivalence."""
    out = []
    for (m, v, lt), g in rows.groupby(["model", "variable", "land_type"]):
        per_fold = g.groupby("fold")["seed"].nunique()
        if per_fold.nunique() != 1 or sorted(per_fold.index.astype(int)) != FOLDS:
            raise ProvenanceError(f"{m} x {v} x {lt}: unbalanced or incomplete folds {per_fold.to_dict()}")
        fold_mean_r2 = g.groupby("fold")["r2"].mean().mean()
        if not np.isclose(fold_mean_r2, g["r2"].mean(), rtol=0, atol=1e-12):
            raise ProvenanceError("row mean differs from fold-balanced mean")
        out.append({
            "model": m, "variable": v, "land_type": lt,
            "mean_r2": g["r2"].mean(), "sd_r2": g["r2"].std(ddof=1),
            "mean_normalized_rmse": g["normalized_rmse"].mean(), "sd_normalized_rmse": g["normalized_rmse"].std(ddof=1),
            "available_folds": int(g["fold"].nunique()), "available_seeds": int(g["seed"].nunique()), "available_rows": len(g),
            "lookback": int(g["lookback"].iloc[0]), "preprocessing": g["preprocessing"].iloc[0], "scaling": g["scaling"].iloc[0],
            "configuration_id": g["configuration_id"].iloc[0], "role": g["role"].iloc[0], "figure7_label": g["figure7_label"].iloc[0],
        })
    return pd.DataFrame(out)


def check_task_table(table: pd.DataFrame, registry: pd.DataFrame, label_column: str = "selected_configuration",
                     allow_roles: Iterable[str] = ("main",)) -> None:
    """For downstream aggregators: every cell must carry a configuration label whose registry role is allowed."""
    if label_column not in table.columns:
        raise ProvenanceError(f"task table lacks provenance column {label_column!r}")
    roles = dict(zip(registry["figure7_label"], registry["role"]))
    roles.pop("", None)
    unknown = set(table[label_column]) - set(roles)
    if unknown:
        raise ProvenanceError(f"unregistered configurations in task table: {sorted(unknown)}")
    bad = {lab for lab in set(table[label_column]) if roles[lab] not in set(allow_roles)}
    if bad:
        raise ProvenanceError(f"task table contains non-main configurations {sorted(bad)}")
    keys = [c for c in ("land_type", "target", "model") if c in table.columns]
    if table.duplicated(keys).any():
        raise ProvenanceError("duplicate task cells")


def build_from_registry(registry: pd.DataFrame, project_root: Path, substitutions: Optional[Dict[str, str]] = None,
                        loader=pd.read_csv) -> pd.DataFrame:
    """Assemble validated raw rows for the main benchmark. `substitutions` maps a main configuration_id to a
    non-main one; it exists only to reproduce historical published variants and marks those rows explicitly."""
    substitutions = substitutions or {}
    cache: Dict[str, pd.DataFrame] = {}
    frames = []
    for _, entry in main_entries(registry).iterrows():
        use = entry
        if entry["configuration_id"] in substitutions:
            use = registry.set_index("configuration_id").loc[substitutions[entry["configuration_id"]]].copy()
            use["configuration_id"] = substitutions[entry["configuration_id"]]
        path = Path(project_root) / use["source_table"]
        if str(path) not in cache:
            if not path.is_file():
                raise ProvenanceError(f"missing source table {path}")
            cache[str(path)] = loader(path)
        frames.append(validate_entry_rows(select_rows(cache[str(path)], use), use))
    rows = pd.concat(frames, ignore_index=True)
    if not substitutions:
        assert_main_design(rows)
    elif rows.duplicated(KEY).any():
        raise ProvenanceError("duplicate task/fold/seed records across configurations")
    return rows

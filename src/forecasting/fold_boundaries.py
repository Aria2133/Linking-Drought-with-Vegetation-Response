"""Rolling-origin fold boundaries and preprocessing-fit windows (pure Python; no data or heavy imports).

Each expanding rolling-origin fold has:
  core training   : first available month ... train_end              (model weights are fitted here)
  internal valid. : validation_start ... validation_end (24 months)   (early stopping / SVR grid selection)
  outer test      : test_start ... test_end (24 months)               (scored once; never used for fitting)

Two preprocessing boundaries are supported. "Preprocessing" = calendar-month imputation, calendar-month linear
detrending and calendar-month z-scaling (or min-max) parameters.

  legacy_through_validation  (label "reviewer_mode_b")
      Preprocessing parameters are fitted on all dates <= validation_end, i.e. core training PLUS the internal
      validation window. This is how every published benchmark run (Figure 7, Tables 4-5, S8/S14 statistics)
      was produced. It does not use outer-test observations, but the internal-validation targets influence the
      imputation, detrending and scaling applied to the training inputs (internal-validation contamination).

  fit_subset  (label "reviewer_mode_b_fitsubset")  -- default for new runs
      Preprocessing parameters are fitted on dates <= train_end only and applied unchanged to validation and
      test inputs. Gaps inside the validation/test windows are not imputed (they are masked, as test gaps
      always were). Where a final refit on training + validation is performed (RBF-SVR), preprocessing is refitted
      on dates <= validation_end for that refit only (`refit_preprocessing_fit_end`); the outer test stays untouched.
"""
from typing import Dict, Iterable, List

LEGACY = "legacy_through_validation"
FIT_SUBSET = "fit_subset"
BOUNDARIES = {
    LEGACY: "reviewer_mode_b",
    FIT_SUBSET: "reviewer_mode_b_fitsubset",
}
VALIDATION_MONTHS = 24


def previous_month(value: int, count: int = 1) -> int:
    year, month = divmod(int(value), 100)
    absolute = year * 12 + month - 1 - count
    return (absolute // 12) * 100 + (absolute % 12) + 1


def check_spec(spec: Dict) -> Dict:
    """Raise if a fold specification lets preprocessing or fitting reach the outer test window."""
    tr, vs, ve, ts, te = (int(spec[k]) for k in ("train_end", "validation_start", "validation_end", "test_start", "test_end"))
    pf = int(spec["preprocessing_fit_end"])
    rf = int(spec.get("refit_preprocessing_fit_end", pf))
    if not (tr < vs <= ve < ts <= te):
        raise ValueError(f"fold windows out of order: {spec}")
    if pf >= ts or rf >= ts:
        raise ValueError(f"preprocessing fit window reaches the outer test period: {spec}")
    if spec.get("preprocessing_boundary") == FIT_SUBSET and pf > tr:
        raise ValueError(f"fit_subset boundary but preprocessing is fitted beyond train_end: {spec}")
    return spec


def rolling_fold_specs(rolling_folds: Iterable[Dict], boundary: str, mode_name: str = "Mode_B_rolling_origin") -> List[Dict]:
    """Expand candidate_config.ROLLING_FOLDS into fold specifications for the chosen preprocessing boundary."""
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown preprocessing boundary {boundary!r}; choose one of {sorted(BOUNDARIES)}")
    specs = []
    for fold in rolling_folds:
        validation_end = int(fold["train_end"])
        validation_start = previous_month(validation_end, VALIDATION_MONTHS - 1)
        train_end = previous_month(validation_start, 1)
        spec = {
            "mode": mode_name,
            "fold": int(fold["fold"]),
            "train_end": train_end,
            "preprocessing_fit_end": validation_end if boundary == LEGACY else train_end,
            "refit_preprocessing_fit_end": validation_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "test_start": int(fold["test_start"]),
            "test_end": int(fold["test_end"]),
            "preprocessing_mode": BOUNDARIES[boundary],
            "preprocessing_boundary": boundary,
        }
        specs.append(check_spec(spec))
    return specs


def window_usage(spec: Dict) -> Dict[str, str]:
    """Human-readable statement of which months each operation may use (for reports and tests)."""
    pf, rf = spec["preprocessing_fit_end"], spec["refit_preprocessing_fit_end"]
    return {
        "model_weight_fit": f"<= {spec['train_end']}",
        "early_stopping_or_selection": f"{spec['validation_start']}-{spec['validation_end']}",
        "imputation_detrending_scaling_fit": f"<= {pf}",
        "refit_preprocessing_fit (SVR only)": f"<= {rf}",
        "outer_test_scoring": f"{spec['test_start']}-{spec['test_end']} (never fitted)",
    }

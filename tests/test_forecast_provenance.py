"""Tiny-fixture tests of the main-benchmark provenance checks."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "forecasting"))
from provenance import (  # noqa: E402
    LAND_TYPES, ProvenanceError, aggregate_cells, assert_main_design, build_from_registry, check_task_table,
    load_registry, main_entries, validate_entry_rows,
)

ENTRY = pd.Series({"configuration_id": "toy_main", "role": "main", "figure7_label": "common_reviewer_safe", "models": "LSTM",
                   "targets": "TWS", "expected_lookback": "12", "expected_preprocessing": "reviewer_mode_b",
                   "expected_scaling": "monthwise_z", "expected_seeds": "42;123"})


def toy_rows(seeds=(42, 123), lookback=12, role="main", configuration_id="toy_main"):
    rows = []
    for lt in LAND_TYPES:
        for fold in (1, 2, 3, 4):
            for s in seeds:
                rows.append({"model": "LSTM", "variable": "TWS", "land_type": lt, "fold": fold, "seed": s,
                             "lookback": lookback, "preprocessing": "reviewer_mode_b", "scaling": "monthwise_z",
                             "status": "success", "r2": 0.1 * fold + 0.01 * (s % 7), "normalized_rmse": 1 - 0.1 * fold,
                             "configuration_id": configuration_id, "role": role, "figure7_label": "x"})
    return pd.DataFrame(rows)


def test_valid_rows_pass_and_aggregate():
    rows = validate_entry_rows(toy_rows(), ENTRY)
    cells = aggregate_cells(rows)
    assert len(cells) == 4 and set(cells["available_rows"]) == {8} and set(cells["available_seeds"]) == {2}
    g = rows[rows.land_type == LAND_TYPES[0]]
    assert np.isclose(cells.loc[0, "mean_r2"], g["r2"].mean())
    assert np.isclose(cells.loc[0, "sd_r2"], g["r2"].std(ddof=1))


def test_wrong_lookback_rejected():
    with pytest.raises(ProvenanceError, match="lookback"):
        validate_entry_rows(toy_rows(lookback=18), ENTRY)


def test_duplicate_records_rejected():
    rows = toy_rows()
    with pytest.raises(ProvenanceError, match="duplicate"):
        validate_entry_rows(pd.concat([rows, rows.iloc[[0]]]), ENTRY)


def test_missing_fold_or_seed_rejected():
    rows = toy_rows()
    with pytest.raises(ProvenanceError, match="grid"):
        validate_entry_rows(rows.drop(index=3), ENTRY)


def test_unbalanced_seeds_rejected_in_aggregation():
    rows = toy_rows().drop(index=1)  # fold 1 of one task loses a seed
    with pytest.raises(ProvenanceError, match="unbalanced"):
        aggregate_cells(rows)


def test_sensitivity_cannot_enter_main_aggregation():
    rows = pd.concat([toy_rows(), toy_rows(seeds=(7,), lookback=18, role="sensitivity", configuration_id="toy_sens")])
    with pytest.raises(ProvenanceError):
        assert_main_design(rows)


def test_registry_has_one_main_configuration_per_model_target():
    reg = load_registry(REPO / "config" / "forecast_result_registry.csv")
    main = main_entries(reg)
    assert set(main["expected_lookback"]) == {"12"}
    assert set(reg.loc[reg.configuration_id.str.startswith("sens_"), "role"]) == {"sensitivity"}


def test_downstream_table_gate():
    reg = load_registry(REPO / "config" / "forecast_result_registry.csv")
    table = pd.read_csv(REPO / "results" / "forecasting" / "figure7_final_task_metrics.csv")
    check_task_table(table, reg)  # current results table: main configurations only
    bad = table.copy()
    bad.loc[0, "selected_configuration"] = "bilstm_special_tws_state_space"
    with pytest.raises(ProvenanceError, match="non-main"):
        check_task_table(bad, reg)
    with pytest.raises(ProvenanceError, match="unregistered"):
        check_task_table(bad.assign(selected_configuration="mystery_run"), reg)


PROJECT_ROOT = os.environ.get("DROUGHT_PROJECT_ROOT")


@pytest.mark.skipif(not PROJECT_ROOT or not Path(PROJECT_ROOT, "Docs").is_dir(), reason="project data folder not available")
def test_published_and_main_variants_rebuild_exactly():
    """Lineage check on the real saved runs (only when DROUGHT_PROJECT_ROOT points at the project folder)."""
    reg = load_registry(REPO / "config" / "forecast_result_registry.csv")
    main = aggregate_cells(build_from_registry(reg, Path(PROJECT_ROOT)))
    ref = pd.read_csv(REPO / "results" / "forecasting" / "figure7_final_task_metrics.csv")
    assert len(main) == 216 and set(main["lookback"]) == {12}
    # The registry points at the saved pre-correction runs; results/ holds the GPP-2022-corrected cells. Non-GPP cells must
    # match exactly; the rebuilt GPP cells must equal the "before" values recorded by the correction.
    non_gpp = main[main.variable != "GPP"].groupby("model")["mean_r2"].mean()
    ref_non_gpp = ref[ref.target != "GPP"].groupby("model")["mean_r2"].mean()
    assert np.allclose(non_gpp.sort_index(), ref_non_gpp.sort_index(), atol=1e-12)
    before = pd.read_csv(REPO / "results" / "gpp_2022_correction" / "figure7_gpp_cells_old_vs_corrected.csv")
    gpp = main[main.variable == "GPP"].groupby("model")["mean_r2"].mean().sort_index()
    assert np.allclose(gpp, before.groupby("model")["mean_r2"].mean().sort_index(), atol=1e-12)
    pub = aggregate_cells(build_from_registry(reg, Path(PROJECT_ROOT), substitutions={"main_reviewer_safe_bilstm_tws": "sens_tws_bilstm_18m_state_space"}))
    tws = pub[(pub.model == "BiLSTM") & (pub.variable == "TWS")]
    assert set(tws["lookback"]) == {18}

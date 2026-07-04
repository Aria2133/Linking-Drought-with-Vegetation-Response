from __future__ import annotations

import csv
import hashlib
import math
import re
import textwrap
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd


DESKTOP = Path(r"C:\Users\LOQ\OneDrive\Desktop")
OUT = DESKTOP / "major_revision_analyses" / "FINAL_section4_results_technical_audit"
TABLES = OUT / "tables"
REPORT = OUT / "report"
EXTRACT = OUT / "manuscript_extract"

MANUSCRIPT = DESKTOP / "Document" / "Paper Draft after addressing comments_AA.docx"

MAIN_FIG_DIR = DESKTOP / "ECOINF_revision_final_figures_package" / "1_paper_main_figures"
SOURCE_TABLE_DIR = DESKTOP / "ECOINF_revision_final_figures_package" / "4_source_tables_used"
FINAL_GNDVI_REBUILD = DESKTOP / "major_revision_analyses" / "final_previous_GNDVI_with_test_rebuild"
FORECAST_AUDIT = DESKTOP / "major_revision_analyses" / "FINAL_forecasting_framework_technical_audit"
DL_GNDVI_RERUN = DESKTOP / "DL_GNDVI_previous_noQA_rerun"
FINAL_PAPER_CORRECTED = DESKTOP / "major_revision_analyses" / "final_paper_figures_corrected"
SUBMISSION_AUDIT = DESKTOP / "major_revision_analyses" / "FINAL_submission_figure_directory_audit"


def ensure_dirs() -> None:
    for d in [OUT, TABLES, REPORT, EXTRACT]:
        d.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_status(path: Path, role: str) -> dict:
    row = {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": "",
        "modified_time": "",
        "sha256": "",
    }
    if path.exists():
        st = path.stat()
        row.update(
            {
                "size_bytes": st.st_size,
                "modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(sep=" ", timespec="seconds"),
                "sha256": sha256(path),
            }
        )
    return row


def docx_paragraphs(path: Path) -> list[str]:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    paragraphs: list[str] = []
    for p in root.findall(".//w:p", ns):
        parts: list[str] = []
        for node in p.iter():
            tag = node.tag.split("}")[-1]
            if tag == "t" and node.text:
                parts.append(node.text)
            elif tag == "tab":
                parts.append("\t")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(re.sub(r"\s+", " ", text))
    return paragraphs


def extract_section4(paragraphs: list[str]) -> pd.DataFrame:
    start = None
    for i, p in enumerate(paragraphs):
        if re.match(r"^\s*4\.\s*Results\b", p, flags=re.I) or p.strip() == "4. Results":
            start = i
            break
    if start is None:
        raise RuntimeError("Could not find '4. Results' in manuscript")
    end = len(paragraphs)
    for i in range(start + 1, len(paragraphs)):
        if re.match(r"^\s*5(\.|\s)", paragraphs[i]):
            end = i
            break
    rows = []
    for i in range(start, end):
        p = paragraphs[i]
        kind = "body"
        if re.match(r"^\s*4\.\s*Results\b", p, flags=re.I) or re.match(r"^\s*4\.\d+\s+", p):
            kind = "heading"
        elif re.match(r"^\s*Table\s+\d+", p, flags=re.I):
            kind = "table_caption_or_row"
        elif re.match(r"^\s*Figure\s+\d+", p, flags=re.I):
            kind = "figure_caption"
        rows.append({"paragraph_index": i, "kind": kind, "text": p})
    return pd.DataFrame(rows)


def risk_scan(section_df: pd.DataFrame) -> pd.DataFrame:
    phrases = [
        "six deep learning",
        "six deep-learning",
        "216 forecasting tasks",
        "2024-only",
        "held-out 2024",
        "BiLSTM",
        "34/36",
        "94%",
        "TCN",
        "36%",
        "raw RMSE",
        " RMSE",
        "six deep learning architectures",
        "2-17-fold",
        "17-fold",
        "fold",
        "resilience paradox",
        "bimodal split",
        "bifurcation",
        "collapse zones",
        "collapse",
        "stress trap",
        "stress traps",
        "regime shift",
        "causal",
        "carbon",
        "intervention",
        "dominance",
        "dominated",
    ]
    rows = []
    for _, row in section_df.iterrows():
        text = row["text"]
        low = text.lower()
        for phrase in phrases:
            if phrase.lower() in low:
                rows.append(
                    {
                        "paragraph_index": row["paragraph_index"],
                        "phrase": phrase,
                        "context": text[:500],
                        "risk": classify_risk(phrase),
                    }
                )
    return pd.DataFrame(rows)


def classify_risk(phrase: str) -> str:
    p = phrase.lower()
    if "bilstm" in p or "34/36" in p or "94" in p or "domin" in p:
        return "Final forecasting counts contradict old dominance wording; verify or replace."
    if "rmse" in p:
        return "Final Figure 7/S6 use normalized RMSE/nRMSE, not raw RMSE."
    if "six deep" in p or "216" in p:
        return "XGBoost is non-deep ML; final benchmark is 36 land-type x target tasks across six learned models."
    if "2024" in p:
        return "Final benchmark uses four rolling-origin test periods, not 2024-only evidence."
    if "fold" in p or "collapse" in p or "bifurcation" in p or "stress trap" in p or "regime" in p:
        return "Reviewer-safe wording should avoid fold-change overclaiming or mechanistic regime language."
    return "Needs evidence check against final source files."


def unique_pair_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["variable_1"] != df["variable_2"]].copy()
    d["pair_key"] = d.apply(lambda r: "–".join(sorted([r["variable_1"], r["variable_2"]])), axis=1)
    # Symmetric matrices may contain both directions. Correlations should match; keep the first.
    return d.sort_values(["land_type", "pair_key"]).drop_duplicates(["land_type", "pair_key"])


def pair_range(d: pd.DataFrame, a: str, b: str) -> dict:
    key = "–".join(sorted([a, b]))
    sub = d[d["pair_key"] == key]
    if sub.empty:
        return {"pair": key, "min": math.nan, "max": math.nan, "land_type_min": "", "land_type_max": "", "n": 0}
    min_row = sub.loc[sub["correlation"].idxmin()]
    max_row = sub.loc[sub["correlation"].idxmax()]
    return {
        "pair": key,
        "min": float(sub["correlation"].min()),
        "max": float(sub["correlation"].max()),
        "land_type_min": min_row["land_type"],
        "land_type_max": max_row["land_type"],
        "n": len(sub),
    }


def load_figure5_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = FINAL_GNDVI_REBUILD / "tables" / "Figure_5_previous_GNDVI_with_test_values.csv"
    df = pd.read_csv(path)
    u = unique_pair_df(df)
    veg = ["GNDVI", "GOSIF", "GPP", "NIRv", "NDVI"]
    veg_pairs = u[u["variable_1"].isin(veg) & u["variable_2"].isin(veg)]
    veg_non_gndvi = veg_pairs[(veg_pairs["variable_1"] != "GNDVI") & (veg_pairs["variable_2"] != "GNDVI")]
    vpd_veg = u[((u["variable_1"] == "VPD") & u["variable_2"].isin(veg)) | ((u["variable_2"] == "VPD") & u["variable_1"].isin(veg))]
    rows = [
        {
            "claim_topic": "All vegetation/productivity pairs including GNDVI",
            "min": veg_pairs["correlation"].min(),
            "max": veg_pairs["correlation"].max(),
            "n_landtype_pairs": len(veg_pairs),
            "note": "Includes GNDVI; broad high-correlation claim must account for lower GNDVI-NDVI/NIRv values.",
        },
        {
            "claim_topic": "Vegetation/productivity pairs excluding GNDVI",
            "min": veg_non_gndvi["correlation"].min(),
            "max": veg_non_gndvi["correlation"].max(),
            "n_landtype_pairs": len(veg_non_gndvi),
            "note": "Better represents non-GNDVI vegetation/productivity coherence.",
        },
        {
            "claim_topic": "VPD versus vegetation/productivity indicators",
            "min": vpd_veg["correlation"].min(),
            "max": vpd_veg["correlation"].max(),
            "n_landtype_pairs": len(vpd_veg),
            "note": "Check sign before claiming universally negative VPD relationships.",
        },
    ]
    for a, b in [("NIRv", "GPP"), ("NDVI", "GPP"), ("SPEI", "TWS"), ("LST", "GPP"), ("LST", "NIRv")]:
        r = pair_range(u, a, b)
        rows.append(
            {
                "claim_topic": r["pair"],
                "min": r["min"],
                "max": r["max"],
                "n_landtype_pairs": r["n"],
                "note": f"Min land type {r['land_type_min']}; max land type {r['land_type_max']}.",
            }
        )
    return df, pd.DataFrame(rows)


def load_figure6_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = FINAL_GNDVI_REBUILD / "tables" / "Figure_6_ACCEPTED_reference_previous_GNDVI_values_used_for_font_fix.csv"
    df = pd.read_csv(path)
    rows = []
    for (land, pred, target), g in df.groupby(["land_type", "predictor", "target"]):
        lag0 = g.loc[g["lag"] == 0, "ccf"]
        max_pos = g.loc[g["ccf"].idxmax()]
        max_abs = g.loc[g["ccf"].abs().idxmax()]
        rows.append(
            {
                "land_type": land,
                "predictor": pred,
                "target": target,
                "lag0_ccf": float(lag0.iloc[0]) if not lag0.empty else math.nan,
                "max_positive_ccf": float(max_pos["ccf"]),
                "lag_of_max_positive": int(max_pos["lag"]),
                "max_abs_ccf": float(max_abs["ccf"]),
                "lag_of_max_abs": int(max_abs["lag"]),
                "involves_TWS_or_SPEI": pred in ["TWS", "SPEI"] or target in ["TWS", "SPEI"],
                "involves_vegetation_productivity": pred in ["GNDVI", "GOSIF", "GPP", "NDVI", "NIRv"] and target in ["GNDVI", "GOSIF", "GPP", "NDVI", "NIRv"],
            }
        )
    return df, pd.DataFrame(rows)


def load_figure8_values() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(FINAL_PAPER_CORRECTED / "tables" / "figure_8_nirv_spatial_response_summary.csv")
    auth = pd.read_csv(SOURCE_TABLE_DIR / "stress_zone_authoritative_values_final.csv")
    return summary, auth


def load_figure9_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(SOURCE_TABLE_DIR / "figure_9_transition_probabilities_deseasonalized.csv")
    wanted = [
        ("Wet / Healthy", "Wet / Healthy", "Wet/Healthy persistence"),
        ("Wet / Suppressed", "Wet / Suppressed", "Wet/Suppressed persistence"),
        ("Dry / Resilient", "Dry / Resilient", "Dry/Resilient persistence"),
        ("Dry / Stressed", "Dry / Stressed", "Dry/Stressed persistence"),
        ("Wet / Healthy", "Dry / Stressed", "Wet/Healthy -> Dry/Stressed"),
        ("Dry / Stressed", "Wet / Healthy", "Dry/Stressed -> Wet/Healthy"),
    ]
    rows = []
    for land, g in df.groupby("land_type"):
        for f, t, label in wanted:
            sub = g[(g["from_state"] == f) & (g["to_state"] == t)]
            rows.append(
                {
                    "land_type": land,
                    "transition": label,
                    "probability_percent": float(sub["probability_percent"].iloc[0]) if not sub.empty else math.nan,
                    "source_count": int(sub["source_state_transition_count"].iloc[0]) if not sub.empty else "",
                }
            )
        off = g[g["from_state"] != g["to_state"]].sort_values("probability_percent", ascending=False).head(1)
        if not off.empty:
            r = off.iloc[0]
            rows.append(
                {
                    "land_type": land,
                    "transition": f"largest off-diagonal: {r['from_state']} -> {r['to_state']}",
                    "probability_percent": float(r["probability_percent"]),
                    "source_count": int(r["source_state_transition_count"]),
                }
            )
    return df, pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def fmt(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return "not verified"
    return f"{x:.{nd}f}"


def build_claim_table(section_df: pd.DataFrame, fig5_summary: pd.DataFrame, fig8_summary: pd.DataFrame, fig9_summary: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "current_manuscript_claim": "Vegetation and productivity indicators are highly correlated at r = 0.82-0.98 across all land-cover types.",
            "location_or_subsection": "4.1 paragraph after heading",
            "supported": "Partly / needs revision",
            "final_evidence_source": str(FINAL_GNDVI_REBUILD / "tables" / "Figure_5_previous_GNDVI_with_test_values.csv"),
            "required_action": "Replace broad range; all vegetation/productivity pairs including GNDVI have a wider/lower range.",
            "reviewer_safe_replacement_wording": "Non-GNDVI vegetation/productivity indicators were generally positively associated, while GNDVI relationships were weaker for some pairs and land types; report pair-specific ranges.",
        },
        {
            "current_manuscript_claim": "NIRv correlated more strongly than NDVI with GPP at r = 0.62-0.82 versus 0.55-0.73.",
            "location_or_subsection": "4.1",
            "supported": "Check final values; old range not directly supported",
            "final_evidence_source": str(FINAL_GNDVI_REBUILD / "tables" / "Figure_5_previous_GNDVI_with_test_values.csv"),
            "required_action": "Use final Figure 5 pair-specific values.",
            "reviewer_safe_replacement_wording": "Report the final NIRv-GPP and NDVI-GPP ranges rather than old ranges.",
        },
        {
            "current_manuscript_claim": "Six deep learning models were evaluated across 216 forecasting tasks.",
            "location_or_subsection": "4.2",
            "supported": "No",
            "final_evidence_source": str(FORECAST_AUDIT / "tables" / "forecasting_claim_verification.csv"),
            "required_action": "Remove.",
            "reviewer_safe_replacement_wording": "Five deep-learning models plus XGBoost were evaluated across 36 land-type x target tasks.",
        },
        {
            "current_manuscript_claim": "BiLSTM had best RMSE in 34/36 or 94% of tasks / model-family dominance.",
            "location_or_subsection": "4.2 and Table 2",
            "supported": "No / obsolete",
            "final_evidence_source": str(FORECAST_AUDIT / "tables" / "final_best_model_counts.csv"),
            "required_action": "Replace Table 2 and all dominance wording.",
            "reviewer_safe_replacement_wording": "XGBoost had the largest final best-count totals, but rankings were target-, land-type-, fold-, metric-, and scale-dependent.",
        },
        {
            "current_manuscript_claim": "Figure 7 reports RMSE.",
            "location_or_subsection": "4.2 / Figure 7 text",
            "supported": "No",
            "final_evidence_source": str(FORECAST_AUDIT / "tables" / "metric_definitions.csv"),
            "required_action": "Use nRMSE / normalized RMSE.",
            "reviewer_safe_replacement_wording": "Figure 7 reports R2 and nRMSE.",
        },
        {
            "current_manuscript_claim": "Severe stress expansion should be expressed as fold-change ratios such as 17x.",
            "location_or_subsection": "4.3 / Table 4 if present",
            "supported": "Unsafe / not recommended",
            "final_evidence_source": str(SOURCE_TABLE_DIR / "stress_zone_authoritative_values_final.csv"),
            "required_action": "Report absolute km2 and percentage-point changes with CIs.",
            "reviewer_safe_replacement_wording": "Severe-response area changed by X km2 and Y percentage points of eligible analyzed land area.",
        },
        {
            "current_manuscript_claim": "Natural Forests recovered strongly from -8.9% to -0.8%, while Grasslands fell from -2.3% to -2.6%.",
            "location_or_subsection": "4.3 paragraph 309",
            "supported": "No",
            "final_evidence_source": str(FINAL_PAPER_CORRECTED / "tables" / "figure_8_nirv_spatial_response_summary.csv"),
            "required_action": "Replace with final Figure 8 period means.",
            "reviewer_safe_replacement_wording": "At the 40% threshold source table, Natural Forest mean NIRv change was -0.87% in 2012-2016 and -8.83% in 2020-2024; Grasslands were -3.63% and -2.08%, respectively.",
        },
        {
            "current_manuscript_claim": "Natural Forest severe stress increased 17 times from 0.14% to 2.42%; Savannas increased 14 times; Grasslands 4.6 times.",
            "location_or_subsection": "4.3 paragraph 310 / Table 4",
            "supported": "No / superseded",
            "final_evidence_source": str(SOURCE_TABLE_DIR / "stress_zone_authoritative_values_final.csv"),
            "required_action": "Remove fold-change-first values and replace with final fractional-mask eligible-area values.",
            "reviewer_safe_replacement_wording": "At the 40% decline threshold, recent-minus-historic severe-area changes were 950.4 km2 for Grasslands, 1853.7 km2 for Natural Forests, 908.0 km2 for Savannas, and 292.5 km2 for Shrublands, with percentage-point changes and CIs reported.",
        },
        {
            "current_manuscript_claim": "Persistent hotspots had Moran's I = 0.28-0.34, p < 0.001.",
            "location_or_subsection": "4.3 paragraph 348",
            "supported": "Not verified in final Figure 8 source tables used in this audit",
            "final_evidence_source": str(SOURCE_TABLE_DIR / "stress_zone_authoritative_values_final.csv"),
            "required_action": "Remove unless a final Moran/hotspot source table is supplied and matches the final fractional-mask Figure 8 definition.",
            "reviewer_safe_replacement_wording": "If retained, cite a final hotspot table; otherwise restrict Figure 8 text to severe-response area, eligible-area percentages, and bootstrap uncertainty.",
        },
        {
            "current_manuscript_claim": "Ecosystem states indicate stress traps/regime shifts/bifurcation.",
            "location_or_subsection": "4.4 / Figure 9 interpretation",
            "supported": "No",
            "final_evidence_source": str(SOURCE_TABLE_DIR / "figure_9_transition_probabilities_deseasonalized.csv"),
            "required_action": "Use descriptive transition language.",
            "reviewer_safe_replacement_wording": "Transition probabilities are descriptive and classification-dependent; they do not establish mechanistic regime shifts.",
        },
        {
            "current_manuscript_claim": "Shrublands have maximum Dry/Stressed persistence of 74% and stress traps with 26% monthly recovery probability.",
            "location_or_subsection": "4.4 paragraph 351",
            "supported": "Partly / wording unsafe",
            "final_evidence_source": str(SOURCE_TABLE_DIR / "figure_9_transition_probabilities_deseasonalized.csv"),
            "required_action": "Update final probability and remove stress-trap/recovery overclaim.",
            "reviewer_safe_replacement_wording": "Final Dry/Stressed persistence was 71.3% for Shrublands; transitions should be described as descriptive and classification-dependent.",
        },
        {
            "current_manuscript_claim": "Transitions from Wet/Healthy to Dry/Stressed exceeded Dry/Stressed to Wet/Healthy across all land types.",
            "location_or_subsection": "4.4 paragraph 352",
            "supported": "No",
            "final_evidence_source": str(SOURCE_TABLE_DIR / "figure_9_transition_probabilities_deseasonalized.csv"),
            "required_action": "Replace with land-type-specific final transition probabilities.",
            "reviewer_safe_replacement_wording": "Wet/Healthy -> Dry/Stressed and Dry/Stressed -> Wet/Healthy probabilities differed by land type; do not state a universal asymmetry unless supported by final values.",
        },
    ]
    return pd.DataFrame(rows)


def build_table_recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table": "Table 2",
                "current_topic": "Forecasting model ranking / best counts",
                "recommendation": "Replace if retained",
                "evidence_source": str(FORECAST_AUDIT / "tables" / "final_best_model_counts.csv"),
                "reason": "Old BiLSTM/RMSE counts are contradicted by final XGBoost-best-count and nRMSE evidence.",
            },
            {
                "table": "Table 3",
                "current_topic": "Forecasting performance by target/land type",
                "recommendation": "Replace or move to Supplementary Information",
                "evidence_source": str(DL_GNDVI_RERUN / "tables" / "figure7_combined_previous_noQA_GNDVI_panel_values.csv"),
                "reason": "Values must match final Figure 7 and use R2/nRMSE; do not reuse old 2024-only or raw-RMSE tables.",
            },
            {
                "table": "Table 4",
                "current_topic": "Drought response / severe area",
                "recommendation": "Replace",
                "evidence_source": str(SOURCE_TABLE_DIR / "stress_zone_authoritative_values_final.csv"),
                "reason": "Use fractional-mask, area-weighted values; report km2, eligible-area percentage, percentage-point differences, and bootstrap CIs.",
            },
        ]
    )


def report_md(
    section_df: pd.DataFrame,
    risks: pd.DataFrame,
    fig_files: pd.DataFrame,
    fig5_summary: pd.DataFrame,
    fig6_pair_summary: pd.DataFrame,
    fig7_counts: pd.DataFrame,
    fig8_period: pd.DataFrame,
    fig8_auth: pd.DataFrame,
    fig9_summary: pd.DataFrame,
    claims: pd.DataFrame,
) -> str:
    headings = section_df[section_df["kind"] == "heading"]["text"].tolist()
    fig5_veg = fig5_summary[fig5_summary["claim_topic"] == "All vegetation/productivity pairs including GNDVI"].iloc[0]
    fig5_veg_no = fig5_summary[fig5_summary["claim_topic"] == "Vegetation/productivity pairs excluding GNDVI"].iloc[0]
    nirv_gpp = fig5_summary[fig5_summary["claim_topic"] == "GPP–NIRv"].iloc[0]
    ndvi_gpp = fig5_summary[fig5_summary["claim_topic"] == "GPP–NDVI"].iloc[0]
    spei_tws = fig5_summary[fig5_summary["claim_topic"] == "SPEI–TWS"].iloc[0]
    vpd_veg = fig5_summary[fig5_summary["claim_topic"] == "VPD versus vegetation/productivity indicators"].iloc[0]

    veg_pairs = fig6_pair_summary[fig6_pair_summary["involves_vegetation_productivity"]]
    veg_lag0_abs = int((veg_pairs["lag_of_max_abs"] == 0).sum())
    veg_total = len(veg_pairs)
    hydro_pairs = fig6_pair_summary[fig6_pair_summary["involves_TWS_or_SPEI"]]
    hydro_nonzero_abs = int((hydro_pairs["lag_of_max_abs"] != 0).sum())

    threshold40 = fig8_auth[fig8_auth["threshold"] == -40].copy()
    fig8_rows = []
    for _, r in fig8_period.iterrows():
        fig8_rows.append(
            f"- {r['land_type']} {r['period']}: mean NIRv change {r['mean_nirv_change_percent']:.2f}%, "
            f"severe area {r['stressed_area_km2']:.1f} km2 ({r['stressed_percent']:.2f}% eligible area)."
        )

    diff_rows = []
    for _, r in threshold40.iterrows():
        diff_rows.append(
            f"- {r['land_type']}: recent-minus-historic severe-area change {r['difference_km2']:.1f} km2 "
            f"({r['difference_percentage_points']:.2f} percentage points), 95% CI "
            f"{r['ci95_low_km2']:.1f} to {r['ci95_high_km2']:.1f} km2."
        )

    f9_persist = fig9_summary[fig9_summary["transition"].str.contains("persistence", na=False)]
    ds = fig9_summary[fig9_summary["transition"] == "Dry/Stressed persistence"]
    f9_lines = [
        f"- {r['land_type']}: {r['transition']} = {r['probability_percent']:.1f}%"
        for _, r in fig9_summary.iterrows()
        if "persistence" in r["transition"] or "->" in r["transition"]
    ]

    risk_preview = "\n".join(
        f"- Paragraph {r.paragraph_index}: `{r.phrase}` - {r.risk}"
        for r in risks.itertuples(index=False)
    ) or "- No configured red-flag phrases found."

    fig7_line = "; ".join(
        f"{r['model']} R2={int(float(r['best_r2_count']))}, nRMSE={int(float(r['best_normalized_rmse_count']))}"
        for _, r in fig7_counts.iterrows()
    )

    return f"""# Section 4 Results Technical Audit

Generated: {datetime.now().isoformat(sep=" ", timespec="seconds")}

## Evidence Boundary

This audit did not edit manuscript, figure, table, or model files. It read the current manuscript draft at:

`{MANUSCRIPT}`

and final figure/source packages under the Desktop major-revision folders. Where a claim could not be tied to a final source table, final script/report, replacement log, or figure file, it is marked as not verified.

## 1. Current Section 4 Structure and Risk Scan

Detected Section 4 headings:

{chr(10).join(f"- {h}" for h in headings)}

Primary red flags detected in the current Section 4 extraction:

{risk_preview}

The highest-risk paragraphs are the forecasting paragraphs and Table 2/Table 3 material, which still use old phrasing such as “six deep learning models,” “216 forecasting tasks,” “RMSE,” and BiLSTM-dominance counts. These are contradicted by the final forecasting audit.

## 2. Figure 5 Audit

Final accepted Figure 5 is in the main figure directory:

- `{MAIN_FIG_DIR / 'Figure_5_final.png'}`
- `{MAIN_FIG_DIR / 'Figure_5_final.pdf'}`

The accepted source lineage is the previous/no-QA GNDVI rebuild:

- values table: `{FINAL_GNDVI_REBUILD / 'tables' / 'Figure_5_previous_GNDVI_with_test_values.csv'}`
- rebuild report: `{FINAL_GNDVI_REBUILD / 'report' / 'FINAL_previous_GNDVI_with_test_rebuild_report.md'}`

The final GNDVI rebuild report states that Figure 5 was rebuilt using the exact old Figure 5 logic with completed previous/no-QA GNDVI and that strict-QA GNDVI was not used as the active final source. This supersedes older provenance rows that described a standardized-anomaly/QA-controlled Figure 5.

Supported final Figure 5 correlation ranges:

- All vegetation/productivity pairs including GNDVI: r = {fig5_veg['min']:.2f} to {fig5_veg['max']:.2f}.
- Vegetation/productivity pairs excluding GNDVI: r = {fig5_veg_no['min']:.2f} to {fig5_veg_no['max']:.2f}.
- NIRv-GPP: r = {nirv_gpp['min']:.2f} to {nirv_gpp['max']:.2f}.
- NDVI-GPP: r = {ndvi_gpp['min']:.2f} to {ndvi_gpp['max']:.2f}.
- SPEI-TWS: r = {spei_tws['min']:.2f} to {spei_tws['max']:.2f}.
- VPD versus vegetation/productivity indicators: r = {vpd_veg['min']:.2f} to {vpd_veg['max']:.2f}.

Conclusion: Figure 5 should be described as the original/raw monthly inter-variable correlation matrix using accepted previous/no-QA GNDVI. The current broad claim that all vegetation/productivity indicators are highly correlated at r = 0.82-0.98 is not supported when GNDVI is included.

## 3. Figure 6 Audit

Final accepted Figure 6 is in the main figure directory:

- `{MAIN_FIG_DIR / 'Figure_6_final.png'}`
- `{MAIN_FIG_DIR / 'Figure_6_final.pdf'}`

Accepted source evidence:

- values used for font fix: `{FINAL_GNDVI_REBUILD / 'tables' / 'Figure_6_ACCEPTED_reference_previous_GNDVI_values_used_for_font_fix.csv'}`
- replacement manifest: `{FINAL_GNDVI_REBUILD / 'audit' / 'Figure6_reference_font_fix_replacement_manifest.csv'}`
- rebuild report: `{FINAL_GNDVI_REBUILD / 'report' / 'FINAL_previous_GNDVI_with_test_rebuild_report.md'}`

The rebuild report states that Figure 6 values were not recomputed for the final replacement; the accepted source was the recovered previous-GNDVI CCF figure and values table. Formatting-only fixes were larger fonts and `NaturalForests` changed to `Natural Forests`.

Cross-correlation logic verified from the values table:

- Unique predictor-target pairs are represented with lags from -6 to +6 months.
- Negative lag means the predictor leads the target, matching the recovered old script convention.
- Vegetation/productivity pairs with maximum absolute CCF at lag 0: {veg_lag0_abs} of {veg_total} pair-land-type combinations.
- Pairs involving TWS or SPEI with maximum absolute CCF at nonzero lag: {hydro_nonzero_abs} of {len(hydro_pairs)} pair-land-type combinations.

Safe interpretation: Figure 6 can support a descriptive statement that many vegetation/productivity pairs peak near lag 0, while hydrometeorological relationships often show nonzero lag structure. Avoid causal language such as rooting-strategy proof unless tied to external references and framed as interpretation.

## 4. Figure 7 Final Forecasting Audit

Final Figure 7 files:

- `{MAIN_FIG_DIR / 'Figure_7_final.png'}`
- `{MAIN_FIG_DIR / 'Figure_7_final.pdf'}`

Accepted GNDVI-rerun source:

- `{DL_GNDVI_RERUN / 'figures' / 'Figure_7_previous_noQA_GNDVI_same_visualization.png'}`
- `{DL_GNDVI_RERUN / 'figures' / 'Figure_7_previous_noQA_GNDVI_same_visualization.pdf'}`
- replacement log: `{SUBMISSION_AUDIT / 'tables' / 'replacement_log.csv'}`

Final forecasting audit evidence:

- learned models: LSTM, BiLSTM, TCN, N-BEATS, N-HiTS, and XGBoost.
- XGBoost is a non-deep machine-learning comparator among the learned models.
- final task count: 4 land types x 9 targets = 36 land-type x target tasks.
- final design: one-month-ahead, single-target forecasting with a common 12-month lookback.
- rolling-origin folds: 2017-2018, 2019-2020, 2021-2022, and 2023-2024.
- Figure 7 metrics: R2 and normalized RMSE / nRMSE, not raw RMSE.
- SARIMA is not part of pixel-level Figure 7; it is monthly land-type aggregate only and belongs in supplementary baseline analysis.

Final best-model counts across 36 tasks:

{fig7_line}

Conclusion: old statements about six deep-learning models, 216 tasks, 2024-only tests, RMSE dominance, or BiLSTM winning 34/36 tasks are obsolete and should be removed.

## 5. Figure 8 Old-Versus-Final Audit

Final Figure 8 files:

- `{MAIN_FIG_DIR / 'Figure_8_final.png'}`
- `{MAIN_FIG_DIR / 'Figure_8_final.pdf'}`

Authoritative final tables:

- final period summary: `{FINAL_PAPER_CORRECTED / 'tables' / 'figure_8_nirv_spatial_response_summary.csv'}`
- authoritative stress-zone table: `{SOURCE_TABLE_DIR / 'stress_zone_authoritative_values_final.csv'}`
- source reconciliation: `{SOURCE_TABLE_DIR / 'stress_zone_source_reconciliation.csv'}`

The final source reconciliation identifies the later fractional-mask, latitude-aware, area-weighted pipeline as authoritative. Percentages refer to eligible analyzed land area, not the entire Figure 1 land-type area.

A distinct older Figure 8 source image/table was not identified as an authoritative comparison target during this audit. The old-versus-final risk assessment is therefore based on the current manuscript wording and the final authoritative Figure 8 tables, not on a verified old-image numerical extraction.

Final Figure 8 period values at the 40% NIRv decline threshold:

{chr(10).join(fig8_rows)}

Recent-minus-historic severe-area changes with uncertainty:

{chr(10).join(diff_rows)}

Fold-change claims should be avoided, especially where historic severe area is zero or near-zero. The final table explicitly recommends reporting absolute km2 and percentage-point change first.

## 6. Figure 9 Old-Versus-Final Audit

Final Figure 9 files:

- `{MAIN_FIG_DIR / 'Figure_9_final.png'}`
- `{MAIN_FIG_DIR / 'Figure_9_final.pdf'}`

Authoritative final transition table:

- `{SOURCE_TABLE_DIR / 'figure_9_transition_probabilities_deseasonalized.csv'}`

Final selected transition probabilities:

{chr(10).join(f9_lines)}

The final Figure 9 caption/provenance describes states as median quadrants of baseline-referenced deseasonalized-only EWBI and NIRv anomalies. The state labels and transitions are descriptive and threshold-dependent; they do not establish mechanistic regime shifts or stress traps.

The earlier comparison table `{DESKTOP / 'major_revision_analyses' / 'tables_revision' / 'figure9_submitted_vs_recomputed_comparison.csv'}` shows that submitted and recomputed transition probabilities differed in the earlier revision workflow, but the final authoritative values are the `figure_9_transition_probabilities_deseasonalized.csv` values listed above. Any old Results text that cites different transition percentages should be updated to this final table.

## 7. Tables in Section 4

- Table 2 should be replaced if retained. Use final best-model counts from `{FORECAST_AUDIT / 'tables' / 'final_best_model_counts.csv'}`.
- Table 3 should be replaced or moved to Supplementary Information. Any retained values must come from final Figure 7/S6 source tables and use R2/nRMSE.
- Table 4 should be replaced. Use `{SOURCE_TABLE_DIR / 'stress_zone_authoritative_values_final.csv'}` and avoid fold-change-first reporting.

## 8. Recommended New Section 4 Structure

1. `4.1 Inter-index relationships and response lags` - Figures 5 and 6. Safe claims: raw monthly correlations and descriptive lag structure. Avoid causal rooting-strategy claims unless softened.
2. `4.2 Forecasting benchmark performance` - Figure 7 and Supplementary Figure S6. Safe claims: reviewer-safe one-month-ahead, common-lookback benchmark; R2/nRMSE; XGBoost largest final best-count totals; rankings are metric/target/scale dependent.
3. `4.3 Spatial NIRv drought-response patterns` - Figure 8. Safe claims: mean NIRv response and severe-response area at 40% threshold; use km2, eligible-area percentages, percentage-point changes, and bootstrap CIs.
4. `4.4 Descriptive ecosystem-state transition probabilities` - Figure 9. Safe claims: descriptive persistence and transition probabilities; classification-dependent, no mechanistic regime-shift language.

## 9. Claim-by-Claim Verification Table

See `{TABLES / 'claim_by_claim_verification.csv'}`. The central actions are: replace old forecasting claims, replace broad Figure 5 correlation ranges, replace fold-change stress-zone wording, and soften Figure 9 mechanism language.

## 10. Manuscript-Ready Results Guidance

- Figure 5: state that final matrices show raw/original monthly inter-variable Pearson correlations by land type using accepted previous/no-QA GNDVI. Report pair-specific ranges rather than broad all-variable claims.
- Figure 6: say lag relationships are descriptive; negative lag means predictor leads target. Many vegetation/productivity pairs peak near lag 0, while hydroclimate variables show nonzero lag structure.
- Figure 7: describe five deep-learning models plus XGBoost, one-month-ahead single-target forecasting, common 12-month lookback, four expanding rolling-origin test periods, training-only preprocessing/scaling, and R2/nRMSE.
- Forecasting interpretation: XGBoost had the largest final best-count totals, but model rankings were target-, land-type-, fold-, metric-, and scale-dependent. Do not claim deep-learning dominance.
- Figure 8: report severe-response area in km2 and percentage points of eligible analyzed land area with bootstrap CIs. Avoid fold-change claims where historic area is near zero.
- Figure 9: use descriptive ecosystem-state transition wording. Avoid “stress trap,” “bifurcation,” “collapse,” or mechanistic transition language.
"""


def main() -> None:
    ensure_dirs()

    paragraphs = docx_paragraphs(MANUSCRIPT)
    section4 = extract_section4(paragraphs)
    write_csv(section4, TABLES / "current_section4_paragraphs.csv")
    (EXTRACT / "current_section4_results_extracted.txt").write_text("\n\n".join(section4["text"].tolist()), encoding="utf-8")

    risks = risk_scan(section4)
    write_csv(risks, TABLES / "section4_risk_scan.csv")

    fig_files = pd.DataFrame(
        [
            file_status(MAIN_FIG_DIR / f"Figure_{i}_final.png", f"main Figure {i} PNG")
            for i in [5, 6, 7, 8, 9]
        ]
        + [
            file_status(MAIN_FIG_DIR / f"Figure_{i}_final.pdf", f"main Figure {i} PDF")
            for i in [5, 6, 7, 8, 9]
        ]
        + [
            file_status(FINAL_GNDVI_REBUILD / "report" / "FINAL_previous_GNDVI_with_test_rebuild_report.md", "previous/no-QA GNDVI rebuild report"),
            file_status(FORECAST_AUDIT / "report" / "FINAL_forecasting_framework_technical_audit.md", "forecasting framework audit report"),
            file_status(SUBMISSION_AUDIT / "tables" / "replacement_log.csv", "final replacement log"),
        ]
    )
    write_csv(fig_files, TABLES / "final_figure_file_evidence_manifest.csv")

    fig5_values, fig5_summary = load_figure5_summary()
    write_csv(fig5_summary, TABLES / "figure5_correlation_support_summary.csv")

    fig6_values, fig6_summary = load_figure6_summary()
    write_csv(fig6_summary, TABLES / "figure6_ccf_pair_summary.csv")

    fig8_period, fig8_auth = load_figure8_values()
    write_csv(fig8_period, TABLES / "figure8_final_period_values.csv")
    write_csv(fig8_auth, TABLES / "figure8_authoritative_threshold_values.csv")

    fig9_values, fig9_summary = load_figure9_summary()
    write_csv(fig9_summary, TABLES / "figure9_selected_transition_summary.csv")

    fig7_counts = pd.read_csv(FORECAST_AUDIT / "tables" / "final_best_model_counts.csv")
    write_csv(fig7_counts, TABLES / "figure7_final_best_model_counts.csv")

    claims = build_claim_table(section4, fig5_summary, fig8_period, fig9_summary)
    write_csv(claims, TABLES / "claim_by_claim_verification.csv")
    write_csv(build_table_recommendations(), TABLES / "section4_table_recommendations.csv")

    md = report_md(
        section4,
        risks,
        fig_files,
        fig5_summary,
        fig6_summary,
        fig7_counts,
        fig8_period,
        fig8_auth,
        fig9_summary,
        claims,
    )
    (REPORT / "FINAL_section4_results_technical_audit.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

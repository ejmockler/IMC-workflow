"""Classify composite lineage categories by frozen D7 self-enrichment."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
SPEC_PATH = REPO_ROOT / "analysis_plans" / "composite_focalization_spec.md"
INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "biological_analysis"
    / "spatial_neighborhoods_composite"
    / "temporal_neighborhood_enrichments.csv"
)
OUTPUT_PATH = (
    REPO_ROOT / "results" / "biological_analysis" / "composite_focalization.csv"
)
OUTPUT_COLUMNS = [
    "category",
    "timepoint",
    "self_enrichment",
    "d7_self_enrichment",
    "category_type",
    "status",
]
CAVEAT = (
    "descriptive within-10um focalization; n=2/timepoint; "
    "not interface validation"
)


def _read_threshold() -> float:
    """Read the pre-registered D7 cutoff from the pinned specification."""
    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    matches = re.findall(
        r"D7\s+self-enrichment\s+(?:is\s+)?>=\s*(\d+(?:\.\d+)?)x",
        spec_text,
        flags=re.IGNORECASE,
    )
    distinct = sorted({float(m) for m in matches})
    if len(distinct) != 1:
        raise ValueError(
            "Expected a single distinct D7 self-enrichment threshold in "
            f"{SPEC_PATH}, found {distinct or 'none'}"
        )
    return distinct[0]


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Return one focalization row per self category and timepoint."""
    required_columns = {
        "focal_cell_type",
        "neighbor_cell_type",
        "timepoint",
        "enrichment_score",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    self_rows = df.loc[
        df["focal_cell_type"].eq(df["neighbor_cell_type"]),
        ["focal_cell_type", "timepoint", "enrichment_score"],
    ].copy()
    self_rows.columns = ["category", "timepoint", "self_enrichment"]

    if self_rows.empty:
        raise ValueError("No self-enrichment rows found")
    if self_rows.isna().any().any():
        raise ValueError("Self-enrichment rows contain missing values")

    self_rows["category"] = self_rows["category"].astype(str)
    self_rows["timepoint"] = self_rows["timepoint"].astype(str)
    self_rows["self_enrichment"] = pd.to_numeric(
        self_rows["self_enrichment"], errors="raise"
    )

    activation_marker_mask = self_rows["category"].str.contains(
        r"cd44|cd140b", case=False, regex=True
    )
    if activation_marker_mask.any():
        markers = sorted(self_rows.loc[activation_marker_mask, "category"].unique())
        raise ValueError(f"Activation markers are not lineage categories: {markers}")

    duplicate_keys = self_rows.duplicated(["category", "timepoint"])
    if duplicate_keys.any():
        duplicates = self_rows.loc[
            duplicate_keys, ["category", "timepoint"]
        ].to_dict("records")
        raise ValueError(f"Duplicate self-enrichment rows: {duplicates}")

    d7_rows = self_rows.loc[
        self_rows["timepoint"].eq("D7"), ["category", "self_enrichment"]
    ].rename(columns={"self_enrichment": "d7_self_enrichment"})
    missing_d7 = sorted(set(self_rows["category"]) - set(d7_rows["category"]))
    if missing_d7:
        raise ValueError(f"Categories missing D7 self-enrichment: {missing_d7}")

    result = self_rows.merge(d7_rows, on="category", how="left", validate="many_to_one")
    result["category_type"] = "pure_lineage"
    result.loc[result["category"].eq("none"), "category_type"] = "no_lineage"
    result.loc[
        result["category"].str.contains("+", regex=False), "category_type"
    ] = "interface"

    threshold = _read_threshold()
    result["status"] = result["d7_self_enrichment"].ge(threshold).map(
        {True: "focal", False: "diffuse"}
    )

    return (
        result.loc[:, OUTPUT_COLUMNS]
        .sort_values(["category", "timepoint"], kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    source = pd.read_csv(INPUT_PATH)
    classify(source).to_csv(OUTPUT_PATH, index=False)
    print(CAVEAT)


if __name__ == "__main__":
    main()

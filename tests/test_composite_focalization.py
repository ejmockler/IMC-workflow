from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from classify_composite_focalization import classify


REPO_ROOT = Path(__file__).resolve().parents[1]
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
EXPECTED_COLUMNS = [
    "category",
    "timepoint",
    "self_enrichment",
    "d7_self_enrichment",
    "category_type",
    "status",
]
EXPECTED_INTERFACE_CATEGORIES = {
    "endothelial+immune",
    "immune+stromal",
    "endothelial+stromal",
    "endothelial+immune+stromal",
}


@pytest.fixture(scope="module")
def classified() -> pd.DataFrame:
    return classify(pd.read_csv(INPUT_PATH))


def test_triple_lineage_category_is_diffuse(classified: pd.DataFrame) -> None:
    rows = classified.loc[
        classified["category"].eq("endothelial+immune+stromal")
    ]
    assert set(rows["category_type"]) == {"interface"}
    assert set(rows["status"]) == {"diffuse"}


def test_endothelial_stromal_category_is_focal(
    classified: pd.DataFrame,
) -> None:
    rows = classified.loc[
        classified["category"].eq("endothelial+stromal")
    ]
    assert set(rows["category_type"]) == {"interface"}
    assert set(rows["status"]) == {"focal"}


def test_none_category_has_no_lineage_type(classified: pd.DataFrame) -> None:
    category_types = set(
        classified.loc[classified["category"].eq("none"), "category_type"]
    )
    assert category_types == {"no_lineage"}


def test_immune_category_has_pure_lineage_type(classified: pd.DataFrame) -> None:
    category_types = set(
        classified.loc[classified["category"].eq("immune"), "category_type"]
    )
    assert category_types == {"pure_lineage"}


def test_triple_positive_is_diffuse(classified: pd.DataFrame) -> None:
    """The triple-positive interface is diffuse — the claim that survived scrutiny.

    This test previously asserted that it was the ONLY diffuse interface. That framing was
    withdrawn (see analysis_plans/composite_focalization_spec.md, k-sensitivity amendment):
    endothelial+immune sits on the 2.0x cutoff and flips to diffuse at k=10/20, and also flips
    depending on whether the per-ROI ratios are averaged arithmetically or geometrically. The
    test asserted the withdrawn claim and passed green, because the shipped classifier reads
    the arithmetic column. Assert only what is robust: triple-positive is diffuse (it clears
    the cutoff downward by ~33%, on every estimator and every k tested).
    """
    interface_rows = classified.loc[classified["category_type"].eq("interface")]
    assert set(interface_rows["category"]) == EXPECTED_INTERFACE_CATEGORIES
    diffuse = set(interface_rows.loc[interface_rows["status"].eq("diffuse"), "category"])
    assert "endothelial+immune+stromal" in diffuse


def test_processed_categories_exclude_activation_markers(
    classified: pd.DataFrame,
) -> None:
    categories = {category.casefold() for category in classified["category"].unique()}
    assert not any(
        marker in category
        for category in categories
        for marker in ("cd44", "cd140b")
    )


def test_output_columns_are_exact(classified: pd.DataFrame) -> None:
    output = pd.read_csv(OUTPUT_PATH)
    assert list(classified.columns) == EXPECTED_COLUMNS
    assert list(output.columns) == EXPECTED_COLUMNS
    assert output.groupby("category")["status"].nunique(dropna=False).eq(1).all()

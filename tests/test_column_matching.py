"""Regression tests for the optimized IMC column-matching path."""

from src.utils.column_matching import match_imc_columns


def test_match_imc_columns_accepts_bare_and_tagged_marker_names() -> None:
    columns = ["CD45", "CD11b(Nd143Di)", "CD31(Sm154Di)"]

    matches = match_imc_columns(["cd45", "CD11b", "CD31"], columns)

    assert matches == {
        "cd45": "CD45",
        "CD11b": "CD11b(Nd143Di)",
        "CD31": "CD31(Sm154Di)",
    }


def test_match_imc_columns_does_not_confuse_numeric_prefix_markers() -> None:
    columns = ["CD38(Nd143Di)", "CD3(Y89Di)"]

    matches = match_imc_columns(["CD3", "CD38"], columns)

    assert matches == {
        "CD3": "CD3(Y89Di)",
        "CD38": "CD38(Nd143Di)",
    }

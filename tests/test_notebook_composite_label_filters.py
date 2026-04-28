"""
Phase 7 MH-4 — runtime/static rename assertion.

Brutalist round 3 finding F7: enumerate-before-rename file (static grep) is
necessary but not sufficient. A missed rename target produces a notebook cell
that filters on an unprefixed composite_label string and silently returns
zero rows. The test below catches this failure mode by scanning every code
cell across every notebook for string literals appearing in composite_label
contexts, then validating each literal against the actual values present in
the annotation parquets.

The check is static (AST-based, no notebook execution required) — the spec's
"runtime assertion" target is the same failure mode caught earlier in the
pipeline. If a future revision needs nbconvert-based runtime tracing, extend
this test rather than replace it.
"""
from __future__ import annotations

import ast
import glob
import json
import re
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_GLOB = str(REPO_ROOT / 'notebooks' / '**' / '*.ipynb')
ANNOTATIONS_GLOB = str(REPO_ROOT / 'results' / 'biological_analysis' / 'cell_type_annotations' / '*.parquet')


def _observed_composite_label_values() -> set[str]:
    """Read every annotation parquet and collect the composite_label value set."""
    values: set[str] = set()
    for f in sorted(glob.glob(ANNOTATIONS_GLOB)):
        df = pd.read_parquet(f, columns=['composite_label'])
        values |= set(df['composite_label'].unique())
    return values


def _extract_composite_label_literals_from_cell(source: str) -> list[str]:
    """Find string literals likely intended as composite_label values in this cell.

    Heuristic: parse the cell as Python, walk the AST, and for every Compare
    node where the left or right side touches a `composite_label` Subscript
    (e.g., `df['composite_label']`), record any string literals on the other
    side. Also captures `.isin([...])` and `.query("composite_label == 'X'")`
    patterns. Returns a deduplicated list of detected literals.
    """
    literals: list[str] = []

    def _node_touches_composite_label(node: ast.AST) -> bool:
        """True if `node` is a subscript like df['composite_label'] or df.composite_label."""
        if isinstance(node, ast.Subscript):
            inner = node.slice if isinstance(node.slice, ast.Constant) else getattr(node.slice, 'value', None)
            if isinstance(inner, ast.Constant) and inner.value == 'composite_label':
                return True
            if isinstance(node.slice, ast.Constant) and node.slice.value == 'composite_label':
                return True
        if isinstance(node, ast.Attribute) and node.attr == 'composite_label':
            return True
        return False

    def _strings_in(node: ast.AST) -> list[str]:
        """Collect every string Constant under `node`."""
        out = []
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.append(n.value)
        return out

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Magic commands or other non-Python content — skip.
        return literals

    for node in ast.walk(tree):
        # Pattern: df['composite_label'] == 'X'  or  'X' == df['composite_label']
        if isinstance(node, ast.Compare):
            comparands = [node.left] + node.comparators
            has_cl = any(_node_touches_composite_label(c) for c in comparands)
            if has_cl:
                for c in comparands:
                    literals.extend(_strings_in(c))

        # Pattern: df['composite_label'].isin(['X', 'Y'])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'isin' and _node_touches_composite_label(node.func.value):
                for arg in node.args:
                    literals.extend(_strings_in(arg))

        # Pattern: df.query("composite_label == 'X'")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'query':
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and 'composite_label' in arg.value:
                    # Pull single-quoted or double-quoted literals out of the query string.
                    for m in re.finditer(r"['\"]([^'\"]+)['\"]", arg.value):
                        literals.append(m.group(1))

    # Deduplicate, drop the column-name itself (which is allowed to appear).
    return sorted({lit for lit in literals if lit != 'composite_label'})


def test_every_notebook_composite_label_filter_resolves_to_known_value():
    """For every code cell in every notebook, every string literal compared
    against `composite_label` must be present in the actual annotation
    parquets. A missed rename produces an empty match and a misleading plot;
    this test catches it loudly."""
    observed = _observed_composite_label_values()
    if not observed:
        pytest.skip(
            "No annotation parquets found; run batch_annotate_all_rois.py first."
        )

    failures: list[tuple[str, int, str]] = []
    for nb_path in sorted(glob.glob(NOTEBOOK_GLOB, recursive=True)):
        nb = json.loads(Path(nb_path).read_text())
        for cell_idx, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') != 'code':
                continue
            source = ''.join(cell.get('source', []))
            if 'composite_label' not in source:
                continue
            for literal in _extract_composite_label_literals_from_cell(source):
                if literal not in observed:
                    failures.append((nb_path, cell_idx, literal))

    if failures:
        msg_lines = [
            f"{len(failures)} composite_label literal(s) in notebook code cells "
            f"do not match any value in {len(observed)} parquet-observed values:",
        ]
        for nb_path, cell_idx, literal in failures:
            rel = Path(nb_path).relative_to(REPO_ROOT)
            msg_lines.append(f"  {rel}:cell[{cell_idx}]  literal={literal!r}")
        msg_lines.append("")
        msg_lines.append(f"Observed composite_label values (sample): {sorted(observed)[:5]}")
        msg_lines.append(
            "If a rename target was missed, prefix the literal with 'c:' or "
            "verify the value still exists in the current annotation pipeline."
        )
        pytest.fail("\n".join(msg_lines))

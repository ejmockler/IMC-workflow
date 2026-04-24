"""
Verify the FROZEN_PREREG.md reproducibility-anchors table against the
current working tree. Fails loudly if any pinned SHA-256 has drifted.

Run after every Phase commit to catch silent manifest/artifact drift.
The brutalist Phase 5.5 cycle caught this exact failure mode: the manifest
pinned commit `6563e90` while `run_provenance.json` recorded `ea8f27d`
with `git_dirty=true`.

Exit codes:
  0  — all anchors match (manifest is reproducible)
  1  — at least one SHA mismatched (run with --print-current to see deltas)
  2  — could not parse the manifest table (broken markdown)
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / 'review_packet' / 'FROZEN_PREREG.md'

# Anchors the manifest *should* claim. Each entry maps a manifest cell-label
# (the left column of the reproducibility table) to either a path under
# REPO_ROOT (resolved + sha256) or the literal string 'GIT_HEAD' for the
# current git commit. Adding a new pinned artifact = add one row here.
ANCHORS: Dict[str, str] = {
    '`config.json` SHA-256': 'config.json',
    '`viz.json` SHA-256': 'viz.json',
    '`analysis_plans/temporal_interfaces_plan.md` SHA-256':
        'analysis_plans/temporal_interfaces_plan.md',
    '`results/biological_analysis/sham_reference_10.0um.json` SHA-256':
        'results/biological_analysis/sham_reference_10.0um.json',
}

# Anchors that are *informational* (not gating): files we recompute and
# print but do not fail the verification on, because they are derived from
# audit-script output that is regenerated each run.
INFO_ANCHORS: Dict[str, str] = {
    '`audit_tissue_mask_density.py` SHA-256': 'audit_tissue_mask_density.py',
    '`audit_family_b_raw_markers.py` SHA-256': 'audit_family_b_raw_markers.py',
}


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of file contents, or 'MISSING' if absent."""
    if not path.exists():
        return 'MISSING'
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return 'UNAVAILABLE'


def parse_manifest_anchors() -> Tuple[Dict[str, str], int]:
    """Read FROZEN_PREREG.md, return {label: claimed_value} for the
    Reproducibility-anchors table, plus the total number of rows parsed.

    Tolerates dynamic placeholder rows whose value is a parenthesized note
    (e.g., 'computed at commit time') by recording the literal placeholder.
    """
    if not MANIFEST_PATH.exists():
        raise SystemExit(f'manifest not found at {MANIFEST_PATH}')
    text = MANIFEST_PATH.read_text()
    # Find the Reproducibility-anchors table block. Keep things simple:
    # it's the first markdown table after the heading.
    block_pat = re.compile(
        r'## Reproducibility anchors[\s\S]+?(?=\n## )',
    )
    match = block_pat.search(text)
    if not match:
        raise SystemExit(2)
    block = match.group(0)
    rows: Dict[str, str] = {}
    for line in block.splitlines():
        if '|' not in line:
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) < 2:
            continue
        label, value = cells[0], cells[1]
        # Skip header / separator rows.
        if label in {'Field', '---', '------'} or set(label) <= set('-'):
            continue
        rows[label] = value
    return rows, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--print-current', action='store_true',
        help='Print the current SHA-256 for each anchor and exit (informational; '
             'still exits non-zero on mismatch).',
    )
    args = parser.parse_args()

    try:
        manifest_rows, _ = parse_manifest_anchors()
    except SystemExit as exc:
        print(f'ERROR: could not parse manifest table ({exc.code})', file=sys.stderr)
        return 2

    print('=' * 72)
    print('FROZEN_PREREG.md anchor verification')
    print('=' * 72)
    print(f'  Manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}')
    print(f'  Anchors checked: {len(ANCHORS)} gating + {len(INFO_ANCHORS)} info')
    print(f'  Current git HEAD: {current_git_commit()}')
    print()

    SHA_RE = re.compile(r'[0-9a-f]{64}')
    mismatches = 0
    for label, rel_path in ANCHORS.items():
        claimed = manifest_rows.get(label, '<NOT FOUND IN MANIFEST>')
        actual = sha256_of(REPO_ROOT / rel_path)
        # Extract the first 64-hex-char token; tolerates surrounding
        # backticks, trailing parenthetical notes, etc.
        sha_match = SHA_RE.search(claimed)
        claimed_clean = sha_match.group(0) if sha_match else ''
        ok = claimed_clean == actual
        flag = 'OK ' if ok else 'BAD'
        print(f'  [{flag}] {label}')
        print(f'         claimed: {claimed_clean[:64] or "(no SHA in manifest cell)"}')
        print(f'         actual:  {actual[:64]}')
        if not ok:
            mismatches += 1

    print()
    print('Informational anchors (drift recorded but does not fail this script):')
    for label, rel_path in INFO_ANCHORS.items():
        actual = sha256_of(REPO_ROOT / rel_path)
        claimed = manifest_rows.get(label, '<NOT FOUND>')
        sha_match = SHA_RE.search(claimed)
        claimed_clean = sha_match.group(0) if sha_match else ''
        print(f'  {label}')
        print(f'    actual:  {actual[:64]}')
        if claimed_clean and claimed_clean != actual:
            print(f'    claimed: {claimed_clean[:64]} (drifted)')

    print()
    if mismatches:
        print(f'FAIL — {mismatches} pinned SHA(s) drifted from manifest.')
        print('  Update FROZEN_PREREG.md to match, or revert the artifact change.')
        return 1
    print('PASS — all gating anchors match the working tree.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

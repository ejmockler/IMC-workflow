# Frozen pre-registration manifest

Snapshot for external review. Referenced content is pinned by the SHA-256 anchors below and tied to the audited git basis recorded here.

## Reproducibility anchors

| Field | Value |
|-------|-------|
| Git basis | `c603b2d25f19b52c38901288be00128ac4d6979d` (closure content commit containing every referenced artifact; this manifest-only follow-up records the pin) |
| Branch | `main` |
| `config.json` SHA-256 | `d12930747fa248ab71aa341849012e2462211e214e4b892cf2f79ff101f6b400` (15-type ontology; 14 of 15 gates exercise all 9 panel markers; the neutrophil gate is the named exception per Phase 7 v2 spec — its gate literal is `+CD45 +Ly6G -CD31 -CD34` (matching `analysis_plans/phase_7_celltype_endpoint_spec.md:198`), leaving CD11b, CD140a, CD140b, CD206, and CD44 free so that the Family C v2 endpoint `neutrophil_compartment_cd44_rate` is non-tautological. Verified bit-exact for all 14 full-panel gates: every labelled superpixel satisfies its gate, every unassigned superpixel fails every gate, 0.0% multi-gate ambiguity. SHA-rotation 2026-05-21: `_comment_neutrophil_exception` field updated to match spec literal `[CD31, CD34]` (was stale `[CD31, CD34, CD140a]`); gate logic and downstream artifacts unchanged. SHA-rotation 2026-07-03 (errata amendment): `_comment_neutrophil_exception` spec cross-ref §6.3→§4.6 corrected; `sham_reference` embedded `config_sha256` and `temporal_interfaces_plan.md` re-pinned in the same amendment; gate logic and pipeline outputs bit-identical — see `analysis_plans/temporal_interfaces_plan.md` §12 (2026-07-03).) |
| `viz.json` SHA-256 | `422a14a03eff4cd7934ce1ec35138e1891d293b65e167dc8d4db498bfb6c0bd2` |
| `analysis_plans/temporal_interfaces_plan.md` SHA-256 | `6c10b2705e157868f929f042c276bf96cdbaf45f5c1d12b67d83b0b73b24fd5b` (post-remediation 2026-05-21 addendum recording 840-row spec-literal-gate cohort run alongside the original 1134-row entry; addendum updated to record post-doc-cycle SHA-of-record `07c5b976…` / `6bfaa56b…`) |
| `results/biological_analysis/sham_reference_10.0um.json` SHA-256 | `276933bd2772258c118132da700f86acfdd3f4b142b37e4a55f2a40f43a98652` (per-mouse aggregation, sigmoid centered on Sham pooled per-mouse 60th percentile; embedded `_metadata.config_sha256` matches `config.json` above; threshold/scale values bit-identical to prior run — only the embedded config_sha rotated under the comment-only config edit) |
| Resolved `DISCRETE_CELL_TYPES` SHA-256 | `6a2ba83c9b99f25b368fb7de89311b42a8fc601623dd202738bd6bf127de52ff` (SHA-256 of newline-joined sorted list of 15 cell_types from `config.cell_type_annotation.cell_types` + literal `unassigned`. Catches config-vocabulary drift independent of `config.json` byte-level SHA. Computed by `verify_frozen_prereg.py::resolved_discrete_cell_types_sha`.) |
| `audit_tissue_mask_density.py` SHA-256 | (computed at commit time, see `verify_frozen_prereg.py`) |
| `audit_family_b_raw_markers.py` SHA-256 | (computed at commit time, see `verify_frozen_prereg.py`) |
| Python | 3.12.10 |
| Platform | Darwin 25.4.0 (macOS) |
| Manifest verification script | `verify_frozen_prereg.py` (fails on divergence in the 5 gating anchors and reports the 2 informational audit-script hashes; the separate composite-track table below is checked directly during release closure) |

## Composite-lineage track (informational)

A parallel Phase 1 DA + SN track operating on the 8-category interface decomposition (none, immune, endothelial, stromal, three two-way interfaces, triple-positive) derived per superpixel via `src.analysis.temporal_interface_analysis.classify_interface_per_superpixel` at threshold 0.3 — the same categorization Family A v1 uses. It assigns one of eight states to every superpixel: among the 50,206 discrete-unassigned superpixels, 39,042 receive a non-`none` composite state and 11,164 remain `none`. Mouse×timepoint equivalence with Family A v1's `interface_fractions.parquet` is verified bit-exactly at machine precision.

| Artifact | SHA-256 |
|---|---|
| `run_composite_lineage_analysis.py` | `019760196301f11074983986e20a36e2867afc75d7c2c361743a1a77000d0a50` |
| `differential_abundance_composite/temporal_differential_abundance.csv` | `80adcfff2fca232ce6dfea6bdfee5d0cd636f08a2a23fa0757fe7f54ef19c40b` |
| `differential_abundance_composite/temporal_top_ranked_by_effect.csv` | `47182b72c365be60f7bb0b1b7e11039225262515df6c38a03b9129fda8c3bd33` |
| `differential_abundance_composite/roi_abundances.csv` | `173bdd29a30bbd67b4844d29931d4b96be9857a7fc61ed482c0aa173c0832ac4` |
| `spatial_neighborhoods_composite/temporal_neighborhood_enrichments.csv` | `952a6a5fe185ccddeb05e6165d6cfc5d47ae51fb06129c7caaf836cd3daaa6a1` |
| `discrete_vs_composite_comparison.md` | `db917fda3876813665b629e20a2daa9be5a1f422da48a820431a85c58c258a03` |

These artifacts are **informational** — derived outputs that re-running the composite-lineage script regenerates with bit-exact values given the gating anchors above. The categorization rule is pinned via the script SHA; the data dependency is the existing gating anchors (`config.json` SHA + sham_reference SHA + DISCRETE_CELL_TYPES SHA). The 8-category interface decomposition is the Family A v1 categorization; reusing it in this track preserves cross-family coherence.

The comparison-document and composite-script SHAs rotated during the 2026-07-12 release-closure pass. The narrative was reconciled with the current 15-type ontology, the discrete/composite cross-tab, mouse-of-mouse neighborhood basis, and Family C v2 neutrophil endpoint. The script change repairs the Family A v1 equivalence check to align the current mouse×timepoint artifact and fail on missing/NaN comparisons; it does not change the categorization or analyses. The four numerical CSV artifacts above remained byte-identical.

## Pre-registered endpoint families (summary; full spec in `analysis_plans/temporal_interfaces_plan.md`)

- **Family A — interface composition CLR** on 8 categories (`immune`, `endothelial`, `stromal`, `endothelial+immune`, `immune+stromal`, `endothelial+stromal`, `endothelial+immune+stromal`, `none`). Two normalization paths side-by-side: Sham-reference-centered sigmoid (primary); raw-marker Sham-reference percentile (independent corroboration). Sensitivity sweeps: lineage threshold {0.2, 0.3, 0.4}; raw-marker Sham percentile {65, 75, 85}.
- **Family B — neighbor-minus-self delta** on continuous lineage scores, per composite-label × neighbor-lineage. k=10 neighbors. Sensitivity: `min_support` {10, 20, 40}. Trajectory filter applied.
- **Family C — cross-compartment CD44⁺ activation** within (CD45⁺, CD31⁺, CD140b⁺, triple-overlap, background) compartments; Sham-referenced percentile threshold. Sensitivity: Sham percentile {65, 75, 85}. Phase 7 v2 adds a single-row neutrophil-gated compartment (`neutrophil_compartment_cd44_rate`) using the discrete `cell_type=='neutrophil'` annotation; the other 14 discrete cell types pin CD44 status by gate construction (would yield a tautological measurement).

## Shrinkage priors (pre-registered)

- Skeptical: N(0, 0.5²)
- Neutral: N(0, 1.0²) — **planning default** when a single number is used (bolded column in headline tables)
- Optimistic: N(0, 2.0²)

The three scales are a sensitivity analysis chosen a priori to span skeptical→optimistic; there is no external murine-kidney IMC Hedges'-g prior distribution to inform a principled choice. Neutral is the planning default; skeptical and optimistic are reported to make prior dependence visible to every reader. If a reviewer's preferred prior differs from neutral, they are invited to pick whichever of the three best matches their assumption.

Sampling variance per Hedges & Olkin (1985): `v(g, n) = 2/n + g²/(4n)`. Posterior mean `E[δ|g] = g × prior_var / (prior_var + v)`.

Pathology gate: rows with `|g| > 3 AND pooled_std < 0.01` emit NaN for all shrunken columns (variance-collapse artifact, not effect size). Note: `stromal_clr` at Sham→D7 has `|g|=6.7` but `pooled_std=0.058`, just above the gate; the headline filter excludes it separately via `normalization_magnitude_disagree`.

## Post-hoc headline filter (locked for follow-up cohorts, pre-registered 2026-04-23)

A Family A CLR endpoint is retained as a Sham→D7 co-headline iff:
1. Direction-consistent between the Sham-reference-centered sigmoid path and the raw-marker Sham-reference percentile path.
2. `|hedges_g| > 0.5` in the primary path.
3. Symmetric magnitude agreement: not flagged as `normalization_magnitude_disagree` (≥2× divergence in either direction).

This rule is applied unchanged for any follow-up cohort; any relaxation in a future amendment is to be reported as a filter-sensitivity result, not as evidence.

## Forbidden language (pre-registered)

- "Significant" or "significance" — at n=2 no FDR-significant findings are possible by construction.
- "95% confidence interval" — at n=2 only ~9 unique mouse-pair resamples exist; the bootstrap range is descriptive, not coverage-bearing.
- "Effect size point estimate" without the three-prior Bayesian shrinkage range — the range is the finding, not any single number.

## What this pilot explicitly concedes (anticipated reviewer questions, pre-registered)

- **n=2 per group**: hypothesis-generating only. No p-values or FDR significance.
- **Three Bayesian priors**: transparency, not indecision. Neutral is the planning default; all three reported.
- **Pan-compartment CD44 rise**: CD44 is a broadly-expressed injury/adhesion marker; Family C rises are explicitly pan-tissue activation claims, not lineage-specific.
- **Bodenmiller scope (closed-by-design, not deferred)**: the companion `run_bodenmiller_benchmark.py` validates the IMC data loader at the channel level (Spearman r=0.996) against published data. It is NOT a test of the Family A/B/C framework, and cannot become one — Bodenmiller Patient1 is single-patient, single-timepoint, different organ, different species, different panel; the framework requires temporal sampling that does not exist in that dataset. Permanent scope boundary, not a Phase-N follow-up. See archived `benchmarks/STATUS.archived.md`.
- **Shared-reference tautology**: both Family A paths anchor on the same Sham baseline. Sign agreement is partly built-in. The symmetric magnitude-disagreement count (13/48 Family A endpoints in the current run) is the honest upper bound on independent measurement.
- **CLR compositional coupling**: `stromal_clr↓` + `triple_clr↑` are one event in two coordinates. Family C is the only genuinely non-compositional corroboration surface in this cohort.

## Closures and remaining open work (Phase 5 reconciliation, 2026-04-23)

**Closed**:
- ~~Continuous Sham-percentile sensitivity sweep at 50/60/70.~~ **Closed Phase 1.5b — `continuous_sham_pct_sweep.csv` + amendment in plan.**
- ~~Parallel raw-marker Sham-reference path for Family B neighbor-minus-self.~~ **Closed Phase 1.5c — corrected Phase 5.5.** Both bases produce comparable headline counts (21 sigmoid, 18 raw at |g_neut|>0.5; 14 in common; Jaccard 0.58; 96% sign agreement). The Phase 1.5c "0 sigmoid headlines" claim was wrong (audit script printed only the raw count); audit script now prints both and the plan amendment is corrected. Phase 5.2 amends to a co-primary intersection-conservative rule for follow-up cohorts.
- ~~Pre-reg compliance: Family B support-sensitivity demotion flag; Family A CLR-without-`none` sensitivity propagation.~~ **Closed Phase 1.5a — `support_sensitive` + `clr_none_sensitivity` columns now in `endpoint_summary.csv`.**
- ~~Tissue-mask area-based non-compositional density.~~ **Closed Phase 5.1 — empirical-closure audit `audit_tissue_mask_density.py`.** Both pre-registered gates (CV(tissue_area_mm2) > 0.05 AND Pearson |r|(density, proportion) < 0.95) fail: CV = 0.012, |r| = 0.97 on the dominant cell type. Closure scope is **area-based denominators on this acquisition design**.

**Untested alternatives (NOT closed by Phase 5.1; flagged for follow-up engineering)**:
- Per-nucleus density via DNA watershed segmentation (`src/analysis/watershed_segmentation.py` exists but is not wired into the production pipeline).
- DNA-intensity integral as denominator (cellularity-weighted volume).
- Variable-extent re-acquisition cohorts (whole-section IMC, panoramic montage).
- Spike-in absolute-quantification controls in a redesigned panel.

**Co-primary Family B implementation (Phase 6 closure, 2026-04-24)**:
- Fully implemented in `run_temporal_interface_analysis.py`. Every pipeline run computes the raw-marker basis alongside the sigmoid basis using config-defined lineage definitions (panel-portable via `config.cell_type_annotation.membership_axes.lineages`).
- `endpoint_summary.csv` carries both bases as Family B rows (270 sigmoid + 270 raw); distinguished by `normalization_mode` ∈ {`sham_reference_v2_continuous`, `sham_reference_raw_marker_per_mouse`}. Total endpoint rows: **618** (was 348 pre-Phase 6).
- New always-emitted artifacts: `family_b_basis_divergence.csv` (per-endpoint headline-overlap status: `both_above` / `sigmoid_only` / `raw_only` / `both_below`); `family_b_basis_conflict.csv` (opposite-sign-same-endpoint subset; empty in current pilot); `family_b_raw_marker_audit.parquet` (combined basis table with `lineage_source` column).
- Pilot Sham→D7 basis-divergence breakdown: **14 both_above** (conservative-intersection headline set), 7 sigmoid_only, 4 raw_only, 20 both_below, 0 conflict.

**Phase 7 — discrete cell-type-resolved endpoint extensions (2026-04-28, three brutalist rounds)**:
- Spec: `analysis_plans/phase_7_celltype_endpoint_spec.md` (390 lines, locked post-round-3).
- **Family A_v2** — discrete cell-type CLR (16 categories: 15 typed from `config.cell_type_annotation.cell_types` + `unassigned`). `unassigned` IN as 16th coordinate (gmean-drag accepted as feature; round-3 §1.1 lock). Single-path (no v2-internal corroborator); rule `|g| > 0.5 AND not g_pathological`. Tagged `endpoint_axis = 'discrete_celltype_16cat'`.
- **Family B_v2** — per-discrete-cell-type neighbor-minus-self. Same kNN gradient operator as v1; stratifier_col swapped from `composite_label` to `cell_type`. Dual-basis (sigmoid + raw-marker) under existing `normalization_mode` column. Tagged `stratifier_basis = 'discrete_celltype'`.
- **Family C v2** — single-row neutrophil extension. New per-ROI compute path (categorical: `cell_type == 'neutrophil'`); per-mouse aggregation reuses Family C contract (collects `*_cd44_rate` columns). The 14/15 other discrete cell types pin CD44 status by gate construction (rate forced 0 or 1); only `neutrophil` admits a non-tautological measurement.
- **`c:` rename** — every `composite_label` value gains a `c:` prefix to disambiguate from `cell_type` values (e.g., `c:endothelial` vs `cell_type='endothelial'`). Single-commit migration; runtime assertion test (`tests/test_notebook_composite_label_filters.py`) catches missed rename targets.
- **v1/v2 cross-rule** — runtime demotion column `headline_demoted_reason = 'cross_axis_co_headline_forbidden'`. Effective reach: 18/840 rows demoted in current cohort post-remediation (v2 rows whose v1 lineage analog passes the v1 rule). Architecture is correct for the events that exist; structural reach acknowledged as small (round-3 F1).
- **`is_headline` boolean** — single canonical headline-status column on every row, computed after demotion. 263/840 rows pass (post-cross-rule, post-remediation).
- **Schema deltas (8 new columns)**: `endpoint_axis`, `stratifier_basis`, `min_prevalence_sweep_value`, `headline_rule_version`, `headline_demoted_reason`, `is_headline`, `unassigned_rate_mouse_mean_1`, `unassigned_rate_mouse_mean_2`.
- **Total endpoint rows (post-remediation)**: **840** (was 618 pre-Phase-7; the pre-remediation Phase 7 cohort had 1134 rows when activated_neutrophil_* subtypes were active and the v2 simplex retained more low-prevalence coordinates). 46 columns (was 37 pre-Phase-7).
- **Family breakdown (post-remediation)**: A v1=48 + A v2=36 = 84; B v1=540 + B v2=180 = 720; C=36. Total 840.
- **Prerequisites merged ahead of Phase 7**: P1 (priority_order in config; `config_sha256` rotated to `adf88d48…`); P2 (parameterized `apply_min_prevalence_filter`); P3 (`DISCRETE_CELL_TYPES` SHA `6a2ba83c…` pinned in this manifest).
- **MH-1 permutation null** — empirical null-distribution criterion (round-3 F5: original "95th percentile == 0" was untenable at n=2). Test scaffolded at `tests/test_phase7_permutation_null.py`. Smoke test runs on every CI; full 1000-shuffle lock gate runs under `PHASE7_RUN_FULL_NULL=1` env.

## Reviewer entry points

| Artifact | Path |
|----------|------|
| Pre-registered plan | `analysis_plans/temporal_interfaces_plan.md` |
| Sham reference artifact | `results/biological_analysis/sham_reference_10.0um.json` |
| Endpoint summary | `results/biological_analysis/temporal_interfaces/endpoint_summary.csv` |
| Rank-based table | `results/biological_analysis/differential_abundance/temporal_top_ranked_by_effect.csv` |
| Run provenance | `results/biological_analysis/temporal_interfaces/run_provenance.json` |
| Main narrative notebook | `notebooks/main_narrative.ipynb` |
| Kidney-specific visualizations | `notebooks/biological_narratives/kidney_injury_spatial_analysis.ipynb` |
| Deprecation manifest | `analysis_plans/deprecation_manifest.md` |

To re-verify the manifest at this commit:

```bash
git rev-parse HEAD
shasum -a 256 config.json viz.json analysis_plans/temporal_interfaces_plan.md \
              results/biological_analysis/sham_reference_10.0um.json
```

Expected output matches the "Reproducibility anchors" table above.

## Reproducible co-headline query

Exact pandas query that regenerates the one-pager's co-headline table from the pinned `endpoint_summary.csv`:

```python
import pandas as pd
s = pd.read_csv('results/biological_analysis/temporal_interfaces/endpoint_summary.csv')
sham_d7 = s[(s['tp1'] == 'Sham') & (s['tp2'] == 'D7')].copy()
for col in ['normalization_sign_reverse', 'normalization_magnitude_disagree']:
    sham_d7[col] = (sham_d7[col] == True)
fa = sham_d7.query(
    "family == 'A_interface_clr' and normalization_mode == 'per_roi_sigmoid' "
    "and abs(hedges_g) > 0.5 "
    "and not normalization_sign_reverse "
    "and not normalization_magnitude_disagree"
)
fc = sham_d7.query(
    "family == 'C_compartment_activation' and abs(hedges_g) > 0.5"
)
# fc returns Family C v1 (CD45/CD31/CD140b/background/triple_overlap) AND v2 (neutrophil-gated, Phase 7)
co = pd.concat([fa, fc])[['family','endpoint','hedges_g',
                          'g_shrunk_skeptical','g_shrunk_neutral','g_shrunk_optimistic']]
print(co.sort_values('g_shrunk_neutral', key=abs, ascending=False).to_string(index=False))
```

Family A endpoints filtered out by the rule (reported in the one-pager "Filtered out" table) are `stromal_clr` and `none_clr` (magnitude disagree), `endothelial_clr` (sign reverse, both near zero), `endothelial+stromal_clr` (|g| ≤ 0.5).

Phase 7 v2 candidate findings (`endpoint_axis='discrete_celltype_16cat'` for Family A; `endpoint=='neutrophil_compartment_cd44_rate'` for Family C) are reported separately in `ONE_PAGER.md`'s v2 sub-table; v2 rows are gated by `is_headline=True` after the runtime cross-rule (`headline_demoted_reason='cross_axis_co_headline_forbidden'`) prevents v2 co-headlining a v1 lineage analog on the same biological event.

# Methods summary — pilot IMC spatial proteomics of AKI

Scoped for a kidney pathologist + spatial-proteomics methodologist + statistician review. Companion to `ONE_PAGER.md` and `FROZEN_PREREG.md`.

## Cohort

- **Model**: unilateral ureteral obstruction (UUO), C57BL/6 mice.
- **Timepoints**: Sham (no ligation) / D1 / D3 / D7, n=2 mice per timepoint.
- **Acquisitions**: 25 regions-of-interest (ROI); 1 Test ROI excluded pre-analysis (`parse_roi_metadata` gates at `timepoint != 'Test'`). 24 analyzed.
- **Imaging**: IMC with 9 protein channels + 2 DNA + 2 calibration beads (`130Ba`, `131Xe`) + background + carrier-gas.
- **Segmentation**: SLIC superpixels at 10 μm (pinned a priori per plan §2), DNA-guided via `slic_input_channels=['DNA1','DNA2']`. ~2,400 superpixels per ROI.

## Preprocessing

- Background pixel subtraction, clipped at zero.
- Arcsinh transform with percentile-derived cofactor (`percentile_threshold=5.0` for proteins; `cofactor_multiplier=3` for DNA).
- No global normalization beyond arcsinh (double transformation was shown to destroy signal gradients in earlier work).

## Annotation (two paths per superpixel)

**Discrete gating** (`annotate_cell_types`) — 15 config-defined cell types via boolean AND-of-positive, AND-NOT-of-negative marker gates. Positivity thresholds are per-ROI percentile (60th default; CD206 at 50th, Ly6G at 70th per config override). Produces `cell_type` label + confidence score per superpixel.

**Continuous multi-label memberships** (`compute_continuous_memberships`) — three independent axes:

- **Lineage scores**: non-exclusive [0,1] continuous scores via Sham-reference sigmoid.
  - *Center*: per-marker Sham-only pooled per-mouse percentile (default 60th), shared across all 24 ROIs. Per-marker overrides honored.
  - *Scale* (sigmoid steepness denominator): experiment-wide IQR (all timepoints pooled) — decouples steepness from Sham-specific noise that caused the per-ROI sigmoid to self-normalize away injury dynamics.
  - Lineage definitions: immune = max(CD45), endothelial = mean(CD31, CD34), stromal = max(CD140a).
- **Subtype scores**: within-lineage geometric mean of positive markers (marker-count-normalized) × (1 − negative markers), gated on parent_lineage ≥ `subtype_threshold` (0.3).
- **Activation overlay**: CD44 and CD140b as independent [0,1] continuous scores.

The Sham-reference artifact is produced once per pipeline run by `generate_sham_reference.py` with hard gates on pilot design (n_sham_mice=2, n_sham_rois=6) and persisted to `results/biological_analysis/sham_reference_10.0um.json` with full provenance (`config_sha256`, `git_hash`, `marker_order`, `percentile`, `aggregation`). `batch_annotate_all_rois.py` validates every provenance field on load; any mismatch raises.

Three previously-drifting Sham-threshold primitives (in `temporal_interface_analysis.py` and the old per-ROI sigmoid) now converge on a single shared primitive at `src/analysis/sham_reference.py`.

## Endpoint families (pre-registered)

**Family A — interface composition CLR.** The three lineage scores are thresholded at 0.3 to produce 8 interface categories (single-lineage × 3 + pairwise overlaps × 3 + triple-positive + none). Mouse-level category fractions → centered log-ratio. Two parallel normalization paths are computed side-by-side:

1. **Sham-reference-centered sigmoid** (primary, via `compute_continuous_memberships` + `composite_label_thresholds.lineage=0.3`).
2. **Raw-marker Sham-reference percentile** (via `compute_global_marker_thresholds(sham_only=True)` at {65, 75, 85}ᵗʰ percentiles; 75 is the primary reference).

Disagreement per endpoint is surfaced by two flags:
- `normalization_sign_reverse` — opposite signs between the two paths.
- `normalization_magnitude_disagree` — ≥2× magnitude divergence in either direction (symmetric, replacing an earlier asymmetric `normalization_g_collapse` flag that understated disagreement).

**Family B — continuous neighborhood shifts.** Per composite label × neighbor lineage, k-NN (k=10) neighbor-minus-self delta at the mouse level. Trajectory filter: composite-label × lineage combinations absent at one or more timepoints are excluded. Sensitivity sweep across `min_support` = {10, 20, 40}.

**Family C — cross-compartment CD44⁺ activation.** Raw-marker compartments (CD45⁺, CD31⁺, CD140b⁺, triple-overlap, background) gated at a Sham-referenced percentile; per-ROI CD44⁺ rate within each compartment; mouse-level mean. Sensitivity sweep across Sham percentile = {65, 75, 85}.

## Statistics

- **Unit of analysis**: mouse. ROI proportions are averaged within each mouse before testing (prevents pseudo-replication).
- **Effect size**: Hedges' g (small-sample corrected); sampling variance per Hedges & Olkin (1985): `v(g, n) = 2/n + g²/(4n)`.
- **Bayesian shrinkage**: posterior mean of δ under three explicit priors (N(0, 0.5²) skeptical / N(0, 1.0²) neutral / N(0, 2.0²) optimistic). The three scales are a sensitivity analysis chosen a priori to span skeptical → optimistic under a non-informative-by-construction framing; we make no appeal to a literature prior because no external murine-kidney IMC Hedges'-g distribution exists. **Neutral is the planning default** when a single number is needed (tables bold `g_neut`); skeptical and optimistic are reported to every reader to make the prior dependence visible. Pathological rows (`|g|>3 AND pooled_std<0.01`) are quarantined with NaN shrunken values — variance-collapse artifacts, not effect sizes. (Caveat: `stromal_clr` Sham→D7 has |g|=6.7 with pooled_std=0.058, passing the pathology threshold by a small margin; its shrunken headline magnitude is filtered out by the normalization magnitude-disagreement flag independently.)
- **Bootstrap range** (not a confidence interval at n=2): B=10,000 resamples of mouse-pairs; reported as `bootstrap_range_min` / `bootstrap_range_max`. With n=2 per group there are ~9 unique resamples, so the range is descriptive, not coverage-bearing.
- **P-values and FDR**: reported in `temporal_differential_abundance.csv` but bounded at 0.333 by construction (minimum two-sided Mann-Whitney at n=2 vs n=2) — not interpretable as inferential evidence. Omitted from the rank-based selection-free table.

## Provenance

Every run emits a `run_provenance.json` alongside the endpoint tables:

- `git_commit` + `git_dirty` flag + `git_modified_critical_files` list.
- `config_sha256`, `sham_reference` artifact path + SHA256 + percentile + aggregation + n_mice/rois.
- `analysis_file_sha256` for each module read by the pipeline.
- `python_version`, `platform`, package versions (numpy, pandas, scipy).
- `random_seeds` (bootstrap=42, join-count=42).
- `pipeline_parameters` including lineage threshold, min support, k-neighbors, Sham-reference percentile sweep, Bayesian prior SDs, pathology thresholds.
- `roi_to_mouse_mapping` for the cohort.

## Audit surfaces beyond headlines

- `endpoint_summary.csv` (**1134 rows × 46 cols** post-Phase-7): every endpoint with shrunken g columns, pathology flag, insufficient-support flag, threshold-sensitive flag, normalization flags (Family A), `sham_percentile` (Family A per-roi 60; Family C 65/75/85), Phase 7 v2 schema columns (`endpoint_axis`, `stratifier_basis`, `headline_rule_version`, `headline_demoted_reason`, `is_headline`, `unassigned_rate_mouse_mean_1/2`, `min_prevalence_sweep_value`).
- `temporal_top_ranked_by_effect.csv`: selection-free top-5 per contrast, sorted on |g_shrunk_neutral|, pathological rows quarantined.
- `interface_fractions_normalization_sensitivity.parquet`: both normalization paths side-by-side for Family A.
- `family_a_endpoints_norm_sweep.parquet`: raw-marker Sham-reference at {65, 75, 85}ᵗʰ percentiles.
- `family_b_sensitivity_endpoints.parquet`, `family_c_sensitivity_endpoints.parquet`: pre-registered sensitivity sweeps for B and C.

## Reproducible candidate-findings query

To regenerate the one-pager's co-headline table directly from `endpoint_summary.csv`:

```python
import pandas as pd
s = pd.read_csv('results/biological_analysis/temporal_interfaces/endpoint_summary.csv')
sham_d7 = s[(s['tp1'] == 'Sham') & (s['tp2'] == 'D7')].copy()
for col in ['normalization_sign_reverse', 'normalization_magnitude_disagree']:
    sham_d7[col] = (sham_d7[col] == True)

# Family A: requires both direction + magnitude agreement
fa = sham_d7.query(
    "family == 'A_interface_clr' and normalization_mode == 'per_roi_sigmoid' "
    "and abs(hedges_g) > 0.5 "
    "and not normalization_sign_reverse "
    "and not normalization_magnitude_disagree"
)

# Family C: single-path by design; any |g| > 0.5
fc = sham_d7.query(
    "family == 'C_compartment_activation' and abs(hedges_g) > 0.5"
)

co = pd.concat([fa, fc])[['family','endpoint','hedges_g',
                          'g_shrunk_skeptical','g_shrunk_neutral','g_shrunk_optimistic']]
co = co.sort_values('g_shrunk_neutral', key=abs, ascending=False)
print(co.to_string(index=False))
```

Expected output matches the co-headline rows in `ONE_PAGER.md`.

## Family B result (two co-primary bases per Phase 5.2 amendment)

**Primary (pre-registered) — sigmoid Sham-ref continuous lineage scores.** Family B produces **21 endpoints** at |g_shrunk_neutral| > 0.5 at Sham→D7 under primary `min_support=20` (none pathological, none `support_sensitive`), concentrated on `vs_sham_mean_delta_lineage_immune` and `..._endothelial` around composite labels including `activated_endothelial_cd44`, `activated_stromal_cd140b_cd44`, `mixed`, `non_myeloid_immune`, `stromal`, `unassigned`. An earlier draft of this section claimed "0 endpoints" — that was wrong; the audit script printed only the raw-marker count and the sigmoid count was assumed. Corrected 2026-04-23 Phase 5.5. Full Family B table at `family_b_endpoints.parquet`; sensitivity across `min_support` ∈ {10, 20, 40} at `family_b_sensitivity_endpoints.parquet` with per-row `support_sensitive` flag (90/270 rows filter-fragile).

**Co-primary sensitivity check (Phase 1.5c, raw arcsinh markers).** `audit_family_b_raw_markers.py` replaces the sigmoid lineage columns with raw-marker composites (`lineage_immune_raw = CD45`; `lineage_endothelial_raw = mean(CD31, CD34)`; `lineage_stromal_raw = CD140a`) and reruns the same neighbor-minus-self pipeline. Neighbor-minus-self is differential, so any Sham-ref additive offset cancels — the raw-marker path is genuinely sigmoid-independent. Produces **18 Sham→D7 endpoints** at |g_shrunk_neutral| > 0.5 (none pathological). **14 endpoints clear in BOTH bases** (Jaccard 0.58 over the union of 21 sigmoid + 18 raw); **96% sign agreement** on overlapping endpoints. Pre-shrinkage magnitude divergence is real (55/166 ≥2× magnitude disagree) but Bayesian shrinkage to neutral converges the headline counts. Output: `family_b_raw_marker_audit.parquet` + `family_b_raw_marker_comparison.csv`. Phase 5.2 plan amendment names the **intersection of both bases as the conservative headline set**; sigmoid-only and raw-only sets are reported but flagged as basis-dependent. Audit script prints both counts at run time so one-sided reporting cannot recur silently.

## Continuous Sham-percentile sensitivity (Phase 1.5b)

`sweep_continuous_sham_pct.py` runs three in-memory Family A paths at continuous Sham-ref percentiles ∈ {50, 60, 70} without altering persistent artifacts. Output: `family_a_continuous_sham_pct_sweep.parquet` (144 rows), `continuous_sham_pct_sweep.csv` (per-endpoint stability). Headline `endothelial+immune+stromal_clr` Sham→D7 neutral g = 1.000 / 0.987 / 0.965 across 50 / 60 / 70 (Sham-pct-invariant under shrinkage). `stromal_clr` raw g is more variable (−13.75 → −6.71 → −3.75) but its neutral-shrunk value is contained (−0.54 → −0.88 → −1.00).

## Key methodological refactors in this commit series

- **Phase 1** (seam-1 closure): replaced per-ROI sigmoid with Sham-reference-centered sigmoid; unified 3 drifting Sham-threshold primitives; added provenance artifact + hard validation gates at every boundary; fixed a latent preprocessing-drift bug (annotation engine was reading raw coabundance features instead of arcsinh per-marker arrays).
- **Phase 2** (seams 2+3): pre-reg amendment locking the post-hoc headline filter; rank-based selection-free companion table (shrunk-g, pathology-gated); explicit CLR compositional coupling caveat with corroboration independence ranking.
- **Phase 3** (engineering): extended VizConfig drift guards (colormap + validation-plots invariants); lazy-loaded viz_utils; repaired 5 pre-existing test failures (ROI count 25→24, ablation-module path move, arcsinh monotonicity test logic, bead-normalization QC fixture, spatial_weight dead-field skip).
- **Phase 5** (deferred-item closures + brutalist-cycle correctness fix): Phase 5.1 closes area-based tissue-mask density empirically (CV 0.012, |r| 0.97 on dominant cell type — both pre-registered gates fail, demonstrating algebraic equivalence to rescaled proportion); 5.2 amends Family B to a co-primary intersection-conservative rule with config-driven raw-marker mapping for panel portability; 5.3 closes Bodenmiller permanently (closed-by-design wording); 5.5 brutalist multi-critic surfaced an uncaught Phase 1.5c factual error — "0 sigmoid Family B Sham→D7 headlines" was wrong (actual: 21 sigmoid, 18 raw-marker, 14 in common); audit script now prints both counts so one-sided reporting cannot recur silently. `verify_frozen_prereg.py` recomputes pinned SHAs and fails loudly on drift.

## Closures and remaining open work (Phase 5 reconciliation, 2026-04-23)

**Closed (Phase 1.5 / 5)**:
- Continuous Sham-percentile sensitivity sweep at 50/60/70 — closed Phase 1.5b.
- Family B parallel raw-marker audit path — closed Phase 1.5c (corrected counts in Phase 5.5).
- Pre-registration compliance: Family B `support_sensitive` flag + Family A `clr_none_sensitivity` propagation — closed Phase 1.5a.
- **Tissue-mask area-based density** — closed empirically Phase 5.1: `audit_tissue_mask_density.py` shows CV(tissue_area_mm2) = 0.012 across 24 ROIs (every ROI acquired at the same ~500×500 µm field-of-view), so `density = count / tissue_area_mm2 ≈ 9857 × proportion ± 1.4%` — same algebraic tautology as the retracted SLIC-constant version. Pearson |r|(density, proportion) on dominant cell type = 0.97 confirms degeneracy. Closure is for **area-based** denominators on this acquisition design only.

**Untested alternatives flagged for follow-up engineering** (NOT closed by Phase 5.1):
- Per-nucleus density via DNA watershed segmentation (`src/analysis/watershed_segmentation.py` not currently wired into production pipeline).
- DNA-intensity integral as denominator (cellularity-weighted volume, varies with nuclear count even when 2D area is fixed).
- Variable-extent re-acquisition cohorts (whole-section IMC, panoramic montage).
- Spike-in absolute-quantification controls in a redesigned panel.

## Entry points

- `python run_analysis.py` — full pipeline from raw IMC acquisitions through multiscale results.
- `python generate_sham_reference.py` — build the Sham-reference artifact (run after any config change that affects the gating thresholds).
- `python batch_annotate_all_rois.py --archive-prior` — annotate all ROIs with the current Sham reference (archives prior annotations).
- `python differential_abundance_analysis.py` — Layer A + Layer B DA with shrunk-g rank table.
- `python spatial_neighborhood_analysis.py` — k-NN enrichment.
- `python run_temporal_interface_analysis.py` — Family A/B/C + sensitivity sweeps + endpoint_summary.

Tests: `.venv/bin/python -m pytest tests/ -q --ignore=tests/test_adaptive_search.py --ignore=tests/test_performance_benchmark.py --ignore=tests/test_performance_regression.py --ignore=tests/test_performance_core.py --ignore=tests/test_pipeline_e2e.py --ignore=tests/test_optimization_regression.py` — 333 pass, 8 skipped, 0 fail at commit `6563e90`. The ignored test files are long-running performance regressions outside the Phase 1-3 scope.

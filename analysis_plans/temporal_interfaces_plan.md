# Pre-Registration: Temporal Multi-Lineage Interface Analysis

**Date frozen:** 2026-04-17 (amended same day after Gate 0 brutalist review)
**Status:** Frozen before any analysis output is generated. Subsequent changes require an amendment block at the bottom of this file with date and rationale.
**Scope:** Resolve five gaps in temporal multi-lineage analysis identified during 2026-04-17 review. Hypothesis-generating pilot at n=2 mice/timepoint; no inferential conclusions.

## 1. Frame

This plan converts descriptive temporal observations into effect-size estimates with explicit power requirements. It does **not** attempt to establish significance. With n=2 mice/timepoint, FDR-significant findings are mathematically impossible. Every finding is reported as: mouse-level mean (range), candidate effect size, sample size required for 80% power.

**Reproducibility freeze:** at the start of execution (T17), the git commit hash, `config.json` hash, ROI-to-mouse mapping table, and excluded ROI ID will be written to `results/biological_analysis/temporal_interfaces/run_provenance.json`. Re-running this analysis without an amendment block invalidates the output directory.

## 2. Inclusion / Exclusion

- **25 ROIs acquired; 24 analyzed.** ROI IMC_241218_Alun_TestROI_001 (or equivalent test acquisition) excluded *prior* to any analysis based on the acquisition log designating it a calibration/test scan. Exclusion ID and rationale logged in `run_provenance.json`.
- 4 timepoints: Sham, D1, D3, D7.
- 2 mice per timepoint.
- **Scale:** 10 µm SLIC superpixels selected a priori as the finest spatial resolution. Results at 20 and 40 µm exist as pipeline outputs but are not analyzed in this effort to prevent scale-as-researcher-degree-of-freedom.
- Per-ROI annotation parquets in `results/biological_analysis/cell_type_annotations/` are inputs.
- Continuous neighborhood: minimum support filter `n_superpixels ≥ 20` per (composite_label, ROI) row; tail collapsed to "other".

## 3. Endpoint Families

Three pre-declared endpoint families. **Multiplicity:** at n=2 per group no real p-value exists, so formal BH-FDR (which requires p-values) is not computed. Earlier drafts of this plan called for a normal-CDF-from-|g| proxy with within-family + pooled BH adjustment; Gate 6 removed those proxy columns because reviewers consistently misread them as real q-values regardless of disclaimers. Family-arbitrage concerns (whether splitting endpoints into 3 families vs pooling them changes conclusions) are addressed by sorting endpoint_summary.csv by `|hedges_g|` directly — the rank-ordering is the relevant audit at this sample size, not coverage-bearing FDR.

### Family A — Interface composition (compositional)
- **Endpoints:** mouse-level fraction of superpixels in each interface category {immune, endothelial, stromal, immune+endothelial, immune+stromal, endothelial+stromal, triple, none}.
- **Transform:** centered log-ratio (CLR).
- **Zero handling:** Bayesian-multiplicative replacement (Martin-Fernandez et al. 2003): when category proportion is zero in a given mouse, replaced with `δ = (n_categories+1) / (total_count × (n_categories+1))` rather than additive eps=1e-6. This preserves the simplex geometry rather than dominating it.
- **Minimum prevalence filter:** categories with mouse-level fraction < 1% in *all* timepoints are collapsed into "other_rare" before CLR. Reported separately.
- **CLR sensitivity:** computed both *with* and *without* the "none" category. If trajectories for biologically meaningful categories change qualitatively when "none" is excluded, the finding is flagged.
- **Side report:** absolute lineage-positive superpixel counts per mouse per timepoint, so denominator shifts are visible.

### Family B — Continuous neighborhood lineage shifts (orphaned data wired in)
- **Endpoints:** per (composite_label × neighbor_lineage), the mouse-level mean of `neighbor_lineage_X − self_lineage_X` (neighbor-minus-self delta).
- **Why neighbor-minus-self:** the self quantity is tautological with the cell type definition; the delta partially defuses this circularity.
- **Acknowledged residual bias:** within stratum X, `self_lineage_X` is high by definition, so the delta is biased negative for self-matching lineages. **Only temporal *changes* in the delta are interpretable; absolute delta values are not.** Reported as `delta_change_vs_Sham` in addition to raw delta.
- **Filter:** rows with `n_superpixels < 20` excluded; cell types failing the filter in any timepoint excluded entirely from the trajectory analysis (not just from the failing timepoint). **Missingness table** reported in `endpoint_summary.csv`: which (cell_type × timepoint) combinations were filtered, distinguishing "absent biology" from "excluded due to support".
- **Stratification caveat:** the conditioning variable (`composite_label`) is itself score-derived. This is post-hoc descriptive conditioning, not pre-treatment grouping. Reported as such.

### Family C — Cross-compartment activation trajectories
- **Endpoints:** mouse-level CD44+ rate within {CD45+, CD31+, CD140b+, background} compartments per timepoint, plus triple-overlap fraction.
- **Compartment definition:** marker > **75th percentile of the Sham reference distribution** (pooled across all Sham ROIs). Threshold computed once, applied globally to all timepoints. **This replaces the per-ROI 75th percentile in the original Part 6 code**, which created a moving threshold that confounded temporal comparisons.
- **CD44+ definition:** also Sham-reference 75th percentile.
- **Compartment overlap caveat:** superpixels can belong to multiple compartments simultaneously (CD45+ AND CD31+, etc.). CD44+ rates across compartments are correlated by construction. Each compartment trajectory is interpreted separately; formal cross-compartment comparison is not performed.
- **No CLR:** these are not exhaustive categories.

## 4. Contrasts

All 6 pairwise timepoint comparisons (descriptive enumeration):
`Sham_vs_D1, Sham_vs_D3, Sham_vs_D7, D1_vs_D3, D1_vs_D7, D3_vs_D7`.

The existing DA framework currently runs only 5 (omits D1_vs_D7); fixed in T19. Reframing as "descriptive enumeration of the trajectory" rather than "all pairwise hypothesis tests" — the latter would inflate multiplicity for a trajectory question. Each contrast reports effect size, never a hypothesis-test verdict.

## 5. Statistical Methods

| Step | Method | Rationale |
|------|--------|-----------|
| Aggregation | Per-ROI quantity → per-mouse mean → per-timepoint group of n=2 | Defeats superpixel-level pseudoreplication |
| Effect size | Hedges' g on mouse-level values | Small-sample-corrected Cohen's d |
| Uncertainty | Percentile bootstrap with 10,000 iterations, **reported as "bootstrap range (min, max of 9 unique values at n=2 per group)" — NOT as 95% CI** | The resolution limit is 9 distinct values; CI notation implies coverage that does not exist |
| Shrinkage | Every observed g is annotated with a **per-endpoint Bayesian shrinkage range** under three prior strengths on the true effect δ: skeptical N(0, 0.5²), neutral N(0, 1.0²), optimistic N(0, 2.0²). Posterior mean E[δ \| g_obs] = g_obs × prior_var / (prior_var + sampling_var), where sampling_var(g, n) = 4/n + g²/(2n). Replaces the prior single-scalar Type-M=0.65 correction with a defensible range. | n=2 winner's curse; Bayesian shrinkage is the standard treatment for noisy effect-size estimators under informative priors. |
| Multiplicity (primary) | BH-FDR within each endpoint family separately | Distinct biological questions |
| Multiplicity | At n=2, no real p-values exist; no FDR is computed. Family-arbitrage rank audit done by sorting `endpoint_summary.csv` on `|hedges_g|`. | Earlier proxy-FDR columns removed in Gate 6 (cognitive-anchoring risk) |
| Power | For each observed g, report `n_required` under four effect-size assumptions: the raw observed g (*most optimistic lower bound*) and the three Bayesian-shrunk values under skeptical/neutral/optimistic priors (*sensitivity range*). **No single prior is designated as "default"** — the range itself is the finding. Downstream study designers pick the prior that matches their own scepticism about the observed pilot effect. | Converts findings into study-design statements with a transparent uncertainty band; avoids recreating the magic-scalar problem by anointing one prior |
| Compositional | CLR with Bayesian-multiplicative zero replacement for Family A only | Family B/C endpoints are not exhaustive |
| Pathology flag | `g_pathological: bool` set when `|g| > 3 AND pooled_std < 0.01` | Quarantines variance-collapse artifacts (e.g., g=−4.87 with mean diff 0.001) |

## 6. Spatial Coherence Replacement

The original proposal called for Moran's I on interface labels. **Not used.** Moran's I on categorical variables is mathematically incoherent.

**Join-count statistics** (per binary interface indicator: is_triple, is_immune+endo, etc.) replace it. Specification:
- **Adjacency:** k=10 nearest neighbors in (x, y) space (matches existing neighborhood enrichment).
- **Null model:** complete spatial randomness (label permutation within ROI), 1000 permutations.
- **Statistic:** observed BB joins (positive-positive) standardized by permutation null mean and std.
- **Filter:** join-count computed only for indicators with ≥ 10 positive superpixels per ROI; below-threshold ROIs reported as "insufficient for spatial analysis".
- **Edge handling:** no padding; ROI boundary effects acknowledged as a within-ROI uniform bias.

**Moran's I on raw continuous lineage scores** computed in parallel — coherent because input is continuous. This is what the existing `spatial_clustering.py:compute_spatial_coherence` already does for clusters; we reuse it for `lineage_immune`, `lineage_endothelial`, `lineage_stromal` per ROI per timepoint.

## 7. Threshold Sensitivity

Researcher-degree-of-freedom audits, all pre-specified:

- **Family A lineage threshold:** sweep at {0.2, 0.3, 0.4}. **Demotion criterion:** if the sign of Hedges' g reverses at any threshold for a given (category × contrast), the finding is annotated as threshold-sensitive in `endpoint_summary.csv`.
- **Family A normalization-mode sensitivity** *(Gate 6 addition, 2026-04-18; revised post-brutalist)*: parallel classification using **Sham-reference raw-marker thresholds** swept at {65, 75, 85}ᵗʰ percentiles of the Sham distribution (matches Family C philosophy and avoids outcome contamination from pooling D1/D3/D7 elevated markers into the threshold). The primary comparison is per-ROI-sigmoid vs Sham-ref-75th. Output: `interface_fractions_normalization_sensitivity.parquet`, `family_a_endpoints_global_norm.parquet`, `family_a_endpoints_norm_sweep.parquet`. **Sign-reversal and magnitude-collapse flags now propagate into the per-ROI Family A endpoint table** (`normalization_sign_reverse`, `normalization_g_collapse`, `hedges_g_sham_ref` columns) so they're visible in `endpoint_summary.csv`, not just in console output. Magnitude-stratified counts: total sign-reverse + sign-reverse-among-|g|>0.5 + magnitude-collapse-count are reported, since raw counts overcount compositionally-coupled near-zero flips.
- **Family B minimum support:** sweep at {10, 20, 40} superpixels. **Demotion criterion:** if a (cell_type × neighbor_lineage × contrast) finding appears at one support threshold but not another (due to filter exclusion), flagged as support-sensitive.
- **Family C Sham-reference percentile:** sweep at {65th, 75th, 85th} percentile of Sham distribution. **Demotion criterion:** sign reversal of g.

Sensitivity outputs reported in `sensitivity_thresholds.parquet` and `interface_fractions_normalization_sensitivity.parquet`.

## 8. Pre-Specified Outputs

`results/biological_analysis/temporal_interfaces/`:
- `run_provenance.json` — git commit, config hash, excluded ROI ID, mouse mapping table
- `interface_fractions.parquet` (Family A, primary threshold)
- `sensitivity_thresholds.parquet` (Family A/B/C sensitivity sweeps)
- `join_counts.parquet` (spatial coherence)
- `lineage_morans_i.parquet` (Moran's I on continuous lineage scores)
- `continuous_neighborhood_temporal.parquet` (Family B with neighbor-minus-self and delta-vs-Sham)
- `compartment_activation_temporal.parquet` (Family C with Sham-reference threshold)
- `endpoint_summary.csv` — single PI/reviewer-facing table with columns: family, endpoint, contrast, tp1, tp2, n_mice_1, n_mice_2, insufficient_support, mouse_mean_1, mouse_mean_2, mouse_range_1, mouse_range_2, observed_range, hedges_g, g_shrunk_skeptical, g_shrunk_neutral, g_shrunk_optimistic, pooled_std, g_pathological, bootstrap_range_min, bootstrap_range_max, n_unique_resamples, n_required_80pct, n_required_skeptical, n_required_neutral, n_required_optimistic, composite_label, threshold_sensitive. Pathological rows have NaN g_shrunk_* and NaN n_required_shrunk_*. Insufficient-support rows also have NaN for all derived statistics. (The earlier `p_proxy_from_g` / `q_proxy_*` columns were removed in Gate 6 — see §12 amendment.)

## 9. Pre-Specified Plots (in kidney_injury_spatial_analysis.ipynb)

- **Cell 11 replacement:** mouse-level dot plot. **Per timepoint per category, 2 mouse dots** (one per mouse), colored by mouse, faceted by interface category. Stacked-bar pooled view retained as a small secondary panel labelled "pooled-superpixel descriptive view, not inferential".
- **Cell 12 replacement:** quantified narrative pulled from `endpoint_summary.csv`. Table-first, prose second. Each claim cites the row.
- **New section after Part 2:** continuous neighborhood heatmap (cell_type × neighbor_lineage faceted by timepoint, values = mouse-level neighbor-minus-self delta-vs-Sham), join-count companion plot.
- **Part 6 replacement:** 4-panel grid (one per compartment) showing CD44+ rate Sham→D7 with 2 mouse dots per timepoint per compartment.

## 10. Forbidden Language

Banned in all narrative cells unless backed by a pre-declared statistic in this plan:
- "surge", "decision point", "decision zone", "coordination", "confirms", "establishes", "demonstrates"
- "peak at D3", "increase at D7" without effect size + range + n_required
- Any numeric claim ("18.5% → 21.4%") not pulled from `endpoint_summary.csv`
- "95% CI" applied to bootstrap intervals at n=2

Required language for every temporal direction claim:
- "Mouse-level mean shifted from X (range a-b) to Y (range c-d), Hedges' g = Z. Bayesian-shrunk under three priors (skeptical/neutral/optimistic): Z_sk / Z_ne / Z_op. Bootstrap range over unique resamples: [min, max]. Detecting these shrunk effects at 80% power would require n ≥ N_sk / N_ne / N_op mice per group respectively. Current pilot at n=2 cannot distinguish any of these from sampling variance; the three-prior range transparently exposes the uncertainty."

## 11. What Cannot Be Concluded

This plan generates effect-size candidates for follow-up. It does **not** establish:
- Whether any temporal trajectory is biologically real vs. inter-mouse variance
- Whether interface composition shifts are biology vs. per-ROI normalization artifact (per-ROI sigmoid normalization is a known confound; mouse-level aggregation mitigates but does not eliminate)
- Whether spatial coherence patterns are biology vs. SLIC superpixel artifact
- Whether the 22% no-lineage fraction is genuinely inert or a panel-coverage gap shifting with treatment
- Absolute values of Family B neighbor-minus-self deltas (only temporal changes are interpretable)
- Effect-size point estimates without the three-prior Bayesian shrinkage range (interpretation of a single observed g at n=2 is misleading; the shrinkage range exposes the honest uncertainty)

These limitations are restated in every notebook section consuming this plan.

## 12. Amendments

### 2026-04-18 (Gate 6 — close remaining methodological seams)
- Removed `p_proxy_from_g`, `q_proxy_within_family`, `q_proxy_pooled` columns from endpoint_summary.csv. At n=2 these were normal-CDF approximations from |g|, not real p-values; Gate 4/5 critics flagged them as cognitive-anchoring risk regardless of column-comment disclaimers. The researcher-degrees-of-freedom audit they supported (within-family vs pooled FDR rank comparison) can be recovered by sorting endpoint_summary by |hedges_g|. The `add_pooled_fdr_proxy()` helper was deleted from the module.
- Added Family A **normalization-mode sensitivity**: a parallel classification using **Sham-reference raw-marker thresholds** at {65, 75, 85}ᵗʰ percentile of the Sham distribution. Tests whether headline findings depend on the per-ROI normalization confound. Earlier draft used pooled-60th-percentile but a Gate 6 critique correctly noted that pooling across timepoints lets D7's elevated markers drive the threshold (outcome contamination). Sham-reference avoids this and matches Family C's existing convention. Output: `interface_fractions_normalization_sensitivity.parquet`, `family_a_endpoints_global_norm.parquet` (75th-percentile primary), `family_a_endpoints_norm_sweep.parquet` (all three percentiles). Sign-reversal and magnitude-collapse flags now appear directly in `endpoint_summary.csv` as `normalization_sign_reverse`, `normalization_g_collapse`, `hedges_g_sham_ref` columns on Family A rows — not just in console output.
- Fixed `src/utils/metadata.py` global cache isolation hazard: replaced module-level `_METADATA_CACHE` with `functools.lru_cache` keyed on the path string + a `clear_metadata_cache()` test helper. Different metadata paths get independent cache entries; tests no longer cross-contaminate each other.

### 2026-04-18 (Gate 4 follow-up — replace Type-M scalar with Bayesian shrinkage) + Gate 5 response
- The earlier `TYPE_M_CORRECTION = 0.65` constant was a single-scalar approximation vulnerable to the "where does 0.65 come from?" reviewer attack. Replaced with per-endpoint, per-prior **Bayesian shrinkage** using the posterior mean of δ under three prior strengths (skeptical N(0, 0.5²) / neutral N(0, 1.0²) / optimistic N(0, 2.0²)).
- Sampling variance of Hedges' g uses the standard Hedges & Olkin (1985) asymptotic formula: v(g, n) = 2/n + g²/(4n). An earlier implementation used a looser 4/n + g²/(2n) without literature support; the Gate 5 brutalist review flagged the non-standard inflation and it was switched to the textbook form. The asymptotic formula mildly under-corrects at n=2; the three-prior sensitivity range bounds that residual uncertainty.
- Output schema changes: `g_type_m_corrected` and `n_required_80pct_type_m` REMOVED from endpoint_summary.csv. Added `g_shrunk_skeptical`, `g_shrunk_neutral`, `g_shrunk_optimistic`, `n_required_skeptical`, `n_required_neutral`, `n_required_optimistic`. Pathological rows (|g|>3 AND pooled_std<0.01) now emit NaN for all g_shrunk_* and n_required_* columns — shrinking a variance-collapse artifact is meaningless.
- Narrative cells cite actual computed values from endpoint_summary.csv (triple-overlap Sham→D7: g=+3.29 shrunk to values reported live from the CSV under each prior).
- Reviewer framing: the three priors are an explicit sensitivity analysis, not a Bayesian inference. We do NOT recommend any single prior as default; the range IS the finding.

### 2026-04-18 (Gate 4 capstone review response — superseded by shrinkage amendment above)
- Hardcoded Type-M correction at 0.65 initially documented as a permissive midpoint ballpark. This approach itself was superseded in the next amendment when the full Bayesian shrinkage replacement landed.
- Other Gate 4 findings (p-proxy column naming, bootstrap-range non-coverage, Family B framing) accepted with existing disclaimers — see Gate 4 capstone log.

### 2026-04-17 (Gate 0 brutalist review response)
Incorporated 11 critical/high findings from multi-critic Gate 0 review:
- C1: CLR zero handling specified as Bayesian-multiplicative replacement + min-prevalence filter
- C2: Family B neighbor-minus-self residual bias acknowledged; only temporal changes interpretable
- C3: Type M error caveat added throughout (Gelman & Carlin 2014); n_required framed as lower bound
- H1: Pooled-across-families FDR added as sensitivity check
- H2/F3: Family C compartment threshold switched from per-ROI to Sham-reference (global)
- H3: Threshold sensitivity demotion criterion pre-specified; extended to Families B and C
- H5: Bootstrap notation changed from "95% CI" to "bootstrap range over 9 unique resamples"
- H6: `g_pathological` flag column added to endpoint_summary
- F10/F11: RESULTS.md narrative claims and TBD D9 promoted to concrete deprecation manifest entries
- M3: Scale rationale stated explicitly (10 µm a priori, 20/40 archived unanalyzed)
- M4: Test ROI exclusion criterion stated explicitly with pre-analysis timing claim
- Reproducibility freeze: git/config hash + provenance JSON specified

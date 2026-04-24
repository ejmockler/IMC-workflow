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
| Shrinkage | Every observed g is annotated with a **per-endpoint Bayesian shrinkage range** under three prior strengths on the true effect δ: skeptical N(0, 0.5²), neutral N(0, 1.0²), optimistic N(0, 2.0²). Posterior mean E[δ \| g_obs] = g_obs × prior_var / (prior_var + sampling_var), where sampling_var(g, n) = **2/n + g²/(4n)** (Hedges & Olkin 1985 asymptotic formula; switched to this textbook form in the Gate 5 amendment from an earlier non-standard 4/n + g²/(2n) — see §12). Replaces the prior single-scalar Type-M=0.65 correction with a defensible range. | n=2 winner's curse; Bayesian shrinkage is the standard treatment for noisy effect-size estimators under informative priors. |
| Multiplicity | At n=2, no real p-values exist; no FDR is computed. Family-arbitrage rank audit done by sorting `endpoint_summary.csv` on `|hedges_g|`. | Earlier proxy-FDR and per-family BH rows removed in Gate 6 (cognitive-anchoring risk) |
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

### 2026-04-23 (Phase 1 Seam 1 closure — Sham-reference sigmoid primary) + Phase 2 additive disclosure

**Methodological change.** `compute_continuous_memberships` now centers its sigmoid on a Sham-pooled per-mouse reference threshold and uses experiment-wide IQR as the sigmoid-steepness denominator, replacing the per-ROI sigmoid that was the longest-standing unresolved confound. The reference is produced by `generate_sham_reference.py` and written to `results/biological_analysis/sham_reference_10.0um.json` with full provenance (`config_sha256`, `git_hash`, `n_sham_mice`, `n_sham_rois`, `n_sham_superpixels`, `aggregation='per_mouse'`, `marker_order`, `percentile`). `batch_annotate_all_rois.py` hard-validates artifact scale, marker order, config hash, percentile, and aggregation on every run; any mismatch raises. The three previously drifting Sham-threshold primitives (`compute_global_marker_thresholds`, `compute_sham_reference_thresholds`, and the implicit per-ROI sigmoid path) now converge on a single shared primitive in `src/analysis/sham_reference.py`.

**Family A reports two normalization paths side-by-side**:
1. Sham-reference-centered sigmoid (primary, on continuous memberships)
2. Raw-marker Sham-reference percentile (independent corroboration, on raw ion counts)

Disagreement is reported per row via `normalization_sign_reverse` AND a NEW symmetric `normalization_magnitude_disagree` flag (≥2× magnitude divergence in either direction, replacing the earlier asymmetric `normalization_g_collapse` which understated disagreement). Under the re-run, **0/35 Family A endpoints at |g|>0.5 sign-reverse between paths; 13/48 disagree by ≥2× in magnitude** — the symmetric count is the honest upper bound on how much the two paths still measure differently.

**Post-hoc headline filter (locked for follow-up cohorts).** A Family A CLR endpoint is retained as a Sham→D7 co-headline if: (i) direction-consistent between the two paths, (ii) `|hedges_g| > 0.5` in the primary path, (iii) symmetric magnitude agreement (not flagged as ≥2× disagreement). This rule was defined after seeing the sensitivity sweep (post-hoc disclosure). **It is pre-specified for any follow-up cohort** and will be applied unchanged; any endpoint that passes only under a relaxed magnitude-agreement threshold will be reported as a filter-sensitivity result, not as evidence.

**CLR compositional coupling caveat.** Family A CLR endpoints operate on a closed simplex — a rise in any category mechanically forces other categories down. The Sham→D7 pattern of `stromal_clr↓` + `endothelial+immune+stromal_clr↑` is therefore **one event in two coordinates**, not independent observations. The only genuinely non-compositional corroboration source in this cohort is **Family C raw-marker CD44+ compartment rates** — different markers (CD140b not CD140a, CD31 not CD31+CD34), different threshold primitive (raw-marker Sham percentile, not sigmoid), different geometry (per-compartment rate, not simplex composition). Family B's neighbor-minus-self delta shares the Sham-reference sigmoid with Family A's continuous memberships at the input level, so it is a dependent check (different measurement geometry — graph-topological delta vs per-superpixel composition — but the same underlying normalized lineage scores).

An earlier draft of this amendment also listed a `raw_density_per_mm2` column in `roi_abundances.csv` as non-compositional corroboration. Brutalist review verified that as implemented (`count / (n_total / 2500)`), the density column reduced algebraically to `2500 × proportion` — a rescaled closed simplex, not independent evidence, because `tissue_area_mm2` inherits the SLIC target-density constant and varies by only ~2% CV across ROIs. The column was removed. Rebuilding density from the actual DNA-segmentation-mask area was deferred to Phase 1.5 follow-up; **Phase 5.1 (2026-04-23) audited that approach empirically and retracted it permanently** — the eroded DNA-mask area also has CV ≈ 0.012 across ROIs because every acquisition shares the same ~500×500 µm field-of-view. CLR findings are corroborated by Family C alone in this cohort.

**Provenance augmentation**: `run_provenance.json` now records the continuous-Sham reference artifact path + SHA256 + percentile + aggregation + n_mice/rois. `endpoint_summary.csv` populates `sham_percentile` on every Family A per_roi_sigmoid row (previously blank); Family B rows stamped `normalization_mode='sham_reference_v2_continuous'`; Family C rows stamped `normalization_mode='sham_reference_raw_marker_per_mouse'` with `sham_percentile` per row. Audit now covers all 348 endpoints, not just 48 Family A rows.

**Phase 2 additive disclosure (Seams 2 + 3)**:
- `differential_abundance_analysis.py` emits a rank-based top-5 table at `temporal_top_ranked_by_effect.csv`, sorted by `|g_shrunk_neutral|` (not raw `|g|`), with a `g_pathological` flag that quarantines variance-collapse artifacts (|g|>3 AND pooled_std<0.01) by NaN-ing their shrunken values. The table drops Mann-Whitney p-values and BH-FDR columns (at n=2 vs n=2 these are mathematically floored at 0.333/1.0 for p and 0.5/1.0 for q, inviting the "q=0.5, so it's fine" misread Gate 6 closed). This is the selection-free companion to the threshold-filtered co-headline table.
- `main_narrative` §6a cross-references Family C as the primary non-compositional corroboration for Family A CLR findings and calls out the compositional coupling explicitly.

A density-per-mm² column was drafted for `roi_abundances.csv` and removed after brutalist review (see caveat above).

**Anticipated reviewer questions (pilot limitations this study concedes, not defends)**:
- **n=2 mice per timepoint**: hypothesis-generating pilot, not confirmatory. No p-values, no FDR significance. Hedges' g unshrunken values are not the reported magnitude — the shrinkage range under three priors is.
- **Three Bayesian priors are transparency, not indecision**: neutral (N(0,1²)) is the planning default; skeptical/optimistic bound residual uncertainty. A reviewer preferring a single prior is invited to pick whichever — we report all three so no prior choice is implicit.
- **Pan-compartment CD44 rise at D7**: CD44 is a broadly-expressed injury/adhesion marker. Its rise across multiple compartments is expected biology. Family C findings are explicitly pan-tissue activation claims, not lineage-specific.
- **Bodenmiller scope (closed-by-design, not deferred)**: `run_bodenmiller_benchmark.py` validates the IMC data loader against published channel-level data (Spearman r=0.996). It is NOT a test of Family A/B/C, and **cannot become one** — the Bodenmiller Patient1 dataset is single-timepoint, single-patient, different organ (pancreas vs kidney), different species (human vs mouse), and uses a different antibody panel; the framework requires temporal sampling that does not exist in that dataset. This is a permanent scope boundary, not a Phase-N follow-up. Disclosed in archived `benchmarks/STATUS.archived.md`; `benchmarks/notebooks/CRITICAL_ANALYSIS.md` was retracted (`.retracted.md`) because the original superpixel-count premise was off by ~50–100×.
- **Shared-reference tautology between Family A paths**: both paths anchor on the same Sham baseline. Sign agreement between them is partly built-in. The symmetric magnitude-disagreement count is the honest upper bound on independent measurement; it is not claimed as independent replication.

**Deferred to Phase 1.5 (documented follow-up, not blocking current commit)**:
- ~~Continuous Sham-percentile sensitivity sweep at 50/60/70 pct (companion to the existing raw-marker 65/75/85 sweep).~~ **Closed 2026-04-23 Phase 1.5b — see amendment below.**
- ~~Parallel raw-marker Sham-reference path for Family B neighbor-minus-self (currently Family B inherits the Sham-reference sigmoid but has no independent audit path).~~ **Closed 2026-04-23 Phase 1.5c — see amendment below.**
- ~~Pre-registration obligations §Family B support-sensitivity demotion flag + §Family A CLR-without-`none` sensitivity propagation (pre-existing gaps; unrelated to Seam 1 work).~~ **Closed 2026-04-23 Phase 1.5a — see amendment below.**
- ~~Tissue-mask-based non-compositional density (rebuild `raw_density_per_mm2` from DNA-segmentation mask area rather than the SLIC target-density constant).~~ **Closed 2026-04-23 Phase 5 — investigated and retracted permanently by acquisition-design constraint; see amendment below.**

### 2026-04-23 Phase 5.1 (tissue-mask area-based density — closed for this acquisition; alternatives flagged untested)

Pre-registered non-degeneracy gate for an area-based density column: across ROIs, CV(tissue_area_mm2) must exceed 0.05, **and** for the candidate density column to be independent of proportion, Pearson |r| between `density_per_mm2` and `proportion` across ROIs (per cell type) must fall below 0.95.

`audit_tissue_mask_density.py` recomputes tissue area per ROI from the persisted `superpixel_labels` array (count of pixels with label ≥ 0, times the resolution-squared — the eroded DNA-mask area as actually fed to SLIC). Across all 24 ROIs at 10 µm scale:

**Algebraic reduction (lead finding).** `density_per_mm2 = count / tissue_area_mm2`. Because every ROI is acquired at the same ~500×500 µm field-of-view, `tissue_area_mm2 = 0.246 ± 0.003` (CV 0.012) is dataset-constant. Therefore `density = count / 0.246 ≈ 9857 × proportion ± 1.4%` — algebraically identical (within ~1% noise) to a rescaled proportion. This reduction does not depend on Pearson r; it follows from the constancy of the denominator.

**CV diagnostic (supporting).** Mean tissue_area_mm2 = 0.246; std = 0.003; **CV = 0.012** — below the 0.05 bar by ~4×. The CV is necessary but not sufficient evidence of degeneracy; the algebraic reduction is what closes the door.

**Root cause**: the acquisition design fixes the field-of-view at the calibration step. There are no spike-in normalization beads or panoramic acquisitions that would let tissue extent vary across ROIs. The limitation is the acquisition design, not the SLIC pipeline.

**Closure scope (narrow)**: this finding closes the **area-based** density column on this acquisition design. It does NOT close every conceivable non-compositional corroboration surface. Untested alternatives that a future cohort or a re-segmentation effort could investigate:

- **Per-nucleus density** from a DNA-channel watershed segmentation (`src/analysis/watershed_segmentation.py` exists but is not wired to the production pipeline; the denominator becomes `n_nuclei` per ROI, which would track tissue cellularity rather than tissue area and could vary meaningfully across timepoints).
- **DNA-intensity integral** as the denominator (∑DNA over the eroded mask), which captures cellularity-weighted tissue volume rather than 2D area — varies with nuclear count even when area is fixed.
- **Variable-extent re-acquisition** (whole-section IMC, panoramic montage) on a follow-up cohort, which restores meaningful CV(tissue_area_mm2).
- **Spike-in absolute-quantification controls** in a redesigned panel — would let raw-count comparisons across ROIs be normalized to absolute concentration rather than relative composition.

The Phase 5 scope did not test any of these because (a) per-nucleus density requires running the watershed pipeline end-to-end (not yet wired), (b) DNA-integral requires re-running ion-count aggregation with a new denominator definition, (c) variable-extent re-acquisition is a wet-lab change. The conservative claim is "area-based density on the present acquisition is closed; cellularity-based and absolute-quant alternatives remain open for a future engineering cycle."

Family C (raw-marker CD44+ compartment activation) remains the only currently-implemented non-compositional corroboration for Family A CLR findings in this cohort. The closure is documented empirically — the audit CSV carries the CV, the algebraic reduction, the gate verdict, and the scope language in its trailing metadata so it cannot drift from the numbers.

Output: `results/biological_analysis/tissue_area_audit.csv` (24 rows + gate-verdict footer + closure-scope footer).

### 2026-04-23 Phase 1.5c (Family B parallel raw-marker audit — magnitude divergence, comparable headline yield) — corrected 2026-04-23 Phase 5.5

`audit_family_b_raw_markers.py` runs Family B neighbor-minus-self with the sigmoid Sham-reference lineage scores replaced by raw arcsinh markers (`lineage_immune_raw = CD45`; `lineage_endothelial_raw = mean(CD31, CD34)`; `lineage_stromal_raw = CD140a`). The neighbor-minus-self operation is differential, so any Sham-reference additive offset cancels — the raw-marker path is genuinely sigmoid-independent.

**Correction (Phase 5.5 brutalist gate)**: an earlier draft of this amendment claimed "**0** sigmoid Family B Sham→D7 endpoints clear `|g_shrunk_neutral| > 0.5`" against "18 raw-marker endpoints". The sigmoid count was assumed without verification; the audit script printed only the raw-marker count. Re-running with both counts surfaced **21 sigmoid endpoints** above the same floor (none pathological, none support-sensitive) — comparable to raw, not zero. The corrected picture:

| Path | Sham→D7 endpoints \|g_shrunk_neutral\|>0.5 | max raw \|hedges_g\| |
|------|------:|------:|
| sigmoid (primary, sham_reference_v2_continuous) | **21** | 13.73 |
| raw-marker (sigmoid-independent) | **18** | 10.85 |
| both above floor | 14 (Jaccard 0.58 over union) | — |
| sigmoid sign-agreement with raw at \|g\|>0.5 | 96% (1 reversal in 25 nontrivial) | — |

**Actual finding** (not "discovery raw missed by sigmoid"): the two bases yield comparable headline counts (21 vs 18) with high directional agreement (96%) and substantial overlap (14 shared headlines). The audit's pre-shrinkage magnitude-disagreement count (55/166 ≥2× divergence at |g|>0.5) is real and reflects raw arcsinh markers retaining wider dynamic range than sigmoid lineage scores; Bayesian shrinkage neutralizes most of that divergence in `g_shrunk_neutral`. Top raw-marker rows (raw |g|, then g_neut) sorted by |hedges_g|:

| composite_label | endpoint | raw g | g_neut |
|-----------------|----------|------:|-------:|
| stromal | delta_lineage_immune | +3.98 | +1.00 |
| activated_endothelial_cd44 | delta_lineage_immune | +4.11 | +1.00 |
| activated_stromal_cd140b_cd44 | delta_lineage_endothelial | +4.45 | +0.99 |
| activated_endothelial_cd44 | delta_lineage_endothelial | +3.33 | +0.98 |
| activated_endothelial_cd140b | delta_lineage_immune | +3.08 | +0.97 |

**Pre-registration status of the raw-marker path**: this lineage-source basis was not pre-registered as a primary path in the original plan — the pre-reg pre-specified Family B on continuous memberships. The raw-marker audit is a sensitivity-analysis check, not a replacement. Its substantive contribution is showing that raw effect-size estimates are larger pre-shrinkage but converge under Bayesian shrinkage to a comparable headline set.

Output: `family_b_raw_marker_audit.parquet` (540 rows = 270 primary + 270 raw); `family_b_raw_marker_comparison.csv` (per-endpoint sign_reverse + magnitude_disagree flags across the two lineage sources). Audit script now prints both sigmoid and raw counts at run time so this kind of one-sided reporting cannot recur silently.

### 2026-04-23 Phase 5.2 (Family B lineage-source basis — pre-reg amendment for follow-up cohorts)

Phase 1.5c showed that the sigmoid and raw-marker bases produce comparable headline counts (21 vs 18 at |g_shrunk_neutral|>0.5, 14 shared, 96% sign agreement) but diverge substantially in pre-shrinkage magnitude (55/166 ≥2× divergence at |g|>0.5). For follow-up cohorts the open question is: when the two bases disagree on a specific endpoint or on overall headline yield, what does the cohort report?

**Co-primary, intersection-conservative rule** (locked for follow-up cohorts):

1. **Both lineage-source bases are co-primary.** Every Family B run emits per-basis endpoint tables and a basis-divergence CSV. Neither is demoted to a sidecar.
2. **Lineage-source mapping is panel-defined, not marker-list-defined.** A cohort's `config.json` declares which raw-marker channels compose each lineage axis (current pilot: `immune = max(CD45)`; `endothelial = mean(CD31, CD34)`; `stromal = max(CD140a)`). The principle: each raw-lineage channel-set is the *minimal disjoint subset of panel markers that the same lineage's sigmoid score also draws from*. A panel substitution (e.g., adding CD11b to immune) updates the raw-mapping in `config.json` and the audit re-runs unchanged. **The rule is not the marker list; the rule is "raw-arcsinh composite over the same channel-set the sigmoid lineage draws from."**
3. **Headline qualification is per-basis.** A Family B endpoint is a headline if it clears `|hedges_g_shrunk_neutral| > 0.5` AND is not `g_pathological` AND is not `support_sensitive`.
4. **Cohort report names three sets explicitly.** (a) **Conservative-intersection headline set**: endpoints clearing the floor in *both* bases with same sign — the strongest claim. (b) **Sigmoid-only set** + (c) **Raw-only set**: endpoints that clear in one basis but not the other; reported but flagged as basis-dependent.
5. **Opposite-sign-same-endpoint resolution.** If an endpoint clears |g_neut|>0.5 in both bases but the signs disagree, neither basis enters the headline set; the row is moved to a `family_b_basis_conflict.csv` table for inspection. Conflicting-sign headlines are evidence that the lineage-score primitive is doing different things in the two bases on that endpoint.
6. **Both-zero outcome.** If both bases produce zero headlines, the cohort reports the null result for Family B alongside the basis-divergence summary; the rule does not invent a fallback.
7. **Mandatory divergence summary.** Every run emits `family_b_basis_divergence.csv` with columns `endpoint, composite_label, contrast, hedges_g_sigmoid, g_shrunk_neutral_sigmoid, hedges_g_raw, g_shrunk_neutral_raw, sign_agree, headline_overlap_status` (one of: `both_above`, `sigmoid_only`, `raw_only`, `both_below`, `conflict`).

**Why intersection-conservative not co-primary disjunction**: a brutalist critic correctly noted that reporting *every* headline from either basis as a co-primary claim is a disjunctive OR over correlated tests, inflating effective multiplicity. Naming the intersection set as the strongest claim and the basis-only sets as basis-dependent (not refuting, but qualifying) preserves transparency without overclaiming.

**Why panel-defined mapping not marker list**: the current rule's hard-coded `[CD45]/[CD31, CD34]/[CD140a]` would not survive a panel that includes CD11b, F4/80, or Ly6G (myeloid) alongside CD45. Specifying the raw-mapping at the same level the sigmoid lineage is defined (`config.json` lineage definitions) makes the audit panel-portable.

**Why headline-count disparity is *not* the gating signal**: an earlier draft of this amendment used "0-vs-18 headlines" as the motivating disparity. Phase 5.5 brutalist correction surfaced that the actual counts were 21-vs-18, comparable. The intersection-conservative rule does not depend on a headline-count disparity to justify itself; it is the right discipline for any cohort where two correlated bases co-exist, including the present cohort.

**Implementation status**: the rule above is pre-registered for any follow-up cohort and is partially implemented in the current pilot artifacts. `audit_family_b_raw_markers.py` already prints both sigmoid and raw headline counts at run time. The `family_b_basis_divergence.csv` and `family_b_basis_conflict.csv` outputs and the per-basis row in `endpoint_summary.csv` are pipeline-integration work for Phase 5.6 and beyond; until then the basis comparison is reachable via the existing `family_b_raw_marker_comparison.csv` and the audit script's stdout.

This rule was defined after observing Phase 1.5c (post-hoc for the present pilot, pre-registered for any follow-up cohort). Applied unchanged in any future cohort; any relaxation is a filter-sensitivity result, not evidence.

### 2026-04-23 Phase 1.5b (continuous Sham-percentile sensitivity sweep)

`sweep_continuous_sham_pct.py` tests whether Family A CLR endpoints are stable when the Sham-reference percentile driving `compute_continuous_memberships`' sigmoid center varies across {50, 60, 70}. Runs in-memory (no parquet round-trip, no cascade re-run). Does not alter persistent pipeline artifacts.

**Stability result (48 Family A endpoints × 3 percentiles)**:
- 3/48 endpoints sign-mix across the sweep (all |g|)
- 2/41 endpoints sign-mix at |g|>0.5
- Triple-interface headline is Sham-pct-invariant under neutral shrinkage: `endothelial+immune+stromal_clr` Sham→D7 neutral g = 1.000 / 0.987 / 0.965 at pcts 50 / 60 / 70
- `stromal_clr` raw g varies widely (−13.75 → −6.71 → −3.75) but its neutral-shrunk value is more contained (−0.54 → −0.88 → −1.00) — shrinkage is load-bearing for magnitude interpretation

Output: `family_a_continuous_sham_pct_sweep.parquet` (144 rows); `continuous_sham_pct_sweep.csv` (per-endpoint sign-mix + relative-range stability).

### 2026-04-23 Phase 1.5a (pre-registration compliance closure)

Two pre-existing pre-reg obligations (predating Seam 1 work) were flagged by the Phase 1 post-cascade brutalist gate as missing in code. Both are now in `endpoint_summary.csv`:

- **§Family B support-sensitivity demotion (plan §87)**: `run_family_b` now computes presence sets across the `min_support` ∈ {10, 20, 40} sweep. Every primary (ms=20) endpoint row is stamped `support_sensitive=True` if its `(endpoint, contrast, composite_label)` key is missing at any of the three supports (filter-fragility). Initial run: 90 of 270 Family B endpoints flagged support-sensitive — a non-trivial fraction, reported for every row rather than buried in a sidecar table.

- **§Family A CLR-without-`none` sensitivity (plan §31)**: `run_family_a` now computes Family A endpoints on `interface_clr_no_none` alongside `interface_clr` and surfaces two new columns in the primary Family A endpoint table: `hedges_g_no_none` (the raw g under the 7-category CLR) and `clr_none_sensitivity=True` if the sign of `hedges_g` reverses when `none` is excluded. Initial run: 0 of 48 Family A endpoints flip sign when the `none` category is excluded — Family A trajectories are qualitatively robust to that compositional choice. This is itself a pre-registered result, not a silent pass.

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

### 2026-04-20 (post-commits doc-audit — plan §5 multiplicity row consolidation)
- Collapsed two contradictory multiplicity table rows in §5 (Statistical Methods) into one. The surviving row correctly states that at n=2 no real p-values exist, no FDR is computed, and family-arbitrage is audited by `|hedges_g|` rank. The removed row ("BH-FDR within each endpoint family separately") was a Gate 6 partial-removal residue — proxy-FDR columns were removed but the descriptive row survived and contradicted the row below it. Scope-limited to §5 table; no amendments to endpoints, contrasts, or filters.
- Also updated `analysis_plans/deprecation_manifest.md` D2 entry to reference Bayesian shrinkage instead of the superseded Type-M scalar, pointing to this amendment log as the authoritative transition.

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

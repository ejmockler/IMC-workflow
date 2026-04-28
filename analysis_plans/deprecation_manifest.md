# Deprecation Manifest: Code Paths Replaced or Removed

**Date frozen:** 2026-04-17 (amended 2026-04-18 after Gate 0 brutalist review)
**Scope:** Every code path that becomes orphaned, wrong, or replaced by the temporal interface analysis effort. Each entry: location, defect, action, replacement.

**Rule:** No deprecated path may be left as a `# deprecated` comment, dead branch, or "alongside the old way" display. Every entry below is either deleted or replaced in-place.

> **Phase 1.5 / 5 scope note (2026-04-23).** Phase 1 (Sham-reference normalization), Phase 1.5 (continuous-Sham-pct sweep + Family B raw-marker basis + pre-reg compliance flags), and Phase 5 (deferred-item closures + Phase 1.5c factual correction + freeze-manifest verifier) are **closures of pre-existing deferrals**, not new deprecations. They are tracked as amendment blocks in `analysis_plans/temporal_interfaces_plan.md` and reflected in `review_packet/FROZEN_PREREG.md` "Closures and remaining open work" section. This deprecation manifest remains a Phase 1/2-era historical record; it is not the right place to log Phase 1.5/5 work.

---

## D1 — Pseudoreplicated interface fractions
- **Location:** `notebooks/biological_narratives/kidney_injury_spatial_analysis.ipynb` cell 11
- **Defect:** Computes interface category fractions over all superpixels per timepoint via `tp_df['interface_type'].value_counts(normalize=True)`. No mouse-level aggregation. Treats ~3000 superpixels as independent samples when they come from 2 mice.
- **Action:** REPLACE in-place
- **Replaced by:** Mouse-level dot plot consuming `results/biological_analysis/temporal_interfaces/interface_fractions.parquet` (T27)
- **Stacked bar retention:** Allowed only as secondary descriptive view with explicit "pooled-superpixel view, not inferential" caption.

## D2 — Eyeballed interface narrative claims
- **Location:** `notebooks/biological_narratives/kidney_injury_spatial_analysis.ipynb` cell 12 ("What the Interfaces Reveal")
- **Defect:** Numeric claims ("Triple-positive interfaces increase at D7 (18.5% → 21.4%)", "Immune-stromal interfaces peak at D7", "Endothelial-immune interfaces peak at D3") are not computed in the notebook. Source unknown; possibly chart eyeballing.
- **Action:** REPLACE in-place
- **Replaced by:** Quantified narrative pulled directly from `endpoint_summary.csv`. Each claim accompanied by mouse-level mean, observed range, Hedges' g plus Bayesian-shrunk range under three priors (skeptical / neutral / optimistic), n_required for 80% power under each prior (T28). (The earlier Type-M scalar correction referenced here was replaced by Bayesian shrinkage in the Gate 5 amendment; see `analysis_plans/temporal_interfaces_plan.md` amendment log.)

## D3 — D7-only multi-compartment coordination
- **Location:** `notebooks/biological_narratives/kidney_injury_spatial_analysis.ipynb` cells 30 and 31 (Part 6)
- **Defect:** CD44+ activation rate across CD45+/CD31+/CD140b+ compartments computed only at D7 with per-ROI 75th percentile threshold (floating). Narrative makes implicit temporal claims ("By Day 7, ALL compartments are activating") without temporal contrast.
- **Action:** REPLACE in-place
- **Replaced by:** 4-panel grid showing CD44+ rate across Sham/D1/D3/D7 per compartment, mouse-level dots, **Sham-reference threshold (not per-ROI)**. Consumes `compartment_activation_temporal.parquet` (T29).

## D4 — Circular self-stratified continuous neighborhoods
- **Location:** `spatial_neighborhood_analysis.py` lines 314-330 (within `analyze_roi_neighborhoods_continuous`) and lines 510-532 (orchestration block writing `continuous_neighborhood_summary.csv`)
- **Defect:** Stratifies neighbor lineage scores by `composite_label`, but `composite_label` is itself derived from those same lineage scores. Self quantity is tautological.
- **Action:** REFACTOR
  - Keep: per-superpixel neighbor lineage scores (the `neighborhood_df`).
  - Remove: the `roi_summary` self/neighbor side-by-side dictionary (lines 317-330).
  - Update: orchestration block emits only neighbor-side quantities; writing `continuous_neighborhood_summary.csv` in its current schema is removed (replaced by new module's parquet output).
- **Replaced by:** New module function `compute_continuous_neighborhood_temporal` with neighbor-minus-self framing, delta-vs-Sham, and minimum-support filter (T23).

## D5 — Missing D1_vs_D7 in DA framework
- **Location:** `differential_abundance_analysis.py` lines 284-290 (discrete pairs) AND lines 383-386 (continuous pairs)
- **Defect:** Both `timepoint_pairs` lists omit `('D1', 'D7')`. Documentation refers to "all pairwise comparisons" but only 5 of 6 are computed.
- **Action:** REPLACE in-place
- **Replaced by:** Both lists augmented with `('D1', 'D7')`. Regenerate `temporal_differential_abundance.csv` (expected 138 rows from 115). One-line fix per location (T19).

## D6 — Stale narrative claims in main_narrative
- **Location:** `notebooks/main_narrative.ipynb` cells 9, 11, 26
- **Defect:** Percentages like "30% single-lineage / 48% multi-lineage / 22% no-lineage" recited from older runs. "Cross-type spatial patterns suggest emerging immune–endothelial reorganization at D7" in cell 26 not quantified anywhere.
- **Action:** REPLACE in-place
- **Replaced by:** Numbers pulled from `endpoint_summary.csv` programmatically. Cross-type claim either backed by `continuous_neighborhood_temporal.parquet` finding or removed (T31).

## D7 — Stale narrative in steinbock_concordance
- **Location:** `notebooks/methods_validation/benchmarks/steinbock_concordance.ipynb`
- **Defect:** None known. Audit only.
- **Action:** AUDIT only (T31 sweep includes audit).

## D8 — gradient_discretization API demo
- **Location:** `notebooks/methods_validation/01_technical_methods/gradient_discretization.ipynb`
- **Defect:** None known after cycle 2 fixes. Cross-references should still resolve to new outputs.
- **Action:** AUDIT only (T31 sweep includes audit).

## D9 — Orphan helpers (concrete after Gate 0 audit)
- **Location:**
  - `spatial_neighborhood_analysis.py:339-346` — `parse_roi_metadata` redundant wrapper around `_parse_roi_metadata_canonical` (Codex Z5).
  - `spatial_neighborhood_analysis.py:265-336` — entire `analyze_roi_neighborhoods_continuous` function: after D4 refactor strips `roi_summary`, the remaining `neighborhood_df` is consumed only by the orchestration block in lines 510-532. Decision: subsume into new module's `compute_continuous_neighborhood_temporal`; delete old function entirely.
  - `spatial_neighborhood_analysis.py:540-555` — output writers for `regional_neighborhood_enrichments.csv`, `roi_neighborhood_enrichments.csv` if no consumer remains after sweep (verify via grep).
- **Action:** DELETE in T32 sweep.

## D10 — Memory entries that may become stale
- **Location:** `/Users/noot/.claude/projects/-Users-noot-Documents-IMC/memory/`
  - `annotation_system.md` references "30.2% / 48.1% / 21.7%" decomposition — must be updated with current numbers from new outputs.
  - `MEMORY.md` index needs entry pointing to the new module.
- **Action:** UPDATE (T36).

## D11 — `figure_interface_composition` in viz utils [ADDED 2026-04-18]
- **Location:** `src/viz_utils/comprehensive_figures.py:275-354` (function definition) and line 611 (unconditional call in `generate_figures`)
- **Defect:** Produces pseudoreplicated superpixel-level interface composition figure. Identical defect to D1, but at the pipeline-figure-generation level. Will run on every figure regeneration even after D1 is fixed in the notebook, leaving the deprecated visualization on disk and in any downstream consumption.
- **Action:** DELETE function and call site
- **Replaced by:** New module's mouse-level outputs are consumed by notebook only; no parallel figure-pipeline output for interface composition. If a figure-pipeline consumer is needed later, it consumes `interface_fractions.parquet` not raw annotations.

## D12 — Hardcoded 0.3 lineage threshold in batch_annotate [ADDED 2026-04-18]
- **Location:** `batch_annotate_all_rois.py:184`
- **Defect:** `above_thresh = {ln: (scores > 0.3).sum() for ln, scores in ls.items()}` hardcodes threshold 0.3, while `config.json:406` defines `subtype_threshold: 0.3` as the configurable value. The Family A threshold sensitivity sweep at {0.2, 0.3, 0.4} requires this be config-driven; otherwise the sweep changes downstream consumers but not this script's summary statistics.
- **Action:** REPLACE in-place
- **Replaced by:** `threshold = config.cell_type_annotation.get('subtype_threshold', 0.3)`; pass through to summary computation. Fix in T19 alongside DA bugfix (related single-line correction).

## D13 — RESULTS.md narrative claims using forbidden language [ADDED 2026-04-18]
- **Location:** `RESULTS.md` lines 91, 112, 264, 332, 334, 336, 389
  - Line 91, 264: "decision point" (forbidden term per plan §10)
  - Line 332, 334, 336: "Spatial data confirms" (overclaims; should be "is consistent with")
  - Line 389: "Fibroblast surge" (forbidden term)
  - Line 112: "decision point" / "myeloid-stromal crosstalk zone" (mechanistic overclaim)
- **Action:** REPLACE in-place during T35 (RESULTS.md update)
- **Replaced by:** Language conforming to §10 of pre-registration. Quantitative findings pull from `endpoint_summary.csv`.

## D14 — README and workflow docs endorse old script as biological analysis entry point [ADDED 2026-04-18]
- **Location:**
  - `README.md:16-17` — describes `spatial_neighborhood_analysis.py` as biological analysis script
  - `docs/architecture/WORKFLOW_INTEGRATION.md:147` — same endorsement
- **Defect:** Even after D4 strips circular self-stratification, the documentation still routes users to a script that produces deprecated outputs (per-superpixel-level summaries with no mouse aggregation). New users will run the wrong entry point.
- **Action:** UPDATE in T35 documentation pass
- **Replaced by:** Documentation points to the new temporal interface module as the entry point for multi-lineage analyses. The old script's role is reduced to "discrete neighborhood enrichment for cell-type pairs" (which it does correctly at mouse level).

## D15 — Orphaned spatial neighborhood output files [ADDED 2026-04-18]
- **Location:** `results/biological_analysis/spatial_neighborhoods/`
  - `regional_neighborhood_enrichments.csv` — no current consumer in notebooks or scripts.
  - `roi_neighborhood_enrichments.csv` (953 KB) — no current consumer in notebooks or scripts.
  - `continuous_neighborhood_summary.csv` — replaced by new parquet outputs.
- **Action:** DELETE generation in T32 sweep (after confirming via grep no consumer exists)
- **Replaced by:** Either nothing (if truly unused) or routed through the new module's typed parquet outputs.

## D16 — Test files asserting D7-only narrative claims [ADDED 2026-04-18]
- **Location:** `tests/test_biological_metrics.py:196, 201, 209, 214, 241`
- **Defect:** Tests encode biological assertions tied to per-ROI 75th percentile compartments and D7-only patterns. After D3/D11 changes (Sham-reference threshold, deleted figure function), these tests will either pass irrelevantly (testing on old code path that no longer exists) or fail (asserting deprecated behavior).
- **Action:** AUDIT in T32; REPLACE assertions with tests against new module's outputs OR DELETE if no longer applicable
- **Replaced by:** Tests in `tests/test_temporal_interface_analysis.py` (T18) cover the new module. Old assertions about D7-only behavior are removed since the new analysis is temporal across all 4 timepoints.

---

## Action summary

| ID | Action | Phase task |
|----|--------|------------|
| D1 | Replace cell 11 | T27 |
| D2 | Replace cell 12 | T28 |
| D3 | Replace Part 6 (cells 30+31) | T29 |
| D4 | Refactor self-stratification block | T23 (and T32 sweep) |
| D5 | Fix DA D1_vs_D7 omission | T19 |
| D6 | Update main_narrative narrative cells | T31 |
| D7 | Audit steinbock | T31 |
| D8 | Audit gradient_discretization | T31 |
| D9 | Sweep orphan helpers (concrete list) | T32 |
| D10 | Update memory | T36 |
| D11 | Delete `figure_interface_composition` | T32 |
| D12 | Fix hardcoded 0.3 in batch_annotate | T19 |
| D13 | RESULTS.md narrative cleanup | T35 |
| D14 | README/WORKFLOW_INTEGRATION docs update | T35 |
| D15 | Delete orphaned output writers | T32 |
| D16 | Audit/replace test assertions | T32 |

## What is explicitly NOT deprecated

- The discrete boolean gating engine (`cell_type_annotation.py`) — still used; produces inputs for the new module. Phase 7 P1 promoted the implicit dict-iteration priority order to an explicit `config.cell_type_annotation.priority_order` list.
- The continuous membership engine — still used; produces inputs. Phase 7 P1 (`_derive_composite_labels`) now emits a `c:` prefix on every value to disambiguate from discrete `cell_type` values.
- The existing `temporal_differential_abundance.csv` — fixed (D5) and `g_pathological` flagged but kept; the new effort adds parallel CLR-corrected interface analyses, doesn't replace the cell-type DA.
- The Steinbock benchmark, gradient_discretization, INDRA notebooks — no defects identified.
- The discrete neighborhood enrichment in `spatial_neighborhood_analysis.py` (the cell-type-pair mouse-level path) — operates correctly at mouse level; only the continuous self-stratified path (D4) is removed.

## Phase 7 deprecations (2026-04-28; locked after three brutalist rounds)

Phase 7 introduced four families of dropped designs. Each is encoded as a machine-checkable invariant in `tests/test_phase7_dropped_designs.py` so it cannot silently return.

| ID | Dropped | Reason | Round | Guard test |
|----|---------|--------|-------|------------|
| D17 | `composite_label_v1` deprecation column | No release process to amortize against; single-commit migration with notebook regen + runtime assertion is the operationally honest replacement | Round 1 (Codex High; Claude Architect) | `test_no_composite_label_v1_column_in_endpoint_summary` + `test_no_composite_label_v1_in_annotation_parquets` |
| D18 | ALR-style CLR with chosen reference category | Math error confused with CLR; revised CLR has no reference (geometric-mean denominator, all N coordinates) | Round 1 (Codex Critical #1) | `test_no_clr_reference_category_column` |
| D19 | Confidence sweep `{0.5, 0.7, 0.9}` for Family A_v2 | Empirically meaningless — actual confidence values are exactly `{0.0, 0.333, 0.5, 1.0}` (per `cell_type_annotation.py::annotate_cell_types` confidence = 1/n_assignments); floors of 0.7 and 0.9 partition identically | Round 1 (Codex Critical #5; verified 58,137 superpixels) | `test_no_confidence_floor_sweep_column_in_endpoint_summary` |
| D20 | Raw-marker corroborating path for Family A_v2 (`classify_celltype_per_superpixel_global_markers`) | Would not be corroboration: shares the entire taxonomy (panel + 15 cell types + priority resolution + lineage groupings + activation suffix logic) with the primary path; only thresholds differ. The conservative-intersection rule does no real multiple-comparison work in this geometry — it filters threshold-edge cases | Round 1 (Codex Critical #1; Claude Finding B) | `test_no_raw_marker_corroborating_path_function_in_temporal_interface_module` |

**Two further round-2 designs were considered and revised, not dropped**:
- "Drop `unassigned` from CLR simplex" — round-1 patch; round-2 found this changed the estimand from "do cell-type proportions shift?" to "do typed proportions shift among assigned superpixels?". Revised: `unassigned` is the 16th coordinate; gmean-drag accepted as feature. Documented in Phase 7 spec §4.1.
- "Never co-headlined" cross-rule as prose — round-2 found prose conventions don't bind readers. Revised: runtime demotion column `headline_demoted_reason = 'cross_axis_co_headline_forbidden'`; small structural reach (~6/1296 endpoints) acknowledged.

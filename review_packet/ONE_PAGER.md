# Spatial proteomics of murine AKI — pilot one-pager

**Study design.** Unilateral ureteral obstruction (UUO) in C57BL/6 mice, n=2 mice per timepoint at Sham / Day 1 / Day 3 / Day 7, 24 ROIs total. Imaging mass cytometry (IMC) with a 9-marker panel. SLIC superpixels at 10 μm (pinned a priori). Pilot, hypothesis-generating — not confirmatory.

**Panel → kidney cell types (brief).** CD45 marks all leukocytes. CD11b is a myeloid common marker (neutrophils, monocytes, macrophages). Ly6G is a neutrophil marker in mice (not cleanly neutrophil-specific: some monocytes express it transiently). CD206 marks alternatively-activated (M2-like) macrophages. CD31 + CD34 together mark endothelium; neither alone is specific (CD34 also appears on hematopoietic progenitors and some fibroblasts). CD140a (PDGFRα) marks fibroblasts broadly; CD140b (PDGFRβ) marks pericytes and activated mesenchymal cells. CD44 is a hyaluronan receptor that rises broadly under tissue injury and adhesion — expected to be pan-compartment, not lineage-specific. The panel **excludes tubular epithelium** (no E-cadherin, no KIM-1, no aquaporin), **lymphocyte subsets** (no CD3/CD4/CD8/CD20), and advanced macrophage polarization markers. ~86% of tissue does not match any discrete cell-type gate as a consequence of panel design; the continuous lineage system partially recovers this by allowing multi-lineage scoring.

**Question.** Does tissue-patch-resolution spatial proteomics at this scale surface candidate findings worth powering in a larger cohort?

**Methods (at a glance).** Each 10 μm superpixel is annotated two ways: (i) discrete cell-type labels via boolean positive/negative gating rules (15 config-defined types + `unassigned`); (ii) continuous lineage memberships on three non-exclusive axes (immune, endothelial, stromal) via Sham-reference sigmoid (center = Sham pooled per-mouse 60th percentile; scale = experiment-wide IQR). Phase 7 (2026-04-28) extends each endpoint family with a discrete-cell-type counterpart; v1 and v2 surfaces are reported in parallel under a runtime cross-rule that prevents v2 from co-headlining with v1 on the same biological event.

| Family | Unit | v1 metric | v2 metric (Phase 7) |
|--------|------|-----------|---------------------|
| A | Mouse-level interface composition | CLR of 8 interface categories | CLR of 16 discrete cell types (incl. `unassigned`) |
| B | Mouse-level neighbor-minus-self | Continuous lineage delta around each `c:`-prefixed composite label | Continuous lineage delta around each discrete `cell_type` |
| C | Mouse-level compartment activation | Raw-marker CD44⁺ rate within each Sham-referenced compartment | Single-row CD44⁺ rate within `cell_type == 'neutrophil'` (the only non-tautological discrete-celltype compartment) |

Sample-size constraint at n=2 vs n=2 sets a hard floor on Mann-Whitney p (0.333) — no FDR-significant findings are possible by construction. Reported magnitudes are Hedges' g under three Bayesian shrinkage priors (skeptical N(0, 0.5²) / neutral N(0, 1.0²) / optimistic N(0, 2.0²)).

**Candidate findings (Sham → D7, neutral-shrunk g).** Passes the pre-registered filter: |hedges_g| > 0.5 AND direction-consistent between the two Family A normalization paths AND not flagged as ≥2× symmetric magnitude disagreement. Family C v1 (raw-marker compartments) and v2 (neutrophil-gated compartment) are both single-path; the magnitude-disagreement check does not apply.

**Family A v1 (composite_label 8-cat) + Family C (v1 + v2):**

| Endpoint | Family | g_skep | **g_neut** | g_opt | n_req (neutral) | Corroboration status |
|----------|:------:|-------:|-----------:|------:|---------------:|:--------------------|
| endothelial+immune+stromal_clr | A v1 | 0.32 | **+0.99** | 2.11 | 17 | Sham-ref sigmoid |
| triple_overlap_fraction | C v1 | 0.32 | **+0.98** | 2.08 | 17 | Raw markers, Sham-ref pct — independent of CLR closure |
| neutrophil_compartment_cd44_rate | C v2 | 0.30 | **+1.00** | 2.34 | 16 | Phase 7 neutrophil-gated compartment — non-tautological because `cell_type=='neutrophil'` is the only discrete celltype not pinned by gate construction |
| background_compartment_cd44_rate | C v1 | 0.31 | **+0.95** | 1.91 | 18 | Raw markers — independent of CLR closure |
| CD140b_compartment_cd44_rate | C v1 | 0.24 | **+0.64** | 1.11 | 38 | Raw markers — independent of CLR closure |
| endothelial+immune_clr | A v1 | 0.21 | **+0.54** | 0.90 | — | Sham-ref sigmoid |
| immune+stromal_clr | A v1 | 0.19 | **+0.50** | 0.83 | — | Sham-ref sigmoid |
| immune_clr | A v1 | –0.15 | **–0.39** | –0.63 | — | Sham-ref sigmoid |

**Family A v2 (discrete cell-type 16-cat, Phase 7) — single-path, rule = `|g|>0.5 AND not g_pathological`:** the v1/v2 cross-rule (`headline_demoted_reason='cross_axis_co_headline_forbidden'`) prevents v2 from co-headlining a v1 lineage analog on the same biological event. After the spec-literal-gate restoration (post-remediation), the v2 simplex collapses to **4 typed coordinates + `other_rare` + `unassigned`** at the default min-prevalence threshold (0.01); the activated_*_cd44 / activated_*_cd140b subtypes fall below threshold and collapse into `other_rare`. Sorted by |g_neut|:

| Endpoint | g_skep | **g_neut** | g_opt | n_req (neutral) | is_headline |
|----------|-------:|-----------:|------:|---------------:|:-----------|
| endothelial_clr | +0.31 | **+1.00** | +2.27 | 16 | ✓ |
| immune_cells_clr | –0.31 | **–1.00** | –2.29 | 16 | demoted (cross_axis_co_headline_forbidden — v1 immune_clr already headlines) |
| unassigned_clr | –0.30 | **–1.00** | –2.38 | 16 | ✓ |
| fibroblast_clr | –0.31 | **–0.91** | –1.77 | 19 | ✓ |
| neutrophil_clr | +0.28 | **+0.77** | +1.38 | 27 | ✓ |
| other_rare_clr | –0.10 | **–0.26** | –0.41 | 240 | ✓ |

The v2 set extends — and in some places contradicts — the v1 candidate read. The bare `endothelial_clr` (v2 discrete-celltype) rises Sham→D7 (+1.00 g_neut) while v1's `endothelial_clr` (single-lineage interface category in the 8-cat simplex) is filtered out by `normalization_sign_reverse` (both near zero). The v2 result reflects the discrete bare-`endothelial` cell-type proportion rising 1.31% → 3.93% (mouse-mean across 6 ROIs/timepoint), which the 8-cat lineage-interface CLR doesn't surface because it pools the bare lineage with all its overlaps. Phase 7 v2 operates on a closed simplex of 4 active discrete cell-type coordinates + `other_rare` + `unassigned` (after min-prevalence collapse), with `unassigned` carrying ~86% of the simplex mass on the mouse-level mean (Sham 87.9% → D7 84.2%, a 3.7-percentage-point drop).

Under the skeptical prior (N(0, 0.5²)), no Sham→D7 co-headline exceeds |g| ≈ 0.32; the neutral-prior magnitudes in the table above are therefore the upper end of a shrinkage range, not point estimates. These discrete cell-type CLR endpoints — including the bare-`endothelial` 1.31% → 3.93% rise — are descriptive-of-segmentation, never significance-bearing: they track how the strict-gate segmentation partitions tissue over time and are candidates against the full 840-endpoint denominator (`endpoint_summary.csv`), not established findings; the `is_headline` marks flag pre-registered-filter passage, not established significance.

**Family A endpoints filtered out (full disclosure).**

| Endpoint | Raw g (sigmoid) | Raw g (raw-marker Sham) | Reason filtered |
|----------|---------------:|------------------------:|:----------------|
| stromal_clr | −6.71 | −3.11 | `normalization_magnitude_disagree` (2.16× divergence); also suggestive variance-collapse geometry (pooled_std = 0.058) — do not treat as a magnitude estimate |
| none_clr | −2.39 | −12.75 | `normalization_magnitude_disagree` (5.3× divergence); also the CLR closure dual of triple-positive — a drop here is mechanically implied by any rise in lineage-positive categories |
| endothelial_clr | −0.007 | +0.18 | `normalization_sign_reverse` (both near zero; opposite sign) |
| endothelial+stromal_clr | −0.48 | −0.43 | |raw g| ≤ 0.5 in primary path |

All three Family A paths (sigmoid, raw-marker Sham-ref, Family C Sham-ref compartments) anchor on the same Sham distribution, so "independent" above means *independent of CLR closure*, not statistically independent from Sham baseline. Every claim inherits the Sham-reference anchoring.

**Rank-based selection-free companion.** `temporal_top_ranked_by_effect.csv` ranks on |g_shrunk_neutral|, with `g_pathological` rows (|g|>3 AND pooled_std<0.01) quarantined. Sign convention: `g = (mean_Sham − mean_D7) / pooled_std`, so a negative g means D7 > Sham. Sham→D7 top-5: `neutrophil` (−0.90, **D7 > Sham** — proportion rises 4.82% → 6.85%), `fibroblast` (+0.85, D7 < Sham — proportion drops 1.07% → 0.67%), `activated_m2_cd140b` (−0.73, D7 > Sham — proportion rises 0.05% → 0.26%), `activated_fibroblast_cd140b` (+0.71, D7 < Sham — proportion drops 0.80% → 0.40%), `activated_endothelial_cd140b` (−0.65, D7 > Sham — proportion rises 0.35% → 0.75%). Quarantined as `g_pathological=True`: `endothelial` (|g|=3.22, pooled_std=0.0046; proportion 1.31% → 3.93%), `activated_myeloid_cd44` (|g|=4.27, pooled_std=0.000418), `immune_cells` (|g|=4.03, pooled_std=0.000497) — each has near-zero within-group variance that breaks the small-sample Hedges' g formula.

**Family B (primary, sigmoid Sham-ref continuous lineage scores) — corrected 2026-04-23 Phase 5.5.** Sham→D7 produces **21 endpoints clearing |g_shrunk_neutral| > 0.5** (none `g_pathological`, none `support_sensitive`). An earlier version of this one-pager claimed "0 endpoints" — that claim was wrong; the audit script printed only the raw-marker count and the sigmoid count was assumed without verification. The 21 sigmoid headlines concentrate on `vs_sham_mean_delta_lineage_immune` and `..._endothelial` around composite labels like `activated_endothelial_cd44`, `mixed` (`c:mixed` is a reproducible multi-lineage co-expression class, never a cell type), `non_myeloid_immune`. They are not promoted to the cross-family co-headline table because Family B does not pass the same headline-filter pipeline as Family A; Family B's null result against the redistribution narrative is a per-row visibility, not a global "no signal" claim.

**Family B sigmoid-independent sensitivity check (Phase 1.5c — magnitude divergence with comparable headline yield).** A parallel Family B computed on raw arcsinh markers instead of sigmoid lineage scores (`audit_family_b_raw_markers.py`) produces **18 Sham→D7 endpoints at |g_neut|>0.5** (none `g_pathological`). Comparable to the 21 sigmoid headlines, **14 in common** (Jaccard 0.58 over the union), 96% sign agreement on overlapping endpoints. The sigmoid/raw disagreement at the magnitude level is real (55/166 ≥2× pre-shrinkage divergence) but Bayesian shrinkage to neutral converges the headline counts. Top raw-marker rows (sorted by |hedges_g|) include `vs_sham_mean_delta_lineage_immune` for `stromal` and `activated_endothelial_cd44` composites — consistent with published UUO biology (immune infiltration into non-immune compartments) and with the sigmoid path's own headlines on the same composite axes. The raw-marker basis was **not pre-registered as a primary path** in the original plan; Phase 5.2 amends the rule for follow-up cohorts to treat both bases as co-primary with an intersection-conservative headline set (see plan §5.2).

**Reading (compatible with, not equivalent to).** At Day 7, tissue that was Sham-only on a single lineage is classified less often as stromal-only and more often as multi-lineage: the triple-positive CLR rises (Family A) in parallel with the raw-marker triple-overlap fraction (Family C). Background compartments acquire broad CD44⁺ activation (Family C), which the panel cannot attribute to a specific cell type — the "background" compartment excludes CD45/CD31/CD140b positivity but its biological referent is ambiguous given the 9-marker panel gap (no tubular-epithelium markers). This convergent pattern is *compatible with* a redistribution story at D7; it is *not* independent replication, because all three analytical paths anchor on Sham.

**Limitations this pilot explicitly concedes.**

- **n=2 per group**: no conventional significance possible (Mann-Whitney p-floor at 0.333). Unshrunken Hedges' g values are not the reported magnitudes; the three-prior Bayesian shrinkage range is. Under the **skeptical prior (N(0, 0.5²))** no Sham→D7 co-headline exceeds |g| = 0.32 — interpret accordingly.
- **CLR closure**: Family A endpoints sum to zero on a closed simplex, so `triple_clr ↑` and `stromal_clr ↓` and `none_clr ↓` are mechanically linked — all three are visible as filtered-out rows above because the arithmetic forces them to co-move. Family C provides non-compositional corroboration (independent of CLR closure but still Sham-anchored).
- **Shared-reference tautology**: all three analytical paths (Family A sigmoid, Family A raw-marker Sham-ref, Family C raw-marker Sham-ref) anchor on the same pooled-Sham distribution from n=2 Sham mice × 3 ROIs. Sign agreement across paths is therefore partly built-in. The symmetric magnitude-disagreement count (13/48 Family A endpoints ≥2× divergence in `endpoint_summary.csv`) is the honest upper bound on how much the two Family A paths measure differently.
- **Cross-sectional design, n=2 mice-per-timepoint**: different mice per timepoint (not longitudinal) AND strain/surgery/batch/housing effects are perfectly confounded with timepoint at n=2 per side. The Sham↔D7 anchor is a 2-vs-2 comparison between different animals; trajectory-shape claims (Sham→D1→D3→D7 monotonicity) are weaker still.
- **Panel coverage**: The 9-marker panel excludes tubular epithelium, lymphocyte subsets, dendritic cells. ~86% of tissue is unassigned under strict discrete gating; the continuous annotation recovers part of this but does not resolve the panel gap. The "background compartment" in Family C therefore has no single biological referent.

**Post-hoc headline filter.** The co-headline selection rule `|g| > 0.5 AND direction-consistent AND symmetric magnitude agreement` was defined after seeing the Family A sensitivity sweep. It is pre-registered for follow-up cohorts unchanged; any relaxation in a future amendment will be flagged as a filter-sensitivity result, not as evidence.

**Reproducibility freeze.** See `FROZEN_PREREG.md` for the git SHA, `config_sha256`, `sham_reference` artifact hash, and frozen plan pointer at the time of this one-pager.

---
*Prepared 2026-04-23. Methodology details in `METHODS_SUMMARY.md`. Pipeline artifacts in `results/biological_analysis/`. Entry point: `run_analysis.py` + `batch_annotate_all_rois.py` + `run_temporal_interface_analysis.py`.*

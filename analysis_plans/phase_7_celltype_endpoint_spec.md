# Phase 7 spec — discrete cell-type-resolved endpoint extensions

**Status**: LOCKED post-brutalist round 2. Pre-registration draft for Phase 7 amendment.
**Drafted**: 2026-04-27. **Locked**: 2026-04-27 (post-round-2, all decisions resolved).
**Iteration log**: round 1 + round 2 patches, structural restructure under "defer nothing" directive. Process history in Appendix A.

---

## 1. Motivation

The Phase 1–6 pipeline already produces a `cell_type` column with 15 config-defined discrete categories (boolean-gated from the 9 markers) plus `unassigned`. The pre-registered Family A/B/C endpoints **never read this column**. They read either:
- `composite_label` (Family B stratifier; Family A interface labels also derived from continuous lineage scores at threshold 0.3), or
- raw markers under Sham-reference thresholds (Family C compartments, Family B raw-marker basis).

So the discrete cell types — the most config-driven, most panel-portable cell-typing surface in the project — sit on disk and only get used by the descriptive `differential_abundance_analysis.py` script and a few notebook narrative cells. They are absent from the formal pre-registered endpoint families.

Phase 7 extends Family A and Family B with the discrete `cell_type` column as a parallel axis (alongside the existing 8-category interface CLR and composite-label stratifier), folds a single non-tautological discrete-celltype endpoint into Family C, and resolves a name-collision risk that already exists between the two label spaces.

The framework is a hypothesis-generating pilot at n=2 per timepoint. Phase 7 does not change that: same forbidden language, same three-prior shrinkage, same conservative-intersection headline rule across normalization bases.

## 1.1 Locked decisions

Every Phase 7 design choice is resolved here. No "open decisions" remain. Brutalist iteration history in Appendix A documents the dropped alternatives. **Round 3 patches** (post-2026-04-27): F1, F2, F3, F4, F5, F6, F7, F8 + Codex F1, F3, F4, F5, F6 — applied below.

| § | Decision | Locked position | One-sentence rationale |
|---|---|---|---|
| 4.1 | A_v2 fate | **Keep** | Dropping defers the question to a future cohort that may not arrive; the design works once `unassigned` is included as a coordinate and the headline-rule asymmetry is exposed at row level. |
| 4.1 | A_v2 simplex composition | **16-cat at input gate; ≤16 + `other_rare` at analysis vocabulary after `min_prevalence` pooling** | Round 3 F2 caught that the "16-cat" claim is falsified by pooling. Honest framing: the input vocabulary lock is 16; the analysis vocabulary collapses sparse categories. |
| 4.1 | `unassigned` handling | **IN as a simplex coordinate; per-mouse `unassigned_rate_mouse_mean_1/2` reported on every Family A endpoint row** | Round 2: dropping `unassigned` changes the estimand. Round 3 Codex F1: per-ROI provenance is impossible at endpoint-row granularity (aggregation happens before CLR); use mouse-mean per contrast side. |
| 4.1 | A_v2 corroboration | **Single-path (no v2-internal raw-marker path)** | Round 1 found the proposed corroborating path shared the entire taxonomy with the primary path; a "two-paths converge" rule there would be theatre. |
| 4.2 | B_v2 fate | **Keep, parallel to A_v2** | kNN-gradient on a discrete stratifier is a different question from CLR-of-proportions. |
| 4.2 | B_v2 dual-basis indicator | **Reuse existing `normalization_mode` column** (values `sham_reference_v2_continuous` and `sham_reference_raw_marker_per_mouse`); no new `lineage_source` column | Round 3 Codex F4: spec drafted `lineage_source` but actual column produced by Phase 6 productization is `normalization_mode`. Align with code, not with prose. |
| 4.3 | Naming-collision rename | **`c:` prefix on every `composite_label` value, single-commit migration** | Round 1 dropped the deprecation column (no release process to amortize). |
| 4.4 | v1/v2 cross-rule enforcement | **Runtime demotion column with explicit join key; structural reach acknowledged as small** | Round 3 F1: cross-rule needs a defined v1↔v2 join predicate. Family A: only 3 of 16 v2 categories (`endothelial`, `immune_cells`, `stromal`/`fibroblast` sibling) have v1-INTERFACE_CATEGORIES analogs; Family B: same set after stripping `c:` prefix. The cross-rule fires on ~6 endpoints out of ~1296. The mechanism is correct architecture for those events; for the rest, v1 and v2 are independently reported and there is no clash to demote. |
| 4.5 | A_v1/A_v2 headline-rule asymmetry | **Row-level `headline_rule_version` column on Family A; row-level `is_headline` boolean on every row** | Round 3 Codex F3: query suggested in §4.4 used Family-A-only column to filter all families. Replace with all-family `is_headline` boolean computed after demotion + family-specific gate. |
| 4.6 | Family C neutrophil scalar | **New per-ROI compute path (categorical, not threshold-based) feeding existing Family C aggregation contract** | Round 3 F4 + Codex Family-C check: the per-ROI compute path needs a new function (`cell_type == 'neutrophil'` is categorical, not `df[m] > sham_thresholds[m]`); the per-mouse aggregation reuses `aggregate_compartment_activation_to_mouse` because it collects `*_cd44_rate` columns generically. The fold-in is at the aggregation+headline layer, not at the per-ROI layer. |
| 7a/7b | Phase split | **Bundle as single Phase 7 amendment** | "Defer nothing" directive. |

**Must-haves (unconditional, not user choices)**:
- **MH-1 (revised post-round-3 F5)**: permutation null **quality metric**, not binary gate. Run 1000 timepoint-label shuffles in null-mode evaluator (skip bootstrap + spatial permutations to make wall-clock tractable; just compute headline-passing endpoints under each shuffle). Report the empirical headline-count distribution. **Lock criterion**: median headline-count under H0 is reported alongside observed headline-count; observed must exceed median + 2× MAD to be reportable. (Original "95th percentile == 0" was mathematically untenable: under H0 at n=2, P(|g|>0.5) ≈ 0.62 per endpoint, so 1296 endpoints produce ~800 expected false headlines per shuffle. No filter present in this spec drops 99.93% of these. Honest framing: report the false-positive rate and gate against it, don't pretend it's zero.)
- **MH-2 (revised post-round-3 F8)**: explicit `priority_order` list in `config.cell_type_annotation.priority_order` (prerequisite P1). Rationale: today's priority is implicit dict iteration order, which works on Python ≥3.7 but is fragile under panel ports / config tooling that doesn't preserve insertion order. P1 makes priority load-bearing-explicit, not load-bearing-implicit.
- **MH-3**: resolved `DISCRETE_CELL_TYPES` list pinned in `FROZEN_PREREG.md` (catches config drift at the resolved-vocabulary level, not just source-file SHA).
- **MH-4 (revised post-round-3 F7)**: post-notebook-execution validation (not in-cell instrumentation): a CI step or pre-merge check that runs every notebook end-to-end and asserts every Pandas DataFrame mutation that filters on `composite_label` produces a non-empty result. Specifically: a wrapper around `nbconvert --execute` that traces `__getitem__` and `query` calls referencing the column, recording (notebook, cell, filter_value) tuples; the test asserts each tuple resolved to >0 rows. The static enumerate-before-rename file is necessary but not sufficient (Round 2 Critical #6); this is the runtime gate that catches what static enumeration misses.

## 2. What exists today (grounding for the diff)

Read directly from current main, not from memory:

**Family A — `INTERFACE_CATEGORIES`** in `src/analysis/temporal_interface_analysis.py:45-50`:
```
('immune', 'endothelial', 'stromal',
 'endothelial+immune', 'immune+stromal', 'endothelial+stromal',
 'endothelial+immune+stromal', 'none')
```
8 categories. Derived from continuous lineage scores at threshold `DEFAULT_LINEAGE_THRESHOLD=0.3`. The discrete `cell_type` column is **not** consulted.

**Family B — stratifier**: `composite_label` (per `temporal_interface_analysis.py:431, 439`). Stratifier values observed in `endpoint_summary.csv` Sham→D7 rows (with sufficient support):
```
endothelial, m2_macrophage, neutrophil, non_myeloid_immune, stromal, mixed, unassigned,
activated_endothelial_cd44, activated_endothelial_cd140b, activated_endothelial_cd140b_cd44,
activated_stromal_cd44, activated_stromal_cd140b, activated_stromal_cd140b_cd44,
activated_neutrophil_cd44, activated_m2_macrophage_cd140b_cd44, ...
```

**Family C — compartments** in `compute_compartment_activation_per_roi` (`temporal_interface_analysis.py:579-614`): 5 raw-marker compartments — `CD45+`, `CD31+`, `CD140b+`, triple-overlap, background. Activation marker = `CD44`.

**Discrete `cell_type` column** observed values (one ROI sample, `roi_data/.../cell_types.parquet`):
```
endothelial, fibroblast, immune_cells, m2_macrophage, myeloid, neutrophil,
activated_endothelial_cd44, activated_endothelial_cd140b,
activated_fibroblast_cd44, activated_fibroblast_cd140b,
activated_immune,
activated_m2_cd44, activated_m2_cd140b,
activated_myeloid_cd44, activated_myeloid_cd140b,
unassigned
```
15 typed categories + `unassigned`. Derived in `cell_type_annotation.py::annotate_cell_types()` by strict boolean gating against the 9 raw markers.

**Naming surfaces that already conflict between the two columns**:
| String | Meaning in `cell_type` (discrete boolean gate) | Meaning in `composite_label` (continuous threshold + dominance) |
|---|---|---|
| `endothelial` | CD31+ AND CD34+ AND CD45− AND CD140a− AND CD44− AND CD140b− | dominant lineage = endothelial AND CD44 < 0.3 AND CD140b < 0.3 |
| `activated_endothelial_cd44` | strict gate above + CD44+ | dominant endothelial + activation_cd44 ≥ 0.3 |
| `activated_endothelial_cd140b` | strict gate + CD140b+ | dominant endothelial + activation_cd140b ≥ 0.3 |
| `m2_macrophage` | CD45+ AND CD11b+ AND CD206+ AND lineage exclusive | immune subtype = m2_macrophage by geometric mean argmax |
| `unassigned` | no boolean gate matched (~66% of superpixels, 38157/58137) | top lineage score < 0.3 (~22% of superpixels) |

**These are different entities under the same string.** A reader downstream cannot disambiguate without also reading the column name. That's a reproducibility bug already present pre-Phase-7; Phase 7 will tip the rate of confusion higher because both labelings appear as endpoint stratifiers.

**Naming surfaces that diverge silently** (different string, same conceptual lineage):
| Lineage | `cell_type` string | `composite_label` string |
|---|---|---|
| stromal | `fibroblast`, `activated_fibroblast_*` | `stromal`, `activated_stromal_*` |
| broad immune | `immune_cells`, `activated_immune` | (subtypes only: `myeloid`/`m2_macrophage`/`neutrophil`/`non_myeloid_immune`) |
| immune subtype `m2` | `activated_m2_cd44` | `activated_m2_macrophage_cd44` |

A user comparing the two axes will need a key. Either we ship one in the spec, or we rename one side.

## 3. Scope

### In scope
- **Family A_v2**: 16-category CLR over the discrete `cell_type` column (15 typed + `unassigned`), parallel to Family A_v1's 8-interface CLR. `unassigned` is a real coordinate of the simplex (the gmean-drag is acknowledged as a feature; the alternative bias of "measuring among-assigned only" is worse for the question A_v2 is for).
- **Family B_v2**: per-discrete-`cell_type` stratification of neighbor-minus-self lineage delta, parallel to Family B_v1's per-`composite_label` stratification. Same `min_support` sweep. Same dual-basis (sigmoid + raw-marker) productization Phase 6 established.
- **Family C v2 single-row neutrophil extension**: `cd44_rate_in_neutrophils_per_mouse`, computed within Ly6G+ superpixels (the only discrete cell type whose gate does not pin CD44 status). Folded into the existing Family C surface with full Bayesian shrinkage, headline rule, pathology gate. No carve-out.
- **Naming-collision resolution**: `c:` prefix on every `composite_label` value so the value space is disjoint from the discrete `cell_type` value space. Single-commit migration: rename in one PR that simultaneously regenerates every notebook and every consuming script in the same diff. No deprecation column.
- **Cross-rule enforcement**: v1 and v2 rows for the same family are reported in parallel. A v2 row that would headline under its own rule is **demoted via a runtime column** (`headline_demoted_reason = 'cross_axis_co_headline_forbidden'`) when a corresponding v1 row passes the v1 rule on the same biological event. The cross-rule is architecture, not prose.
- **Headline-rule-version visibility**: `headline_rule_version` column on every Family A row tags which rule a flagged row passed. Required because A_v1 enforces dual-normalization agreement and A_v2 does not.

### Out of scope (closed-by-design)
- **Family C with discrete `cell_type` compartments other than `neutrophil`**: 14 of 15 typed cell types pin CD44 status by gate construction (`activated_*_cd44` requires CD44+ → rate = 1; bare types and `*_cd140b` activations require CD44− → rate = 0). The neutrophil exception is the single non-tautological measurement; it ships as Family C v2. The other 14 are not deferred — the structure forbids them.
- **Cross-cohort validation**: no second cohort exists.
- **Cross-vocabulary integration of v1 and v2 evidence**: deferred to a follow-up cohort with pre-registered cross-axis hypothesis. Phase 7 reports them in parallel and demotes co-headlining via the cross-rule column.

## 4. Pre-registration changes (locked)

### 4.1 Family A_v2 — discrete cell-type CLR

- **Categories** (16): the 15 strings emitted by `cell_type_annotation.py::annotate_cell_types()` plus `unassigned`. Defined exhaustively as the keys of `config.cell_type_annotation.cell_types` plus the literal `unassigned`. Runtime gate asserts the observed `cell_type` set ⊆ this 16-element set; any new value triggers fail-fast. Adding a cell type to the config is a pre-reg amendment, not a transparent ride-along.
- **Question**: across timepoint contrasts, do discrete cell-type proportions shift? The 15 typed categories carry activation state, so a Sham→D7 rise in `activated_fibroblast_cd44` at the expense of bare `fibroblast` is a discriminable event Family A_v1 cannot resolve.
- **Closure**: 16-category simplex per ROI, including `unassigned`. CLR (per `temporal_interface_analysis.py:316-326`) produces all 16 coordinates with the geometric mean as denominator and rowsums to 0 in CLR space. **No reference category.** The geometric-mean drag toward `unassigned`'s log-fraction is a known property; it is not removed because doing so changes the estimand from "do cell-type proportions shift?" to "do typed proportions shift among assigned superpixels?", and `unassigned` mass itself moves with injury (Sham aggregate 68.48%, D7 62.72%; per-ROI range 56–72%, verified empirically in current cohort). Reporting the 16-category CLR with `unassigned` IN keeps the question intact.
- **Sensitivity sweep**: `min_prevalence ∈ {0.005, 0.01, 0.02}` (default 0.01). Categories below threshold across all timepoints are pooled into `other_rare`. The 16-category set is the **input vocabulary lock**; the post-pooling **analysis vocabulary** is ≤16 categories + `other_rare`.
- **Headline rule**: single-path. Family A_v2 has no v2-internal corroborating path because any such path would share the panel + taxonomy + priority-resolution logic with the primary path; agreement would be threshold-edge filtering, not corroboration. A_v2 headlines on `|hedges_g| > 0.5 AND not g_pathological`. The asymmetry with A_v1 (which additionally requires `not normalization_magnitude_disagree`) is exposed at row level via the `headline_rule_version` column (§4.5).
- **Pathology gate**: `|g| > 3 AND pooled_std < 0.01` emits NaN for shrunken columns. Same as v1.
- **Per-row provenance**: `unassigned_rate_mouse_mean_1` and `unassigned_rate_mouse_mean_2` are written on every Family A endpoint row, recording the per-mouse-mean unassigned fraction in each contrast side. Not independent endpoints; provenance for the gmean-drag.

### 4.2 Family B_v2 — per-discrete-cell-type neighbor-minus-self

- **Stratifier**: discrete `cell_type` column (16 values).
- **Neighbor lineage scores**: same as v1 — continuous lineage scores from `compute_continuous_memberships()`, unchanged.
- **k**: 10. **Sensitivity**: `min_support ∈ {10, 20, 40}`.
- **Dual basis**: both the Phase-1 sigmoid lineage basis AND the Phase-6 raw-marker per-mouse basis, indicated by the existing `normalization_mode` column ∈ {`sham_reference_v2_continuous`, `sham_reference_raw_marker_per_mouse`} (per Phase 6 productization). No new `lineage_source` column is introduced; reuse the existing schema.
- **Headline rule**: conservative-intersection across `normalization_mode` bases — clears the headline gate (`|g_shrunk_neutral| > 0.5 AND not g_pathological AND not support_sensitive`) in **both** bases. Same divergence and conflict CSVs as Phase 6 v1, namespaced `family_b_v2_basis_divergence.csv` / `family_b_v2_basis_conflict.csv`.
- **Conflict policy**: a v2 headline is demoted if it appears in `family_b_v2_basis_conflict.csv` (opposite-sign-same-endpoint between bases). Demotion writes `headline_demoted_reason = 'family_b_basis_conflict'` on the row.
- **Cross-rule demotion** (§4.4): a v2 row passing the v2 rule is also demoted if a corresponding v1 row passes the v1 rule under the join key defined in §4.4. The runtime mechanism sets `headline_demoted_reason = 'cross_axis_co_headline_forbidden'`.

### 4.3 Naming-collision resolution

- All values written to the `composite_label` column gain a `c:` prefix. Examples: `c:endothelial`, `c:activated_endothelial_cd44`, `c:mixed`, `c:unassigned`. The discrete `cell_type` column is unchanged.
- **Single-commit migration**: rename in one PR that simultaneously regenerates every notebook and every consuming script. No deprecation column.
- **Enumerate-before-rename gate**: before any code change, `grep -rn` for every value that would change produces `analysis_plans/phase_7a_rename_targets.txt`. The PR diff must touch every file in the list. Files in scope:
  - `notebooks/**/*.ipynb` — narrative cells, plotting code, gating logic
  - `viz.json` — color/label keys
  - `differential_abundance_analysis.py`, `spatial_neighborhood_analysis.py` — descriptive scripts
  - `audit_family_b_raw_markers.py`, `audit_tissue_mask_density.py`
  - `tests/**` referencing composite labels by string
- **Runtime assertion** (necessary because the static list can miss): notebook execution gate or pipeline-run gate verifies every `composite_label` filter in every consumer resolves to >0 rows. Failure to match is a hard error, not a silent zero-row return.

### 4.4 v1/v2 cross-rule (runtime architecture)

The cross-rule is enforced as a runtime column. **Structural reach is small** — most v1 and v2 vocabularies do not overlap, so most v2 rows have no v1 counterpart to clash with.

**Join key (Family A)**: a v2 row at `(endpoint='X_clr', endpoint_axis='discrete_celltype_16cat')` clashes with a v1 row at `(endpoint='X_clr', endpoint_axis='composite_label_8cat')` iff the v2 cell-type maps to the same biological compartment as the v1 interface category, by this fixed table:

| v2 `cell_type` | v1 `INTERFACE_CATEGORIES` member |
|---|---|
| `endothelial`, `activated_endothelial_cd44`, `activated_endothelial_cd140b` | `endothelial` |
| `immune_cells`, `activated_immune` | `immune` |
| `fibroblast`, `activated_fibroblast_cd44`, `activated_fibroblast_cd140b` | `stromal` |
| (m2_macrophage, myeloid, neutrophil, activated_m2_*, activated_myeloid_*, unassigned) | (no v1 analog → no clash possible) |

Activated v2 categories collapse to their bare-lineage v1 analog because Family A v1 cannot resolve activation; the cross-rule's job is to prevent v2 from "discovering" what v1 already discovered at the lineage level.

**Join key (Family B)**: a v2 row at `(endpoint, contrast, normalization_mode, cell_type=X)` clashes with a v1 row at `(endpoint, contrast, normalization_mode, composite_label=Y)` iff `Y == 'c:' + X` after the rename (single-cell-type composite label) OR iff `X` and `Y` map to the same lineage via the table above. Composite labels like `c:mixed`, `c:unassigned` have no v2 analog and produce no clash.

**Demotion procedure**: for each clash group:
1. Compute the per-row "would-headline" status using each row's own headline rule.
2. If both v1 and v2 rows would headline: v1 passes through; v2 is demoted with `headline_demoted_reason = 'cross_axis_co_headline_forbidden'`.

**Effective reach**: ~6 endpoint clashes out of ~1296 endpoint rows. The cross-rule is correct architecture for these events; the bulk of v1+v2 evidence is reported in parallel without clash.

**Headline-status column (NEW post-round-3)**: rather than ask consumers to compose multiple columns, emit a single `is_headline` boolean (all families) computed after all demotions:
```
is_headline = (passes_family_specific_rule) AND (headline_demoted_reason is null)
```
Downstream consumers query `is_headline == True` to get the canonical headline set. The `headline_rule_version` column (Family A only) and `headline_demoted_reason` column (all families) provide the audit trail; `is_headline` is the question consumers actually want answered.

### 4.5 Headline-rule-version column

`headline_rule_version` (Family A only): str ∈ {`v1_dual_normalization_intersection`, `v2_pathology_only`}. Tags every Family A row with the rule that decided its headline status:
- `v1_dual_normalization_intersection`: row's family is `A_interface_clr` AND endpoint_axis is `composite_label_8cat`. Rule: `|g| > 0.5 AND not g_pathological AND not normalization_magnitude_disagree`.
- `v2_pathology_only`: row's family is `A_interface_clr` AND endpoint_axis is `discrete_celltype_16cat`. Rule: `|g| > 0.5 AND not g_pathological`.

A reader scanning headline-flagged rows can now see which rule the row passed. Joint display without joint headline is honest.

### 4.6 Family C v2 — single-row neutrophil extension

- **Endpoint**: `neutrophil_compartment_cd44_rate` per mouse, computed as the fraction of `cell_type == 'neutrophil'` superpixels also marked CD44+ under the existing Sham-reference percentile threshold for CD44.
- **Per-ROI compute path**: NEW function `compute_neutrophil_compartment_activation_per_roi()` parallel to `compute_compartment_activation_per_roi`. Necessary because Family C v1 defines compartments by `df[m] > sham_thresholds[m]` (threshold-based); the neutrophil compartment is defined by `cell_type == 'neutrophil'` (categorical). Honest framing: per-ROI compute is new code; per-mouse aggregation reuses the existing `aggregate_compartment_activation_to_mouse` (`temporal_interface_analysis.py:617-640`) because that function generically collects `*_cd44_rate` and `*_compartment_n` columns.
- **Headline rule**: same as Family C v1 — `|g_shrunk_neutral| > 0.5 AND not g_pathological`.
- **Sensitivity**: same as Family C v1 — Sham percentile {65, 75, 85}.
- **Why this and only this**: 14 of 15 discrete cell types include CD44 in their boolean gate (forced 0 or 1 rate); `neutrophil` is the only type whose gate (`+CD45 +Ly6G −CD31 −CD34`) leaves CD44 status free. This is the one and only discrete-cell-type compartment that admits a non-tautological CD44 rate.

### 4.7 Forbidden language (unchanged from Phase 5)
- "Significant" / "significance"
- "95% confidence interval"
- "Effect size point estimate" without the three-prior shrinkage range

### 4.8 Headline filter (unchanged from Phase 5, with cross-rule extension)
A Phase 7 endpoint is a headline iff it satisfies the family-specific rule above AND is not demoted by any of:
- support sensitivity (`support_sensitive == True`)
- pathology gate (`g_pathological == True`)
- normalization magnitude disagreement (Family A v1 only, applied per existing column)
- cross-axis co-headline forbidden (Family A and Family B v2 only, per §4.4)

## 5. Anticipated reviewer questions

- **Sparse discrete categories at n=2 mice/timepoint**: 15 typed categories over 24 ROIs with 65.6% unassigned (38,157/58,137 superpixels) means many activated/lineage combinations have <10 superpixels per ROI. Family B_v2 handles this via `min_support` sweep. Family A_v2 applies `apply_min_prevalence_filter` (parameterized in Phase 7 prereq P2 to accept arbitrary category set, not just `INTERFACE_CATEGORIES`); categories below threshold pool into `other_rare`. The 16-category vocabulary lock applies at the input gate; the realized analysis vocabulary is ≤16 + `other_rare`.
- **Activation circularity in Family A_v2 (NOT Family C)**: discrete cell types like `activated_endothelial_cd44` encode CD44 in their definition. So a Family A_v2 endpoint "activated_endothelial_cd44 fraction rises Sham→D7" is indistinguishable from "endothelial CD44 rises in CD31+/CD34+ cells". This is the intended event — Family A_v2's value-add is precisely that it can resolve activation co-occurring with lineage. The CD44 question is not double-counted because Family C reports a per-compartment **rate** while A_v2 reports a per-ROI **proportion**; both are valid statistics about the same biology.
- **Multiple-comparison surface growth**: Family A 8 v1 + 16 v2 = 24 endpoints × 6 contrasts = 144; Family B 540 v1 + 16 × 3 lineages × 2 bases × 6 contrasts = 540 + 576 = 1116; Family C 30 v1 + 6 v2 (neutrophil contrasts) = 36; total ~1296. n=2 still produces no FDR-significant findings; the headline-pass count is what matters and is gated by (a) the family-specific rule, (b) the v1/v2 cross-rule (runtime column), (c) the permutation null acceptance test (§6.2).
- **Why include `unassigned` in the CLR simplex?**: dropping it makes A_v2 ask a different question than the one it is for. `unassigned` mass moves with injury (Sham 68%, D7 63%); reporting it separately while excluding it from the simplex makes the simplex a derived statistic conditioned on a moving denominator. Including it accepts the gmean-drag as the cost of measuring the right thing.

## 6. Implementation outline

### 6.0 Prerequisites (must merge before Phase 7 PR)

**Ordering (round 3 F3 + Codex F6)**: P3 first, then P1, then P2. P1 mutates `config.json` and rotates `config_sha256`; the `verify_frozen_prereg.py` SHA-pins (Phase 5.6) will fail until SHAs are recomputed. P3 must extend the manifest verifier's tolerance for resolved-vocabulary entries BEFORE P1 lands so the chain doesn't break main between merges. Each P-PR includes its own SHA-recompute commit so the manifest is always consistent at HEAD.

1. **P3 (MH-3)**: Extend `review_packet/FROZEN_PREREG.md` schema and `verify_frozen_prereg.py` to pin the resolved `DISCRETE_CELL_TYPES` list (15 typed from config + `unassigned`). Catches config drift at the resolved-vocabulary level. Own PR; no-op cost. Ships first because subsequent P-PRs need the verifier to accept their manifest changes.
2. **P1 (MH-2)**: (a) Update `src/config_schema.py:520` to allow `priority_order` as a known key on `cell_type_annotation` (current schema explicitly rejects unknown keys per round-3 Codex F6 verification). (b) Move `annotate_cell_types()` priority order from `src/analysis/cell_type_annotation.py` into `config.cell_type_annotation.priority_order`. (c) Recompute `config_sha256` in `FROZEN_PREREG.md` in the same PR. Regression: cell_type assignments on all 24 ROIs identical before/after refactor (priority loaded from config matches today's implicit dict-iteration order).
3. **P2**: Parameterize `apply_min_prevalence_filter` (`temporal_interface_analysis.py:329`) to accept an arbitrary category set, not just `INTERFACE_CATEGORIES`. Regression: Family A v1 row counts and effect-size values bit-identical. Own PR; no manifest impact.

### 6.1 Code touchpoints
1. `src/analysis/temporal_interface_analysis.py`:
   - Add `DISCRETE_CELL_TYPES: Tuple[str, ...]` module constant, populated at import time from `config.raw['cell_type_annotation']['cell_types']` keys + `('unassigned',)`.
   - Add `compute_celltype_fractions_per_roi()` parallel to `compute_interface_fractions_per_roi`, using `cell_type` column.
   - Add `run_family_a_v2()`, `run_family_b_v2()`, `extend_family_c_with_neutrophil_v2()` orchestrator entry points.
   - Add `apply_cross_rule_demotion()` helper that reads paired v1/v2 row sets and writes `headline_demoted_reason` on v2 rows that clash with passing v1 rows.
2. `src/analysis/cell_type_annotation.py`:
   - Apply the `c:` prefix in `_derive_composite_labels()` output.
3. `run_temporal_interface_analysis.py`:
   - Wire the four v2 entry points into the orchestrator.
   - Emit new endpoint_summary columns (§6.3) on every row.
   - Emit `family_b_v2_basis_divergence.csv`, `family_b_v2_basis_conflict.csv`, `family_b_v2_raw_marker_audit.parquet`.
   - Apply cross-rule demotion as a final pass before writing `endpoint_summary.csv`.
4. `audit_family_b_raw_markers.py`: extend the comparison helpers to accept stratifier-column name as a parameter, so the same audit code handles Family B v1 and v2.
5. `analysis_plans/temporal_interfaces_plan.md`: amend with Phase 7 section that incorporates this spec's §4 verbatim.
6. `review_packet/FROZEN_PREREG.md`: add Phase 7 entry to "Closures and remaining open work"; recompute SHAs for config.json (changed by P1), viz.json (unchanged), plan (changed), sham reference (unchanged).
7. `verify_frozen_prereg.py`: extended in P3 to validate the resolved discrete-celltype list.

### 6.2 Tests
- **Existing must pass**: all Phase 5/6 tests.
- **MH-1 permutation null (revised post-round-3 F5)**: `tests/test_phase7_permutation_null.py`. Run 1000 timepoint-label shuffles in **null-mode evaluator** (skip bootstrap, skip spatial permutations, skip neighbor-graph computation; just compute Hedges' g for each endpoint and apply the headline filter). For each shuffle, count rows with `is_headline == True`. **Lock criterion**: the observed (real-label) headline count must exceed `median(null_distribution) + 2 * MAD(null_distribution)`. Report median, MAD, observed, and excess-over-null in the test output and as a pinned line in `FROZEN_PREREG.md` Phase 7 entry. (Original "95th percentile == 0" was mathematically untenable — under H0 at n=2 the expected per-shuffle false-headline count is ~800; see §1.1 MH-1 rationale.)
- **DISCRETE_CELL_TYPES vocabulary lock**: `DISCRETE_CELL_TYPES` set ⊇ observed `cell_type` values across all 24 ROI annotation parquets at orchestration start. Fail-fast on drift.
- **CLR closure**: 16-category CLR rowsums to 0 in CLR space (within float tolerance) PRE-pooling. Post-pooling, the realized-vocabulary CLR (≤16 + `other_rare`) also rowsums to 0.
- **Schema rename**: `composite_label` values all begin with `c:` after Phase 7. No `composite_label_v1` column exists anywhere in pipeline outputs.
- **MH-4 runtime rename assertion (revised post-round-3 F7)**: post-execution validation. CI step runs `nbconvert --execute` on every notebook with a wrapper that traces `__getitem__` and `query` calls referencing the `composite_label` column; records (notebook, cell, filter_value) tuples; asserts each tuple resolved to >0 rows. Implementation in `tests/test_notebook_composite_label_filters.py` plus a CI workflow step.
- **Cross-rule mechanism (revised post-round-3 F1)**: synthetic fixture exercising the explicit join-key table (§4.4). Assert v2 rows mapping to a v1-passing row via the table get `headline_demoted_reason='cross_axis_co_headline_forbidden'`; assert v2 rows with no v1-table mapping pass through unchanged.
- **Headline-rule-version**: every Family A row has `headline_rule_version` populated; v1 rows tagged `v1_dual_normalization_intersection`, v2 rows tagged `v2_pathology_only`.
- **`is_headline` boolean** (NEW post-round-3): every row has `is_headline` populated; values consistent with `(passes_family_rule) AND (headline_demoted_reason is null)`.
- **Regression**: Phase 5/6 row counts unchanged in v1 axis (`endpoint_axis == 'composite_label_8cat'` rows = 48 Family A; v1 Family B rows in `stratifier_basis == 'composite_label'` = 540 — already includes Phase 6 dual-basis 270+270); v1 statistical content (effect sizes, shrunken values, support flags) bit-identical post-rename modulo the `c:` prefix on `composite_label` strings.

### 6.3 Schema deltas (endpoint_summary.csv)

New columns:
- `endpoint_axis`: str ∈ {`composite_label_8cat`, `discrete_celltype_16cat`}. Family A only; default `composite_label_8cat` for existing v1 rows. Family B/C rows: NaN.
- `stratifier_basis`: str ∈ {`composite_label`, `discrete_celltype`}. Family B only; default `composite_label` for existing v1 rows. Family A/C rows: NaN.
- `min_prevalence_sweep_value`: float ∈ {0.005, 0.01, 0.02}. Family A_v2 only. NaN for other rows.
- `headline_rule_version`: str ∈ {`v1_dual_normalization_intersection`, `v2_pathology_only`}. Family A only; NaN for Family B/C.
- `headline_demoted_reason` (closed enum, round 3 F6): str ∈ {`null`, `cross_axis_co_headline_forbidden`, `family_b_basis_conflict`}. All families. Null means the row is a candidate headline; the two non-null values are the only valid reasons. Existing demotion columns (`g_pathological`, `support_sensitive`, `normalization_magnitude_disagree`) remain as separate columns and are NOT migrated into this enum (they predate Phase 7; round 3 F6 confirmed they should stay separate to keep v1 statistical content bit-stable).
- `is_headline` (NEW post-round-3, all families): bool. Computed as `(passes_family_specific_rule) AND (g_pathological == False) AND (support_sensitive == False) AND (normalization_magnitude_disagree == False if Family A_v1 else True) AND (headline_demoted_reason is null)`. Single canonical headline-status column; downstream consumers query this rather than composing the audit columns.
- `unassigned_rate_mouse_mean_1`, `unassigned_rate_mouse_mean_2` (Codex F1 fix): float, Family A only. Per-mouse mean unassigned rate in each contrast side (`tp1`, `tp2`). Documents the denominator dynamic alongside the CLR coordinates. (Original `unassigned_rate_per_roi` proposal was schema-incoherent because endpoint rows are post-aggregation per mouse-pair contrast, not per-ROI.)

Existing 37 columns unchanged. Predicted row count: ~1296 (Family A 144 = 8 v1 + 16 v2 categories × 6 contrasts; Family B 1116 = 540 v1 dual-basis + 576 v2 dual-basis × stratifiers × contrasts; Family C 36 = 30 v1 + 6 v2 neutrophil contrasts).

### 6.4 Orchestration cost
Family A_v2: O(n_superpixels × n_categories) — trivial. Family B_v2: kNN reused from v1; second groupby pass on `cell_type`; estimated ~10s wall-clock add. Family C v2: new per-ROI compute pass over `cell_type == 'neutrophil'` subset (24 ROIs × small subset); trivial. Cross-rule demotion: linear pass over endpoint_summary; trivial.

**MH-1 permutation null cost (revised post-round-3 Codex F5)**: full pipeline orchestration runs 10K bootstraps per endpoint plus 1000 spatial permutations per ROI/category. 1000 × full pipeline ≈ 500+ hours wall-clock; infeasible. The MH-1 null-mode evaluator skips bootstrap, spatial, and side-statistics computation; it computes Hedges' g per endpoint per shuffle and applies the headline filter. Budget: 1000 × ~5s per shuffle (no bootstrap, no spatial) = ~1.5 hours wall-clock. Tractable as a release gate, not a per-run cost.

## 7. Residual risks (post-locked-decisions)

These are risks that remain even with every decision locked. They are not blockers; they are honest-status notes for future cohorts.

1. **`unassigned`-as-coordinate accepts the geometric-mean drag.** The 16-category CLR's per-coordinate values are partly mechanical because `unassigned` holds 60–70% of the simplex mass. The decision to keep `unassigned` IN preserves the question; the cost is that a "shift in `activated_fibroblast_cd44` CLR" is partly the inverse of the `unassigned` shift. The `unassigned_rate_mouse_mean_1/2` columns let a reader see the denominator dynamic alongside the CLR coordinate; that is the mitigation, not a fix.
2. **The cross-rule's structural reach is small** (round 3 F1). Only ~6 of ~1296 endpoints have a v1↔v2 join-key match per the §4.4 table; the rest are independently reported. For those independent rows the cross-rule does not fire and v1+v2 stand as parallel evidence. This is correct — most of the time there's nothing to demote because the vocabularies don't overlap — but the spec's "cross-rule" rhetoric is doing less work than it sounds like.
3. **The cross-rule demotes v2, never v1.** Asymmetric by design — v1 is the primary surface. A genuinely-superior v2 finding cannot promote over a v1 row that happens to also pass the v1 rule. For follow-up cohorts where v1 vs v2 evidence is weighed jointly, the cross-rule will need to change.
4. **Activation-encoding asymmetry in Family B_v2.** Family B's lineage scores are not activation-augmented. A B_v2 row "immune-lineage gradient around `activated_endothelial_cd44`" measures gradient around CD31+CD34+CD44+ superpixels. The semantic is by-name and load-bearing; readers must read the stratifier name to know the population. Documented in §4.2; no test enforces semantic-clarity beyond row-naming.
5. **MH-1 lock criterion is empirical, not theoretical** (round 3 F5). The null-mode evaluator returns an empirical headline-count distribution; the lock criterion is "observed > median + 2×MAD." This is not a sharp test and a single boundary observation could flip the lock outcome. The criterion is a discipline floor, not a guarantee; future cohorts should re-derive a power-justified criterion if MH-1 results are within 1× MAD of the null median.
6. **Pre-reg amendment fatigue.** Phase 7 is amendment #6 in 6 weeks. Each amendment regenerates notebooks, recomputes manifest SHAs, re-runs the cohort. Operational cost is roughly linear in amendments. Phase 7 was intended to be additive; the rename mutates v1 row strings and the cross-rule changes how readers integrate v1 evidence. Honest status: amendment #7+ should be gated on a second cohort or an external validation, not on internal iteration.

## 8. Acceptance criteria

- All Phase 5/6 tests pass.
- Phase 5/6 endpoint_summary rows are bit-identical in **statistical content** (effect sizes, shrunken values, support flags); `composite_label` value strings change due to the `c:` rename only.
- Family A_v2 produces 16-category CLR per ROI at the input gate; the realized analysis vocabulary after `min_prevalence` pooling is ≤16 + `other_rare`. CLR rowsums to 0 in CLR space at both pre-pooling and post-pooling vocabulary widths (per round 3 F2 — original "16-category" criterion was falsified by pooling).
- Family B_v2 produces per-discrete-cell-type stratified rows in both `normalization_mode` bases (`sham_reference_v2_continuous` and `sham_reference_raw_marker_per_mouse`; reuses existing column, no new `lineage_source` column).
- Family C v2 single-row neutrophil endpoint emitted via new per-ROI compute path; per-mouse aggregation reuses Family C contract (per round 3 F4 — honest framing of split between new compute path and reused aggregation).
- `composite_label` post-Phase-7 values all start with `c:`. No `composite_label_v1` column exists.
- `headline_rule_version` column populated on every Family A row.
- `headline_demoted_reason` column populated on every row from the closed enum (per round 3 F6).
- `is_headline` boolean column populated on every row; downstream consumers query this.
- `unassigned_rate_mouse_mean_1`, `unassigned_rate_mouse_mean_2` populated on every Family A row (per round 3 Codex F1 fix).
- `verify_frozen_prereg.py` passes with the resolved `DISCRETE_CELL_TYPES` list pinned (per P3) AND with config_sha256 recomputed (per P1; round 3 F3).
- All notebooks regenerate without errors AND the post-execution runtime assertion (MH-4) confirms every `composite_label` filter resolves to >0 rows.
- MH-1 permutation null test passes: observed headline-count exceeds `median(null_distribution) + 2 × MAD(null_distribution)` (per round 3 F5 — original "95th percentile == 0" was untenable).
- Cross-rule mechanism test passes: synthetic v1+v2 fixture exercises the §4.4 join-key table and verifies demotion fires for matched pairs and passes through for unmatched.
- Appendix A.5 propagation-checklist methodology is signed off — every round-1/round-2/round-3 finding has status `applied` or `rejected with reason`.

## 9. What round 3 should attack on the locked spec

Locked decisions in §1.1 close every previously-open question. Round 3 should look for what locking missed — defects produced by consolidation, hidden dependencies between locked decisions, or load-bearing assumptions still hidden.

1. **Internal coherence after consolidation**: did the §9-§12 collapse into §1.1 Locked Decisions + Appendix A drop a constraint that §3-§8 silently assume? Re-read §3-§8 and verify every claim has a backing decision in §1.1.
2. **`unassigned` as 16th coordinate**: §1.1 keeps `unassigned` IN the simplex; §4.1 acknowledges the gmean-drag. Are the `unassigned_rate_mouse_mean_1/2` provenance columns (§6.3) honest mitigation, or a marketing fig leaf? Specifically: does the per-mouse-mean granularity license a reader to "back out" the unassigned dynamic from a CLR coordinate, and is that operation defensible?
3. **Cross-rule asymmetry**: §4.4 demotes v2 only, never v1. Defensible for primary/secondary surface framing, but is the demotion mechanism observable? If a reader queries headlines without joining on `headline_demoted_reason`, they'll see the v2 row as a candidate. The mechanism is correct only if every consumer reads the column. List every consumer; verify each.
4. **Neutrophil Family C v2 row's compatibility with existing Family C aggregation**: §4.6 folds it in with same shrinkage and headline rule. Read `compute_compartment_activation_per_roi` and `aggregate_compartment_activation_to_mouse` in `src/analysis/temporal_interface_analysis.py` — does adding a per-celltype-stratified row fit the existing per-mouse aggregation contract, or does it need its own scaffolding (in which case it's structurally a new family, not a v2 extension)?
5. **MH-1 permutation null as a hard gate**: 1000 shuffles × Phase 7 orchestration ≈ 8 hours wall-clock per the §6.4 budget. If the test fails, the spec doesn't lock. Is the spec prepared to invalidate itself, or does the gate become a soft promise? Specifically: what does the rollback path look like if MH-1 returns a non-zero 95th percentile?
6. **Phase 7 prereqs P1+P2+P3 ordering**: P1 changes `config.json`, which rotates the FROZEN_PREREG SHA. P3 extends FROZEN_PREREG. If P1 ships before P3, the manifest temporarily lacks the resolved-vocabulary pin while config has a new shape. Is the prereq ordering safe, or does it need P3 to ship first?
7. **The activation-encoding asymmetry in Family B_v2** (§7.3): documented but not enforced. A reader joining B_v2 stratifier name to lineage outcome could mis-interpret. Is row-naming sufficient, or does the spec need a structured-data clarifier (e.g., a `stratifier_includes_activation` boolean column)?
8. **Process meta-attack**: this is round 3 of iteration on a single spec. The locked decisions in §1.1 are the spec's commitment that no further structural changes are needed. Is that commitment defensible against a fourth round, or is round 3 the point at which iteration becomes its own bad engineering?

## Appendix A. Process history (rounds 1, 2, and 3)

This appendix preserves the brutalist iteration log for audit. The locked spec in §1-§8 is the current truth; this appendix is provenance.

### A.1 Round 1 findings

Critic panel: Codex + Claude. Gemini rate-limited.

**Verified factual errors (patched in round 1):**
- §4.1 wrote ALR semantics ("N−1 components, reference = unassigned") while calling it CLR. Real CLR per `temporal_interface_analysis.py:316-326` has all N components with gmean denominator. Patched.
- Confidence sweep `{0.5, 0.7, 0.9}` is meaningless: empirical confidence values are exactly `{0.0, 0.333, 0.5, 1.0}` across 58,137 superpixels. Sweep dropped; replaced with `min_prevalence ∈ {0.005, 0.01, 0.02}`.
- Closed-by-design Family C argument was incomplete: 14/15 typed cell types pin CD44 status by gate construction; only `neutrophil` escapes. Argument rewritten with correct enumeration; `neutrophil` exception became Family C v2 single-row extension after round 2.
- Spec wrote 76% unassigned; actual 65.6% (38,157/58,137). Corrected.
- `apply_min_prevalence_filter` hard-codes `INTERFACE_CATEGORIES`. Phase 7 prereq P2 parameterizes it.

**Structural findings (patched in round 1 + restructure):**
- "Two paths" Family A_v2 corroboration was theatre (shared taxonomy). Single-path framing locked (§4.1).
- `composite_label_v1` deprecation column had no payer (no release process). Dropped; single-commit migration with enumerate-before-rename gate.
- v1/v2 cross-rule was absent (would have functioned as optional stopping). Locked as runtime demotion column (§4.4).
- 16-category lock contradicted `other_rare` pooling. Reconciled: 16-cat at input gate, ≤16 + `other_rare` at analysis vocabulary.

### A.2 Round 2 findings

Critic panel: Codex + Claude. Gemini rate-limited.

**Section-drift propagation (Critical, both critics):** round-1 patches were applied in §4 only. §5/§6/§7/§8 still described un-patched behavior. 11 specific stale references identified and patched. The round-2 patches added a propagation-checklist methodology that is re-run in T5 (§9 round 3 attack #1).

**New substantive design findings (escalated to locked decisions):**
- `unassigned`-as-scalar trades CLR-bias for conditional-on-assignment-success bias. Round 1 had dropped `unassigned`; round 2 found this changed the estimand. Locked: §1.1 Decision 4A brings `unassigned` back as 16th coord.
- A_v2 headline rule looser than A_v1; same CSV co-displays them. Locked: §4.5 row-level `headline_rule_version` column.
- "Never co-headlined" cross-rule was prose, not architecture. Locked: §4.4 runtime demotion column.
- Neutrophil scalar = Family C in costume. Locked: §4.6 fold into Family C with full discipline.
- Enumerate-before-rename without runtime assertion. Locked: MH-4 runtime rename assertion.
- §11 decision matrix was tilted (in-line "Default: yes" + recommendation paragraph). Removed in restructure.
- Round-1 deferrals (perm null, priority order, SHA pin) routed to "user choice" when they are must-haves. Locked: MH-1, MH-2, MH-3.

### A.3 Findings rejected after verification (round 1)

- "Priors are calibrated for v1's CLR effect-size distribution and don't transport." Partially rejected: priors are not v1-tuned but philosophical scales chosen a priori (`temporal_interface_analysis.py:64-73`). Underlying concern (different effect-size distribution in v2) mitigated by cross-rule.

### A.4 Dropped designs (do not re-implement)

These trajectories were considered and ruled out. Phase 7 implementation guards (Task D1) encode them as machine-checkable invariants so they cannot silently return.

- **`composite_label_v1` deprecation column**: no release process to amortize against; dropped in round 1.
- **ALR-style CLR with chosen reference category**: math error confused with CLR; dropped in round 1.
- **Confidence sweep `{0.5, 0.7, 0.9}`**: meaningless because actual values are `{0.0, 0.333, 0.5, 1.0}`; dropped in round 1.
- **Raw-marker corroborating path for Family A_v2** (shared taxonomy with primary path; would not be corroboration): dropped in round 1; A_v2 single-path locked.
- **Drop `unassigned` from CLR simplex** (round 1 patch superseded by round 2 finding): re-included in round 2 as 16th coord.
- **"Never co-headlined" cross-rule as prose**: not enforceable; replaced with runtime demotion column in round 2.
- **Tilted §11 decision matrix**: removed in restructure; locked decisions in §1.1 deliberately do not vote.

### A.5 Propagation checklist methodology

Used at T5 in the restructure task graph; re-run at every future patch:

For every locked decision in §1.1, verify it propagates to §3-§8. Status maintained in the spec's iteration commit messages. If a future patch introduces a new fix, this methodology must be re-run before the patch is claimed "applied" — round 2 found that skipping this is the dominant failure mode.

### A.6 Round 3 findings disposition

Critic panel: Codex + Claude. Gemini rate-limited a third time.

| Finding | Source | Disposition |
|---|---|---|
| F1: Cross-rule has no defined v1↔v2 join key | Codex Critical, Claude Critical | §4.4 patched with explicit join-key table; structural reach (~6/1296) acknowledged |
| F2: §8 "16-cat CLR" criterion contradicts §4.1 pooling | Claude High | §8 patched: criterion now spans pre-pooling and post-pooling vocabulary |
| F3: Prereq ordering breaks `verify_frozen_prereg.py` between merges | Both High | §6.0 reordered P3→P1→P2; each PR recomputes its own SHA |
| F4: §4.6 "no carve-out" framing was misleading | Claude High | §4.6 patched: per-ROI compute is new code; aggregation reuses Family C contract |
| F5: MH-1 binary criterion mathematically untenable | Claude Medium | Verified empirically (P(\|g\|>0.5\|H0)≈0.62 at n=2; ~800 expected false headlines per shuffle); replaced with empirical null-distribution criterion + null-mode evaluator (skip bootstrap+spatial); §1.1 MH-1 + §6.2 + §8 patched |
| F6: `headline_demoted_reason` enum was open-ended | Claude Medium | §6.3 patched: closed enum {null, cross_axis_co_headline_forbidden, family_b_basis_conflict} |
| F7: MH-4 discovery mechanism unspecified | Claude Medium | §1.1 MH-4 + §6.2 patched: post-execution validation via `nbconvert` + `__getitem__`/`query` trace |
| F8: MH-2 framing was misleading | Claude Low | §1.1 MH-2 patched: "load-bearing-explicit, not implicit" |
| Codex F1: `unassigned_rate_per_roi` schema-incoherent | Codex High | §6.3 patched: replaced with `unassigned_rate_mouse_mean_1/2` per contrast side |
| Codex F3: Self-contradictory headline query | Codex High | §4.4 patched: introduces `is_headline` boolean (all families) as canonical column |
| Codex F4: `lineage_source` doesn't exist (column is `normalization_mode`) | Codex High | §4.2 patched: align with code |
| Codex F5: MH-1 cost wildly understated (would be 500h, not 8h) | Codex High | §6.4 patched: null-mode evaluator with skip-bootstrap; ~1.5h estimate |
| Codex F6: P1 schema rejects unknown keys | Codex Medium | §6.0 P1 patched: schema update is part of P1 |
| Codex F2: Acceptance test 16-cat criterion (same as F2) | Codex High | Same patch as F2 |
| Codex F7: Doc drift in repo (348 vs 618 endpoint rows) | Codex Medium | Out of Phase 7 spec scope; flagged for R1 review-packet refresh |

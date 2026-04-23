# Frozen pre-registration manifest

Snapshot for external review. Every file referenced is pinned by SHA-256 + git commit so the reviewer can reproduce exactly what generated the one-pager and methods summary.

## Reproducibility anchors

| Field | Value |
|-------|-------|
| Git commit | `6563e90f84ff84eb7183762c188bcbd7609b2429` |
| Branch | `main` |
| `config.json` SHA-256 | `85c314245576f72b2f24e03240bc077e7608619d3627fb88de4d0101e6548170` |
| `viz.json` SHA-256 | `422a14a03eff4cd7934ce1ec35138e1891d293b65e167dc8d4db498bfb6c0bd2` |
| `analysis_plans/temporal_interfaces_plan.md` SHA-256 | `468d6c2c7da656e24f11b4ce73a4f8a87b3186824294ad574cc8530ba9fc375d` |
| `results/biological_analysis/sham_reference_10.0um.json` SHA-256 | `35c3faed6d05d12597cbf545c5a9b4acd5557b8d026abc5ade119c1be824800b` |
| Sham-ref artifact `_metadata.git_hash` | `6563e90f84ff84eb7183762c188bcbd7609b2429` (regenerated from the pinned commit; the earlier `cf0e337` / `git_dirty=True` build captured by any notebook pre-dating this manifest has been superseded) |
| Python | 3.12.10 |
| Platform | Darwin 25.4.0 (macOS) |
| Snapshot timestamp (UTC) | 2026-04-23 |

## Pre-registered endpoint families (summary; full spec in `analysis_plans/temporal_interfaces_plan.md`)

- **Family A — interface composition CLR** on 8 categories (`immune`, `endothelial`, `stromal`, `endothelial+immune`, `immune+stromal`, `endothelial+stromal`, `endothelial+immune+stromal`, `none`). Two normalization paths side-by-side: Sham-reference-centered sigmoid (primary); raw-marker Sham-reference percentile (independent corroboration). Sensitivity sweeps: lineage threshold {0.2, 0.3, 0.4}; raw-marker Sham percentile {65, 75, 85}.
- **Family B — neighbor-minus-self delta** on continuous lineage scores, per composite-label × neighbor-lineage. k=10 neighbors. Sensitivity: `min_support` {10, 20, 40}. Trajectory filter applied.
- **Family C — cross-compartment CD44⁺ activation** within (CD45⁺, CD31⁺, CD140b⁺, triple-overlap, background) compartments; Sham-referenced percentile threshold. Sensitivity: Sham percentile {65, 75, 85}.

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
- **Bodenmiller scope**: the companion `run_bodenmiller_benchmark.py` validates the IMC data loader at the channel level (Spearman r=0.996) against published data. It is NOT a test of the Family A/B/C framework, which requires temporal sampling absent from the Bodenmiller Patient1 dataset. See archived `benchmarks/STATUS.archived.md`.
- **Shared-reference tautology**: both Family A paths anchor on the same Sham baseline. Sign agreement is partly built-in. The symmetric magnitude-disagreement count (13/48 Family A endpoints in the current run) is the honest upper bound on independent measurement.
- **CLR compositional coupling**: `stromal_clr↓` + `triple_clr↑` are one event in two coordinates. Family C is the only genuinely non-compositional corroboration surface in this cohort.

## Known deferred follow-ups (non-blocking for this pilot's one-pager)

- ~~Continuous Sham-percentile sensitivity sweep at 50/60/70.~~ **Closed Phase 1.5b — `continuous_sham_pct_sweep.csv` + amendment in plan.**
- ~~Parallel raw-marker Sham-reference path for Family B neighbor-minus-self.~~ **Closed Phase 1.5c — `family_b_raw_marker_comparison.csv` + amendment in plan.** Surfaced 18 post-hoc discovery endpoints at Sham→D7; reported as sensitivity analysis, not pre-registered.
- ~~Pre-reg compliance: Family B support-sensitivity demotion flag; Family A CLR-without-`none` sensitivity propagation.~~ **Closed Phase 1.5a — `support_sensitive` + `clr_none_sensitivity` columns now in `endpoint_summary.csv`.**
- Tissue-mask-based non-compositional density (SLIC-constant surface retracted as algebraically equivalent to `2500 × proportion`) — remains open.

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
co = pd.concat([fa, fc])[['family','endpoint','hedges_g',
                          'g_shrunk_skeptical','g_shrunk_neutral','g_shrunk_optimistic']]
print(co.sort_values('g_shrunk_neutral', key=abs, ascending=False).to_string(index=False))
```

Family A endpoints filtered out by the rule (reported in the one-pager "Filtered out" table) are `stromal_clr` and `none_clr` (magnitude disagree), `endothelial_clr` (sign reverse, both near zero), `endothelial+stromal_clr` (|g| ≤ 0.5).

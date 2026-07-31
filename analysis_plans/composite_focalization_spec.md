# Composite focalization discriminator (pre-registered)

## Purpose

Report separate spatial and structural descriptors for each of the eight composite lineage categories (`none`, `immune`, `endothelial`, `stromal`, `endothelial+immune`, `immune+stromal`, `endothelial+stromal`, `endothelial+immune+stromal`) from its own within-scale (10 um) spatial self-enrichment and lineage composition. These are descriptive tissue-patch-state labels, explicitly not cell phenotypes.

## Metric and pinned rule

Self-enrichment is the `enrichment_score` for rows where `focal_cell_type == neighbor_cell_type` in the frozen product `results/biological_analysis/spatial_neighborhoods_composite/temporal_neighborhood_enrichments.csv`.

A category's spatial `status` is **focal** if and only if its D7 self-enrichment is >= 2.0x; otherwise it is **diffuse**. This threshold is frozen before the discriminator is run and must not be re-tuned after seeing outputs.

## Review clarification (2026-07-19)

`status` in {`focal`, `diffuse`} is a within-10um **spatial** self-enrichment statement applicable to any category: focal iff D7 self-enrichment >= 2.0x (threshold unchanged). `category_type` in {`no_lineage`, `pure_lineage`, `interface`} is **structural**: `none` -> `no_lineage`; a single lineage (`immune`, `endothelial`, or `stromal`) -> `pure_lineage`; and a multi-lineage category (`endothelial+immune`, `immune+stromal`, `endothelial+stromal`, or `endothelial+immune+stromal`) -> `interface`. The substantive focal-versus-diffuse interface reading applies only to those four `interface` categories; the informative result is that `endothelial+immune+stromal` is the lone diffuse interface.

## Mandatory interpretation caveats

Every use of the descriptor must carry all of the following caveats:

- **DESCRIPTIVE ONLY**: it makes no inferential or biological claim.
- The study has **n=2 mice per timepoint**.
- The analysis is confounded by DNA-only SLIC superpixels (segmentation uses a DNA1+DNA2 composite via `channels.dna_channels`), no spillover compensation anywhere in the codebase, and injury-driven cellularity that inflates multi-lineage rates.
- **Dilution corollary:** cross-scale robustness is **NOT** evidence of interface biology. A genuine focal co-expression dilutes at coarser grain, so scale-robust co-positivity indicates diffuse mixing, not a validated interface.

## Amendment — kNN size (k) sensitivity outcome

The kNN neighborhood size `k = 10` underlying every self-enrichment value here was never swept
when this spec was written. It has since been swept under a pre-registered, report-either-way
protocol (`analysis_plans/k_sensitivity_precommitment.md`, committed *before* the sweep ran;
output `results/biological_analysis/k_sensitivity_focalization.csv`). Outcome, at 10 µm pooled D7:

| category | k=5 | k=10 | k=20 | status stable? |
|---|---|---|---|---|
| endothelial+stromal | 3.43× | 3.05× | 2.63× | yes — focal |
| immune+stromal | 2.99× | 2.56× | 2.16× | yes — focal |
| **endothelial+immune** | **2.24× focal** | **1.99× diffuse** | **1.74× diffuse** | **NO — flips** |
| endothelial+immune+stromal (triple) | 1.45× | 1.35× | 1.27× | yes — diffuse |

**Consequences, per the pre-commitment (outcome 2):**

1. The "lone diffuse interface" framing is **withdrawn**. At k=10 on the scale×region basis
   endothelial+immune is *also* diffuse (1.99×), so the triple-positive is not the only one.
2. `endothelial+immune` sits on the 2.0× cutoff and its label is basis-dependent as well as
   k-dependent (2.03× on the temporal-composite basis vs 1.99× on the scale×region basis at the
   same k=10). Any statement about it must carry that caveat.
3. **The core reading survives**: the triple-positive is diffuse at every k tested, 27–35% below
   the cutoff, while remaining the largest and most-grown interface category. The mixing
   signature does not depend on k.
4. The **rank order** endothelial+stromal > immune+stromal > endothelial+immune > triple-positive
   is identical at every k. Self-enrichment magnitude declines monotonically with k for all
   categories (larger neighborhoods dilute), so magnitudes are only comparable at fixed k.
5. The fragile element is the **binary 2.0× cutoff**, not the measurement. The cutoff is retained
   (changing it post hoc would be exactly the forking-paths move this spec exists to prevent),
   but any category within ~5% of it must be reported as on-the-cutoff rather than as a status.

`k = 10` remains the primary value. This sweep is a disclosure, not a re-selection.

## Category-definition exclusion

Activation markers CD44 and CD140b are excluded from every category definition: the categories are lineage-only. This preserves the non-tautological neutrophil-CD44 headline.

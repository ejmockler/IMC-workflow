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
- The analysis is confounded by DNA-only SLIC superpixels (`config.json` has `slic_input_channels = DNA1, DNA2`), no spillover compensation anywhere in the codebase, and injury-driven cellularity that inflates multi-lineage rates.
- **Dilution corollary:** cross-scale robustness is **NOT** evidence of interface biology. A genuine focal co-expression dilutes at coarser grain, so scale-robust co-positivity indicates diffuse mixing, not a validated interface.

## Category-definition exclusion

Activation markers CD44 and CD140b are excluded from every category definition: the categories are lineage-only. This preserves the non-tautological neutrophil-CD44 headline.

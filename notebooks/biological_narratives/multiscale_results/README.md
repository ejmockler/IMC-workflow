# multiscale_results/ — orphan pre-Phase-7 outputs

**Status (2026-05-07): legacy / stale.**

The four CSVs in this directory (`enrichment_10um.csv`, `enrichment_20um.csv`, `enrichment_40um.csv`, `scale_coherence.csv`, all dated November 2025) were emitted by an early-phase pipeline run. Column labels in these files (`Resting_Endothelial`, `Activated_Fibroblast` without the CD44/CD140b suffix, `Activated_Immune_CD44`, `Activated_Immune_CD140b`) reflect a **pre-Phase-7 ontology** that is not present in the current 15-type config. No current notebook, script, or doc references these files.

The reviewer-facing analysis is in `results/biological_analysis/temporal_interfaces/endpoint_summary.csv` (840 rows × 46 cols, post-remediation) and `results/biological_analysis/spatial_neighborhoods/temporal_neighborhood_enrichments.csv` (current cell-type-pair enrichments). Treat the files in this directory as historical artifacts only.

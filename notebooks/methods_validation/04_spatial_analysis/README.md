# Spatial Analysis (Legacy)

**Created**: 2025-11-08
**Status**: 📦 Legacy descriptive content — superseded by Phase 2/Phase 7 temporal-interface analysis (`endpoint_summary.csv`, 840 rows × 46 cols, post-remediation). Numbers below are from a pre-current-ontology run (~59,972 superpixels under an earlier segmentation; current cohort is 58,137 superpixels at 10 µm).

## Purpose

Analyze spatial tissue organization using superpixel-level clustering. Discover tissue domains and track their evolution during kidney injury. The reviewer-facing analysis has moved to the pre-registered Family A/B/C framework; this notebook is retained as descriptive context only.

## Working Notebooks

### `01_spatial_tissue_domains.ipynb`

**Clusters 59,972 superpixels into 6 tissue domains** *(legacy run; current pipeline uses 58,137 superpixels)*

**Key findings** *(legacy; not regenerated)*:
- Domain 4 (Fibrotic/immune): CD44=4.37, CD11b=4.09 → Grows 3.5% → 20% (Sham to D7)
- Domain 2 (Vascular): CD31=3.36, CD34=2.89 → Expands 10.9% → 22%
- Domain 1 (Quiescent): Low all markers → Shrinks 19.3% → 12.7%

**Method**: K-means clustering on arcsinh-transformed marker expression

**Script version**: `test_domain_clustering.py` (standalone Python script)

## Why This Works

**Biology happens in space**:
- Fibrosis forms in focal domains (Domain 4)
- Vascular regions expand (Domain 2)
- Tissue reorganizes at superpixel scale
- The 92.8% superpixel variance IS the biology

**Superpixels are the right unit** for tissue micro-environment analysis. (The "(40-50 cells each)" estimate from earlier drafts has been retracted — see `benchmarks/notebooks/CRITICAL_ANALYSIS.retracted.md`. Per `METHODS.md` §SLIC and `RESULTS.md` §How-Processing-Shapes-Results, a 10 µm superpixel covers ~100 µm² of tissue and may span parts of multiple cells or extracellular matrix — not "40-50 cells each".)

## Next Steps

1. Assign biological names to all 6 domains
2. Visualize spatial domain maps (scatter plots)
3. Domain-level hierarchical statistics (Superpixels → Domains → ROIs → Mice)

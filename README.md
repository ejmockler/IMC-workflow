## IMC Spatial Analysis and Visualization Pipeline

This repository provides a transparent, no-code description of how we quantify spatial protein organization from Imaging Mass Cytometry (IMC) data and communicate results to investigators and peers. It explains what the pipeline does, why the methods are appropriate, what the outputs mean, and how to reproduce the figures—without requiring readers to inspect the source code.

## What the pipeline does (executive summary)

- **Purpose**: Reveal tissue organization and protein co-localization directly from IMC pixels, then summarize findings across time, condition, and region.
- **Approach**: Build local neighborhoods at biologically relevant distances, compute robust protein–protein co-localization, detect expression domains and their contacts, and derive interpretable protein networks.
- **Outputs**: Per-ROI visuals, temporal and condition comparisons, replicate variability, and network plots that highlight hubs and cross-compartment communication.

## How it works (high-level, no code)

### Inputs and preprocessing
- Reads ROI files from `data/` and selects proteins defined in `config.json` functional groups.
- Transforms intensities with an arcsinh cofactor (arcsinh(x/5)) to stabilize variance.
- Extracts metadata per ROI (timepoint, condition, tissue region, replicate) using standardized parsing rules and consistent label normalization.

### Neighborhood model (spatial scaffolding)
- Builds a spatial index and defines distance bands (in μm) with a small tolerance to capture interactions at specific scales.
- Uses these neighborhoods to evaluate local organization while controlling runtime through sampling and caps.

### Per‑protein spatial organization
- For each protein and distance band, estimates a neighborhood correlation metric, with robust fallbacks when variance is low.
- Produces a per-protein autocorrelation profile over distance (an “organization scale” view).

### Protein–protein co‑localization
- For every pair of proteins, collects local Pearson correlations across sampled neighborhoods (center pixel plus nearby neighbors).
- Aggregates using the median absolute correlation, which is robust to outliers and heterogeneous neighborhood sizes.
- The result is a co-localization score per protein pair for each ROI.

### Expression domains and contacts

> Terminology — "blob": In this documentation, a "blob" means an expression domain: a contiguous patch of pixels with similar multi‑protein expression, obtained by clustering pixel expression vectors. A "blob type" refers to the cluster label (the domain identity).

- Clusters pixels into expression “domains” (blob types) based on multi-protein profiles.
- Quantifies normalized contact frequencies between domain types using neighbor queries within a biologically motivated radius (e.g., ~15 μm).
- These contacts summarize how tissue regions interface in space.

### Aggregation across the experiment
- Summaries are stratified by injury day, condition, tissue region, and mouse replicate.
- Figures show trajectories over time, experimental contrasts, and replicate variability for the most informative interactions.

### Network analysis and interpretation
- Converts co-localization into a protein network: proteins are nodes; edges reflect spatial co-occurrence strength.
- Annotates proteins by functional groups from `config.json` and computes standard graph metrics (e.g., modularity, clustering, centrality, density).
- Highlights communities and hubs, and counts inter-group edges to characterize cross-compartment crosstalk (e.g., immune–vascular).

## What you can expect in the results

The pipeline writes a structured results file and a set of publication-ready figures in `results/`.

- `analysis_results.json`: ROI-level analysis summary used by the visualization pipeline.
- Per‑ROI detailed figures in `results/per_roi/`: domain maps, domain contact matrices, and protein–domain interaction context.
- Temporal and condition views (study-specific names may vary):
  - `kidney_healing_timeline.png`: key interactions over injury days.
  - `kidney_condition_comparison.png`: distributions and contrasts across conditions.
  - `kidney_region_time_grid.png`: region × time grids for domain contacts.
- Replicates:
  - `replicate_variance_analysis.png` and/or `kidney_replicate_variance.png`: robustness of top interactions across replicates.
- Network:
  - `network_analysis_comprehensive.png`: protein–protein co-localization network with functional group annotations.

Note: Exact filenames depend on which runner you use (standard visualization vs study scripts). All outputs are saved to `results/`.

## How to run (no code changes required)

```bash
# 1) Run the analysis pipeline (produces results/analysis_results.json)
python run_analysis.py

# 2) Generate standard figures from the analysis results
python run_visualization.py

# 3) Run the complete kidney-healing study workflow (optional)
python scripts/experiments/kidney_healing/run_full_report.py
```

## Configuration at a glance

All study parameters live in `config.json` and are loaded throughout the pipeline to ensure consistency.

- **Proteins and functional groups**: which channels are analyzed and how they are biologically grouped.
- **Spatial settings**: distance bands (μm), neighbor caps, and domain contact radius.
- **Visualization**: color palettes, figure sizes, and output destinations.
- **Network**: thresholds for including edges; functional annotations used for labeling.

Typical defaults that influence interpretation (report these in methods):
- Intensity transform: arcsinh(x/5)
- Distance bands with ±2 μm tolerance
- Co-localization neighborhoods: center pixel + up to 10 neighbors (sampled)
- Domain contact radius: ~15 μm; pixel subsampling for efficiency
- Network edge thresholds: low (≈0.05) for exploratory visualization; higher (≈0.25–0.30) for conservative summaries

## Detailed methods for PIs and peers

This section provides the methodological depth typically expected in lab meetings, manuscripts, and code reviews. It complements the high‑level overview above.

### Data model and preprocessing

- **ROI input format**: Tab‑separated text with pixel coordinates `X, Y` and protein channels named with isotopes (e.g., `CD31(146Nd)`). Coordinates are treated in microns; no explicit cell segmentation is used.
- **Protein selection**: Channels are chosen by name via `config.json → proteins.functional_groups`, excluding structural controls (e.g., DNA). Names are mapped back to base protein symbols for readability.
- **Intensity transform**: `arcsinh(x/5)` stabilizes variance and reduces heavy tails common in IMC data, improving correlation estimates.
- **Metadata parsing**: Filenames encode condition, injury day, region, and replicate. We parse and normalize labels (e.g., region aliases mapped to consistent names), falling back to title‑case for unknowns.

### Spatial scaffolding (neighborhoods by distance)

- **Indexing**: A spatial tree accelerates neighbor queries across the ROI.
- **Distance bands**: For each distance `d` (μm), neighbors are taken from a band `[d−2, d+2]`, removing self. Bands model interactions at distinct biological scales and reduce sensitivity to coordinate noise.
- **Sampling and caps**: Neighbor lists are truncated (e.g., first 10–20) to bound runtime while preserving local signal; pixel subsampling is used where needed (e.g., every 10th pixel for domain contacts).

### Per‑protein spatial organization (autocorrelation over distance)

- For each protein and distance band, we estimate a neighborhood correlation between pixel intensities and their neighbors. When variance is too low for a stable estimate, a robust fallback computes pairwise correlations on sampled neighbor pairs.
- Output is a per‑protein function over distance (non‑negative), interpretable as an “organization profile.” Peaks at specific distances suggest organized structures at that scale.

### Protein–protein co‑localization (robust neighborhood correlations)

- **Local windows**: For each pixel, create a small window consisting of the center pixel and up to N nearby neighbors from the chosen distance band.
- **Correlation measure**: Compute Pearson correlation for each pair of proteins within each window; take absolute values to focus on co‑occurrence strength irrespective of direction.
- **Aggregation**: Use the median of local correlations (and the standard deviation across windows) to summarize per‑ROI co‑localization. Median improves robustness to outliers and heterogeneity in local window size.
- **Thresholds**: Thresholds are not used to filter during computation; they are applied later for visualization or for building conservative networks.

### Expression domains and domain–domain contacts

- **Domain detection**: Pixels are clustered into expression “domains” (blob types) using multi‑protein profiles with a minimum domain size. Small or near‑duplicate clusters are merged to avoid fragmentation.
- **Contact quantification**: For a subsample of pixels, neighbors within ~15 μm are examined; counts of domain–domain adjacencies are accumulated and then normalized by total contacts per domain to produce frequencies.
- **Interpretation**: Contact frequencies summarize how distinct tissue regions interface; stable, high‑frequency contacts often reflect anatomical or microenvironmental structure.

### Aggregation and stratification

- ROI‑level metrics are aggregated by injury day, condition, tissue region, and mouse replicate. Figures focus on top‑informative interactions and report central tendencies alongside replicate variability.

### Network construction and analysis

- **Nodes and edges**: Proteins are nodes; edges connect protein pairs with co‑localization scores above a threshold. Edges carry weight (score), uncertainty (score standard deviation), and counts (number of local windows).
- **Annotations**: Each node is annotated with a functional group and textual annotation from `config.json` to support biological interpretation.
- **Graph metrics**: We compute modularity (community structure), clustering coefficient, average shortest path length (weighted), and density. Hub proteins are defined as the top ≈20% by betweenness centrality.
- **Crosstalk**: Inter‑group edges are counted; specific patterns (e.g., immune–vascular) are tracked as indicators of cross‑compartment communication.
- **Thresholds in practice**: Visualization may use a permissive threshold (≈0.05) to reveal structure; analyses reporting network metrics typically use a more conservative cutoff (≈0.25–0.30).

### Parameter defaults and rationale

- **arcsinh(x/5)**: Standard in cytometry; balances dynamic range and noise.
- **Distance bands with ±2 μm tolerance**: Captures interactions at intended scales while reducing sensitivity to discretization.
- **Local window size (up to ~10 neighbors)**: Preserves locality and reduces influence from distant pixels.
- **Domain contact radius (~15 μm)**: Approximates short‑range tissue adjacency at the pixel scale; adjustable per tissue.
- **Sampling caps**: Keep complexity manageable on large ROIs while maintaining stable estimates.

### Data dictionary (outputs)

Each ROI entry in `results/analysis_results.json` contains:

- `filename`: ROI file name.
- `metadata`: `{ condition, injury_day, tissue_region, mouse_replicate }`.
- `spatial_autocorrelation` (per protein): `{ protein: { distance_um: autocorr_value, ... }, ... }`.
- `colocalization`: `{ "PROT1↔PROT2": { colocalization_score, score_std, n_measurements }, ... }`.
- Optional fields used by detailed ROI figures (when full per‑ROI analysis is run): `coords`, `blob_labels`, `blob_type_mapping`, `canonical_contacts`.

Note: Field names and exact thresholds are governed by `config.json`; keys above reflect the current pipeline.

### Figure‑by‑figure interpretation guide

- **Per‑ROI detailed**: Domain map + contact matrix; look for coherent domains and dominant interfaces.
- **Temporal progression**: Track emergence/peak/resolution of top interactions across injury days; consistent patterns across ROIs indicate robust biology.
- **Condition comparison**: Contrast distributions between experimental and control; highlight shifts in key interactions.
- **Region × time grid**: Identify region‑specific temporal trajectories and divergent microenvironments.
- **Replicate variance**: Assess robustness; high variance suggests context dependence or batch effects.
- **Network analysis**: Communities indicate modules; hubs suggest coordination points; inter‑group edges support crosstalk hypotheses.

### Limitations and failure modes

- Pixel neighborhoods approximate, but do not guarantee, cellular context; segmentation is not used here.
- Clustered domains are heuristic; overly small domains may be merged, and boundaries may not align with histology.
- Pearson correlation captures linear associations; absolute values discard directionality.
- Spatial parameters (distance, radius, sampling) affect magnitudes; all should be reported with results.
- Co‑localization implies spatial co‑occurrence, not causality; interpret with annotations and prior knowledge.

### Validation and recommended diagnostics

- Report replicate variability and confidence summaries for top interactions.
- Sensitivity checks: vary distance bands, window sizes, and contact radii; report stability.
- Optional: permutation/rotation controls within ROIs to estimate null distributions for co‑localization.
- Optional: bootstrap network metrics to provide intervals for modularity, centrality rankings, and density.

## How to read the figures

- **Per‑ROI detailed plots**: show domain maps and domain–domain contact matrices. Use these to inspect local tissue architecture in each ROI.
- **Temporal progression**: tracks how strong interactions emerge, peak, and resolve across injury days.
- **Condition comparison**: contrasts interaction strengths or frequencies between experimental groups.
- **Region × time grid**: reveals spatial heterogeneity across tissue regions over time.
- **Replicate variance**: quantifies robustness across mice; large variance flags context-specific effects.
- **Network analysis**: nodes are proteins (colored by functional group); edges are co-localization strengths; hubs and communities point to coordination modules and cross-compartment crosstalk.

## Validation and quality considerations

- **Robust statistics**: median of local absolute correlations reduces outlier influence and handles varied neighborhood sizes.
- **Subsampling**: deliberate caps on neighbor counts and pixel sampling keep runtime manageable while preserving trends.
- **Normalization of contacts**: domain contact frequencies are normalized to reduce domain-size bias.
- **Replicates-first mindset**: dedicated replicate variance figures surface stability of key effects.

Recommended additions for manuscripts or peer review (optional):
- Permutation or rotation tests within ROIs to estimate null distributions of co-localization scores.
- Sensitivity analyses for distance bands, neighbor caps, and contact radii.
- Bootstrap intervals for network metrics and top interactions.

## Assumptions and limitations (state explicitly)

- Pixel‑level neighborhoods approximate cellular contexts; no cell segmentation is used here.
- Expression domains (clusters) are heuristic; merging/splitting of biological structures is possible.
- Pearson correlation captures linear association; using absolute values loses directionality.
- Spatial parameters (distance bands, radii, sampling) affect scores and should be reported alongside results.
- Co-localization reflects spatial co-occurrence, not necessarily direct signaling; interpret in biological context.

## Reproducibility and provenance

- All figures and tables are derived from `data/` using parameters in `config.json` and saved in `results/`.
- Outputs are not tracked in version control to keep the repository lean; they can be regenerated at any time.
- Utility scripts in `scripts/utilities/` help with housekeeping (e.g., checking for large files prior to committing).

## Repository layout (for orientation)

```
IMC/
├── src/
│   ├── analysis/        # Spatial organization, co-localization, domains, networks
│   ├── visualization/   # Per‑ROI, temporal, condition, replicate, network figures
│   └── utils/           # Configuration and helpers
├── scripts/
│   └── experiments/     # Study‑specific workflows (e.g., kidney healing)
├── data/                # Input IMC ROI files (TSV)
└── results/             # Generated outputs (JSON, PNG)
```

## One‑slide talk track (optional)

- We quantify spatial protein organization directly from pixels using biologically meaningful neighborhoods.
- Robust median neighborhood correlations yield stable co-localization scores for all protein pairs.
- Expression domains and their contacts summarize how tissue regions interface.
- Aggregated figures trace trajectories over time, compare conditions, and show replicate robustness.
- A functional protein network highlights hubs, communities, and cross-compartment crosstalk.

# Implementation Plan: Publication-Ready IMC Pilot

> **Constraint**: n=2 mice/group, 9 markers, 25 ROIs, zero FDR-significant findings.
> **Frame**: Hypothesis-generating spatial proteomics pilot with INDRA-grounded biological context.
> **Principle**: Every output must survive a reviewer who reads the Methods before the Results.

---

## Architecture Invariants

These patterns pervade the codebase. Every work package respects them.

1. **Config-driven** — `config.json` is the single source of truth. No hardcoded parameters.
2. **Analysis → artifact → visualization** — Analysis scripts produce CSV/JSON. Viz scripts consume them. Never mix.
3. **Mouse-level aggregation** — The biological replicate is the mouse (n=2), not the ROI (n=6). Every statistical claim operates at mouse level.
4. **Provenance** — Every output traces to inputs + code version + config hash.

---

## Work Packages

Six packages. WP1-WP4 are independent and run in parallel. WP5 depends on WP1+WP3. WP6 is a verification gate.

```
WP1 (INDRA refinement) ──────┐
WP2 (spatial figures)  ──────┤
WP3 (enrichment heatmaps) ───┼──> WP5 (narrative assembly) ──> WP6 (verification)
WP4 (power + benchmark) ─────┘
```

---

### WP1: Refine INDRA to Biological Context Layer

**Agent background**: Computational biologist fluent in knowledge graphs, INDRA/CoGEx architecture, and the distinction between evidence and context. Understands Fisher's exact test limitations on small gene sets and why ORA on a designed panel is circular.

**Why**: The brutalist review correctly identified that the current evidence table annotates 104/126 findings at `evidence_tier=none`, inflating null results with quantitative-looking columns. The framework must be restructured from "evidence table" to "biological context" — Discussion fuel, not Results decoration.

**Read first**:
- `build_indra_evidence_table.py` (the entire file — 1016 lines, 5 knowledge layers)
- `results/biological_analysis/indra_knowledge_base.json`
- `results/biological_analysis/indra_evidence_table.csv` (first 30 rows to see the problem)

**Do**:

1. **Add shared upstream regulators to knowledge base**. These were queried via Cypher and are biologically potent:
   ```
   TGF-β → 5/8 panel genes (PECAM1, PDGFRB, MRC1, CD44, CD34)
   VEGF/VEGFA → 4/8 (PECAM1, PDGFRB, CD44, CD34)
   KDR (VEGFR2) → 4/8 (PECAM1, PDGFRB, PDGFRA, CD34)
   LPS → 4/8 (PECAM1, MRC1, CD44, CD34)
   TNF → 3/8 (PECAM1, CD44, CD34)
   IL-6 → 3/8 (MRC1, CD44, CD34)
   ```
   Add as `SHARED_UPSTREAM_REGULATORS` dict in `build_indra_evidence_table.py`. Include in JSON export.

2. **Add mediated paths to knowledge base**. Key intermediaries connecting panel genes through third parties:
   ```
   CD44 ──Complex(288)──> CD74 ──Complex(29)──> PDGFRB   (MIF receptor axis)
   CD44 ──Complex(218)──> EGFR ──Complex(45)──> PDGFRA   (RTK crosstalk)
   CD44 ──Complex(238)──> Integrins ──Complex(16)──> PECAM1  (adhesion cascade)
   PDGFRB ──Complex(236)──> PDGF-BB ──Activation(37)──> PDGFRA  (ligand-receptor)
   ```
   Add as `MEDIATED_PATHS` dict. These explain WHY co-localized markers interact.

3. **Restructure CSV output**. Split into two files:
   - `indra_panel_context.json` — Per-gene profiles, shared regulators, mediated paths, panel-level summary. This is the Methods/Discussion reference. Static, doesn't depend on analysis results.
   - `indra_finding_annotations.csv` — Only findings where `indra_evidence_count > 0` OR `aki_gene_association = True`. Drop the 104 "none" rows. Rename columns: `evidence_tier` → `indra_context_tier`, `mechanistic_note` → `biological_context`.

4. **Add panel coherence summary**. Compute and report: "8 grounded genes, 91 intra-panel causal statements, 5/8 regulated by TGF-β, 4/8 by VEGF. Panel captures immune infiltration (PTPRC, ITGAM, CD44), vascular integrity (PECAM1, CD34), stromal/fibrotic response (PDGFRA, PDGFRB), and resolution (MRC1) axes." This is the one-paragraph panel justification for Methods.

**Do NOT**:
- Add ORA, RCR, or source-target explanation. These are methodological theater for n=2 with 8 genes.
- Add any new INDRA queries. All data is pre-queried and sufficient.
- Frame any INDRA output as "validation" or "evidence" for spatial findings.

**Acceptance criteria**:
- `indra_panel_context.json` exists with shared regulators, mediated paths, panel coherence summary
- `indra_finding_annotations.csv` has zero rows with `indra_evidence_count=0` AND `aki_gene_association=False`
- `build_indra_evidence_table.py` runs cleanly: `.venv/bin/python build_indra_evidence_table.py`
- Script prints panel coherence summary on stdout

**Files modified**: `build_indra_evidence_table.py`
**Files created**: None (outputs go to gitignored `results/`)

---

### WP2: Spatial Figures — Representative IMC Images

**Agent background**: Image processing engineer familiar with IMC/multiplexed imaging data formats, matplotlib compositing, and publication figure standards. Understands that spatial biology papers live and die on images — a reviewer needs to SEE the tissue architecture, not just read statistics about it.

**Why**: Both brutalist critics identified representative images as the #1 missing deliverable. The neighborhood enrichment statistics are meaningless without spatial context showing WHERE these cell types are and what the tissue looks like.

**Read first**:
- `config.json` — channel definitions (lines ~30-90), cell type annotations (lines ~120-200)
- `src/viz_utils/plotting.py` — existing `plot_segmentation_overlay()`, `plot_protein_expression()`
- `src/analysis/main_pipeline.py` — how ROI data is loaded and stored (HDF5 structure)
- `src/utils/helpers.py` — metadata classes, ROI naming conventions
- One ROI result dir: `results/cross_sectional_kidney_injury/roi_results/` (pick any, read metadata.json)

**Data paths**:
- Raw IMC: `data/241218_IMC_Alun/*.txt` (32MB each, 54 channels including 9 protein + 2 DNA)
- Cell type annotations: `results/biological_analysis/cell_type_annotations/*.json`
- ROI metadata: `data/241218_IMC_Alun/Metadata-Table 1.csv`

**Do**:

1. **Create `generate_spatial_figures.py`** (standalone script, root level). This script:
   - Reads config.json for channel definitions and cell type colors
   - Loads 1 representative ROI per timepoint (4 ROIs: Sham, D1, D3, D7) — choose cortex ROIs from mouse 1 for consistency
   - For each ROI, generates a figure panel:
     - **Panel A**: 3-channel composite (DNA=blue, CD44=green, CD31=red) showing tissue architecture + injury marker + vasculature
     - **Panel B**: Cell type overlay — superpixels colored by assigned cell type, unassigned in light gray
     - **Panel C**: Key marker expression heatmap across the spatial field
   - Saves to `results/figures/spatial_overview_{timepoint}.png` at 300 DPI

2. **Use existing infrastructure**. The pipeline already loads `.txt` files via `ion_count_processing.py` and produces superpixel-level data. Reuse the data loading path — don't rewrite it. The cell type annotation JSONs contain superpixel-to-celltype mappings.

3. **Color scheme**: Use the existing marker group colors from `comprehensive_figures.py` (immune=Reds, vascular=Blues, stromal=Greens). Cell type colors should be consistent with any existing heatmaps.

4. **Figure size**: 180mm wide (Nature standard), 4-panel layout (one column per timepoint).

**Do NOT**:
- Add statistical annotations to images
- Generate images for all 25 ROIs — 4 representative is sufficient
- Implement new data loaders — use existing pipeline infrastructure

**Acceptance criteria**:
- 4 PNG files in `results/figures/` showing Sham, D1, D3, D7 spatial overviews
- Each figure has 3 panels (composite, cell types, marker heatmap)
- Unassigned tissue (~79%) is visible but de-emphasized (light gray)
- Color legend present for cell types
- Script runs end-to-end: `.venv/bin/python generate_spatial_figures.py`

**Files created**: `generate_spatial_figures.py`
**Dependencies**: numpy, matplotlib, scipy (all in .venv)

---

### WP3: Neighborhood Enrichment Heatmaps

**Agent background**: Data visualization specialist with spatial statistics knowledge. Understands enrichment scores (observed/expected ratios), permutation-based p-values, and how to present matrix data for biological interpretation. Knows that self-clustering (diagonal) vs cross-type enrichment (off-diagonal) tells different biological stories.

**Why**: Neighborhood enrichment is the strongest finding in this dataset. Self-clustering of cell types (2-3x enrichment) validates the segmentation and reveals spatial organization. The temporal evolution of these patterns (Sham→D1→D3→D7) is the core spatial narrative. Currently this exists only as CSV — it needs visual form.

**Read first**:
- `results/biological_analysis/spatial_neighborhoods/temporal_neighborhood_enrichments.csv` (257 rows — this is the primary data)
- `spatial_neighborhood_analysis.py` (the script that generates the data, especially the display sections at lines 390-440)
- `src/viz_utils/comprehensive_figures.py` — existing `_plot_neighborhood_composition()` function

**Do**:

1. **Create `generate_enrichment_heatmaps.py`** (standalone script, root level). This script:
   - Reads `temporal_neighborhood_enrichments.csv`
   - Produces a 2×2 grid of heatmaps (one per timepoint: Sham, D1, D3, D7)
   - Each heatmap: rows = focal cell type, columns = neighbor cell type, values = log2(enrichment_score)
   - Color scale: diverging (blue=depleted, white=neutral, red=enriched), centered at 0
   - Diagonal (self-clustering) highlighted with border or annotation
   - Only include cell types with ≥10 focal cells (drop rare types that produce noisy enrichments)
   - FDR significance overlay: cells where `fraction_significant_fdr > 0.5` get a dot or asterisk

2. **Summary figure**: Below the 4-panel grid, add a line plot showing how the top 3-5 enrichment scores change across timepoints (Sham→D1→D3→D7). This reveals temporal dynamics — e.g., does immune-endothelial proximity increase after injury?

3. **Exclude unassigned**. The `unassigned` cell type dominates at ~79% and produces trivially high self-clustering. Exclude from heatmaps.

4. **Figure size**: 180mm wide, ~200mm tall (4 heatmaps + summary).

**Do NOT**:
- Include p-value annotations (non-significant at mouse level, only ROI-level permutation significance)
- Generate per-ROI heatmaps — temporal aggregation is the right granularity
- Add INDRA annotations to the figure — keep spatial results separate from knowledge context

**Acceptance criteria**:
- Single PNG: `results/figures/neighborhood_enrichment_temporal.png` at 300 DPI
- 4 heatmaps (Sham/D1/D3/D7) with consistent color scale across panels
- Temporal trajectory subplot showing top enrichment trends
- Self-clustering diagonal visually distinct
- Unassigned excluded
- Script runs end-to-end: `.venv/bin/python generate_enrichment_heatmaps.py`

**Files created**: `generate_enrichment_heatmaps.py`

---

### WP4: Power Analysis & Bodenmiller Benchmark

**Agent background**: Biostatistician with experience in spatial proteomics experimental design and method benchmarking. Understands that n=2 is a pilot, not a study — the value is in estimating effect sizes for the follow-up. Familiar with Steinbock/DeepCell segmentation pipelines and how to compare SLIC superpixel-based approaches against single-cell segmentation.

**Why**: The two highest-value non-visualization deliverables: (1) a power analysis telling experimentalists how many mice they need for the follow-up, and (2) a benchmark proving the pipeline works on established data before trusting it on novel data.

**Read first**:
- `notebooks/methods_validation/power_analysis.py` (existing power analysis — read fully)
- `differential_abundance_analysis.py` (the mouse-level Hedges' g computation, lines 200-280)
- `results/biological_analysis/differential_abundance/temporal_differential_abundance.csv`
- `benchmarks/configs/bodenmiller_benchmark_config.json`
- `benchmarks/data/bodenmiller_example/Patient1/steinbock_outputs/Patient1/steinbock_workdir/` — list contents

**Part A: Power Analysis Script**

1. **Create `generate_power_analysis.py`** (standalone, root level). This script:
   - Reads `temporal_differential_abundance.csv` to extract observed Hedges' g values per cell type per comparison
   - For each cell type, computes the largest absolute Hedges' g across all temporal comparisons (the "best-case" pilot estimate)
   - Computes required n per group for 80% power at alpha=0.05 (two-sided Mann-Whitney) to detect that effect size
   - Produces a table: `cell_type | max_hedges_g | ci_lower | ci_upper | n_required_80pct | n_required_90pct`
   - Saves to `results/power_analysis/sample_size_requirements.csv`
   - Generates a forest plot of effect sizes with CIs: `results/figures/pilot_effect_sizes.png`

2. **Report the honest answer**: "With n=2, the largest observed effect was |g|=X.X (cell type Y, comparison Z). To detect this effect at 80% power requires n=N per group. Most observed effects have CIs crossing zero, indicating n=2 is insufficient for any individual comparison."

3. **Use scipy.stats for power computation**. For Mann-Whitney with small n, use simulation-based power (bootstrap from observed distributions) rather than parametric approximation.

**Part B: Bodenmiller Benchmark Comparison**

1. **Create `run_bodenmiller_benchmark.py`** (standalone, root level). This script:
   - Loads Steinbock outputs: `benchmarks/data/bodenmiller_example/Patient1/steinbock_outputs/.../intensities/*.csv` (cell × channel matrices from DeepCell segmentation)
   - Loads our pipeline outputs for the same ROIs
   - Computes concordance metrics:
     - Per-marker mean expression correlation (Steinbock single-cell vs our superpixel means)
     - Spatial neighborhood overlap (Steinbock Delaunay neighbors vs our radius-based neighbors)
     - Cluster assignment similarity (if Steinbock provides cell type labels)
   - Produces a summary table and correlation scatter plot
   - Saves to `results/benchmark/bodenmiller_concordance.csv` and `results/figures/benchmark_concordance.png`

2. **Scope carefully**. Bodenmiller has 54 markers, we process a subset. Compare only overlapping markers. The goal is pipeline validation (does SLIC produce reasonable aggregations?), not equivalence proof.

**Do NOT**:
- Overfit the power analysis to look optimistic — report CIs honestly
- Claim SLIC equals DeepCell — they're fundamentally different approaches. Show concordance, not superiority.
- Implement mixed-effects or hierarchical power models — keep it simple (per-group sample size for Mann-Whitney)

**Acceptance criteria**:
- `results/power_analysis/sample_size_requirements.csv` exists with all cell types
- `results/figures/pilot_effect_sizes.png` shows forest plot with CIs
- `results/benchmark/bodenmiller_concordance.csv` exists with per-marker correlations
- Both scripts run end-to-end with `.venv/bin/python`

**Files created**: `generate_power_analysis.py`, `run_bodenmiller_benchmark.py`

---

### WP5: Narrative Assembly — METHODS.md and README

**Agent background**: Scientific writer with spatial proteomics publication experience. Understands how to frame a pilot study for maximum utility: the value is in the spatial patterns, the pipeline, and the hypotheses — not in confirmatory statistics. Familiar with INDRA/CoGEx as a knowledge grounding tool. Ruthlessly honest about limitations.

**Depends on**: WP1 (INDRA panel context), WP3 (enrichment heatmaps data)

**Read first**:
- `METHODS.md` (current version — check for remaining discrepancies with code)
- `README.md` (current framing)
- `build_indra_evidence_table.py` (after WP1 modifications — for panel coherence summary)
- `results/biological_analysis/spatial_neighborhoods/temporal_neighborhood_enrichments.csv` (for spatial narrative)

**Do**:

1. **Update METHODS.md Section: Panel Design Justification** (new subsection after marker descriptions). Use the INDRA panel coherence summary from WP1:
   ```
   Panel biological coverage was assessed against the INDRA/CoGEx knowledge
   graph (queried 2026-02-20). The 8 groundable markers (Ly6G excluded as
   murine-specific) encode genes with 91 known intra-panel causal relationships,
   spanning immune infiltration (PTPRC, ITGAM), tissue injury/adhesion (CD44),
   vascular integrity (PECAM1, CD34), stromal/fibrotic response (PDGFRA, PDGFRB),
   and anti-inflammatory resolution (MRC1). Five of 8 genes are regulated by
   TGF-β, the master regulator of renal fibrosis. The panel was not designed
   for pathway enrichment analysis (n=8 genes precludes meaningful ORA).
   ```

2. **Update METHODS.md Section: Spatial Neighborhood Analysis**. Add:
   - The ablation result: spatial_weight=0 produces identical enrichment scores (r=1.000), confirming self-clustering reflects marker co-expression via boolean gating, not spatial weighting artifacts.
   - FDR correction: BH within each ROI, then fraction of ROIs reaching FDR significance reported.

3. **Add METHODS.md Section: Limitations** (if not already present). Include:
   - n=2 per group: insufficient for inferential statistics, zero FDR-significant findings
   - 79% unassigned tissue: 9-marker panel cannot annotate the full cellular landscape
   - Near-zero clustering stability (bootstrap ARI) at all scales
   - Pilot framing: effect sizes reported for follow-up study design, not for confirmatory claims

4. **Update README.md**: Add a "Quick Start" section pointing to the 3 standalone analysis scripts and the figure generation scripts. Remove any language suggesting validated findings.

**Do NOT**:
- Add claims about INDRA "validating" spatial findings
- Remove the honest limitation statements
- Add new Methods sections for analyses that weren't performed (ORA, RCR)

**Acceptance criteria**:
- METHODS.md contains panel justification with INDRA provenance
- METHODS.md contains spatial_weight ablation result
- METHODS.md contains explicit limitations section
- No remaining discrepancies between METHODS.md and code
- README.md has clear pilot framing

**Files modified**: `METHODS.md`, `README.md`

---

### WP6: Verification Gate

**Agent background**: Skeptical reviewer persona. Reads the outputs of WP1-WP5 and tries to find overclaiming, inconsistency, or methodological errors. Has access to all results files and the code.

**Depends on**: All previous WPs complete.

**Do**:

1. **Grep METHODS.md for every number** and cross-reference with code/config:
   - `grep -oP '\d+\.?\d*' METHODS.md` → verify each against source
   - Check: marker count (9), ROI count (25), timepoint count (4), mouse count (2/group), scales (10/20/40μm)

2. **Check figure consistency**: Do the heatmap color scales match? Are cell type colors consistent between spatial overlays and heatmaps? Are the same cell types excluded (unassigned) everywhere?

3. **Check INDRA outputs**: Run `build_indra_evidence_table.py` and verify:
   - No rows in `indra_finding_annotations.csv` with `indra_evidence_count=0` AND `aki_gene_association=False`
   - `indra_panel_context.json` contains shared regulators and mediated paths
   - Panel coherence summary is printed

4. **Check for overclaiming**: Search all `.py` and `.md` files for:
   - "significant" without "non-significant" or "not significant" qualification
   - "validates" or "confirms" (should be "contextualizes" or "is consistent with")
   - "p < 0.05" without FDR context
   - Any implication that n=2 supports inferential conclusions

5. **Run all scripts** in sequence and verify clean execution:
   ```
   .venv/bin/python build_indra_evidence_table.py
   .venv/bin/python generate_spatial_figures.py
   .venv/bin/python generate_enrichment_heatmaps.py
   .venv/bin/python generate_power_analysis.py
   .venv/bin/python run_bodenmiller_benchmark.py
   ```

**Acceptance criteria**:
- Zero overclaiming instances found
- All scripts run without error
- All figures render at 300 DPI
- METHODS.md numbers match code
- Pilot framing is consistent across all outputs

---

## Explicitly Killed

These were considered and rejected based on brutalist peer review:

| Capability | Why Killed |
|---|---|
| **Discrete ORA** (go_ora, reactome_ora) | Circular: panel was designed to capture these pathways. n=8 genes from a designed panel tests panel selection, not data. |
| **Reverse Causal Reasoning** | Zero FDR-significant differential expression → no directional signal to infer from. Would fit noise to causal model. |
| **Source-Target Explanation** | Without significant changes, output is entirely knowledge-graph-driven. Produces plausible-looking narratives explaining nothing observed. |
| **GSEA** | Requires genome-wide ranked list. We have 9 markers. |
| **Subnetwork construction** | Interesting but not available via MCP; direct Neo4j access required. Pre-queried mediated paths (WP1) capture the key biology. |

---

## Dependency Graph

```
                    ┌─── WP1: INDRA refinement ──────────┐
                    │                                     │
Data (existing) ────┼─── WP2: Spatial figures ────────────┼──> WP5: Narrative ──> WP6: Verify
                    │                                     │
                    ├─── WP3: Enrichment heatmaps ────────┘
                    │
                    └─── WP4: Power + Benchmark (independent)
```

WP1-WP4: fully parallel, no dependencies between them.
WP5: reads outputs of WP1 (panel context) and WP3 (enrichment data for narrative).
WP6: reads everything, runs everything, flags inconsistencies.

# Results — Kidney Healing 2024 (kidney_healing_2024)

## What This Experiment Is

A hypothesis-generating pilot using Imaging Mass Cytometry to map the spatial protein landscape of murine kidney tissue across an acute injury time course. Eight C57BL/6 mice — two per timepoint (Sham, Day 1, Day 3, Day 7 post-injury) — each contributing 2-3 regions of interest (24 ROIs total, cortex and medulla). A 9-marker antibody panel measures immune infiltration, vascular integrity, stromal remodeling, and tissue activation at single-micron resolution.

The pipeline converts raw ion count images into multi-scale spatial features, assigns superpixels to cell phenotypes via boolean gating, quantifies temporal and regional abundance changes with effect sizes, maps spatial neighborhood architecture through permutation testing, and grounds findings in published molecular biology via the INDRA knowledge graph.

With n=2 mice per group, no comparison can reach conventional statistical significance. Mann-Whitney U on two observations per group produces only three possible p-values (~0.33, 0.67, 1.0). Zero findings survive FDR correction. The value of this work is in establishing the analytical framework, measuring effect sizes for power calculations, and identifying patterns worth pursuing in adequately powered follow-up studies.

**Config**: `config.json` | **Panel**: 9 markers | **ROIs**: 24 (18 injury + 6 sham) | **Timepoints**: Sham, D1, D3, D7

---

## The Marker Panel

Nine protein markers cover five biological axes relevant to kidney injury and repair:

| Marker | Gene | CURIE | Biological Axis |
|--------|------|-------|-----------------|
| CD45 | PTPRC | hgnc:9666 | Pan-leukocyte (immune infiltration) |
| CD11b | ITGAM | hgnc:6149 | Myeloid cells (neutrophils, macrophages) |
| Ly6G | Ly6g | uniprot:P35461 | Murine neutrophils |
| CD140a | PDGFRA | hgnc:8803 | Fibroblasts, mesenchymal cells |
| CD140b | PDGFRB | hgnc:8804 | Pericytes, vascular mural cells |
| CD31 | PECAM1 | hgnc:8823 | Endothelial cells |
| CD34 | CD34 | hgnc:1662 | Endothelial progenitors |
| CD206 | MRC1 | hgnc:7228 | M2/alternatively activated macrophages |
| CD44 | CD44 | hgnc:1681 | Tissue injury, hyaluronan receptor |

**Immune infiltration** (CD45, CD11b, Ly6G) distinguishes neutrophil-dominant acute inflammation from broader myeloid recruitment. **Myeloid polarization** (CD206) marks alternatively-activated macrophages associated with tissue repair, though the panel lacks M1 markers (would need CD86 or TNF-a). **Vascular integrity** (CD31, CD34) identifies the vascular compartment and endothelial stress states. **Stromal/fibrotic response** (CD140a, CD140b) captures the PDGF receptor axis central to the repair-versus-fibrosis decision. **Tissue injury** (CD44) is the only panel gene with a direct MESH annotation to acute kidney injury (MESH:D058186), upregulated broadly during damage and remodeling.

All 9 markers were grounded via the INDRA knowledge graph, yielding 117 unique published molecular relationships among the 8 groundable genes (Ly6G is murine-specific with limited INDRA coverage). Five of eight genes are regulated by TGF-beta, the master regulator of kidney fibrosis. Four are regulated by VEGF. The panel sits at the intersection of six kidney-relevant pathways including nephrogenesis (WP4823), neutrophil degranulation (R-HSA-6798695), and glomerular endothelium development (GO:0072011).

**What the panel cannot do**: With 9 markers, approximately 77% of tissue superpixels remain unassigned to any cell type. T cells, B cells, dendritic cells, and epithelial cells cannot be identified. Macrophage polarization is partial. Fibroblast activation states beyond CD44 positivity are invisible. These are the expected limitations of a 9-marker panel, not failures of the analysis.

---

## How Processing Shapes Results

Understanding the results requires understanding three methodological choices that directly affect what the numbers mean.

**Arcsinh transformation with data-driven cofactors.** Raw ion counts span 0 to ~10,000. The arcsinh transform (cofactor = 5th percentile of positive counts per marker) stabilizes variance and compresses this range into approximately 0-8, making markers with vastly different dynamic ranges comparable. All expression values reported below are arcsinh-transformed.

**SLIC superpixels, not cells.** The pipeline segments tissue into morphologically-coherent patches using DNA channel morphology, not into individual cells. At 10 um scale, a superpixel covers ~100 um^2 — potentially spanning parts of multiple cells or extracellular matrix. This means "cell type" is shorthand for "tissue microregion with a cell-type-consistent marker profile." Three scales (10/20/40 um) capture different organizational levels of kidney anatomy: capillary diameter, tubular cross-section, and glomerular diameter respectively.

**Boolean gating with per-ROI thresholds.** Cell types are assigned by marker positivity at the 60th percentile within each ROI (with marker-specific overrides: CD206 at 50th, Ly6G at 70th). This normalizes inter-ROI technical variation but partially suppresses genuine biological differences — a D7 ROI with more CD45+ cells gets a higher threshold, potentially masking the very signal of interest. This is the single biggest methodological concern for follow-up work (see Discretization Cost below). The gradient discretization notebook quantifies the cost directly: **82.9% of Shannon entropy is lost** when continuous marker distributions are collapsed into binary gates.

---

## 1. Tissue Composition

### Cell Type Assignment

Assignment rate across 24 ROIs: **14.8%–31.8% (mean 22.4%)**. The remaining ~77% of superpixels lack sufficient marker combinations for annotation — a direct consequence of the 9-marker panel.

| Cell Type | Overall | D1 | D3 | D7 |
|-----------|---------|----|----|-----|
| Activated Immune CD44+ | 4.7% | — | — | — |
| Activated Fibroblast | 4.4% | 4.0% | 2.9% | 1.7% |
| Activated Endothelial CD44+ | 4.2% | 5.3% | 5.4% | 2.5% |
| Neutrophil | 2.5% | 6.4% | 5.8% | 9.1% |
| Resting Endothelial | 2.0% | — | — | — |
| Activated Endothelial CD140b+ | 1.9% | — | — | — |
| Activated Immune CD140b+ | 1.5% | — | — | — |
| M2 Macrophage | 1.3% | 6.8% | 9.7% | 14.3% |
| Unassigned | 77.4% | — | — | — |

M2 macrophages more than double from D1 to D7 (6.8% to 14.3%). Neutrophils rise from 6.4% to 9.1%. Activated fibroblasts decline (4.0% to 1.7%). These trajectory patterns — rising resolution-phase markers, declining acute-phase markers — are consistent with the expected injury-to-repair transition, though all proportions carry the caveat that they are relative to total tissue (including the 77% unassigned).

### Marker Expression (arcsinh-transformed, 10 um scale)

| Marker | Mean | Std | CV across ROIs | Variability |
|--------|------|-----|----------------|-------------|
| CD44 | 2.510 | 1.131 | 0.179 | HIGH |
| CD140b | 2.276 | 0.874 | 0.124 | MODERATE |
| CD45 | 1.693 | 0.591 | 0.122 | MODERATE |
| CD11b | 2.392 | 1.020 | 0.104 | MODERATE |
| CD206 | 2.203 | 0.922 | 0.102 | MODERATE |
| CD34 | 2.111 | 0.745 | 0.083 | LOW |
| Ly6G | 1.745 | 0.555 | 0.080 | LOW |
| CD31 | 2.423 | 0.921 | 0.074 | LOW |
| CD140a | 2.192 | 0.765 | 0.064 | LOW |

CD44 (tissue injury/activation) is the most heterogeneous marker across ROIs (CV=0.179), consistent with its role as a dynamic injury response marker that varies with local damage severity. Structural markers (CD31, CD140a) are the most stable — the vasculature and stroma have consistent baseline expression regardless of injury state.

---

## 2. Marker Correlation Structure

At D7, pairwise correlations at 10 um reveal three tight biological axes:

| Axis | Markers | Pearson r | Interpretation |
|------|---------|-----------|----------------|
| Vascular | CD31 — CD34 | 0.851 | Endothelial identity axis |
| Myeloid | CD45 — CD11b | 0.848 | Immune infiltrate axis |
| Stromal | CD140a — CD140b | 0.814 | PDGFR signaling axis |

These three axes are independently strong (r > 0.81) and match known biology — CD31/CD34 co-express on endothelial cells, CD45/CD11b co-express on myeloid cells, CD140a/CD140b heterodimerize on fibroblasts/pericytes.

Cross-axis correlations reveal interaction zones:

| Pair | r | Significance |
|------|---|--------------|
| CD11b — CD140b | 0.755 | Myeloid-stromal interface |
| CD11b — CD44 | 0.753 | Myeloid activation |
| CD140b — CD44 | 0.664 | Stromal activation |
| CD45 — CD140a | 0.659 | Immune-fibroblast |
| CD45 — CD140b | 0.657 | Immune-pericyte |
| CD45 — CD44 | 0.653 | Immune activation |

The CD11b-CD140b correlation (r=0.755) is notable — it suggests superpixels where myeloid and stromal markers co-exist. INDRA-grounded literature includes a PDGFRB-CD44 physical complex (10 evidence, belief=0.74).

---

## 3. Temporal Dynamics

### Marker Trajectories (D1 to D7)

| Marker | D1 | D7 | Change | Interpretation |
|--------|----|----|--------|----------------|
| CD45 | 1.576 | 1.945 | +23.4% | Immune infiltration intensifying |
| CD44 | 2.397 | 2.830 | +18.1% | Pan-tissue activation expanding |
| CD206 | 2.132 | 2.409 | +13.0% | M2 macrophage accumulation |
| CD11b | 2.334 | 2.571 | +10.2% | Myeloid compartment expanding |
| Ly6G | 1.809 | 1.783 | -1.4% | Neutrophil mean expression flat |

The immune compartment shifts from acute (neutrophil-dominant) toward resolution (macrophage/activation-dominant) character. The flat Ly6G trajectory is deceptive — mean expression is stable but spatial organization changes (see Neutrophil Foci below). CD44 rising 18% across all tissue suggests pan-compartmental activation, not a cell-type-specific response.

### Differential Abundance

Unit of analysis: mouse (ROI proportions averaged within each mouse to prevent pseudoreplication). All q-values >= 0.86. Raw p-values are 0.333, 0.667, or 1.000 — the only values Mann-Whitney U can produce at n=2. Bootstrap ranges are inherently degenerate: with 2 observations per group, only 4 unique bootstrap samples exist, yielding 9 unique Hedges' g values per comparison. Ranges are bounds on observed values, not coverage-bearing CIs.

**Top effects by |Hedges' g|:**

| Cell Type | Comparison | Hedges' g | Bootstrap range | Fold Change |
|-----------|-----------|-----------|--------|-------------|
| Activated Fibroblast | Sham vs D1 | **-8.74** | [-15.8, -8.7] | 1.54x increase |
| Activated Fibroblast | Sham vs D3 | **-4.21** | [-15.6, -4.2] | 1.50x increase |
| Activated Endo CD140b+ | Sham vs D1 | **3.81** | [3.8, 172.8] | 0.70x decrease |
| Neutrophil | Sham vs D7 | **-3.70** | [-11.2, -3.7] | 1.36x increase |
| M2 Macrophage | Sham vs D1 | **-2.83** | [-4.8, -2.8] | 2.63x increase |
| M2 Macrophage | D1 vs D3 | **2.10** | [2.1, 5.3] | 0.36x decrease |
| Neutrophil | D3 vs D7 | **-1.99** | [-18.2, -1.9] | 1.63x increase |
| Activated Fibroblast | Sham vs D7 | **-1.65** | [-16.4, -1.6] | 1.44x increase |
| Neutrophil | Sham vs D1 | **1.32** | [1.2, 6.4] | 0.87x decrease |
| Activ. Immune CD140b+ | Sham vs D3 | **-1.22** | [-62.6, -1.2] | 1.85x increase |

**Activated fibroblasts show a large observed effect immediately post-injury** (g=-8.74, the largest |g| in the discrete cell-type DA, but flagged as variance-collapse: pooled_std<0.01, see g_pathological column). This is a CD140b+/CD44+ population — pericytes or mesenchymal cells co-expressing the injury marker. The effect size is enormous but the CI is wide [-15.8, -8.7], reflecting n=2 uncertainty. **M2 macrophages rise sharply at D1** (g=-2.83, 2.63x) then decline by D3 (g=2.10), suggesting rapid myeloid remodeling. **Neutrophils accumulate progressively** through D7 (g=-3.70 Sham vs D7) — an unexpected pattern if confirmed, as classical AKI models show neutrophil resolution by D3-D7. **Endothelial CD140b+ cells decline acutely** (g=3.81, 0.70x), suggesting pericyte-endothelial coupling disruption at D1.

No finding has both a large effect size AND high INDRA support. The largest effects (activated fibroblast, g=-8.74) lack specific INDRA annotations because the fibroblast phenotype is defined by CD140b+CD44+ without a single gene anchor. The best-grounded gene (CD44, direct AKI annotation, 17 evidence statements) shows only moderate effect sizes.

### Regional Comparisons (Cortex vs Medulla)

No regional comparison reaches FDR significance (all q >= 0.600). Cortex/medulla from the same mouse are paired observations, but Mann-Whitney treats them as independent — a caveat for interpretation.

| Cell Type | Timepoint | Region Enriched | Hedges' g | Fold Change |
|-----------|-----------|-----------------|-----------|-------------|
| Activated Fibroblast | D7 | Medulla | -4.71 | 2.38x |
| M2 Macrophage | D1 | Medulla | -1.58 | 2.42x |
| Activated Endo CD44+ | D1 | Medulla | -1.16 | 1.45x |
| Neutrophil | D3 | Medulla | -1.05 | 3.98x |

Medullary enrichment of fibroblasts at D7 (2.38x) and neutrophils at D3 (3.98x) is consistent with the medulla's susceptibility to ischemic injury — lower oxygen tension and higher tubular density make it more vulnerable to damage and more active in the remodeling response.

---

## 4. Spatial Architecture

### Clustering Quality

Leiden community detection at 10 um scale across 18 injury ROIs:

| Metric | Mean +/- SD | Range | Interpretation |
|--------|-------------|-------|----------------|
| Silhouette score | -0.142 +/- 0.058 | -0.249 to -0.047 | Substantial cluster overlap |
| Moran's I | 0.041 +/- 0.015 | 0.022 to 0.096 | Slightly better than random spatial contiguity |
| Clusters per ROI | 14.6 +/- 7.8 | 7 to 30 | High variability |

Negative silhouette scores mean clusters overlap substantially in feature space — no well-separated communities exist. Bootstrap stability analysis confirms this: near-zero ARI at all scales and resolutions. Cluster assignments should be treated as descriptive, not definitive. Cell type annotation relies on boolean gating (independent of clustering), not cluster membership.

Some clusters do map to interpretable biology. In a representative D7 ROI (M2_01_24, 2,490 superpixels, 16 Leiden clusters):

| Cluster | Dominant Phenotype | Enrichment | Purity |
|---------|-------------------|------------|--------|
| 15 | M2 Macrophage | 6.29x | 58.8% |
| 13 | Neutrophil | 4.45x | 68.4% |
| 9 | Endothelial | 4.33x | 40.1% |
| 8 | Immune (CD45+) | 6.18x | 6.5% |

Cluster 15 is 58.8% M2 macrophage at 6.29x enrichment — a spatially coherent macrophage niche. Cluster 13 is 68.4% neutrophil — a focal immune aggregate. Others are mixed. The clustering captures real spatial structure but does not resolve it cleanly.

### Scale-Dependent Complexity

| Scale | Clusters per ROI | Range |
|-------|-----------------|-------|
| 10 um | 13.7 +/- 7.2 | 7–28 |
| 20 um | 9.8 +/- 4.2 | 3–15 |
| 40 um | 4.1 +/- 1.6 | 2–7 |

**3.4x complexity reduction** from 10 um to 40 um. This is not just mathematical aggregation — it reflects genuine hierarchical organization. The spatial maps (now rendered with DNA tissue morphology underlays) show the difference directly: 10 um captures individual capillary-scale microenvironments (immune foci, vascular networks, stromal patches); 40 um reveals regional cortex-versus-medulla patterns (injured versus spared zones). The implication is that local cell-level decisions aggregate into regional tissue-level outcomes.

### The Neutrophil Paradox

Mean Ly6G expression is flat across timepoints (-1.4% D1 to D7), but neutrophils form consistent spatial foci:

| Timepoint | Foci (top 10%) | Mean Ly6G in Foci | Mean Ly6G Outside | Enrichment |
|-----------|---------------|-------------------|-------------------|------------|
| D1 | 1,446 spx (10%) | 2.894 | 1.687 | 1.60x |
| D3 | 1,441 spx (10%) | 2.726 | 1.641 | 1.56x |
| D7 | 1,470 spx (10%) | 2.918 | 1.655 | 1.64x |

Foci enrichment is remarkably consistent (1.56-1.64x) despite injury progression. DNA-underlaid spatial maps show these as discrete clusters against tissue morphology, not diffuse gradients. The paradox: neutrophil mean expression suggests nothing is changing, but spatial organization tells a different story — neutrophils are concentrating into focal aggregates whose internal intensity and spatial extent remain stable across the injury time course.

### Neighborhood Self-Enrichment

All cell types cluster with themselves at 2-4x enrichment (permutation-tested, 1,000 shuffles per ROI, Phipson & Smyth corrected):

| Cell Type | Sham | D1 | D3 | D7 | Temporal Trend |
|-----------|------|----|----|-----|----------------|
| M2 Macrophage | 4.25x | 2.90x | 2.95x | 2.88x | High baseline, stable |
| Activated Endo CD140b+ | 2.98x | 2.59x | 4.01x | 2.99x | D3 peak |
| Resting Endothelial | 2.29x | 2.72x | 3.27x | **3.31x** | **Rising** |
| Activated Immune CD140b+ | 3.20x | 3.00x | 3.14x | 3.05x | Stable |
| Neutrophil | 3.02x | 3.02x | 3.36x | 2.62x | D3 peak, D7 decline |
| Activated Fibroblast | 2.46x | 1.99x | 1.88x | 2.35x | U-shaped |
| Activated Immune CD44+ | 1.98x | 2.41x | 1.98x | 2.07x | D1 peak |
| Activated Endo CD44+ | 2.01x | 2.24x | 1.95x | **2.62x** | **Rising** |

The most informative temporal signal is resting endothelial self-enrichment rising from 2.29x (Sham) to 3.31x (D7). The vascular network becomes more spatially distinct during injury — endothelial cells that remain resting consolidate into tighter spatial domains as the surrounding tissue remodels. Activated endothelial CD44+ shows the same trend (2.01x to 2.62x), suggesting that both resting and activated endothelial populations are spatially tightening.

Self-clustering is partially confounded by marker sharing — cell types defined by overlapping markers (e.g., activated immune CD44+ and activated endothelial CD44+ both require CD44+) will trivially co-localize. The spatial_weight=0 ablation confirmed that self-clustering scores are identical with and without spatial coordinate weighting (Pearson r=1.000), meaning the signal comes from marker co-expression and boolean gating, not from the clustering algorithm's spatial component.

### Cross-Type Spatial Associations

Cross-type patterns are more informative than self-enrichment. Several pairs show consistent avoidance:

| Focal | Neighbor | D1 | log2 | INDRA Prediction |
|-------|----------|-----|------|------------------|
| Endo CD44+ | Macrophage | 0.43x | -1.52 | Co-localize (extravasation) |
| Neutrophil | Resting Endothelial | 0.50x | -1.21 | Co-localize (adhesion cascade) |
| M2 Macrophage | Activated Fibroblast | 0.52x | -0.97 | Co-localize (repair niche) |
| Immune CD44+ | Resting Endothelial | 0.56x | -1.02 | Co-localize (CD44-PECAM1 adhesion) |
| Fibroblast | Macrophage | 0.60x | -0.79 | Co-localize (fibrosis niche) |

These avoidance patterns persist across timepoints. INDRA predicts proximity for several of these pairs based on published molecular biology: PDGFRB-MRC1 physical complex should bring fibroblasts and macrophages together in the repair/fibrosis niche; PECAM1-CD44 sequential adhesion should place immune cells near endothelium. The concordance table:

| Cell Type Pair | Spatial | INDRA Prediction | Agreement |
|---------------|---------|------------------|-----------|
| Endothelial — Endothelial | 2.59x | Co-localize (PECAM1 homophilic adhesion) | YES |
| Fibroblast — Fibroblast | 1.99x | Co-localize (interstitial compartment) | YES |
| Macrophage — Macrophage | 2.90x | Co-localize (immune niche formation) | YES |
| Endo — Fibroblast | 0.53x | Co-localize (vascular-interstitial interface) | **NO** |
| Endo — Macrophage | 0.43x | Co-localize (leukocyte extravasation) | **NO** |
| Fibro — Macrophage | 0.60x | Co-localize (repair/fibrosis niche) | **NO** |
| Immune — Endothelial | 0.56x | Co-localize (CD44-PECAM1 adhesion) | **NO** |

**Self-clustering predictions match. Every cross-type proximity prediction fails.** This is the most provocative finding in the dataset. Either the expected cellular niches do not form at superpixel resolution in this AKI model, or per-ROI gating normalization obscures the signal, or the molecular interactions documented in INDRA operate at subcellular/membrane-contact scales invisible to 10 um superpixels. Distinguishing these explanations requires higher-resolution analysis (single-cell segmentation) or an expanded marker panel that reduces the 77% unassigned fraction.

---

## 5. Multi-Lineage Coordination at D7

D7 candidate trajectories diverge along the repair-vs-fibrosis axis. CD44 activation rates across biological compartments (all D7 ROIs):

| Compartment | CD44+ Rate | SD |
|-------------|-----------|-----|
| Immune (CD45+) | 49.4% | 35.0% |
| Stromal (CD140b+) | 46.7% | 23.4% |
| Vascular (CD31+) | 28.4% | 23.8% |
| Background (none+) | 12.8% | 9.0% |

Nearly half of immune and stromal superpixels co-express CD44 at D7, versus only 13% of background tissue. This is not a compartment-specific response — immune, vascular, and stromal cells all show elevated CD44 activation, suggesting coordinated tissue-wide remodeling.

Multi-compartment overlap quantifies the extent of co-expression:

| Category | Tissue Fraction | SD |
|----------|----------------|-----|
| No compartment | 52.4% | 15.0% |
| Single compartment | 25.1% | 4.5% |
| Dual compartment | 17.1% | 11.7% |
| Triple compartment | 5.4% | 4.9% |

5.4% of tissue is triple-positive (immune + vascular + stromal markers simultaneously). The high standard deviations (35%, 23.8%) reflect substantial inter-ROI variability — some D7 ROIs show intense pan-tissue activation while others are relatively quiescent, consistent with the patchy nature of kidney injury.

---

## 6. INDRA Knowledge Graph Context

### Panel Grounding and Relationships

All 9 markers grounded (9/9). 117 unique intra-panel INDRA statements (175 raw edges aggregated by source/type/target).

| Marker | Gene | INDRA Relationships |
|--------|------|---------------------|
| CD45 | PTPRC | 39 |
| CD34 | CD34 | 36 |
| CD11b | ITGAM | 35 |
| CD31 | PECAM1 | 27 |
| CD44 | CD44 | 27 |
| CD140b | PDGFRB | 22 |
| CD206 | MRC1 | 20 |
| CD140a | PDGFRA | 19 |
| Ly6G | Ly6g | 9 |

**Strongest intra-panel relationships:**

| Gene Pair | Type | Evidence | Belief | Biological Significance |
|-----------|------|----------|--------|------------------------|
| PDGFRA — PDGFRB | Complex | 183 | 0.89 | Heterodimer — core PDGF receptor signaling in fibroblasts/pericytes |
| CD44 — PTPRC | Complex | 17 | 0.77 | Immune activation axis |
| CD44 — PDGFRB | Complex | 10 | 0.74 | Immune-stromal interaction — molecular basis for myeloid-pericyte crosstalk |
| MRC1 — PTPRC | Complex | 20 | 0.61 | M2 macrophage surface identity |
| ITGAM — Ly6G | Complex | 15 | 0.44 | Canonical neutrophil marker combination |
| ITGAM — PECAM1 | Complex | 3 | 0.45 | Transendothelial migration machinery |

The PDGFRA-PDGFRB complex (183 evidence, belief=0.89) is by far the strongest relationship — the two PDGF receptors heterodimerize in fibroblasts and pericytes, and this physical complex is what makes the stromal correlation axis (r=0.814) biologically expected. The CD44-PDGFRB complex (10 evidence, belief=0.74) provides the molecular basis for the myeloid-stromal interface seen in the correlation data (CD11b-CD140b r=0.755).

### Upstream Regulators

- **TGF-beta**: 5/8 panel genes (PDGFRA, PDGFRB, CD44, MRC1, PECAM1) — the master regulator of renal fibrosis touches more than half the panel
- **VEGF**: 4/8 panel genes (PDGFRA, PDGFRB, PECAM1, CD34) — angiogenesis
- **TNF**: 3-4 genes (ITGAM, PTPRC, PECAM1, CD44) — canonical AKI driver
- **IL-6**: 3 genes (PTPRC, ITGAM, CD44) — pro-inflammatory cytokine

The panel is positioned to detect downstream consequences of these cytokine axes even though the cytokines themselves are not measured.

### Mechanistic Narratives Cross-Referenced with Spatial Data

The publication notebook generates 8 mechanistic narratives from INDRA, each cross-referenced with observed spatial enrichment:

- **Endothelial self-clustering** (2.59x at D1): PECAM1 mediates homophilic adhesion (GO:0007156). Both CD31 and CD34 share glomerular endothelium development (GO:0072011). Spatial data is consistent with expected vascular network topology.

- **Fibroblast self-clustering** (1.99x at D1): PDGFRA is expressed in kidney cortex and nephron tubule interstitium. In nephrogenesis (WP4823), fibroblasts form the structural scaffold. Spatial data is consistent with interstitial compartment organization.

- **Macrophage self-clustering** (2.90x at D1): ITGAM+ cells accumulate at injury sites via complement-mediated adhesion. MRC1+ M2 macrophages cluster in resolution zones. Spatial data is consistent with immune niche formation.

- **Endothelial-fibroblast avoidance** (0.53x at D1): INDRA predicts proximity via PDGFRB pericyte-endothelial contact. Observed avoidance contradicts this. The vascular-interstitial interface, reported in literature as the origin of capillary rarefaction and fibrosis in AKI, is not spatially evident at superpixel resolution.

- **Fibroblast-macrophage avoidance** (0.60x at D1): PDGFRB forms a physical complex with MRC1 (3 evidence). M2 macrophages are reported to secrete TGF-beta that activates fibroblasts via PDGFR signaling. The repair/fibrosis niche predicted by this interaction is spatially absent.

### Finding Annotations

79 findings annotated: 46 involve genes with known AKI literature links. 31 have medium-tier INDRA context, 16 low-tier. CD44 is the only panel gene with direct AKI disease annotation (MESH:D058186). INDRA provides Discussion-level interpretation, not Results-level evidence — a distinction maintained throughout the analysis.

---

## 7. Discretization Cost

The gradient discretization notebook quantifies what boolean gating sacrifices (58,137 superpixels, 9 markers, adaptive percentile-based binning for zero-inflated distributions):

| Marker | H_continuous (bits) | H_discrete (bits) | Information Loss |
|--------|--------------------|--------------------|-----------------|
| CD45 | 5.64 | 0.97 | 82.8% |
| CD11b | 5.64 | 0.97 | 82.8% |
| Ly6G | 5.64 | 0.88 | 84.4% |
| CD140a | 5.64 | 0.97 | 82.8% |
| CD140b | 5.64 | 0.97 | 82.8% |
| CD31 | 5.64 | 0.97 | 82.8% |
| CD34 | 5.64 | 0.97 | 82.8% |
| CD206 | 5.64 | 1.00 | 82.3% |
| CD44 | 5.64 | 0.97 | 82.8% |

**Mean information loss: 82.9%.** UMAP embeddings show smooth, continuous protein gradients in expression space; boolean gates carve artificial boundaries through them. IMC markers are zero-inflated (68% of tissue is background CD11b, 57% background CD140a), and gating collapses the heterogeneity within positive populations into a single binary state. The trade-off: lose 83% of gradient information, gain interpretable cell type categories that integrate domain knowledge about marker co-expression. Ly6G at the stricter 70th percentile gate loses the most (84.4%); CD206 at the permissive 50th percentile loses the least (82.3%).

---

## 8. Benchmark Validation

The Steinbock concordance notebook validates the processing pipeline against a reference implementation using the Bodenmiller Patient1 dataset (3 ROIs, 10,711 DeepCell-segmented cells, 47 channels):

| Metric | Value |
|--------|-------|
| Spearman rho (channel means) | **0.9962** |
| Pearson r (normalized means) | **1.0000** |
| Max rank discordance | 4 positions (noise-floor channels only) |
| Per-ROI consistency | All 3 ROIs independently rho > 0.996 |

This validates data I/O integrity — no channel transpositions, scaling artifacts, or silent corruption. KS statistics scale with marker sparsity as expected: ubiquitous markers show small distributional divergence, sparse markers show the largest. The benchmark does not validate methodological equivalence between SLIC superpixels and DeepCell cells; these answer different biological questions (tissue composition versus individual cell phenotyping).

---

## 9. Power Analysis for Follow-Up

Based on observed effect sizes, required sample sizes for 80% power (alpha=0.05, two-sided):

| Target Effect (|g|) | Required n per group | Detectable Signal |
|---------------------|---------------------|-------------------|
| 8.0+ | 2-3 | Variance-collapse-flagged g (g_pathological=true; treat as artifact, not biology) |
| 3.5-4.0 | 3-4 | Endothelial CD140b, neutrophil |
| 2.0-3.0 | 5 | M2 macrophage dynamics |
| 1.0-1.5 | 10 | Moderate temporal changes |
| 0.5-1.0 | 17 | Small effects |

**Recommendation for follow-up**: n >= 10 per group (captures moderate effects), expand panel to 20+ markers (reduce 77% unassigned below 30%), longitudinal design (same animals across timepoints, enabling paired analyses), global-threshold sensitivity analysis (compare per-ROI versus pooled-global gating to quantify signal compression), Wilcoxon signed-rank for paired regional comparisons.

---

## How the Notebooks Fit Together

Four notebooks present the analysis in a pedagogical arc:

**Gradient Discretization** (`methods_validation/01_technical_methods/`) asks: *what do we lose by categorizing?* Answer: 83% of Shannon entropy. This is the acknowledged price of boolean gating — established upfront, not hidden.

**Steinbock Concordance** (`methods_validation/benchmarks/`) asks: *can we trust the raw numbers?* Answer: yes. Spearman rho=0.9962 against an independent reference. The data pipeline reads, transforms, and aggregates ion counts correctly.

**Kidney Injury Spatial Analysis** (`biological_narratives/`) asks: *what does the tissue look like?* It reveals marker correlation axes matching known biology, spatial clusters with imperfect but interpretable phenotype enrichments, the neutrophil paradox (stable mean but focal spatial organization), multi-lineage CD44 activation at D7, and hierarchical tissue organization across scales. All spatial ROI plots show DNA tissue morphology as a gray underlay beneath the data, providing anatomical context.

**Publication Narrative** (`biological_narratives/`) asks: *what can we claim?* Answer: nothing with statistical confidence. But effect sizes — large observed activated-fibroblast g (g=-8.74, variance-collapse-flagged), M2 macrophage dynamics (g=2.10), sustained neutrophil accumulation (g=-3.70) — are large enough to be real, large enough to detect with modest sample increases, and grounded in 117 INDRA-curated gene relationships. The spatial data adds a second dimension: cell types self-cluster, the vascular network tightens during injury, and the predicted fibroblast-macrophage repair niche is spatially absent at the scales measured. The honest conclusion is a set of testable hypotheses, power calculations to test them, and a validated analytical framework ready for the adequately powered follow-up.

---

## Glossary

**ARI (Adjusted Rand Index)** — Agreement between two clusterings, adjusted for chance. 1.0 = perfect, 0.0 = chance-level.

**Arcsinh transformation** — arcsinh(x) = ln(x + sqrt(x^2 + 1)). Variance-stabilizing for count data; behaves like log for large values but is defined at zero.

**BH FDR** — Benjamini-Hochberg False Discovery Rate. Controls expected proportion of false positives among rejected hypotheses.

**Boolean gating** — Binary classification by marker expression thresholds. Superpixel is "positive" if expression exceeds a percentile-based cutoff.

**Bootstrap range** — Bounds on observed effect size from resampling with replacement. At n=2, only 9 unique values exist per comparison; the range is a bound on observed values, not a coverage-bearing CI.

**CLR (Centered Log-Ratio)** — Compositional data transform: CLR(x_i) = log(x_i / geometric_mean(x)). Removes sum-to-one constraint.

**Cofactor** — Scaling parameter in arcsinh. Determines linear-to-logarithmic transition point. Estimated from noise-floor percentile.

**Compositional closure** — Proportions must sum to 1.0, creating spurious negative correlations unless addressed (e.g., via CLR).

**Hedges' g** — Small-sample-corrected standardized effect size. Like Cohen's d but with bias correction factor for small n.

**IMC (Imaging Mass Cytometry)** — Metal-tagged antibodies + laser ablation + mass spectrometry. Multiplexed protein imaging at ~1 um resolution.

**INDRA** — Integrated Network and Dynamical Reasoning Assembler. Extracts causal biological relationships from literature. Belief scores (0-1) reflect evidence strength.

**Leiden algorithm** — Graph community detection. Improvement over Louvain: guarantees connected communities and monotonic quality improvement.

**Moran's I** — Spatial autocorrelation statistic. +1 = clustered, 0 = random, -1 = dispersed.

**Permutation test** — Null distribution constructed by randomly rearranging labels. P-value = fraction of permutations as extreme as observed.

**Phipson & Smyth correction** — p = (n_extreme + 1) / (n_total + 1). Prevents permutation p-values from being exactly zero.

**Pseudoreplication** — Treating non-independent observations (multiple ROIs from one mouse) as independent, inflating sample size.

**ROI** — Region of Interest. ~500 x 500 um tissue area selected for IMC acquisition.

**Shannon entropy** — Information content of a distribution (bits). Higher = more uniform = more information.

**SLIC** — Simple Linear Iterative Clustering. Superpixel algorithm using local k-means on spatial + intensity features.

**Superpixel** — Aggregated pixel group (~100 um^2 at 10 um scale). Not a cell — may span parts of multiple cells or include extracellular matrix.

**Spatial weight** — Parameter (0-1) controlling feature-vs-coordinate balance in clustering. 0.2 = feature-dominant, 0.4 = spatial-dominant.

---

## File Manifest

| Output | Path |
|--------|------|
| Temporal DA | `results/biological_analysis/differential_abundance/temporal_differential_abundance.csv` |
| Regional DA | `results/biological_analysis/differential_abundance/regional_differential_abundance.csv` |
| ROI abundances | `results/biological_analysis/differential_abundance/roi_abundances.csv` |
| Temporal enrichments | `results/biological_analysis/spatial_neighborhoods/temporal_neighborhood_enrichments.csv` |
| Regional enrichments | `results/biological_analysis/spatial_neighborhoods/regional_neighborhood_enrichments.csv` |
| INDRA panel context | `results/biological_analysis/indra_panel_context.json` |
| INDRA finding annotations | `results/biological_analysis/indra_finding_annotations.csv` |
| Steinbock concordance | `results/benchmark/bodenmiller_concordance.csv` |
| Pipeline metadata | `results/run_summary.json` |

---

*Experiment: kidney_healing_2024. Config: `config.json`. Methods: `METHODS.md`. Last Phase 1 pipeline run: 2026-03-31 (24 ROIs × 3 scales). Last Phase 2 temporal-interface run: 2026-04-23 (348 endpoint rows × 37 columns in `endpoint_summary.csv`). Phase 5 deferred-item closures, Phase 1.5c factual correction, and freeze-manifest verifier (`verify_frozen_prereg.py`) added 2026-04-23 — see `review_packet/FROZEN_PREREG.md` for the pinned reproducibility anchors and `analysis_plans/temporal_interfaces_plan.md` for the full amendment log.*


---

## Temporal Interface Analysis (Pre-Registered, Phase 2)

A separate pre-registered analysis (`analysis_plans/temporal_interfaces_plan.md`, frozen 2026-04-17) computes three endpoint families on the continuous lineage scores: interface composition (CLR-transformed compositional vector), continuous neighborhood lineage shifts (neighbor-minus-self delta), and cross-compartment activation (Sham-reference threshold). Outputs at `results/biological_analysis/temporal_interfaces/` (consumed by `notebooks/biological_narratives/kidney_injury_spatial_analysis.ipynb` Parts 2 + 2.5 and `notebooks/main_narrative.ipynb` Sections 2 + 6).

**Headline observations** (from `endpoint_summary.csv`). Effect sizes reported as observed Hedges' g plus a per-endpoint Bayesian shrinkage range under three priors on the true effect δ (skeptical N(0, 0.5²), neutral N(0, 1.0²), optimistic N(0, 2.0²)). n_required values are 80%-power sample sizes per Bayesian-shrunk g.

**Sham→D7 co-headline findings** (post-hoc selection rule defined after the Family A normalization sensitivity sweep; the rule itself is **pre-registered for follow-up cohorts** in `temporal_interfaces_plan.md` §Phase 2 amendment).

**Current selection rule** (locked unchanged for any follow-up cohort): a Family A CLR endpoint is retained as a Sham→D7 co-headline iff (1) **direction-consistent** between the Sham-reference-centered sigmoid path (primary) and the raw-marker Sham-reference percentile path (corroboration), (2) `|hedges_g| > 0.5` in the primary path, (3) **symmetric magnitude agreement** — not flagged as `normalization_magnitude_disagree` (≥2× divergence in either direction). The earlier "Sham-ref magnitude ≥ 20% of per-ROI" rule was an asymmetric one-sided check; it has been replaced by the symmetric flag (the older `normalization_g_collapse` column is retained alongside for backward compatibility, but the headline filter no longer uses it). Phase 1 (Sham-reference sigmoid for continuous memberships) replaced the earlier per-ROI sigmoid as the primary normalization, removing the temporal-compression confound; the comparison reported below is between the two **Sham-anchored** paths (sigmoid vs raw-marker percentile), not per-ROI vs Sham-ref. Effect sizes from `endpoint_summary.csv` reported as observed Hedges' g + per-endpoint Bayesian shrinkage range under three priors on the true effect δ (skeptical N(0, 0.5²), neutral N(0, 1.0²) — **planning default**, optimistic N(0, 2.0²)) with Hedges & Olkin (1985) sampling variance v = 2/n + g²/(4n).

**Pilot Sham→D7 co-headline table** (verbatim from `review_packet/ONE_PAGER.md` candidate-findings table; reproducible via the pandas query in `review_packet/FROZEN_PREREG.md`):

| Endpoint | Family | g_skep | g_neut | g_opt | Corroboration |
|---|---|---:|---:|---:|---|
| endothelial+immune+stromal_clr | A | 0.32 | +0.99 | 2.11 | Sham-ref sigmoid |
| triple_overlap_fraction | C | 0.32 | +0.98 | 2.08 | Raw markers, Sham-ref pct — independent of CLR closure |
| background_compartment_cd44_rate | C | 0.31 | +0.95 | 1.91 | Raw markers — independent of CLR closure |
| CD140b_compartment_cd44_rate | C | 0.24 | +0.64 | 1.11 | Raw markers — independent of CLR closure |
| endothelial+immune_clr | A | 0.21 | +0.54 | 0.90 | Sham-ref sigmoid |
| immune+stromal_clr | A | 0.19 | +0.50 | 0.83 | Sham-ref sigmoid |
| immune_clr | A | -0.15 | -0.39 | -0.63 | Sham-ref sigmoid |

The triple-positive CLR rise (Family A) and the raw-marker triple-overlap fraction rise (Family C) are *compatible with* a redistribution-at-D7 reading. **Independence ranking**: all three reported analytical paths (Family A sigmoid, Family A raw-marker Sham-ref, Family C Sham-ref compartments) anchor on the same Sham baseline, so "independent" above means *independent of CLR closure*, not statistically independent of Sham. The symmetric magnitude-disagreement count (13/48 Family A endpoints in the current run) is the honest upper bound on how much the two Family A paths measure differently.

**Family A endpoints filtered out by the rule** (full disclosure in `review_packet/ONE_PAGER.md`): `stromal_clr` and `none_clr` (`normalization_magnitude_disagree`), `endothelial_clr` (`normalization_sign_reverse`, both near zero), `endothelial+stromal_clr` (|g| ≤ 0.5).

**Threshold sensitivity**: Family A findings largely robust to lineage threshold sweep {0.2, 0.3, 0.4} (3/48 sign-reverse) and to continuous Sham-percentile sweep {50, 60, 70} (closed Phase 1.5b: triple-interface neutral g 1.000 / 0.987 / 0.965, Sham-pct-invariant under shrinkage). Family C robust to Sham-percentile sweep {65, 75, 85}. Family B carries `support_sensitive=True` on rows that change presence across `min_support` ∈ {10, 20, 40} (closed Phase 1.5a: 90/270 rows flagged); Family A carries `clr_none_sensitivity` on rows whose sign flips when the `none` category is excluded (closed Phase 1.5a: 0/48 flips). Phase 5.5 brutalist correction: Family B sigmoid-basis Sham→D7 produces 21 endpoints clearing `|g_shrunk_neutral|>0.5` (not 0 as a prior draft mis-stated); raw-marker basis (closed Phase 1.5c) produces 18; 14 in common.

**Pathology and missingness**: At n=2 per group no real p-value exists, so no BH-FDR is computed (earlier drafts used a normal-CDF-from-|g| proxy that Gate 6 removed as cognitive-anchoring risk). Endpoint rows with `|g|>3 AND pooled_std<0.01` (variance-collapse artifacts) are flagged `g_pathological=True` with NaN shrunk values; rows with `insufficient_support=True` (n<2 mice in one group) are preserved as NaN-with-flag rather than silently dropped. The selection-free rank companion is `temporal_top_ranked_by_effect.csv` (sorted on `|g_shrunk_neutral|`, pathology-quarantined). Reviewer-checkable in `endpoint_summary.csv`.

**Bayesian shrinkage rationale**: per-endpoint Bayesian shrinkage under three explicit priors (replaced an earlier single-scalar Type-M correction). The shrinkage factor for an endpoint with observed g and Hedges & Olkin (1985) asymptotic sampling variance v = 2/n + g²/(4n) is `prior_var / (prior_var + v)`. The three priors — skeptical N(0, 0.5²), **neutral N(0, 1.0²) (planning default)**, optimistic N(0, 2.0²) — are a **pre-registered sensitivity analysis**, not a Bayesian inference. Neutral is the planning default when a single number is needed for design (bolded column in headline tables); skeptical and optimistic bracket residual prior dependence. A reviewer preferring a different prior is invited to read whichever column matches their assumption.

**Phase 5 closures (deferred-item resolution)**: Phase 5.1 closed area-based tissue-mask density empirically (`audit_tissue_mask_density.py` + `tissue_area_audit.csv`): both pre-registered gates fail (CV(tissue_area_mm2) = 0.012, Pearson |r|(density, proportion) = 0.97 on the dominant cell type), so density = count / 0.246 ≈ 9857 × proportion ± 1.4% — same algebraic tautology as the retracted SLIC-constant version, one layer deeper. Closure scope is **area-based denominators on this acquisition design**; per-nucleus density (watershed), DNA-intensity integral, and variable-extent re-acquisition cohorts remain untested alternatives. Phase 5.2 added a co-primary intersection-conservative rule for Family B with config-driven raw-marker mapping (panel-portable). Phase 5.3 closed Bodenmiller as closed-by-design (not "tabled"). Phase 5.6 added `verify_frozen_prereg.py` (recomputes pinned SHAs, fails on drift).

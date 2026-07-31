# Discrete vs Composite Multi-Lineage Comparison

**Discrete track** (15-type ontology; 14 gates exercise all 9 panel markers, with the locked neutrophil gate as the named exception): 7,931 of 58,137 superpixels labelled (13.64%); 50,206 (86.36%) unassigned.
**Composite track** (8 composite categories from continuous lineage scores at threshold 0.3): 100% of 58,137 superpixels classified into 8 mutually-exclusive lineage combinations

The two tracks partition tissue differently. The discrete track assigns 13.64% of all superpixels to one of 15 gated cell types. The composite track classifies all superpixels: 80.78% have at least one continuous lineage active, while 19.22% fall in the `none` category. Of the 50,206 discrete-unassigned superpixels, 39,042 (77.76%) acquire a non-`none` composite category and 11,164 remain `none`.

> **Neighborhood-basis note (read first).** Unless explicitly labelled canonical mouse-of-mouse, the neighborhood-enrichment magnitudes below — both the discrete-track column of the Track-vs-track table and the composite-track self/cross values — are **ROI-level reproduction-anchor** values (discrete read from the frozen `spatial_neighborhoods/temporal_neighborhood_enrichments.csv`; composite from the frozen `spatial_neighborhoods_composite/` copy). They are retained for reproduction continuity and are **not** the canonical reporting basis. The **canonical mouse-of-mouse** discrete reading is RESULTS.md §4 (self-enrichment table) and the §4/§4b track-coherence rows (`activated_fibroblast_cd140b` self **1.33 → 2.00×**; `activated_fibroblast_cd140b` ↔ `m2_macrophage` **symmetric 1.19× / 1.15×**; `activated_myeloid_cd44` ↔ `endothelial` **0.67×**), traceable to `spatial_neighborhoods/temporal_neighborhood_enrichments_mouse.csv`. In particular, under mouse-of-mouse the `activated_fibroblast_cd140b` ↔ `m2_macrophage` pair is **already symmetric** (1.19× / 1.15×), so the ROI-level "asymmetric repair niche" and the "COMPOSITE-RESCUES-SYMMETRY" framing in the Track-vs-track table are ROI-level artifacts superseded by the mouse-of-mouse reconciliation (RESULTS.md classifies both tracks as symmetric / COHERENT).

## The composition (58,137 superpixels)

| Composite category | Count | Tissue % |
|---|---:|---:|
| **endothelial+immune+stromal (triple)** | **16,833** | **28.95%** |
| none (no lineage active) | 11,173 | 19.22% |
| endothelial only | 7,851 | 13.50% |
| endothelial+stromal | 5,778 | 9.94% |
| endothelial+immune | 5,310 | 9.13% |
| immune+stromal | 4,967 | 8.54% |
| immune only | 3,415 | 5.87% |
| stromal only | 2,810 | 4.83% |

The single largest tissue category is **endothelial+immune+stromal (triple-positive interface) at 29%** — superpixels where all three lineages score above threshold simultaneously. Of these 16,833 triple-positive superpixels, 15,489 (92.02%) are `unassigned` by the discrete ontology and 1,344 (7.98%) receive a discrete cell-type label; the continuous triple-positive state is therefore largely, but not exclusively, outside the strict discrete gates.

## DA findings — composite multi-lineage track

**Top Sham→D7 effects** (raw proportion; CLR-transformed g in parentheses):

| Composite category | Sham → D7 | Hedges' g | CLR g | Direction |
|---|---|---:|---:|---|
| **endothelial+immune+stromal** | **20.3% → 38.5%** | **−3.98** | **−3.40** | **▲ expands 1.90-fold** — triple-positive interface expands dramatically |
| stromal (pure) | 6.4% → 2.7% | +6.75 | +6.71 | ▼ contracts — pure-stromal shrinks |
| none | 24.9% → 14.4% | +3.73 | +2.39 | ▼ contracts — lower at D7 |
| immune (pure) | 8.0% → 4.8% | +1.65 | (sparse) | ▼ contracts |
| endothelial (pure) | 13.9% → 11.3% | +0.46 | — | ▼ contracts moderately |
| endothelial+immune | (rises moderately) | −0.66 | −1.17 | ▲ |
| endothelial+stromal | 9.8% → 5.9% | +0.72 | — | ▼ |
| immune+stromal | 8.7% → 11.8% | −0.67 | — | ▲ |

**The dominant cross-sectional compositional pattern across the injury course is expansion of multi-lineage interface tissue, especially the triple-positive (endothelial+immune+stromal) compartment, alongside contraction of no-lineage and pure-lineage categories.** 20.3% of tissue at Sham scores positive on all three lineages; by D7 that is 38.5%. Pure-stromal falls 58% (6.4% → 2.7%), pure-immune falls 40% (8.0% → 4.8%), pure-endothelial falls 19% (13.9% → 11.3%), and no-lineage tissue falls 42% (24.9% → 14.4%). These cohort-level compositions do not track individual superpixel state transitions.

**This pattern aligns with the §8 Phase 2/7 reviewer-facing headlines** (`endothelial+immune+stromal_clr` g_neut +0.99; `triple_overlap_fraction` g_neut +0.98), now observed at the compositional Phase 1 level rather than only in the pre-registered endpoint families.

## SN findings — composite multi-lineage cross-type

**Self-enrichment by timepoint** (within-10 µm composite-category focalization):

| Composite category | Sham | D1 | D3 | D7 | Trajectory |
|---|---:|---:|---:|---:|---|
| **stromal (pure)** | 1.66× | 1.66× | 2.83× | **3.57×** | ▲ stromal foci tighten progressively |
| **endothelial+stromal** | 1.44× | 1.48× | 1.67× | **3.26×** | ▲ E+S interface tightens at D7 |
| **immune+stromal** | 1.72× | 2.58× | 2.68× | 2.58× | ▲ I+S interface tightens by D1, plateaus |
| endothelial | 1.46× | 1.40× | 1.63× | 2.48× | ▲ |
| immune | 1.51× | 1.65× | 2.37× | 2.43× | ▲ |
| endothelial+immune | 1.36× | 1.54× | 1.78× | 2.03× | ▲ |
| none | 1.33× | 1.47× | 1.60× | 2.26× | ▲ |
| **endothelial+immune+stromal (triple)** | 1.50× | 1.34× | 1.25× | **1.35×** | ▬ flat — diffusely distributed despite compositional growth |

**Descriptive focalization discriminator.** The composite categories now carry a **DESCRIPTIVE within-10 µm focalization status** from `composite_focalization.csv`, as defined in `analysis_plans/composite_focalization_spec.md`: `focal` iff D7 self-enrichment is ≥2.0×, otherwise `diffuse`. Among the four structural `interface` categories, endothelial+stromal (3.26×) and immune+stromal (2.58×) are focal by D7 and endothelial+immune+stromal is diffuse at 1.35×. The endothelial+immune status is **not robust** — a pre-registered kNN-size sweep (`k_sensitivity_focalization.csv`) gives 2.24× (focal) at k=5 but 1.99× / 1.74× (diffuse) at k=10 / k=20 — so the earlier "lone diffuse interface" framing is withdrawn. Robust across k ∈ {5,10,20}: the rank order (E+S > I+S > E+I > triple), the focal status of E+S and I+S, and the **diffuse status of triple-positive** at every k. The largest, most-grown interface category therefore does not spatially concentrate, a descriptive pattern consistent with spatial mixing rather than a focal interface; these statuses make no inferential or biological claim.

**Interpretive limits.** Under the **dilution corollary**, cross-scale robustness is not evidence of interface biology: genuine focal co-expression dilutes at coarser grain, so scale-robust co-positivity indicates diffuse mixing, not a validated interface. At DNA-superpixel resolution, co-positivity is a co-localization tissue-patch state, not a cell phenotype. DNA-only SLIC segmentation, no spillover compensation, injury-driven cellularity, and n=2 mice per timepoint limit this descriptive reading. CD44 and CD140b are excluded from these lineage-only category definitions, so the neutrophil–CD44 headline remains non-tautological.

**Selected cross-type attractions at D1** (symmetric pairs; log₂ enrichment shown):

| Pair | log₂ enrichment (both directions) |
|---|---|
| immune+stromal ↔ stromal | +0.63 / +0.65 |
| immune+stromal ↔ immune | +0.48 / +0.49 |
| immune ↔ none | +0.42 / +0.45 |
| endothelial+immune ↔ endothelial | +0.30 / +0.30 |
| stromal ↔ none | +0.22 / +0.22 |

**Descriptive pattern: two-lineage co-positive patches preferentially sit next to their constituent single-lineage patches.** The immune+stromal category is enriched near both pure immune and pure stromal, while endothelial+immune is enriched near pure endothelial. This describes local tissue-patch topology without establishing an interface cell phenotype.

**Top cross-type avoidances at D1** (log₂ < −0.3):

| Pair | log₂ enrichment (symmetric) |
|---|---|
| endothelial ↔ immune+stromal | −0.78 / −0.82 |
| endothelial+immune ↔ stromal | −0.66 / −0.66 |
| endothelial+stromal ↔ immune | −0.57 / −0.58 |
| none ↔ endothelial+immune+stromal | −0.51 / −0.53 |

**Descriptive pattern: cross-pair categories avoid each other.** Endothelial-only patches are depleted near immune+stromal patches, stromal-only patches are depleted near endothelial+immune patches, and no-lineage patches are depleted near triple-positive patches. These co-localization tissue-patch states occupy different local zones at the 100 µm² superpixel scale; this does not establish mutually exclusive interface biology.

## Track-vs-track classification of major findings

| Finding | Discrete track result | Composite track result | Class |
|---|---|---|---|
| Stromal-niche emergence by D7 | `activated_fibroblast_cd140b` self-clustering 1.35 → 2.68× (ROI-level reproduction anchor; canonical mouse-of-mouse 1.33 → 2.00×) | **stromal** self-clustering 1.66 → 3.57× (stronger); **endothelial+stromal** also 1.44 → 3.26× | **COHERENT** — the composite track descriptively parallels the discrete pattern with larger ratios |
| Endothelial deactivation + expansion | bare `endothelial` rises 1.31% → 3.92%; `activated_endothelial_cd44` declines 0.36% → 0.24% | pure endothelial contracts 13.9% → 11.3%; endothelial+stromal contracts 9.8% → 5.9%; triple-positive expands 20.3% → 38.5% | **COMPLEMENTARY** — the discrete bare-endothelial compartment increases while composite endothelial-bearing tissue shifts from pure/two-lineage states toward triple-positive |
| Repair niche (PDGFRβ-MRC1) | `activated_fibroblast_cd140b` ↔ `m2_macrophage` **symmetric under mouse-of-mouse** (1.19× / 1.15× at D1; the ROI-level 1.30× / 0.94× asymmetry is a superseded anchor value) | immune+stromal ↔ stromal is 1.62× / 1.59×; immune+stromal ↔ immune is 1.42× / 1.41× | **COHERENT (both symmetric)** — both tracks descriptively show bidirectional adjacency at their respective tissue-labeling resolutions; the earlier COMPOSITE-RESCUES-SYMMETRY reading was a ROI-level pseudoreplication artifact |
| Endothelial–myeloid low-proximity interface | `activated_myeloid_cd44` ↔ `endothelial` 0.63× / 0.75× at D1 (ROI-level reproduction anchor; canonical mouse-of-mouse 0.67× / 0.67×) | pure immune ↔ pure endothelial also show below-chance proximity at D1 (0.80× / 0.79×; log₂ −0.34 / −0.35), but cannot isolate the activated-myeloid subtype | **COHERENT AT COARSER GRAIN** — broad immune/endothelial avoidance is present in both; subtype attribution is discrete-only |
| Neutrophil compartment CD44 activation | CD44⁺ rate within neutrophil-typed tissue rises 31.8% → 81.1% Sham→D7 (`g_neut` +1.00; Family C v2 headline) | **N/A** — composite track does not compute activation rates | **DISCRETE-ONLY** — the locked neutrophil gate leaves CD44 free, making this the only non-tautological discrete-cell-type CD44-rate endpoint |
| Pan-tissue triple-positive interface growth (NEW) | not represented as a discrete category; 15,489/16,833 triple-positive superpixels are unassigned and 1,344 are absorbed into single cell-type labels | **endothelial+immune+stromal** rises 20% → 38% Sham → D7 (g = −3.98) | **COMPOSITE-ONLY AS AN INTERFACE STATE** — this is the biggest single compositional shift in the dataset |
| Multi-lineage cross-interface avoidance (NEW) | invisible | endothelial ↔ immune+stromal at 0.58-0.60× symmetric avoidance; multiple cross-interface pairs at 0.58-0.71× | **COMPOSITE-ONLY** — co-positive tissue-patch states occupy different local zones at superpixel scale |

## Conclusions

1. **The composite track recovers the largest single compositional finding in the dataset**: triple-positive (E+I+S) interface tissue rises 1.90-fold Sham → D7 (g = −3.98 raw, −3.40 CLR). The triple-positive tissue-patch state is not represented in the discrete-cell-type ontology: 92.02% of these superpixels are unassigned and the remaining 7.98% are absorbed into single cell-type labels.

2. **The stromal-tightening pattern appears in both tracks and is stronger in the composite track.** Discrete `activated_fibroblast_cd140b` self-enrichment is 1.35 → 2.68× at the ROI-level reproduction anchor and 1.33 → 2.00× under canonical mouse-of-mouse; composite `stromal` is 1.66 → 3.57× and `endothelial+stromal` is 1.44 → 3.26×. The composite tissue-patch description covers more of the tissue.

3. **The repair-niche adjacency is symmetric at both levels (mouse-of-mouse).** The earlier ROI-level reading showed the discrete `activated_fibroblast_cd140b ↔ m2_macrophage` pair as asymmetric (1.30× / 0.94×), "rescued" by the composite symmetric pair — but under mouse-of-mouse the discrete pair is **already symmetric** (1.19× / 1.15× at D1), so there is no asymmetry to rescue. The composite `immune+stromal ↔ stromal` pair (1.62× / 1.59×) and the discrete pair agree: immune-stromal interface tissue sits next to pure stromal tissue, bidirectionally.

4. **The transmigration-interface negative finding is directionally coherent at composite resolution but subtype-specific only in the discrete track.** Pure immune and pure endothelial tissue also show below-chance proximity at D1, while the composite categories cannot distinguish CD44-activated myeloid tissue from other immune states.

5. **Two new descriptive cross-interface avoidance patterns emerge from the composite track**: cross-pair categories avoid each other at superpixel scale (endothelial ↔ immune+stromal at 0.58–0.60×, endothelial+immune ↔ stromal at 0.64×, etc.). These tissue-patch states occupy different local zones at 100 µm² resolution; this is not evidence of mutually exclusive interface biology.

## What this means for the PI presentation

The discrete track is a **sharp story about gated discrete cell types** (13.64% assigned under 14 full-panel gates plus the locked neutrophil exception). The composite track is a **broader description of lineage co-localization tissue-patch states** (100% coverage; 80.78% with at least one active lineage and 19.22% `none`). They describe different tissue-labeling surfaces.

The strongest single descriptive message is **a cross-sectional shift from no-lineage and pure-single-lineage composition toward multi-lineage interface composition across the injury course**, with the triple-positive interface compartment rising 1.90-fold by D7. This is directly observed in the composite track and echoed by the pre-registered Family A v1 `endothelial+immune+stromal_clr` and Family C `triple_overlap_fraction` headlines.

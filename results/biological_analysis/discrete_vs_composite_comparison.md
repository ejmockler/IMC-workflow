# Discrete vs Composite Multi-Lineage Comparison

**Discrete track** (17-type ontology, every gate at full 9/9 panel coverage): 4,780 of 58,137 superpixels labelled (8.2%); 91.8% unassigned
**Composite track** (8 interface categories from continuous lineage scores at threshold 0.3): 100% of 58,137 superpixels classified into 8 mutually-exclusive interface combinations

The two tracks see different parts of the tissue. The discrete track sees the 8% that satisfies strict 9/9 gate criteria. The composite track sees the 81% of discrete-unassigned tissue **recovered as multi-lineage interface tissue** plus the 19% truly off-map (`none`).

## The composition (58,137 superpixels)

| Interface category | Count | Tissue % |
|---|---:|---:|
| **endothelial+immune+stromal (triple)** | **16,833** | **28.95%** |
| none (no lineage active) | 11,173 | 19.22% |
| endothelial only | 7,851 | 13.50% |
| endothelial+stromal | 5,778 | 9.94% |
| endothelial+immune | 5,310 | 9.13% |
| immune+stromal | 4,967 | 8.54% |
| immune only | 3,415 | 5.87% |
| stromal only | 2,810 | 4.83% |

The single largest tissue category is **endothelial+immune+stromal (triple-positive interface) at 29%** — superpixels where all three lineages score above threshold simultaneously. This is the interface tissue that the discrete ontology forces into `unassigned` because no single-cell-type gate accepts a CD45⁺ CD31⁺ CD140a⁺ co-expressing region.

## DA findings — composite multi-lineage track

**Top Sham→D7 effects** (raw proportion; CLR-transformed g in parentheses):

| Interface category | Sham → D7 | Hedges' g | CLR g | Direction |
|---|---|---:|---:|---|
| **endothelial+immune+stromal** | **20.3% → 38.5%** | **−3.98** | **−3.40** | **▲ doubles** — triple-positive interface expands dramatically |
| stromal (pure) | 6.4% → 2.7% | +6.75 | +6.71 | ▼ contracts — pure-stromal shrinks |
| none | 24.9% → 14.4% | +3.73 | +2.39 | ▼ contracts — no-lineage tissue converts to interface |
| immune (pure) | 8.0% → 4.8% | +1.65 | (sparse) | ▼ contracts |
| endothelial (pure) | (rises moderately) | +0.95 | — | ▲ |
| endothelial+immune | (rises moderately) | −0.66 | −1.17 | ▲ |
| endothelial+stromal | 9.8% → 5.9% | +0.72 | — | ▼ |
| immune+stromal | 8.7% → 11.8% | −0.67 | — | ▲ |

**The dominant compositional shift across the injury course is the conversion of "no-lineage" and "pure single-lineage" tissue into multi-lineage interface tissue, especially the triple-positive (endothelial+immune+stromal) compartment.** 20% of tissue at Sham scores positive on all three lineages; by D7 that's 38%. Pure-stromal halves (6.4% → 2.7%). Pure-immune halves (8% → 5%). No-lineage tissue drops 25% → 14%.

**This is the same finding as the §8 Phase 2/7 reviewer-facing headlines** (`endothelial+immune+stromal_clr` g_neut +0.99; `triple_overlap_fraction` g_neut +0.98), now observed at the compositional Phase 1 level rather than only in the pre-registered CLR family.

## SN findings — composite multi-lineage cross-type

**Self-enrichment by timepoint** (interface tissue spatial focalization):

| Interface | Sham | D1 | D3 | D7 | Trajectory |
|---|---:|---:|---:|---:|---|
| **stromal (pure)** | 1.66× | 1.66× | 2.83× | **3.57×** | ▲ stromal foci tighten progressively |
| **endothelial+stromal** | 1.44× | 1.48× | 1.67× | **3.26×** | ▲ E+S interface tightens at D7 |
| **immune+stromal** | 1.72× | 2.58× | 2.68× | 2.58× | ▲ I+S interface tightens by D1, plateaus |
| endothelial | 1.46× | 1.40× | 1.63× | 2.48× | ▲ |
| immune | 1.51× | 1.65× | 2.37× | 2.43× | ▲ |
| endothelial+immune | 1.36× | 1.54× | 1.78× | 2.03× | ▲ |
| none | 1.33× | 1.47× | 1.60× | 2.26× | ▲ |
| **endothelial+immune+stromal (triple)** | 1.50× | 1.34× | 1.25× | **1.35×** | ▬ flat — diffusely distributed despite compositional growth |

**The triple-positive compartment, even though it more than doubles compositionally, does not focalize spatially.** Its self-enrichment hovers around 1.35× across all timepoints — close to chance. Meanwhile, every other interface category (including pure-stromal which is compositionally contracting) shows D7 self-enrichment > 2×.

**Reading: by D7, the tissue contains MORE triple-positive interface area (38%) but it's diffusely spread, not focal.** The focal organizing principle is two-lineage interfaces (E+S, I+S) and single lineages — these concentrate into discrete foci.

**Top cross-type attractions at D1** (symmetric pairs only, log₂ > 0.3):

| Pair | log₂ enrichment (both directions) |
|---|---|
| immune+stromal ↔ stromal | +0.63 / +0.65 |
| immune+stromal ↔ immune | +0.48 / +0.49 |
| immune ↔ none | +0.42 / +0.45 |
| endothelial+immune ↔ endothelial | +0.30 / +0.30 |
| stromal ↔ none | +0.22 / +0.22 |

**Pattern: two-lineage interfaces preferentially sit next to their constituent single lineages.** The immune+stromal interface is enriched near both pure immune AND pure stromal. The endothelial+immune interface is enriched near pure endothelial. This is the expected biological topology of tissue interfaces: the interface zone touches both compartments it bridges.

**Top cross-type avoidances at D1** (log₂ < −0.3):

| Pair | log₂ enrichment (symmetric) |
|---|---|
| endothelial ↔ immune+stromal | −0.78 / −0.82 |
| endothelial+immune ↔ stromal | −0.66 / −0.66 |
| endothelial+stromal ↔ immune | −0.57 / −0.58 |
| none ↔ endothelial+immune+stromal | −0.51 / −0.53 |

**Pattern: cross-pair interfaces avoid each other.** An endothelial-only region avoids an immune+stromal interface; a stromal-only region avoids an endothelial+immune interface; pure no-lineage tissue avoids the triple-positive interface. The biological reading is that these are mutually exclusive tissue zones — interfaces of one type displace interfaces of another type at the local 100 µm² scale.

## Track-vs-track classification of major findings

| Finding | Discrete track result | Composite track result | Class |
|---|---|---|---|
| Stromal-niche emergence by D7 | activated_fibroblast_cd140b self-clustering 1.24 → 2.54× | **stromal** self-clustering 1.66 → 3.57× (stronger); **endothelial+stromal** also 1.44 → 3.26× | **COHERENT** (composite track corroborates and amplifies the discrete finding) |
| Endothelial deactivation + expansion | bare endothelial 1.31% → 3.92%; activated_endothelial_cd44 halves | pure-endothelial composition flat-to-slightly-rising; endothelial+stromal and triple-positive both expand into endothelial-rich regions | **COHERENT** (different framing: discrete sees "endothelial deactivating"; composite sees "endothelial tissue becoming more multi-lineage") |
| Repair niche (PDGFRβ-MRC1) | activated_fibroblast_cd140b → m2_macrophage 1.43×; reciprocal 0.93× (asymmetric) | immune+stromal ↔ stromal symmetric at 1.62-1.63×; immune+stromal ↔ immune at 1.41-1.49× | **COMPOSITE-RESCUES-SYMMETRY** — the bidirectional repair-niche-adjacent biology IS present at the multi-lineage level, with cleaner symmetry |
| Endothelial-myeloid transmigration interface | activated_neutrophil_cd44 ↔ endothelial 0.59× (avoidance); activated_myeloid_cd44 ↔ endothelial 0.62× | **NOT VISIBLE** at the 8-category interface level (immune-only and endothelial-only co-localize positively at +0.18) | **DISCRETE-ONLY** — transmigration interface is a sub-category phenomenon that the 8-category interface system cannot resolve |
| Neutrophil compartment CD44 activation | g_neut +0.28 (below headline gate; pure neutrophil pool too sparse) | **N/A** — composite track doesn't compute activation rates (activation overlay is dropped in the 8-category decomposition) | **N/A** (use Family C v1 for raw-marker compartment activation; that finding is g_neut +0.95 for background, +0.98 for triple-overlap, both robust) |
| Pan-tissue triple-positive interface growth (NEW) | invisible — discrete labels treat triple-positive tissue as unassigned | **endothelial+immune+stromal** rises 20% → 38% Sham → D7 (g = −3.98) | **COMPOSITE-ONLY** — this is the biggest single compositional shift in the dataset; the discrete track can't see it |
| Multi-lineage cross-interface avoidance (NEW) | invisible | endothelial ↔ immune+stromal at 0.58-0.60× symmetric avoidance; multiple cross-interface pairs at 0.58-0.71× | **COMPOSITE-ONLY** — the interface-zone biology is mutually exclusive at superpixel scale |

## Conclusions

1. **The composite track recovers the largest single compositional finding in the dataset**: triple-positive (E+I+S) interface tissue more than doubles Sham → D7 (g = −3.98 raw, −3.40 CLR). This is invisible to the discrete-cell-type track because the discrete ontology forces multi-lineage tissue into `unassigned`.

2. **The stromal-tightening story holds across both tracks and is stronger in the composite track.** Discrete `activated_fibroblast_cd140b` self-enrichment 1.24 → 2.54×; composite `stromal` 1.66 → 3.57× and `endothelial+stromal` 1.44 → 3.26×. The interface-level reading is more comprehensive.

3. **The repair-niche asymmetry at the discrete level is resolved at the composite level.** The discrete `activated_fibroblast_cd140b → m2_macrophage` at 1.43× has no reciprocal direction (0.93×). The composite `immune+stromal ↔ stromal` symmetric pair at 1.62-1.63× restores the symmetric reading — the underlying biology is "immune-stromal interface tissue sits next to pure stromal tissue," which is bidirectionally true.

4. **The transmigration-interface negative finding is a discrete-track phenomenon**, not visible at composite resolution. The composite tracks cannot distinguish CD44-activated from CD140b-co-expressing endothelium (it collapses activation), so the cleaner discrete reading is more diagnostic here.

5. **Two new cross-interface avoidance findings emerge from the composite track**: cross-pair interfaces avoid each other at superpixel scale (endothelial ↔ immune+stromal at 0.58×, endothelial+immune ↔ stromal at 0.66×, etc.). The biological reading is that these are mutually exclusive tissue zones — different multi-lineage interfaces don't co-reside at 100 µm² resolution.

## What this means for the PI presentation

The discrete track is a **sharp story about pure cell types** (the 8% that pass strict 9/9 gates). The composite track is a **broader story about multi-lineage interface tissue** (the 81% of tissue that has at least one lineage active). They describe different biological phenomena.

The strongest single message the data supports is **the conversion of no-lineage and pure-single-lineage tissue into multi-lineage interface tissue across the injury course**, with the triple-positive interface compartment more than doubling by D7. This finding is in both tracks (composite as a compositional headline; pre-registered as the Family A v1 / Family C v1 CLR-based §8 headlines), but is most directly observed in the composite-lineage Phase 1 track.

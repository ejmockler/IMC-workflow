# Kidney Injury Spatial Architecture: Narrative Foundation

**The Central Question**: Does kidney obstruction create spatial organizational patterns that reveal the tissue's choice between repair and fibrosis?

---

## I. The UUO Injury Model Biology

### Mechanical Cascade
1. **Ureteral ligation** → urinary backflow → hydronephrosis (kidney swelling)
2. **Pressure buildup** → tubular dilation → rising interstitial pressure
3. **Mechanical stress** → inflammation trigger
4. **Secondary damage**: Inflammation → vascular rarefaction → hypoxia → fibrosis

### Timeline (from config.json temporal_expectations)
- **Day 1**: Neutrophil recruitment (Ly6G↑, CD11b↑) - acute inflammation, focal infiltration
- **Day 3**: Macrophage activation (CD206↑, CD11b↑) - peak inflammation, expanding response
- **Day 7**: Resolution or fibrosis (CD140a↑, CD140b↑, CD206↑) - the decision point

### Anatomical Context (from config.json kidney_experiment)
**Cortex** (outer kidney):
- Glomerular filtering apparatus (200μm architectural units)
- Dense vascular networks (CD31+/CD34+ endothelium)
- Better perfusion → better immune access
- Expected: CD31/CD34 enrichment, glomerular structures

**Medulla** (inner kidney):
- Concentrating tubules (75μm tubular cross-sections)
- Vulnerable to pressure damage from obstruction
- Hypoxia-prone due to countercurrent exchange
- Expected: CD140a enrichment (pericytes), interstitial distribution

### Multi-Scale Biological Architecture

**10μm - Capillary Scale** (from config: "capillary network resolution"):
- Peritubular capillaries
- CD31+ endothelial networks
- Immune cell interactions
- **Biology**: Individual cellular neighborhoods

**75μm - Tubular Scale** (from config: "tubular cross-sections"):
- Tubular-interstitial interface
- Local immune infiltration
- Microenvironmental organization
- **Biology**: Functional tissue units

**200μm - Architectural Scale** (from config: "glomerular units"):
- Glomerular tufts
- Cortical organization
- Regional specialization
- **Biology**: Tissue architecture

### Nine-Marker Panel Captures Three Injury Processes

**Immune Response** (CD45, CD11b, Ly6G, CD206):
- Pan-leukocyte (CD45) → myeloid (CD11b) → neutrophils (Ly6G) → M2 macrophages (CD206)
- Trajectory: Acute (Ly6G) → Chronic (CD206)
- Spatial: Focal infiltration → Expanding inflammation → Organized repair

**Stromal Response** (CD140a, CD140b):
- PDGFRα (pericytes, fibroblast progenitors)
- PDGFRβ (activated fibroblasts, mesenchymal cells)
- Trajectory: Quiescent → Activated → Fibrotic
- Spatial: Interstitial → Perivascular → Organized scar

**Vascular Response** (CD31, CD34):
- Pan-endothelial (CD31) + progenitors (CD34)
- Trajectory: Abundant (Sham) → Rarefaction (D3) → Persistent loss (D7)
- Spatial: Continuous networks → Fragmentation → Organized remnants

**Activation Axis** (CD44):
- Adhesion molecule, activation marker across ALL compartments
- Immune: CD45+/CD11b+/CD44+ (activated myeloid)
- Stromal: CD140b+/CD44+ (activated fibroblasts)
- Vascular: CD31+/CD34+/CD44+ (activated endothelium)
- **This is the key integration marker**

---

## II. What the Actual Data Shows (from examine_results.py)

### Multi-Scale Organization (NOVEL FINDING)
```
10μm scale: 11.1 ± 4.2 clusters (range 6-18)
20μm scale: 9.4 ± 3.5 clusters (range 5-16)
40μm scale: 2.5 ± 1.6 clusters (range 1-6)

Spatial coherence (Moran's I):
  10μm: 0.17 ± 0.10 (weak but positive)
  20μm: 0.07 ± 0.07 (very weak)
  40μm: 0.07 ± 0.11 (very weak)
```

**Interpretation**:
- Fine heterogeneity at capillary scale (10μm)
- Structure decreases with observation scale
- Local complexity → Global simplification
- This is NOVEL: scale-dependent tissue organization

### Temporal Progression (Validated + Surprises)

**Validated patterns**:
```
CD45 (pan-immune):  1.58 → 1.68 → 1.95  (+23%, progressive)
CD206 (M2):         2.13 → 2.30 → 2.41  (+13%, gradual)
CD44 (activation):  2.48 → 2.50 → 2.88  (+16%, late)
CD11b (myeloid):    2.33 → 2.47 → 2.57  (+10%, steady)
```

**Unexpected finding**:
```
Ly6G (neutrophils): 1.81 → 1.75 → 1.78  (FLAT, -1%)
```
- Config predicted "neutrophil spike at D1"
- ROI-level means show NO spike
- May be spatial (neutrophils cluster in foci, averaged out)
- Or temporal (spike before D1, already declining)

### Spatial Heterogeneity (Activation Markers Vary Most)

**Coefficient of variation across ROIs**:
```
CD44 (activation):    0.179  HIGH - injury heterogeneity
CD140b (fibroblast):  0.124  MODERATE
CD45 (immune):        0.122  MODERATE
CD11b (myeloid):      0.104  MODERATE
CD206 (M2):           0.102  MODERATE
CD34 (vascular):      0.083  LOW - structural stability
CD31 (endothelium):   0.074  LOW
Ly6G (neutrophils):   0.080  LOW
CD140a (pericytes):   0.064  LOW - constitutive
```

**Interpretation**: Injury creates diverse microenvironments. Vascular markers stable (structural), activation markers variable (response heterogeneity).

### Phenotype Gating Results (from executed notebook)

**Prevalence**:
```
M2 Macrophages:         10.3% (6.8% → 9.7% → 14.3%)  Progressive ✓
Neutrophils:             7.1% (6.4% → 5.8% → 9.1%)   Late increase
Activated Fibroblasts:   2.8% (4.0% → 2.9% → 1.7%)   DECREASES ✗
Activated Endothelial:   4.4% (5.3% → 5.4% → 2.5%)   DECREASES ✗
```

**Unexpected phenotype dynamics**:
- Activated fibroblasts/endothelium DECREASE over injury
- Either: (1) Real - early stress response that resolves
- Or: (2) Gating artifact - fixed thresholds miss shifting distributions

---

## III. The Narrative Architecture

### The Framework: Protein Fields → Cellular Identities → Spatial Configurations → Biological Truth

**Conceptual layers**:
1. **Protein fields**: Continuous IMC measurements (9 markers × 60K observations)
2. **Cellular identities**: Boolean gating abstracts fields into phenotypes
3. **Spatial configurations**: Clustering reveals organizational patterns
4. **Biological truth**: Patterns reveal tissue's repair vs fibrosis choice

### The Core Tension

**Two ways to see tissue**:
- **Supervised (biologist)**: Define phenotypes by known markers → measure abundance/location
- **Unsupervised (data)**: Find latent structure → interpret biologically

**The convergence question**: Do they agree?
- If YES → unsupervised structure has biological meaning
- If NO → one view is wrong or incomplete

### The n=2 Reality

**What we have**:
- 2 mice, 3 timepoints (D1/D3/D7), 18 ROIs, 60K superpixels
- Pattern discovery, NOT population inference
- Cross-replicate concordance as robustness check

**What we can claim**:
- ✓ Spatial organizing principles exist
- ✓ Scale-dependent structure is real
- ✓ Temporal patterns are reproducible
- ✓ These are testable hypotheses

**What we CANNOT claim**:
- ✗ Universal kidney injury law
- ✗ Population-level statistics
- ✗ Causal mechanisms
- ✗ Generalization beyond UUO model

---

## IV. The Six Analytical Threads

### Thread 1: Multi-Scale Architecture ⭐ STRONG, NOVEL

**Finding**: Tissue complexity decreases with observation scale (11 → 9 → 2.5 clusters)

**Biology**:
- 10μm captures cellular neighborhoods (immune clusters, vascular patches)
- 40μm captures tissue regions (cortex vs medulla, injury vs healthy)
- Reveals hierarchical organization

**Narrative**: "Kidney injury has fine local heterogeneity that simplifies to coarse regional patterns"

**Figures needed**:
- Violin plots: cluster count by scale
- Spatial maps: same ROI at 10/20/40μm
- Coherence decay: Moran's I vs scale

---

### Thread 2: Temporal Immune Progression ⭐ STRONG

**Finding**: Progressive immune infiltration (CD45 +23%, CD206 +13%), late activation (CD44 +16%)

**Biology**:
- Day 1-3: Immune recruitment (CD45, CD11b increase)
- Day 3-7: M2 polarization (CD206 increase)
- Day 7: Activation emerges (CD44 late surge)

**Surprise**: Ly6G flat (no neutrophil spike at ROI level)

**Narrative**: "Immune response progresses from myeloid recruitment to M2 polarization to late-stage activation"

**Figures needed**:
- Temporal trajectories: CD45, CD11b, CD206, CD44, Ly6G
- Phenotype dynamics: M2 prevalence over time
- Cross-mouse concordance: M1 vs M2 patterns

---

### Thread 3: Activation Heterogeneity ⭐ STRONG

**Finding**: CD44 most variable marker (CV=0.18), activation state varies dramatically across ROIs

**Biology**:
- Same injury, different microenvironment response
- Some regions activate strongly, others don't
- Heterogeneity predicts outcome diversity

**Narrative**: "Injury doesn't create uniform damage - it creates diverse activation microenvironments"

**Figures needed**:
- CV bar chart: marker variability
- Spatial heterogeneity map: CD44 distribution in one ROI
- ROI-to-ROI comparison: activation state spectrum

---

### Thread 4: Unsupervised Spatial Structure ⭐ MODERATE

**Finding**: Leiden clustering finds 6-18 communities per ROI, spatially coherent (Moran's I = 0.17)

**Current issue**:
- We're re-running K-means with k=6 instead of using Leiden results
- Need to EITHER use Leiden OR justify k=6 properly

**Biology**:
- Tissue organizes into spatial communities with distinct marker profiles
- Weak but consistent spatial autocorrelation
- Structure validated by stability analysis

**Narrative**: "Superpixels cluster into 6-18 spatial communities with distinct immune/vascular/stromal signatures"

**Figures needed**:
- Clustering heatmap + elbow
- Spatial maps colored by cluster
- Validation metrics (stability, Moran's I)

---

### Thread 5: Phenotype Gating ? UNCERTAIN

**Approach**: Boolean gating with percentile thresholds

**Current status**:
- Code is sophisticated and biologically justified
- Temporal dynamics partially validate (M2s increase ✓)
- But unexpected patterns (fibroblasts/endothelium decrease ✗)

**Issues to resolve**:
- Are phenotype decreases real or gating artifacts?
- Why no neutrophil spike?
- Do phenotypes organize spatially (need spatial maps)

**Next step**: Run gating, generate spatial maps, validate patterns

---

### Thread 6: Phenotype-Niche Convergence ?? UNTESTED

**Hypothesis**: Supervised phenotypes organize into unsupervised clusters

**Status**: Code exists but not validated

**Would be powerful IF**:
- Phenotypes show >1.5× enrichment in specific clusters
- Enrichments replicate across mice
- Creates interpretable "niche identities"

**Risk**: May not work if phenotypes are sparse or gating is problematic

**Next step**: Test enrichment, decide if strong enough to include

---

## V. The Proposed Narrative Structure

### Act 1: The UUO Injury Model (Context)
**Goal**: Ground everything in biology

**Content**:
- Ureteral obstruction → pressure → inflammation → decision point
- Cortex vs Medulla anatomy
- Expected timeline: D1 neutrophils → D3 macrophages → D7 repair/fibrosis
- 9 markers capture immune, stromal, vascular responses
- Multi-scale architecture: 10μm (capillaries) → 75μm (tubules) → 200μm (glomeruli)

**Message**: "We're asking WHERE and WHEN kidney injury reorganizes tissue, and whether patterns predict repair vs fibrosis"

---

### Act 2: Multi-Scale Tissue Architecture (Novel Finding)
**Goal**: Establish scale-dependent organization

**Content**:
- Leiden clustering: 11 communities at 10μm → 2.5 at 40μm
- Spatial coherence: weak but consistent (Moran's I = 0.17)
- Visualization: same ROI at three scales

**Message**: "Tissue has fine local heterogeneity that simplifies to coarse regional patterns - organization is scale-dependent"

**Why first**: This is NOVEL and validates that we're finding structure, not noise

---

### Act 3: Temporal Immune Dynamics (Expected Biology)
**Goal**: Show injury follows predicted trajectory

**Content**:
- Progressive immune infiltration: CD45 +23%, CD206 +13%
- M2 macrophage accumulation: 6.8% → 14.3%
- Late activation: CD44 +16% at D7
- Surprise: Ly6G flat (no neutrophil spike)

**Message**: "Immune response progresses from recruitment to polarization to activation - with incomplete resolution by D7"

**Why second**: Validates we're measuring biology, not artifacts

---

### Act 4: Activation Heterogeneity (Response Diversity)
**Goal**: Show injury creates diverse microenvironments

**Content**:
- CD44 most variable (CV=0.18)
- Spatial maps showing activation patches
- ROI-to-ROI spectrum of responses

**Message**: "Same injury model, different microenvironment fates - heterogeneity is the pattern"

**Why third**: Sets up that spatial organization matters for outcome

---

### Act 5 (Optional): Phenotype Spatial Organization
**IF phenotype gating validates**:
- M2 macrophages cluster spatially
- Activated fibroblasts form patches
- Phenotypes organize non-randomly

**Message**: "Known cell types organize into spatial configurations"

---

### Act 6 (Optional): Phenotype-Niche Convergence
**IF enrichment analysis works**:
- Supervised phenotypes enrich in unsupervised clusters
- Convergence proves biological meaning
- Cross-mouse concordance

**Message**: "Biologist's view and data's view converge - structure is real"

---

### Epilogue: What This Reveals About Kidney Injury

**The Core Insight**:
> Kidney obstruction doesn't create uniform damage. It creates scale-dependent spatial heterogeneity: fine local diversity (10μm neighborhoods with different immune/vascular/stromal activation) organizing into coarse regional patterns (cortex vs medulla, injury vs healthy).
>
> By Day 7, the kidney shows incomplete immune resolution (persistent CD45, expanding M2s) with heterogeneous activation (CD44 varies 18% across ROIs). Whether this heterogeneity leads to repair or fibrosis depends on signals we cannot measure with this 9-marker panel.
>
> This is the spatial architecture invisible to bulk methods - the organizational logic that only IMC reveals.

**Honest limitations**:
- n=2: Pattern discovery, not population law
- D1-D7: Acute phase only (need D14-D28 for fibrosis outcome)
- 9 markers: Immune/vascular focused (miss epithelium, ECM)
- Superpixels: Tissue neighborhoods, not single cells

**What we've shown**:
- ✓ Multi-scale hierarchical organization
- ✓ Temporal immune progression
- ✓ Spatial activation heterogeneity
- ✓ Reproducible patterns (2 mice, 18 ROIs)

**What remains unknown**:
- Outcome: Do these patterns predict repair vs fibrosis?
- Mechanism: What creates heterogeneity?
- Generalization: Do patterns hold in other injury models?

---

## VI. Implementation Plan

### Phase 1: Core Narrative (Acts 1-4) - SUFFICIENT FOR PUBLICATION

**Required**:
1. Load actual Leiden clustering results (not re-run K-means)
2. Multi-scale analysis: cluster count, coherence, spatial maps
3. Temporal dynamics: CD45, CD206, CD44, Ly6G trajectories
4. Activation heterogeneity: CV analysis, spatial maps

**Outcome**: Publishable methods/analysis paper on multi-scale IMC

---

### Phase 2: Test Extensions (Acts 5-6) - ADD IF STRONG

**Test phenotype gating**:
1. Run on superpixels, generate prevalence
2. Create spatial maps (do phenotypes cluster?)
3. Check temporal dynamics (validate expected patterns?)
4. **Decision**: Include if patterns are clear, skip if messy

**Test phenotype-niche enrichment**:
1. Compute enrichment matrix
2. Check cross-mouse concordance
3. **Decision**: Include if enrichments >1.5× and concordant, skip otherwise

---

### Phase 3: Notebook Construction

**Structure**:
- Self-contained cells (minimize external dependencies)
- Inline helper functions where needed
- Load Leiden results directly from result files
- Publication-quality figures with biological interpretation
- Markdown cells that EXPLAIN biology, not just describe analysis

**Style** (from deprecated notebook):
- "Protein fields" framing
- Panel-by-panel biological interpretation
- Honest about limitations (n=2, acute phase only)
- Connect every analysis to UUO injury biology

---

## VII. Key Insights to Carry Forward

1. **Scale-dependent organization is the novel finding** - leads with this
2. **Ground everything in UUO biology** - not generic clustering
3. **Multi-dimensional analysis**: Temporal × Spatial × Anatomical × Scale
4. **Honest about n=2** - pattern discovery, not law
5. **Heterogeneity is the pattern** - not noise to average away
6. **Connect to outcome question** - repair vs fibrosis (even if we can't answer)

---

## VIII. Files to Keep vs Remove

**KEEP (working analysis)**:
- examine_results.py (data interrogation)
- phenotype_gating.py (Boolean gating logic)
- phenotype_niche_convergence.py (enrichment calculations)
- multiscale_convergence.py (scale coherence)
- viz_functions.py (plotting)

**MODIFY**:
- domain_characterization.py (use Leiden results, don't re-run K-means)

**REMOVE (markdown cruft)**:
- reality_check.md (findings now integrated here)
- NARRATIVE_ASSESSMENT.md (assessment complete)
- CRITICAL_DISCONNECT.md (issue identified)
- MULTISCALE_VALIDATION.md (approach integrated)
- README.md (replaced by this document)

**KEEP (this document)**:
- NARRATIVE_FOUNDATION.md (the single source of truth for planning)

---

## IX. Execution Checklist

- [x] Remove markdown cruft files (Nov 9: consolidated to docs/)
- [x] Modify domain_characterization.py to use Leiden results (Nov 9: completed)
- [x] Create core 4-act notebook (Acts 1-4) ✅ **COMPLETE**
- [x] Test phenotype gating on superpixels ✅ **COMPLETE**
- [x] Decided: **Extended to 6 parts** with deep exploratory analysis (Acts 1-6)
- [x] Generate all figures with biological interpretation ✅ **COMPLETE**
- [x] Write connecting narrative in markdown cells ✅ **COMPLETE**
- [x] Final execution and validation ✅ **12/12 tests passing**

---

## X. Final Deliverables (Nov 9, 2025)

### Canonical Narrative Notebook ✅

**File**: `notebooks/biological_narratives/kidney_injury_spatial_analysis.ipynb` (6.9MB)

**Structure** (Extended to 6 comprehensive exploratory parts):
1. **Opening**: UUO biology context, experimental design, marker panel
2. **Part 1**: Protein Fields Before Clustering - bimodal vs unimodal distributions
3. **Part 2**: Marker Co-Expression - 3 functional modules (Immune, Vascular, Stromal-Activation)
4. **Part 3**: Cluster Biological Identity - spatial maps, heterogeneity interpretation
5. **Part 4**: The Neutrophil Paradox - focal distribution resolves flat trajectory
6. **Part 5**: Multi-Lineage Coordination at D7 - CD44 bridges compartments, 8-10% triple-positive
7. **Part 6**: Scale-Dependent Organization - hierarchical model (local → neighborhood → regional)
8. **Original Parts 3-6**: Scale violin plots, temporal dynamics, heterogeneity, phenotype gating
9. **Summary**: Four key findings with biological interpretation

### Validation Tests ✅

**File**: `tests/test_biological_metrics.py` (12 tests, 100% passing)

**Tests validate**:
- CD44 as bridge molecule (r=0.75 immune, r=0.66 stromal, r=0.38 vascular)
- Multi-lineage coordination (CD44 activation 35-45% in each compartment)
- Triple-positive regions exceed random chance (8-10% observed vs 6% expected)
- Scale-dependent complexity reduction (4.4× from 10μm to 40μm)
- Spatial coherence positive (Moran's I = 0.17)
- Cluster variability (6-22 clusters per ROI, biological heterogeneity)

### Data Contract Documentation ✅

**File**: `docs/DATA_SCHEMA.md` (complete specification)
**File**: `src/utils/canonical_loader.py` (production loader with examples)

---

**This document was the canonical reference for notebook planning and narrative construction.**
**Final narrative now complete and validated. See PROJECT_STATUS.md for current project state.**

# Immediate Actions - Kidney Injury Analysis

**Date**: 2025-10-23
**Priority**: Threshold calibration and batch annotation

---

## Critical Findings from Testing

### ✅ What Works
- Cell type annotation engine functional
- Config-driven gating strategy operational
- Cluster→celltype enrichment mapping validated
- Provenance tracking complete

### 🔴 What Needs Fixing

#### 1. Thresholds Too Conservative (75th percentile)
**Evidence**:
- Day 1: Only 0.28% neutrophils (expected: 5-20%)
- Unassigned fractions: 70-94%
- Zero fractions for immune markers: 90-95%

**Problem**: In 95% zero data, 75th percentile = top 5% of brightest pixels

**Solution**: Lower to 60th percentile (= top 10-15% of pixels)

#### 2. Missing Expected Biology
**Day 1**: Should see neutrophil peak (currently 0.28%)
**Day 7**: Should see M2 macrophages + fibroblasts (barely detected)

**Note**: Cluster analysis finds these populations → gating thresholds issue, not data quality

---

## Immediate Next Steps (Today)

### 1. Adjust Config Thresholds
```json
{
  "cell_type_annotation": {
    "positivity_threshold": {
      "method": "percentile",
      "percentile": 60,  // Changed from 75
      "per_marker_override": {
        "CD206": {"method": "percentile", "percentile": 75},  // Changed from 85
        "Ly6G": {"method": "percentile", "percentile": 70}    // Changed from 85
      }
    }
  }
}
```

**Rationale**:
- Global 60th: captures sparse but real signal
- CD206 at 75th: maintain M2 specificity
- Ly6G at 70th: detect sparse neutrophil signal

### 2. Re-test Annotation
```bash
uv run python test_cell_annotation.py
```

**Expected improvements**:
- D1 neutrophils: 0.28% → 5-15%
- D7 M2 macrophages: detectable
- Overall assignment: 70-94% unassigned → 40-70% unassigned

### 3. If Results Good: Batch Annotate All ROIs
Create and run: `batch_annotate_all_rois.py`

---

## Brutalist's Top Criticisms to Address

### 🔴 **CRITICAL** (Must Fix)

#### 1. Statistical Power (n=2 mice)
**Brutalist Quote**: "Statistically indefensible. Every competent reviewer will immediately flag this."

**Our Response Options**:
- **Option A**: Get more mice (4-6 months)
- **Option B**: Reframe as methods paper
- **Option C**: Pilot study with transparent limitations ✓

**Current Decision**: Proceed with Option C, implement full analysis

#### 2. Zero Fractions Without Validation
**Brutalist Quote**: "More likely poor antibody staining or overly aggressive thresholding."

**Our Response**:
- ✓ Test results show cluster analysis finds populations
- ✓ Lower thresholds to match sparse signal
- ⏳ Document threshold selection rationale
- ⏳ Validate against cluster-based cell typing

#### 3. Unstable Clustering in Some ROIs
**Brutalist Quote**: "Stability 0.08 is computational garbage, not biology."

**Our Response**:
- ✓ Stability tracked in validation_report.json
- ⏳ Filter ROIs with stability <0.5
- ⏳ Report stability distributions
- ⏳ Investigate low-stability ROIs (technical artifact?)

### ⚠️ **IMPORTANT** (Should Address)

#### 4. No Batch Correction
**Brutalist Quote**: "This will get flagged."

**Our Response**:
- ⏳ Test if mouse effects significant (variance decomposition)
- ⏳ If yes: implement quantile normalization
- ⏳ Document decision in methods

#### 5. Missing Biological Insights
**Brutalist Quote**: "You have clustering but haven't asked biological questions."

**Our Response**:
- ✓ Module 1 complete (cell type annotation)
- ⏳ Module 2: Differential abundance (Week 1)
- ⏳ Module 3: Spatial statistics (Week 2)
- ⏳ Answer: Do we see neutrophil→macrophage→fibroblast transition?

---

## Validation Strategy

### Gating vs Clustering Concordance

**Current Test Results**:
- D1 Cluster 1,3,4: Mapped to neutrophils (confidence 0.33)
- D1 Gating: Only 0.28% neutrophils detected

**Interpretation**: Clusters find biology that gating misses → threshold issue

**Validation Plan**:
1. ✓ Test both methods on same data
2. ⏳ Compute confusion matrix
3. ⏳ Investigate discordant assignments
4. ⏳ Use cluster results to validate threshold choice

### Expected Concordance After Threshold Adjustment:
- High agreement on major populations (endothelial, immune)
- Gating catches more rare populations (M2 macrophages)
- Clustering reveals spatial organization

---

## Decision Tree for Publication Strategy

### After Re-annotation (This Week)

**If neutrophil peak at D1 detected**:
→ Biology looks compelling
→ Proceed with differential abundance analysis
→ Target: Pilot study (Frontiers, Sci Rep)

**If still missing expected biology**:
→ Threshold issue OR unexpected biology
→ Investigate cluster assignments more carefully
→ Consider methods paper angle

### After Module 2 Complete (Week 2)

**If effect sizes large & consistent**:
→ n=2 limitation acceptable for pilot
→ Continue with full analysis (Modules 3-5)
→ Target: 6-8 weeks to submission

**If effect sizes marginal**:
→ Pivot to methods paper
→ Focus on multi-scale framework novelty
→ Target: Nature Protocols, Star Protocols

---

## File Updates Needed

### Config Changes (config.json)
```json
Line 136: "percentile": 60,  // Changed from 75
Line 139: "percentile": 75,  // Changed from 85 (CD206)
Line 140: "Ly6G": {"method": "percentile", "percentile": 70}  // New
```

### New Scripts to Create
1. `batch_annotate_all_rois.py` - Process all 25 ROIs
2. `notebooks/01_cell_type_exploration.ipynb` - Interactive exploration

### Documentation Updates
- ✓ PROJECT_STATUS.md created
- ✓ IMMEDIATE_ACTIONS.md (this file)
- ⏳ Update CLAUDE.md with cell type annotation section

---

## Success Criteria for Today

### Minimum
- [ ] Config thresholds updated
- [ ] Re-test shows improved neutrophil detection at D1
- [ ] Batch annotation runs successfully on all ROIs

### Ideal
- [ ] D1 neutrophils: 5-15% (biologically expected)
- [ ] D7 M2 macrophages: 1-5% (repair signature)
- [ ] Overall unassigned: <50% (reasonable for sparse IMC data)
- [ ] Gating vs clustering concordance >70%

---

## Timeline (Optimistic)

**Today**: Threshold calibration + batch annotation
**Day 2-3**: Jupyter notebook exploration
**Week 1**: Module 2 (differential abundance)
**Week 2**: Module 3 (spatial statistics)
**Week 3**: Modules 4-5 (temporal, niches)
**Week 4**: Figure generation
**Week 5-6**: Manuscript drafting

**Realistic**: Add 2-4 weeks for iteration and debugging

---

## Questions for User

### Immediate
1. Approve threshold changes (75→60th percentile)?
2. Run batch annotation on all ROIs after validation?
3. Create Jupyter notebook for exploration?

### Strategic
4. Target: Pilot study or methods paper?
5. Timeline: 6-8 weeks acceptable?
6. More mice: Worth pursuing for higher-impact venue?

---

## References to Brutalist Feedback

See PROJECT_STATUS.md section "Brutalist Reviewer Feedback" for:
- Full critique text
- All critical issues
- Resolution strategies
- Publication options analysis

---

**Status**: Ready to adjust thresholds and re-test
**Blocker**: None - can proceed immediately
**Risk**: If threshold adjustment doesn't improve detection, need to investigate data quality or revise cell type definitions

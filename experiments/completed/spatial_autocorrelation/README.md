# Spatial Autocorrelation Analysis

## Purpose
Compute Moran's I (spatial autocorrelation coefficient) for all 9 protein markers across all ROIs to:
1. Identify which markers have strongest spatial clustering
2. Evaluate whether specific markers would make better segmentation inputs
3. Understand ROI heterogeneity

## Method
- **Metric**: Moran's I (range: -1 to +1, positive = spatial clustering)
- **ROIs**: All 25 ROIs from kidney injury time course
- **Markers**: CD45, CD11b, Ly6G, CD140a, CD140b, CD31, CD34, CD206, CD44

## Key Findings

### Representative ROIs

**High Stability (D7_M1_01_21, s=0.81)**:
- Top markers: CD45 (0.240), CD34 (0.214), CD31 (0.213)
- DNA-based segmentation works well here

**Low Stability (D1_M1_01_9, s=0.08)**:
- Top markers: CD11b (0.160), Ly6G (0.158), CD45 (0.123)
- Immune markers dominate, CD31/CD140b very low (0.074)

**Medium Stability (D3_M1_01_15, s=0.60)**:
- Top markers: CD206 (0.117), CD44 (0.096), CD140b (0.089)
- Repair markers dominate

### Extreme Cases
**Highest autocorrelation**: D7_M2_01_24
- CD140a (0.563), CD44 (0.515), CD31 (0.402)
- Strong vascular/stromal organization

**Most autocorrelated marker overall**: CD206 @ D7_M2_02_25 (0.346)

## Conclusions

1. **No universal segmentation marker**: Different ROIs have different dominant spatial structures
2. **ROI heterogeneity is real**: Timepoint and region drive which markers cluster
3. **DNA remains best universal choice**: Consistent baseline across all ROIs
4. **Spatial structure varies with injury timeline**:
   - Sham: Moderate clustering (CD11b, CD45)
   - D1: Immune dominance (neutrophils)
   - D3: Repair signatures (CD206)
   - D7: Multi-lineage (vascular + immune + repair)

## Impact
This analysis validated the decision to use DNA-based segmentation rather than structure-guided approaches (see `experiments/completed/structure_segmentation/`).

## Files
- `analyze_spatial_autocorrelation.py` - Analysis script

## Date
October 2025

# Structure-Guided Segmentation Experiment

## Hypothesis
Using vascular (CD31) + stromal (CD140b) markers for SLIC segmentation might improve clustering stability compared to DNA-based segmentation, especially in ROIs with low DNA signal.

## Method
Tested SLIC segmentation on 3 representative ROIs:
- D1_M1_01_9 (LOW stability: 0.08)
- D3_M1_01_15 (MED stability: 0.60)
- D7_M1_01_21 (HIGH stability: 0.81)

Compared:
- **DNA-based**: DNA1 + DNA2 (current approach)
- **Structure-guided**: CD31 + CD140b

## Results

| ROI | Original Stability | New Stability | Change |
|-----|-------------------|---------------|--------|
| D1_M1_01_9 (LOW) | 0.08 | 0.645 | +0.565 ✅ |
| D3_M1_01_15 (MED) | 0.60 | 0.538 | -0.062 ❌ |
| D7_M1_01_21 (HIGH) | 0.81 | 0.408 | -0.402 ❌ |

## Conclusion
**Hypothesis REJECTED**

Structure-guided segmentation showed **ROI heterogeneity issues**:
- Improved worst-case ROI (D1) dramatically
- **Degraded best-case ROI (D7)** by 50%

**Root cause** (via Moran's I analysis):
Different ROIs have different dominant spatial structures:
- D7: CD45 (0.240), CD34 (0.214), CD31 (0.213) - vascular markers work
- D1: CD11b (0.160), Ly6G (0.158) - immune markers dominate
- D3: CD206 (0.117), CD44 (0.096) - neither DNA nor CD31/CD140b capture biology

**Decision**: Retain **DNA-based segmentation** as universal baseline.

## Files
- `test_structure_segmentation.py` - Test script
- Config used: `slic_input_channels: ["CD31", "CD140b"]`

## Date
October 2025

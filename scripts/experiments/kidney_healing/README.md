# Kidney Healing Experiment Scripts

This directory contains experiment-specific scripts for the kidney injury and healing study.

## Scripts

### run_timeline.py
- **Purpose**: Generate temporal progression timeline visualization
- **Output**: `kidney_healing_timeline.png`
- **Use case**: Quick visualization of healing progression over Days 1, 3, 7

### run_full_report.py
- **Purpose**: Generate comprehensive publication-quality figures
- **Outputs**:
  - `kidney_healing_timeline.png` - Temporal progression
  - `kidney_condition_comparison.png` - Sham vs Injury comparison
  - `kidney_replicate_variance.png` - Biological replicate variance
  - `kidney_region_time_grid.png` - Region × Time contact matrix
  - `kidney_healing_metrics.json` - Quantitative metrics

## Usage

```bash
# Quick timeline visualization
python scripts/experiments/kidney_healing/run_timeline.py

# Full publication report
python scripts/experiments/kidney_healing/run_full_report.py
```

## Requirements
- Requires completed analysis (`run_analysis.py` should be run first)
- Expects kidney-specific metadata in CSV files
- Config must include experimental parameters for timepoints, conditions, regions
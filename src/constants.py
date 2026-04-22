"""Frozen constants that span analysis + viz boundaries.

Extracted here so config/viz loaders can import them without pulling in
scipy, pandas, or any heavy analysis code. Changes to anything in this
module are pre-registration-level events — update with care.
"""
from __future__ import annotations

from typing import Tuple


# Timepoint ordering — frozen by analysis_plans/temporal_interfaces_plan.md §2.
# Used by:
#   - src.analysis.temporal_interface_analysis (Family A/B/C contrasts)
#   - src.viz_utils.viz_config (cross-check against viz.json timepoint_display.order)
TIMEPOINT_ORDER: Tuple[str, ...] = ('Sham', 'D1', 'D3', 'D7')


# Continuous lineage score column names in the per-ROI annotation parquet.
# Used by Family A interface classification + viz layer diagnostics.
LINEAGE_COLS: Tuple[str, ...] = ('lineage_immune', 'lineage_endothelial', 'lineage_stromal')

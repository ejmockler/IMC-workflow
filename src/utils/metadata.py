"""Canonical ROI metadata parser. Extracts timepoint, mouse, region from IMC filenames."""

import pandas as pd
from functools import lru_cache
from pathlib import Path
from typing import Optional

_DEFAULT_METADATA_CSV = Path(__file__).parent.parent.parent / "data" / "241218_IMC_Alun" / "Metadata-Table 1.csv"


@lru_cache(maxsize=8)
def _load_metadata_df_cached(metadata_csv_str: str) -> pd.DataFrame:
    """Per-path cached CSV loader. Keyed on the string form of the path so
    different metadata_csv arguments get independent cache entries — tests
    with a custom CSV no longer poison subsequent default-path callers.
    """
    path = Path(metadata_csv_str)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clear_metadata_cache() -> None:
    """Test helper: drop all cached metadata DataFrames."""
    _load_metadata_df_cached.cache_clear()


def parse_roi_metadata(roi_id: str, metadata_csv: Optional[Path] = None) -> dict:
    """
    Extract metadata from ROI ID using the metadata CSV.

    Format: IMC_241218_Alun_ROI_{timepoint}_{mouse}_{replicate}_{index}
    Examples:
      - IMC_241218_Alun_ROI_D1_M1_01_9 (Day 1, Mouse 1)
      - IMC_241218_Alun_ROI_Sam1_01_2 (Sham timepoint)

    Note: M1/M2 in filename = Mouse 1/Mouse 2 (biological replicates), NOT regions.
    Actual anatomical region comes from the metadata "Details" column.

    Returns dict with keys: roi_id, timepoint, region, replicate, mouse.
    """
    csv_path = Path(metadata_csv) if metadata_csv else _DEFAULT_METADATA_CSV
    metadata_df = _load_metadata_df_cached(str(csv_path))

    # Remove 'roi_' prefix if present
    file_name = roi_id.replace('roi_', '')

    # Look up actual anatomical region from metadata
    metadata_row = metadata_df[metadata_df['File Name'] == file_name]

    if len(metadata_row) == 0:
        # Fallback parsing for ROIs not in metadata
        parts = roi_id.split('_')
        if 'Sam' in roi_id:
            timepoint = 'Sham'
            sam_part = [p for p in parts if 'Sam' in p][0]
            replicate = sam_part
            mouse = None
            region = None
        elif 'Test' in roi_id:
            timepoint = 'Test'
            replicate = 'Test01'
            mouse = None
            region = None
        else:
            d_parts = [p for p in parts if p.startswith('D') and len(p) <= 3]
            if d_parts:
                timepoint = d_parts[0]
                mouse_parts = [p for p in parts if p.startswith('M') and len(p) <= 3]
                mouse = mouse_parts[0] if mouse_parts else None
                replicate_idx = [i for i, p in enumerate(parts) if p.startswith('M') and len(p) <= 3]
                if replicate_idx and mouse:
                    next_idx = replicate_idx[0] + 1
                    replicate = f"{timepoint}_{mouse}_{parts[next_idx]}" if next_idx < len(parts) else f"{timepoint}_{mouse}"
                else:
                    replicate = timepoint
            else:
                # Unrecognized format (e.g. Bodenmiller Patient1 ROIs)
                timepoint = 'Unknown'
                mouse = None
                replicate = roi_id
            region = None

        return {
            'roi_id': roi_id,
            'timepoint': timepoint,
            'region': region,
            'replicate': replicate,
            'mouse': mouse,
        }

    # Extract from metadata CSV
    row = metadata_row.iloc[0]
    injury_day = row['Injury Day']
    timepoint = 'Sham' if injury_day == 0 else f"D{int(injury_day)}"
    region = row['Details'].strip() if pd.notna(row['Details']) else None
    mouse = row['Mouse']

    # Build replicate ID
    parts = roi_id.split('_')
    if 'Sam' in roi_id:
        replicate = mouse.replace('MS', 'Sam')  # MS1 -> Sam1, MS2 -> Sam2
    else:
        replicate_num = [p for p in parts if p.isdigit() and len(p) == 2]
        if replicate_num:
            replicate = f"{timepoint}_{mouse}_{replicate_num[0]}"
        else:
            replicate = f"{timepoint}_{mouse}"

    return {
        'roi_id': roi_id,
        'timepoint': timepoint,
        'region': region,
        'replicate': replicate,
        'mouse': mouse,
    }

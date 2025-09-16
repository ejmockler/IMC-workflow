"""
Centralized Data Loading for IMC Analysis
Single source of truth for loading and preprocessing IMC data
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from .helpers import Metadata


@dataclass
class IMCData:
    """Standard container for IMC data"""
    coords: np.ndarray  # Spatial coordinates (n_pixels, 2)
    values: np.ndarray  # Expression values (n_pixels, n_proteins)
    protein_names: List[str]  # Protein names
    roi_id: str  # ROI identifier
    metadata: Optional[Dict[str, Any]] = None


def load_roi_data(roi_file: Path, 
                 config_path: str = 'config.json') -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load ROI data with protein expression and coordinates
    
    This is the primary data loading function used throughout the pipeline.
    Returns raw data tuple for backward compatibility.
    
    Args:
        roi_file: Path to ROI file
        config_path: Path to configuration file
        
    Returns:
        Tuple of (coords, values, protein_names)
    """
    data = load_roi_as_imc_data(roi_file, config_path)
    return data.coords, data.values, data.protein_names


def load_roi_as_imc_data(roi_file: Path,
                         config_path: str = 'config.json') -> IMCData:
    """
    Load ROI data as IMCData object
    
    Modern interface returning structured data object.
    
    Args:
        roi_file: Path to ROI file
        config_path: Path to configuration file
        
    Returns:
        IMCData object with all data and metadata
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Read ROI file
    df = pd.read_csv(roi_file, sep='\t')
    
    # Get proteins from functional groups (excluding DNA/structural controls)
    selected_names = set()
    if 'functional_groups' in config['proteins']:
        for group_name, proteins in config['proteins']['functional_groups'].items():
            if group_name != 'structural_controls':
                selected_names.update(proteins)
    
    # If no functional groups, fall back to protein list
    if not selected_names and 'channels' in config['proteins']:
        # Extract protein names from channel descriptions
        for channel in config['proteins']['channels']:
            protein_name = channel.split('(')[0]
            selected_names.add(protein_name)
    
    # Map to full column names with isotope tags
    available = []
    for protein_name in selected_names:
        for col in df.columns:
            if col.startswith(protein_name + '('):
                available.append(col)
                break
    
    # Extract coordinates and values
    coords = df[['X', 'Y']].values
    
    # Apply arcsinh transformation (standard for IMC/CyTOF data)
    cofactor = config.get('analysis', {}).get('transformation_cofactor', 5.0)
    values = np.arcsinh(df[available].values / cofactor)
    
    # Extract clean protein names
    protein_names = [col.split('(')[0] for col in available]
    
    # Extract ROI ID from filename
    roi_id = Path(roi_file).stem
    
    # Build metadata
    metadata = {
        'filename': str(roi_file),
        'n_pixels': len(coords),
        'n_proteins': len(protein_names),
        'x_range': (coords[:, 0].min(), coords[:, 0].max()),
        'y_range': (coords[:, 1].min(), coords[:, 1].max()),
        'transformation': f'arcsinh(x/{cofactor})'
    }
    
    return IMCData(
        coords=coords,
        values=values,
        protein_names=protein_names,
        roi_id=roi_id,
        metadata=metadata
    )


def load_multiple_rois(roi_files: List[Path],
                      config_path: str = 'config.json') -> List[IMCData]:
    """
    Load multiple ROI files
    
    Args:
        roi_files: List of ROI file paths
        config_path: Path to configuration file
        
    Returns:
        List of IMCData objects
    """
    return [load_roi_as_imc_data(roi_file, config_path) for roi_file in roi_files]


def subsample_data(data: IMCData, 
                  n_samples: int,
                  random_state: int = 42) -> IMCData:
    """
    Subsample IMC data for faster processing
    
    Args:
        data: IMCData object
        n_samples: Number of pixels to sample
        random_state: Random seed for reproducibility
        
    Returns:
        Subsampled IMCData object
    """
    if n_samples >= len(data.coords):
        return data
    
    np.random.seed(random_state)
    indices = np.random.choice(len(data.coords), n_samples, replace=False)
    
    metadata = data.metadata.copy() if data.metadata else {}
    metadata['subsampled'] = True
    metadata['original_n_pixels'] = len(data.coords)
    metadata['subsample_n_pixels'] = n_samples
    
    return IMCData(
        coords=data.coords[indices],
        values=data.values[indices],
        protein_names=data.protein_names,
        roi_id=data.roi_id,
        metadata=metadata
    )


def normalize_expression(data: IMCData,
                       method: str = 'zscore') -> IMCData:
    """
    Normalize expression values
    
    Args:
        data: IMCData object
        method: Normalization method ('zscore', 'minmax', 'percentile')
        
    Returns:
        IMCData with normalized values
    """
    values = data.values.copy()
    
    if method == 'zscore':
        # Z-score normalization per protein
        means = values.mean(axis=0)
        stds = values.std(axis=0)
        stds[stds == 0] = 1  # Avoid division by zero
        values = (values - means) / stds
        
    elif method == 'minmax':
        # Min-max scaling per protein
        mins = values.min(axis=0)
        maxs = values.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        values = (values - mins) / ranges
        
    elif method == 'percentile':
        # Percentile normalization (99th percentile)
        for i in range(values.shape[1]):
            p99 = np.percentile(values[:, i], 99)
            if p99 > 0:
                values[:, i] = values[:, i] / p99
                values[:, i] = np.clip(values[:, i], 0, 1)
    
    metadata = data.metadata.copy() if data.metadata else {}
    metadata['normalization'] = method
    
    return IMCData(
        coords=data.coords,
        values=values,
        protein_names=data.protein_names,
        roi_id=data.roi_id,
        metadata=metadata
    )


# Backward compatibility exports
def load_imc_data(roi_file: Path, config_path: str = 'config.json') -> IMCData:
    """Alias for consistency with pipeline.py"""
    return load_roi_as_imc_data(roi_file, config_path)


def load_metadata_from_csv(csv_path: Path, 
                           config: Dict[str, Any],
                           roi_files: List[Path]) -> Dict[str, Metadata]:
    """
    Load and standardize metadata from CSV using config mapping.
    
    This is the single source of truth for metadata loading.
    Maps user's CSV columns to standardized Metadata objects.
    
    Args:
        csv_path: Path to metadata CSV file
        config: Configuration dictionary with metadata_tracking section
        roi_files: List of ROI files to match
        
    Returns:
        Dictionary mapping ROI filename to Metadata object
    """
    metadata_map = {}
    
    if not csv_path.exists():
        print(f"Warning: Metadata file {csv_path} not found. Using default metadata for all ROIs.")
        # Return default metadata for all ROIs
        for roi_file in roi_files:
            metadata_map[roi_file.stem] = Metadata()
        return metadata_map
    
    # Load CSV with error handling
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("Metadata CSV file is empty")
    except Exception as e:
        raise ValueError(f"Failed to load metadata CSV {csv_path}: {str(e)}. "
                        f"Please check file format and permissions.")
    
    # Get column mappings from config
    tracking = config.get('metadata_tracking', {})
    replicate_col = tracking.get('replicate_column', 'Replicate')
    timepoint_col = tracking.get('timepoint_column', 'Timepoint')
    condition_col = tracking.get('condition_column', 'Condition') 
    region_col = tracking.get('region_column', 'Region')
    filename_col = tracking.get('filename_column', 'File Name')
    
    # Validate required columns exist
    available_columns = list(df.columns)
    required_columns = [filename_col]  # At minimum, need filename to match ROIs
    optional_columns = [replicate_col, timepoint_col, condition_col, region_col]
    
    missing_required = []
    missing_optional = []
    
    def find_column_variant(col_name, available_cols):
        """Find column with potential spacing variations."""
        variants = [col_name, col_name + ' ', ' ' + col_name, col_name.strip()]
        for variant in variants:
            if variant in available_cols:
                return variant
        return None
    
    # Check required columns
    for col in required_columns:
        if not find_column_variant(col, available_columns):
            missing_required.append(col)
    
    # Check optional columns
    for col in optional_columns:
        if not find_column_variant(col, available_columns):
            missing_optional.append(col)
    
    if missing_required:
        raise ValueError(f"Required columns missing from metadata CSV: {missing_required}. "
                        f"Available columns: {available_columns}. "
                        f"Please check your metadata_tracking configuration in config.json.")
    
    if missing_optional:
        print(f"Warning: Optional metadata columns missing: {missing_optional}. "
              f"These ROIs will use default values. Available columns: {available_columns}")
    
    # Update column names to use found variants
    filename_col = find_column_variant(filename_col, available_columns)
    replicate_col = find_column_variant(replicate_col, available_columns)
    timepoint_col = find_column_variant(timepoint_col, available_columns)
    condition_col = find_column_variant(condition_col, available_columns)
    region_col = find_column_variant(region_col, available_columns)
    
    # Handle potential trailing spaces in column names
    # Check both with and without trailing space
    def get_column_value(row, col_name, default='Unknown'):
        """Get column value handling potential trailing spaces."""
        if col_name in row:
            return row[col_name]
        elif col_name + ' ' in row:
            return row[col_name + ' ']
        elif ' ' + col_name in row:
            return row[' ' + col_name]
        return default
    
    # Process each ROI file with validation
    unmatched_rois = []
    validation_warnings = []
    
    for roi_file in roi_files:
        roi_name = roi_file.stem
        
        # Try to find matching row in CSV
        matched = False
        for _, row in df.iterrows():
            file_value = get_column_value(row, filename_col, '')
            if file_value == roi_name:
                try:
                    # Get and validate timepoint (should be numeric if present)
                    timepoint_raw = get_column_value(row, timepoint_col, None)
                    timepoint = None
                    if timepoint_raw is not None and timepoint_raw != 'Unknown':
                        try:
                            timepoint = int(float(timepoint_raw))  # Handle both int and float strings
                        except (ValueError, TypeError):
                            validation_warnings.append(
                                f"ROI {roi_name}: Invalid timepoint value '{timepoint_raw}', using None"
                            )
                            timepoint = None
                    
                    # Create metadata with validated data
                    metadata_map[roi_name] = Metadata(
                        replicate_id=str(get_column_value(row, replicate_col, 'Unknown')),
                        timepoint=timepoint,
                        condition=str(get_column_value(row, condition_col, 'Unknown')),
                        region=str(get_column_value(row, region_col, 'Unknown'))
                    )
                    matched = True
                    break
                    
                except Exception as e:
                    validation_warnings.append(
                        f"ROI {roi_name}: Error processing metadata - {str(e)}. Using defaults."
                    )
                    metadata_map[roi_name] = Metadata()
                    matched = True
                    break
        
        # Use default if no match found
        if not matched:
            unmatched_rois.append(roi_name)
            metadata_map[roi_name] = Metadata()
    
    # Report warnings and unmatched ROIs
    if validation_warnings:
        print(f"Metadata validation warnings:")
        for warning in validation_warnings[:5]:  # Limit to first 5 warnings
            print(f"  - {warning}")
        if len(validation_warnings) > 5:
            print(f"  ... and {len(validation_warnings) - 5} more warnings")
    
    if unmatched_rois:
        print(f"Warning: {len(unmatched_rois)} ROIs not found in metadata CSV: {unmatched_rois[:3]}...")
        print(f"These ROIs will use default metadata values.")
        
        # Suggest possible filename issues
        csv_filenames = set(df[filename_col].astype(str))
        roi_filenames = set(roi_name for roi_name in [f.stem for f in roi_files])
        
        if csv_filenames and roi_filenames:
            # Check for systematic naming differences
            csv_sample = list(csv_filenames)[0] if csv_filenames else ""
            roi_sample = list(roi_filenames)[0] if roi_filenames else ""
            if csv_sample and roi_sample and csv_sample != roi_sample:
                print(f"Filename format mismatch detected:")
                print(f"  CSV format example: '{csv_sample}'")
                print(f"  ROI format example: '{roi_sample}'")
                print(f"Consider checking filename column mapping in config.json")
    
    # Final validation summary
    total_rois = len(roi_files)
    matched_rois = total_rois - len(unmatched_rois)
    print(f"Metadata loading summary: {matched_rois}/{total_rois} ROIs matched successfully")
    
    return metadata_map
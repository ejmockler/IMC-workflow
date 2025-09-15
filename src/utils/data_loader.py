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
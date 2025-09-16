"""
Data loading utilities for visualization.

Helper functions to load analysis results from standard output formats
(HDF5, Parquet, JSON) for use in visualization notebooks.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings


def load_roi_results(
    results_path: Union[str, Path],
    roi_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load ROI analysis results from JSON output.
    
    Args:
        results_path: Path to results file or directory
        roi_name: Specific ROI to load (if None, loads all)
        
    Returns:
        Dictionary containing ROI results
    """
    results_path = Path(results_path)
    
    if results_path.is_dir():
        # Look for JSON files in directory
        json_files = list(results_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {results_path}")
        results_path = json_files[0]
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    if roi_name and 'roi_results' in data:
        # Filter for specific ROI
        roi_results = [r for r in data['roi_results'] 
                      if r.get('roi_name') == roi_name]
        if roi_results:
            return roi_results[0]
        else:
            raise ValueError(f"ROI {roi_name} not found in results")
    
    return data


def load_multiscale_results(
    results_path: Union[str, Path],
    roi_name: Optional[str] = None
) -> Dict[float, Dict]:
    """
    Load multiscale analysis results.
    
    Args:
        results_path: Path to results file
        roi_name: Specific ROI to extract
        
    Returns:
        Dictionary mapping scales to their results
    """
    data = load_roi_results(results_path, roi_name)
    
    if 'multiscale_results' in data:
        # Convert string keys to float if needed
        multiscale = {}
        for key, value in data['multiscale_results'].items():
            try:
                scale = float(key)
            except (ValueError, TypeError):
                scale = key
            multiscale[scale] = value
        return multiscale
    
    return {}


def load_batch_results(
    results_dir: Union[str, Path],
    batch_pattern: str = "*"
) -> pd.DataFrame:
    """
    Load and combine results from multiple batches.
    
    Args:
        results_dir: Directory containing batch results
        batch_pattern: Pattern to match batch files
        
    Returns:
        Combined DataFrame of batch results
    """
    results_dir = Path(results_dir)
    
    # Try different file formats
    parquet_files = list(results_dir.glob(f"{batch_pattern}.parquet"))
    if parquet_files:
        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)
    
    csv_files = list(results_dir.glob(f"{batch_pattern}.csv"))
    if csv_files:
        dfs = [pd.read_csv(f) for f in csv_files]
        return pd.concat(dfs, ignore_index=True)
    
    # Fall back to JSON
    json_files = list(results_dir.glob(f"{batch_pattern}.json"))
    if json_files:
        all_data = []
        for f in json_files:
            with open(f, 'r') as file:
                data = json.load(file)
                if isinstance(data, list):
                    all_data.extend(data)
                elif isinstance(data, dict) and 'results' in data:
                    all_data.extend(data['results'])
        return pd.DataFrame(all_data)
    
    raise FileNotFoundError(f"No data files found in {results_dir} matching {batch_pattern}")


def load_protein_data(
    roi_file: Union[str, Path],
    protein_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Load protein expression data from ROI file.
    
    Args:
        roi_file: Path to ROI data file (TSV format)
        protein_names: Specific proteins to load (None = all)
        
    Returns:
        Dictionary mapping protein names to expression arrays
    """
    roi_data = pd.read_csv(roi_file, sep='\t')
    
    # Get protein columns (exclude coordinates and DNA)
    protein_cols = [col for col in roi_data.columns 
                   if col not in ['X', 'Y'] and 'DNA' not in col]
    
    if protein_names:
        protein_cols = [col for col in protein_cols 
                       if any(name in col for name in protein_names)]
    
    protein_data = {}
    for col in protein_cols:
        # Extract clean protein name (before parenthesis)
        protein_name = col.split('(')[0] if '(' in col else col
        protein_data[protein_name] = roi_data[col].values
    
    return protein_data


def load_coordinates(roi_file: Union[str, Path]) -> np.ndarray:
    """
    Load spatial coordinates from ROI file.
    
    Args:
        roi_file: Path to ROI data file
        
    Returns:
        (N, 2) array of X, Y coordinates
    """
    roi_data = pd.read_csv(roi_file, sep='\t')
    return roi_data[['X', 'Y']].values


def load_clustering_results(
    results_path: Union[str, Path],
    scale: Optional[float] = None
) -> Dict[str, Any]:
    """
    Load clustering results from analysis output.
    
    Args:
        results_path: Path to results
        scale: Specific scale to load (for multiscale analysis)
        
    Returns:
        Dictionary with cluster labels, centroids, and metadata
    """
    data = load_roi_results(results_path)
    
    if scale and 'multiscale_results' in data:
        scale_results = data['multiscale_results'].get(str(scale), {})
        return {
            'labels': scale_results.get('cluster_labels'),
            'centroids': scale_results.get('cluster_centroids'),
            'n_clusters': len(scale_results.get('cluster_centroids', [])),
            'method': scale_results.get('method')
        }
    
    # Try to find clustering results at top level
    clustering = {}
    if 'cluster_labels' in data:
        clustering['labels'] = np.array(data['cluster_labels'])
    if 'cluster_centroids' in data:
        clustering['centroids'] = np.array(data['cluster_centroids'])
    if 'clustering_params' in data:
        clustering.update(data['clustering_params'])
    
    return clustering


def load_validation_results(
    validation_dir: Union[str, Path]
) -> pd.DataFrame:
    """
    Load validation study results.
    
    Args:
        validation_dir: Directory containing validation outputs
        
    Returns:
        DataFrame with validation metrics
    """
    validation_dir = Path(validation_dir)
    
    # Look for validation summary
    summary_file = validation_dir / "validation_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame for easy analysis
        if 'experiments' in data:
            return pd.DataFrame(data['experiments'])
        elif 'results' in data:
            return pd.DataFrame(data['results'])
        else:
            return pd.DataFrame([data])
    
    # Look for CSV results
    csv_files = list(validation_dir.glob("validation*.csv"))
    if csv_files:
        return pd.read_csv(csv_files[0])
    
    raise FileNotFoundError(f"No validation results found in {validation_dir}")


def load_experiment_metadata(
    config_path: Union[str, Path] = "config.json"
) -> Dict[str, Any]:
    """
    Load experiment metadata from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with experiment configuration
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    metadata = {
        'protein_names': config.get('proteins', []),
        'scales': config.get('multiscale_analysis', {}).get('scales_um', []),
        'n_clusters': config.get('ion_count_processing', {}).get('n_clusters', 8),
        'bin_size': config.get('ion_count_processing', {}).get('bin_sizes_um', [20])[0],
        'output_dir': config.get('output', {}).get('results_dir', 'results')
    }
    
    return metadata
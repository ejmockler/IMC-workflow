"""
Data Adapter Layer for IMC Analysis

Provides clean interface between raw analysis outputs and visualization needs.
Implements lazy loading, smart caching, and on-demand calculations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from functools import lru_cache
import json

logger = logging.getLogger(__name__)


class IMCDataAdapter:
    """
    Adapter layer for accessing IMC analysis results.
    
    Provides efficient access to NPZ arrays and JSON metadata without
    creating redundant intermediate files. Uses caching for performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize adapter with configuration.
        
        Args:
            config: Configuration dictionary from config.json
        """
        self.config = config
        
        # Get paths from config
        output_config = config.get('output', {})
        results_dir_path = output_config.get('results_dir', 'results/cross_sectional_kidney_injury')
        
        # Handle relative paths - if results directory doesn't exist, try with ../ prefix
        self.results_dir = Path(results_dir_path)
        if not self.results_dir.exists():
            self.results_dir = Path('../') / results_dir_path
        
        self.roi_dir = self.results_dir / output_config.get('roi_results_dir', 'roi_results')
        
        # Get protein channels from config
        channels_config = config.get('channels', {})
        self.protein_channels = channels_config.get('protein_channels', [
            'CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b', 
            'CD31', 'CD34', 'CD206', 'CD44'
        ])
        
        # Cache for loaded data
        self._npz_cache = {}
        self._metadata_cache = {}
        
    @lru_cache(maxsize=32)
    def list_available_rois(self) -> List[str]:
        """Get list of available ROI IDs.
        
        Returns:
            List of ROI identifiers
        """
        npz_files = list(self.roi_dir.glob("*_arrays.npz"))
        roi_ids = [f.stem.replace('_arrays', '') for f in npz_files]
        return sorted(roi_ids)
    
    def load_roi_arrays(self, roi_id: str, scale: float = 20.0) -> Optional[Dict]:
        """Load spatial arrays from NPZ file.
        
        Args:
            roi_id: ROI identifier
            scale: Scale in micrometers
            
        Returns:
            Dictionary with spatial arrays or None if not found
        """
        cache_key = f"{roi_id}_{scale}"
        if cache_key in self._npz_cache:
            return self._npz_cache[cache_key]
        
        npz_file = self.roi_dir / f"{roi_id}_arrays.npz"
        if not npz_file.exists():
            return None
        
        try:
            data = np.load(npz_file)
            scale_key = f"scale_{scale}"
            
            result = {}
            # Load essential arrays
            for key in data.keys():
                if key.startswith(scale_key):
                    clean_key = key.replace(f'{scale_key}_', '')
                    result[clean_key] = data[key]
            
            self._npz_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error loading {npz_file}: {e}")
            return None
    
    def load_roi_metadata(self, roi_id: str) -> Optional[Dict]:
        """Load ROI metadata from JSON file.
        
        Args:
            roi_id: ROI identifier
            
        Returns:
            Dictionary with metadata or None if not found
        """
        if roi_id in self._metadata_cache:
            return self._metadata_cache[roi_id]
        
        json_file = self.roi_dir / f"{roi_id}_metadata.json"
        if not json_file.exists():
            return None
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            self._metadata_cache[roi_id] = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            return None
    
    def get_expression_matrix(self, roi_ids: Optional[List[str]] = None, 
                            scale: float = 20.0,
                            use_standardized: bool = False) -> pd.DataFrame:
        """Build expression matrix from NPZ files directly.
        
        Args:
            roi_ids: List of ROI IDs to include (None for all)
            scale: Scale to use for expression data
            use_standardized: Whether to use Z-scored arrays
            
        Returns:
            DataFrame with expression data
        """
        if roi_ids is None:
            roi_ids = self.list_available_rois()
        
        expression_data = []
        
        for roi_id in roi_ids:
            # Load arrays
            arrays = self.load_roi_arrays(roi_id, scale)
            if not arrays:
                continue
            
            # Load metadata
            metadata = self.load_roi_metadata(roi_id)
            if not metadata:
                continue
            
            # Get ROI info
            roi_info = metadata.get('roi_metadata', {})
            
            # Get feature matrix
            if 'feature_matrix' in arrays:
                feature_matrix = arrays['feature_matrix']
                
                # Calculate mean expression per ROI
                mean_expression = np.mean(feature_matrix, axis=0)
                
                # Create row with metadata and expression
                row = {
                    'roi_id': roi_id,
                    'condition': roi_info.get('condition', 'Unknown'),
                    'timepoint': roi_info.get('timepoint', 0),
                    'region': roi_info.get('region', 'Unknown'),
                    'replicate': roi_info.get('replicate_id', 'Unknown')
                }
                
                # Add protein expression values
                for i, protein in enumerate(self.protein_channels):
                    if i < len(mean_expression):
                        row[protein] = mean_expression[i]
                
                expression_data.append(row)
        
        return pd.DataFrame(expression_data)
    
    def get_spatial_data(self, roi_id: str, scale: float = 20.0) -> Optional[Dict]:
        """Get spatial data for visualization.
        
        Args:
            roi_id: ROI identifier
            scale: Scale in micrometers
            
        Returns:
            Dictionary with spatial arrays and metadata
        """
        arrays = self.load_roi_arrays(roi_id, scale)
        metadata = self.load_roi_metadata(roi_id)
        
        if not arrays or not metadata:
            return None
        
        # Get scale-specific metadata
        scale_metadata = metadata.get('multiscale_metadata', {}).get(f'scale_{scale}', {})
        
        return {
            'arrays': arrays,
            'metadata': metadata.get('roi_metadata', {}),
            'scale_metadata': scale_metadata,
            'cluster_centroids': scale_metadata.get('cluster_centroids', {}),
            'optimization_results': scale_metadata.get('optimization_results', {})
        }
    
    def calculate_spatial_metrics(self, roi_id: str, params: Dict = None) -> Dict:
        """Calculate spatial metrics on-demand.
        
        Args:
            roi_id: ROI identifier
            params: Parameters for metric calculation
            
        Returns:
            Dictionary with calculated metrics
        """
        params = params or {}
        scale = params.get('scale', 20.0)
        
        data = self.get_spatial_data(roi_id, scale)
        if not data:
            return {}
        
        arrays = data['arrays']
        metrics = {}
        
        # Calculate basic spatial metrics
        if 'cluster_labels' in arrays:
            labels = arrays['cluster_labels']
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            metrics['n_clusters'] = len(unique_labels)
            metrics['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
            metrics['cluster_entropy'] = -np.sum(
                (counts/counts.sum()) * np.log(counts/counts.sum() + 1e-10)
            )
        
        # Calculate spatial heterogeneity
        if 'feature_matrix' in arrays:
            feature_matrix = arrays['feature_matrix']
            metrics['spatial_variance'] = np.mean(np.var(feature_matrix, axis=0))
            metrics['spatial_cv'] = np.mean(
                np.std(feature_matrix, axis=0) / (np.mean(feature_matrix, axis=0) + 1e-10)
            )
        
        return metrics
    
    def get_network_data(self, roi_ids: Optional[List[str]] = None,
                        threshold: float = 0.5) -> Dict:
        """Build protein correlation networks dynamically.
        
        Args:
            roi_ids: ROI IDs to include
            threshold: Correlation threshold for edges
            
        Returns:
            Dictionary with network data
        """
        # Get expression matrix
        expr_df = self.get_expression_matrix(roi_ids)
        
        if expr_df.empty:
            return {}
        
        # Calculate correlations
        protein_cols = [col for col in self.protein_channels if col in expr_df.columns]
        corr_matrix = expr_df[protein_cols].corr()
        
        # Build network structure
        edges = []
        for i, prot1 in enumerate(protein_cols):
            for j, prot2 in enumerate(protein_cols[i+1:], i+1):
                corr = corr_matrix.loc[prot1, prot2]
                if abs(corr) > threshold:
                    edges.append({
                        'source': prot1,
                        'target': prot2,
                        'weight': abs(corr),
                        'correlation': corr
                    })
        
        return {
            'nodes': protein_cols,
            'edges': edges,
            'correlation_matrix': corr_matrix.to_dict()
        }
    
    def get_temporal_trajectories(self, condition: str = None,
                                 region: str = None) -> pd.DataFrame:
        """Get temporal expression trajectories.
        
        Args:
            condition: Filter by condition
            region: Filter by region
            
        Returns:
            DataFrame with temporal data
        """
        expr_df = self.get_expression_matrix()
        
        # Apply filters
        if condition:
            expr_df = expr_df[expr_df['condition'] == condition]
        if region:
            expr_df = expr_df[expr_df['region'] == region]
        
        # Group by timepoint and calculate statistics
        protein_cols = [col for col in self.protein_channels if col in expr_df.columns]
        
        temporal_stats = expr_df.groupby('timepoint')[protein_cols].agg(['mean', 'std', 'sem'])
        
        return temporal_stats
    
    def get_statistical_comparisons(self, groups: Dict[str, List[str]],
                                  metric: str = 'mean') -> pd.DataFrame:
        """Perform statistical comparisons between groups.
        
        Args:
            groups: Dictionary mapping group names to ROI lists
            metric: Metric to compare
            
        Returns:
            DataFrame with statistical results
        """
        from scipy import stats
        
        results = []
        
        for protein in self.protein_channels:
            group_values = {}
            
            for group_name, roi_list in groups.items():
                expr_df = self.get_expression_matrix(roi_list)
                if protein in expr_df.columns:
                    group_values[group_name] = expr_df[protein].values
            
            # Perform pairwise comparisons
            group_names = list(group_values.keys())
            for i, group1 in enumerate(group_names):
                for group2 in group_names[i+1:]:
                    if group1 in group_values and group2 in group_values:
                        statistic, pvalue = stats.mannwhitneyu(
                            group_values[group1], 
                            group_values[group2]
                        )
                        
                        results.append({
                            'protein': protein,
                            'group1': group1,
                            'group2': group2,
                            'statistic': statistic,
                            'pvalue': pvalue,
                            'mean_diff': np.mean(group_values[group1]) - np.mean(group_values[group2])
                        })
        
        return pd.DataFrame(results)
    
    def clear_cache(self):
        """Clear all cached data."""
        self._npz_cache.clear()
        self._metadata_cache.clear()
        self.list_available_rois.cache_clear()
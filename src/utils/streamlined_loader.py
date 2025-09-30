"""
Streamlined IMC Data Loader - Single Source of Truth

Combines all data loading functionality into one clean, efficient interface.
Replaces the multiple competing loaders that caused the plotting issues.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from functools import lru_cache
from scipy import stats

logger = logging.getLogger(__name__)


class StreamlinedIMCLoader:
    """
    Single, streamlined data loader for IMC analysis.
    
    Provides all functionality needed for visualization and analysis:
    - Metadata and expression matrices
    - Spatial array loading  
    - Statistical comparisons
    - Network analysis
    - Quality metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        
        # Get paths
        output_config = config.get('output', {})
        results_dir_path = output_config.get('results_dir', 'results/cross_sectional_kidney_injury')
        
        self.results_dir = Path(results_dir_path)
        if not self.results_dir.exists():
            self.results_dir = Path('../') / results_dir_path
            
        self.roi_dir = self.results_dir / output_config.get('roi_results_dir', 'roi_results')
        
        # Get protein channels
        channels_config = config.get('channels', {})
        self.protein_channels = channels_config.get('protein_channels', [
            'CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b', 
            'CD31', 'CD34', 'CD206', 'CD44'
        ])
        
        # Initialize data containers
        self._metadata_df = None
        self._expression_df = None
        self._quality_df = None
        self._roi_cache = {}
        
    @property 
    def metadata_df(self) -> pd.DataFrame:
        """Get metadata DataFrame (cached)."""
        if self._metadata_df is None:
            self._load_all_data()
        return self._metadata_df.copy() if self._metadata_df is not None else pd.DataFrame()
    
    @property
    def expression_df(self) -> pd.DataFrame:
        """Get expression DataFrame (cached)."""
        if self._expression_df is None:
            self._load_all_data()
        return self._expression_df.copy() if self._expression_df is not None else pd.DataFrame()
    
    @property 
    def quality_df(self) -> pd.DataFrame:
        """Get quality DataFrame (cached)."""
        if self._quality_df is None:
            self._load_all_data()
        return self._quality_df.copy() if self._quality_df is not None else pd.DataFrame()
    
    def _load_all_data(self):
        """Load all ROI data from JSON files."""
        json_files = list(self.roi_dir.glob("*_metadata.json"))
        
        if not json_files:
            logger.warning(f"No metadata files found in {self.roi_dir}")
            self._metadata_df = pd.DataFrame()
            self._expression_df = pd.DataFrame()
            self._quality_df = pd.DataFrame()
            return
        
        metadata_list = []
        expression_list = []
        quality_list = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                roi_id = json_file.stem.replace('_metadata', '')
                
                # Extract metadata
                roi_metadata = data.get('roi_metadata', {})
                metadata_row = {
                    'roi_id': roi_id,
                    'condition': roi_metadata.get('condition', 'Unknown'),
                    'injury_day': roi_metadata.get('timepoint', 0),
                    'mouse': roi_metadata.get('replicate_id', 'Unknown'),
                    'region': roi_metadata.get('region', 'Unknown'),
                    'file_name': roi_metadata.get('filename', roi_id),
                    'timepoint': roi_metadata.get('timepoint', 0),  # Standardized name
                    'replicate': roi_metadata.get('replicate_id', 'Unknown')
                }
                
                # Add batch info if available
                if 'batch_id' in data:
                    metadata_row['batch'] = data['batch_id']
                
                metadata_list.append(metadata_row)
                
                # Extract expression data from multiscale metadata
                multiscale = data.get('multiscale_metadata', {})
                if multiscale:
                    expression_row = self._extract_expression_data(roi_id, multiscale, metadata_row)
                    if expression_row:
                        expression_list.append(expression_row)
                
                # Extract quality data
                quality_row = self._extract_quality_data(roi_id, multiscale, metadata_row)
                if quality_row:
                    quality_list.append(quality_row)
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
        
        # Create DataFrames
        self._metadata_df = pd.DataFrame(metadata_list)
        self._expression_df = pd.DataFrame(expression_list)
        self._quality_df = pd.DataFrame(quality_list)
        
        # Convert types
        if not self._metadata_df.empty:
            self._metadata_df['injury_day'] = pd.to_numeric(
                self._metadata_df['injury_day'], errors='coerce'
            ).fillna(0).astype(int)
            
            self._metadata_df['timepoint'] = self._metadata_df['injury_day']
        
        logger.info(f"Loaded {len(self._metadata_df)} ROIs")
        
    def _extract_expression_data(self, roi_id: str, multiscale: Dict, metadata: Dict) -> Optional[Dict]:
        """Extract expression data from multiscale metadata."""
        # Get preferred scale (20μm)
        scales = self.config.get('segmentation', {}).get('scales_um', [10.0, 20.0, 40.0])
        preferred_scale = scales[1] if len(scales) > 1 else 20.0
        
        scale_data = None
        
        # Handle different multiscale formats
        if isinstance(multiscale, dict):
            scale_key = f'scale_{preferred_scale}'
            if scale_key in multiscale:
                scale_data = multiscale[scale_key]
            else:
                # Get first available scale
                for key, val in multiscale.items():
                    if isinstance(val, dict):
                        scale_data = val
                        break
        elif isinstance(multiscale, list) and multiscale:
            # Find matching scale or use first
            for sd in multiscale:
                if isinstance(sd, dict) and sd.get('scale_um') == preferred_scale:
                    scale_data = sd
                    break
            if not scale_data:
                scale_data = multiscale[0] if isinstance(multiscale[0], dict) else None
        
        if not scale_data:
            return None
        
        # Build expression row
        row = {
            'roi_id': roi_id,
            **{k: v for k, v in metadata.items() if k != 'roi_id'}
        }
        
        # Extract cluster centroids (mean expression per cluster)
        if 'cluster_centroids' in scale_data:
            cluster_centroids = scale_data['cluster_centroids']
            # cluster_centroids is dict: cluster_id -> {protein -> value}
            for protein in self.protein_channels:
                values = []
                for cluster_id, centroid in cluster_centroids.items():
                    if isinstance(centroid, dict) and protein in centroid:
                        values.append(centroid[protein])
                row[protein] = np.mean(values) if values else 0.0
        else:
            # No cluster centroids available - set to 0
            for protein in self.protein_channels:
                row[protein] = 0.0
        
        return row
    
    def _extract_quality_data(self, roi_id: str, multiscale: Dict, metadata: Dict) -> Optional[Dict]:
        """Extract quality metrics from multiscale data."""
        if not multiscale:
            return None
        
        # Get first scale with quality data
        scale_data = None
        if isinstance(multiscale, dict):
            for key, val in multiscale.items():
                if isinstance(val, dict):
                    scale_data = val
                    break
        elif isinstance(multiscale, list) and multiscale:
            scale_data = multiscale[0] if isinstance(multiscale[0], dict) else None
        
        if not scale_data:
            return None
        
        row = {
            'roi_id': roi_id,
            'condition': metadata.get('condition', 'Unknown'),
            'injury_day': metadata.get('injury_day', 0),
            'region': metadata.get('region', 'Unknown'),
            'coordinate_quality': scale_data.get('coordinate_quality', 0.0),
            'ion_count_quality': scale_data.get('ion_count_quality', 0.0),
            'biological_quality': scale_data.get('biological_quality', 0.0),
            'overall_quality': scale_data.get('overall_quality', 0.0)
        }
        
        # Add specific quality metrics if available
        if 'quality_metrics' in scale_data:
            qm = scale_data['quality_metrics']
            row.update({
                'signal_to_noise': qm.get('signal_to_noise', 0.0),
                'dynamic_range': qm.get('dynamic_range', 0.0),
                'tissue_coverage': qm.get('tissue_coverage', 0.0)
            })
        
        return row
    
    @lru_cache(maxsize=32)
    def list_available_rois(self) -> List[str]:
        """Get list of available ROI IDs."""
        return list(self.metadata_df['roi_id'].values) if not self.metadata_df.empty else []
    
    def load_spatial_arrays(self, roi_id: str, scale: float = 20.0) -> Optional[Dict]:
        """Load spatial arrays from NPZ file."""
        cache_key = f"{roi_id}_{scale}"
        if cache_key in self._roi_cache:
            return self._roi_cache[cache_key]
        
        npz_file = self.roi_dir / f"{roi_id}_arrays.npz"
        if not npz_file.exists():
            return None
        
        try:
            data = np.load(npz_file)
            scale_key = f"scale_{scale}"
            
            result = {}
            for key in data.keys():
                if key.startswith(scale_key):
                    clean_key = key.replace(f'{scale_key}_', '')
                    result[clean_key] = data[key]
            
            self._roi_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error loading spatial arrays for {roi_id}: {e}")
            return None
    
    def get_protein_correlations(self, condition: str = None) -> pd.DataFrame:
        """Get protein correlation matrix."""
        expr_df = self.expression_df
        
        if condition:
            expr_df = expr_df[expr_df['condition'] == condition]
        
        protein_cols = [col for col in self.protein_channels if col in expr_df.columns]
        
        if len(protein_cols) < 2:
            return pd.DataFrame()
        
        return expr_df[protein_cols].corr()
    
    def get_temporal_trajectories(self, condition: str = None) -> pd.DataFrame:
        """Get temporal expression trajectories."""
        expr_df = self.expression_df
        
        if condition:
            expr_df = expr_df[expr_df['condition'] == condition]
        
        protein_cols = [col for col in self.protein_channels if col in expr_df.columns]
        
        if expr_df.empty or not protein_cols:
            return pd.DataFrame()
        
        # Group by timepoint and calculate stats
        results = []
        for timepoint in sorted(expr_df['timepoint'].unique()):
            tp_data = expr_df[expr_df['timepoint'] == timepoint]
            
            for protein in protein_cols:
                values = tp_data[protein].dropna()
                if len(values) > 0:
                    results.append({
                        'timepoint': timepoint,
                        'protein': protein,
                        'condition': condition or 'All',
                        'mean': values.mean(),
                        'sem': values.sem(),
                        'count': len(values)
                    })
        
        return pd.DataFrame(results)
    
    def perform_statistical_tests(self, protein: str, group_col: str = 'condition') -> Dict:
        """Perform statistical tests between groups."""
        expr_df = self.expression_df
        
        if protein not in expr_df.columns or group_col not in expr_df.columns:
            return {}
        
        groups = expr_df[group_col].unique()
        if len(groups) < 2:
            return {}
        
        results = {}
        
        # Pairwise comparisons
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                data1 = expr_df[expr_df[group_col] == group1][protein].dropna()
                data2 = expr_df[expr_df[group_col] == group2][protein].dropna()
                
                if len(data1) > 0 and len(data2) > 0:
                    try:
                        statistic, pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        
                        results[f"{group1}_vs_{group2}"] = {
                            'statistic': statistic,
                            'pvalue': pvalue,
                            'mean1': data1.mean(),
                            'mean2': data2.mean(),
                            'fold_change': data1.mean() / (data2.mean() + 1e-10)
                        }
                    except Exception as e:
                        logger.warning(f"Statistical test failed for {protein} {group1} vs {group2}: {e}")
        
        return results
    
    def calculate_spatial_metrics(self, roi_id: str, scale: float = 20.0) -> Dict:
        """Calculate basic spatial metrics for an ROI."""
        arrays = self.load_spatial_arrays(roi_id, scale)
        
        if not arrays:
            return {}
        
        metrics = {}
        
        # Basic clustering metrics
        if 'cluster_labels' in arrays:
            labels = arrays['cluster_labels']
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            metrics['n_clusters'] = len(unique_labels)
            metrics['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
            
            # Calculate entropy
            if len(counts) > 1:
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                metrics['cluster_entropy'] = entropy
            else:
                metrics['cluster_entropy'] = 0.0
        
        # Feature matrix metrics
        if 'feature_matrix' in arrays:
            feature_matrix = arrays['feature_matrix']
            metrics['spatial_variance'] = np.mean(np.var(feature_matrix, axis=0))
            
            means = np.mean(feature_matrix, axis=0)
            stds = np.std(feature_matrix, axis=0)
            cv = stds / (means + 1e-10)
            metrics['spatial_cv'] = np.mean(cv)
        
        return metrics
    
    def get_summary_stats(self) -> Dict:
        """Get comprehensive summary statistics."""
        stats = {
            'n_rois': len(self.metadata_df),
            'conditions': self.metadata_df['condition'].value_counts().to_dict() if not self.metadata_df.empty else {},
            'timepoints': sorted(self.metadata_df['timepoint'].unique()) if not self.metadata_df.empty else [],
            'regions': self.metadata_df['region'].value_counts().to_dict() if not self.metadata_df.empty else {},
            'n_proteins': len([p for p in self.protein_channels if p in self.expression_df.columns]),
            'protein_channels': self.protein_channels
        }
        
        return stats
    
    def clear_cache(self):
        """Clear all caches."""
        self._roi_cache.clear()
        self.list_available_rois.cache_clear()
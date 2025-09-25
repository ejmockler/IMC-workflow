"""Loader for IMC analysis results from JSON files."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class IMCResultsLoader:
    """Load and organize IMC analysis results from ROI JSON files."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the results loader with configuration.
        
        Args:
            config: Configuration dictionary from config.json
        """
        self.config = config
        
        # Get paths from config
        output_config = config.get('output', {})
        self.results_dir = Path(output_config.get('results_dir', 'results/cross_sectional_kidney_injury'))
        self.roi_dir = self.results_dir / output_config.get('roi_results_dir', 'roi_results')
        
        # Get protein channels from config
        channels_config = config.get('channels', {})
        self.protein_channels = channels_config.get('protein_channels', [
            'CD45', 'CD11b', 'Ly6G', 'CD140a', 'CD140b', 
            'CD31', 'CD34', 'CD206', 'CD44'
        ])
        
        # Initialize data containers
        self.roi_data = {}
        self.metadata_df = None
        self.expression_df = None
        self.quality_df = None
        
        # Load all data
        self._load_all_roi_data()
        if self.roi_data:
            self._create_metadata_df()
            self._create_expression_matrix()
            self._create_quality_df()
    
    def _load_all_roi_data(self):
        """Load all ROI JSON files from results directory."""
        json_pattern = "*_metadata.json"
        json_files = list(self.roi_dir.glob(json_pattern))
        
        if not json_files:
            logger.warning(f"No ROI metadata files found in {self.roi_dir}")
            return
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                roi_id = json_file.stem.replace('_metadata', '')
                self.roi_data[roi_id] = data
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.roi_data)} ROI files")
    
    def _create_metadata_df(self):
        """Create metadata DataFrame from loaded ROI data."""
        metadata_list = []
        
        for roi_id, data in self.roi_data.items():
            if 'roi_metadata' not in data:
                continue
                
            metadata = data['roi_metadata']
            
            # Extract key metadata fields using config mappings
            tracking = self.config.get('metadata_tracking', {})
            
            row = {
                'roi_id': roi_id,
                'condition': metadata.get(
                    tracking.get('condition_column', 'Condition'), 'Unknown'
                ),
                'injury_day': metadata.get(
                    tracking.get('timepoint_column', 'Injury Day'), 0
                ),
                'mouse': metadata.get(
                    tracking.get('replicate_column', 'Mouse'), 'Unknown'
                ),
                'region': metadata.get(
                    tracking.get('region_column', 'Details'), 'Unknown'
                ),
                'file_name': metadata.get('File Name', roi_id),
                'roi_number': metadata.get('ROI', 0)
            }
            
            # Add batch info if available
            if 'batch_id' in data:
                row['batch'] = data['batch_id']
            
            metadata_list.append(row)
        
        self.metadata_df = pd.DataFrame(metadata_list)
        
        # Convert types
        self.metadata_df['injury_day'] = pd.to_numeric(
            self.metadata_df['injury_day'], errors='coerce'
        ).fillna(0).astype(int)
        
        logger.info(f"Created metadata DataFrame with {len(self.metadata_df)} rows")
    
    def _create_expression_matrix(self):
        """Create expression matrix from multiscale metadata."""
        expression_data = []
        
        for roi_id, data in self.roi_data.items():
            if 'multiscale_metadata' not in data:
                continue
            
            multiscale = data['multiscale_metadata']
            
            # Get the preferred scale from config or default to 20μm
            scales = self.config.get('segmentation', {}).get('scales_um', [10.0, 20.0, 40.0])
            preferred_scale = scales[1] if len(scales) > 1 else 20.0
            
            # Find the preferred scale data
            scale_data = None
            if isinstance(multiscale, list):
                for sd in multiscale:
                    if isinstance(sd, dict) and sd.get('scale_um') == preferred_scale:
                        scale_data = sd
                        break
            elif isinstance(multiscale, dict):
                # Handle dict format with scale keys
                scale_key = f'scale_{preferred_scale}'
                if scale_key in multiscale:
                    scale_data = multiscale[scale_key]
            
            # Fallback to first available scale
            if not scale_data:
                if isinstance(multiscale, list) and multiscale:
                    scale_data = multiscale[0] if isinstance(multiscale[0], dict) else None
                elif isinstance(multiscale, dict):
                    # Get first scale from dict
                    for key, val in multiscale.items():
                        if isinstance(val, dict):
                            scale_data = val
                            break
            
            if not scale_data:
                continue
            
            # Extract cluster means for each protein
            row = {'roi_id': roi_id}
            
            if 'cluster_means' in scale_data:
                cluster_means = scale_data['cluster_means']
                
                for protein in self.protein_channels:
                    if protein in cluster_means:
                        # Get all cluster values
                        values = []
                        for cluster_id, value in cluster_means[protein].items():
                            if value is not None:
                                values.append(value)
                        
                        # Use mean across clusters
                        if values:
                            row[protein] = np.mean(values)
                        else:
                            row[protein] = 0.0
                    else:
                        row[protein] = 0.0
            else:
                # No cluster means, use zeros
                for protein in self.protein_channels:
                    row[protein] = 0.0
            
            expression_data.append(row)
        
        if expression_data:
            self.expression_df = pd.DataFrame(expression_data)
            
            # Merge with metadata
            if self.metadata_df is not None:
                self.expression_df = self.expression_df.merge(
                    self.metadata_df, on='roi_id', how='left'
                )
            
            logger.info(f"Created expression matrix: {self.expression_df.shape}")
        else:
            # Create empty DataFrame with expected columns
            cols = ['roi_id'] + self.protein_channels
            self.expression_df = pd.DataFrame(columns=cols)
            logger.warning("No expression data found in ROI files")
    
    def _create_quality_df(self):
        """Extract quality scores from ROI data."""
        quality_data = []
        
        for roi_id, data in self.roi_data.items():
            if 'multiscale_metadata' not in data:
                continue
            
            multiscale = data['multiscale_metadata']
            scale_data = None
            
            # Get quality metrics from first available scale
            if isinstance(multiscale, list) and multiscale:
                scale_data = multiscale[0] if isinstance(multiscale[0], dict) else None
            elif isinstance(multiscale, dict):
                # Get first scale from dict
                for key, val in multiscale.items():
                    if isinstance(val, dict):
                        scale_data = val
                        break
            
            if scale_data:
                
                row = {
                    'roi_id': roi_id,
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
                
                quality_data.append(row)
        
        if quality_data:
            self.quality_df = pd.DataFrame(quality_data)
            
            # Merge with metadata
            if self.metadata_df is not None:
                self.quality_df = self.quality_df.merge(
                    self.metadata_df[['roi_id', 'condition', 'injury_day', 'region']], 
                    on='roi_id', how='left'
                )
            
            logger.info(f"Created quality DataFrame with {len(self.quality_df)} rows")
        else:
            self.quality_df = pd.DataFrame()
            logger.warning("No quality data found in ROI files")
    
    def get_metadata(self) -> pd.DataFrame:
        """Get metadata DataFrame."""
        if self.metadata_df is None:
            return pd.DataFrame()
        return self.metadata_df.copy()
    
    def get_expression_matrix(self) -> pd.DataFrame:
        """Get expression matrix DataFrame."""
        if self.expression_df is None:
            return pd.DataFrame()
        return self.expression_df.copy()
    
    def get_quality_scores(self) -> pd.DataFrame:
        """Get quality scores DataFrame."""
        if self.quality_df is None:
            return pd.DataFrame()
        return self.quality_df.copy()
    
    def get_temporal_data(self) -> pd.DataFrame:
        """Get expression data organized by timepoint."""
        if self.expression_df is None or self.expression_df.empty:
            return pd.DataFrame()
        
        # Identify columns to use
        id_vars = ['roi_id', 'condition', 'injury_day', 'mouse', 'region']
        id_vars = [col for col in id_vars if col in self.expression_df.columns]
        
        value_vars = [col for col in self.protein_channels 
                     if col in self.expression_df.columns]
        
        if not value_vars or not id_vars:
            return pd.DataFrame()
        
        # Melt for temporal analysis
        temporal_df = self.expression_df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='protein',
            value_name='expression'
        )
        
        return temporal_df
    
    def get_spatial_neighborhoods(self, roi_id: str) -> Optional[Dict]:
        """Get spatial neighborhood data for a specific ROI.
        
        Args:
            roi_id: ROI identifier
            
        Returns:
            Dictionary with neighborhood data or None if not found
        """
        if roi_id not in self.roi_data:
            return None
        
        data = self.roi_data[roi_id]
        if 'multiscale_metadata' not in data:
            return None
        
        # Extract neighborhood information
        neighborhoods = {}
        for scale_data in data['multiscale_metadata']:
            scale = scale_data.get('scale_um', 0)
            
            neighborhood_data = {
                'scale_um': scale,
                'n_clusters': scale_data.get('n_clusters', 0),
                'consistency_score': scale_data.get('consistency_score', 0.0)
            }
            
            # Add cluster means if available
            if 'cluster_means' in scale_data:
                neighborhood_data['cluster_means'] = scale_data['cluster_means']
            
            # Add cluster sizes if available
            if 'cluster_sizes' in scale_data:
                neighborhood_data['cluster_sizes'] = scale_data['cluster_sizes']
            
            neighborhoods[f'scale_{scale}um'] = neighborhood_data
        
        return neighborhoods if neighborhoods else None
    
    def get_multiscale_consistency(self) -> pd.DataFrame:
        """Get multi-scale consistency scores for all ROIs."""
        consistency_data = []
        
        for roi_id, data in self.roi_data.items():
            if 'multiscale_metadata' not in data:
                continue
            
            row = {'roi_id': roi_id}
            
            for scale_data in data['multiscale_metadata']:
                scale = scale_data.get('scale_um', 0)
                consistency = scale_data.get('consistency_score', 0.0)
                row[f'consistency_{scale}um'] = consistency
            
            consistency_data.append(row)
        
        if consistency_data:
            consistency_df = pd.DataFrame(consistency_data)
            
            # Merge with metadata
            if self.metadata_df is not None:
                consistency_df = consistency_df.merge(
                    self.metadata_df[['roi_id', 'condition', 'injury_day']], 
                    on='roi_id', how='left'
                )
            
            return consistency_df
        
        return pd.DataFrame()
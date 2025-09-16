"""
PFD Pipeline Orchestration

Main entry point for running the complete PFD analysis.
Reads config, processes ROIs, extracts features, saves results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import warnings

from src.config import Config
from src.utils.data_loader import load_roi_data
from .pfd import create_all_protein_fields, extract_roi_features
from .spatial_stats import compute_spatial_correlation
from .threshold_analysis import extract_threshold_features


class PFDPipeline:
    """Main pipeline for Protein Field Dynamics analysis."""
    
    def __init__(self, config: Config):
        """Initialize pipeline with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.results = []
        
    def run_analysis(self) -> pd.DataFrame:
        """
        Run complete PFD analysis on all ROIs.
        
        Returns:
            DataFrame with one row per ROI and extracted features as columns
        """
        print("=" * 60)
        print("Protein Field Dynamics (PFD) Analysis Pipeline")
        print("=" * 60)
        
        # Load metadata
        metadata_df = self._load_metadata()
        print(f"Loaded metadata for {len(metadata_df)} ROIs")
        
        # Get PFD parameters from config
        pfd_params = self.config.raw.get('pfd', {})
        field_params = pfd_params.get('field_modeling', {})
        analysis_params = pfd_params.get('feature_extraction', {})
        
        # Process each ROI
        for idx, row in metadata_df.iterrows():
            roi_name = row.get('File Name', f'ROI_{idx}')
            print(f"\nProcessing {roi_name}...")
            
            try:
                # Load ROI data
                roi_path = self._get_roi_path(row)
                coords, values, protein_names = load_roi_data(roi_path)
                
                # Create protein data dictionary
                protein_data = {
                    protein: values[:, i] 
                    for i, protein in enumerate(protein_names)
                }
                
                # Create continuous fields
                print(f"  Creating protein fields ({len(protein_names)} proteins)...")
                fields = create_all_protein_fields(
                    coords, 
                    protein_data,
                    resolution_um=field_params.get('resolution_um', 1.0),
                    method=field_params.get('method', 'gaussian_kde')
                )
                
                # Extract features
                print("  Extracting ROI features...")
                correlation_pairs = analysis_params.get('correlation_pairs', [])
                features = extract_roi_features(
                    fields,
                    protein_pairs=correlation_pairs,
                    hotspot_percentile=analysis_params.get('hotspot_threshold_percentile', 95)
                )
                
                # Add metadata to features
                features['roi_name'] = roi_name
                features['timepoint'] = row.get('Injury Day', 'Unknown')
                features['region'] = row.get('Details', 'Unknown')  
                features['replicate'] = row.get('Mouse', 'Unknown')
                features['condition'] = row.get('Condition', 'Unknown')
                
                # Store result
                self.results.append(features)
                print(f"  Extracted {len(features)} features")
                
            except Exception as e:
                print(f"  ERROR processing {roi_name}: {e}")
                # Store minimal result with error flag
                self.results.append({
                    'roi_name': roi_name,
                    'error': str(e),
                    'timepoint': row.get('Injury Day', 'Unknown'),
                    'region': row.get('Details', 'Unknown'),
                    'replicate': row.get('Mouse', 'Unknown')
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save results
        self._save_results(results_df)
        
        # Print summary
        self._print_summary(results_df)
        
        return results_df
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata from configured path."""
        metadata_path = self.config.raw.get('metadata', {}).get('path')
        if not metadata_path:
            # Try experimental metadata
            metadata_path = self.config.raw.get('experimental', {}).get('metadata_file')
        
        if not metadata_path:
            raise ValueError("No metadata path configured")
        
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        return pd.read_csv(metadata_path)
    
    def _get_roi_path(self, metadata_row: pd.Series) -> Path:
        """Get ROI data file path from metadata row."""
        # Try different path patterns
        roi_pattern = self.config.raw.get('metadata', {}).get('roi_path_pattern')
        
        if roi_pattern:
            # Use configured pattern
            roi_path = roi_pattern.format(**metadata_row.to_dict())
            return Path(roi_path)
        
        # Default: look in data directory
        data_dir = self.config.data_dir
        file_name = metadata_row.get('File Name', '')
        
        # Try with .txt extension
        roi_path = data_dir / f"{file_name}.txt"
        if roi_path.exists():
            return roi_path
        
        # Try without extension
        roi_path = data_dir / file_name
        if roi_path.exists():
            return roi_path
        
        raise FileNotFoundError(f"ROI file not found for {file_name}")
    
    def _save_results(self, results_df: pd.DataFrame):
        """Save results to configured output directory."""
        output_config = self.config.raw.get('output', {})
        results_dir = Path(output_config.get('results_dir', 'results/pfd'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature matrix as CSV
        feature_file = results_dir / output_config.get('feature_matrix', 'roi_features.csv')
        results_df.to_csv(feature_file, index=False)
        print(f"\nSaved feature matrix to {feature_file}")
        
        # Save as JSON for detailed inspection
        json_file = results_dir / 'roi_features.json'
        results_df.to_json(json_file, orient='records', indent=2)
        print(f"Saved JSON results to {json_file}")
    
    def _print_summary(self, results_df: pd.DataFrame):
        """Print analysis summary."""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Basic statistics
        n_rois = len(results_df)
        n_successful = len(results_df[~results_df.get('error', pd.Series()).notna()])
        
        print(f"Total ROIs processed: {n_rois}")
        print(f"Successful: {n_successful}")
        print(f"Failed: {n_rois - n_successful}")
        
        if n_successful > 0:
            # Group by experimental variables
            print("\nROIs by condition:")
            for condition, group in results_df.groupby('timepoint'):
                print(f"  {condition}: {len(group)} ROIs")
            
            print("\nROIs by region:")
            for region, group in results_df.groupby('region'):
                print(f"  {region}: {len(group)} ROIs")
            
            # Feature statistics
            feature_cols = [c for c in results_df.columns 
                          if c.startswith(('mean_', 'std_', 'hotspot_', 'correlation_'))]
            print(f"\nExtracted {len(feature_cols)} features per ROI")
        
        print("\n" + "=" * 60)
        print("CRITICAL LIMITATIONS (n=2 pilot study)")
        print("=" * 60)
        print("- This is a PILOT STUDY with n=2 biological replicates")
        print("- All findings are DESCRIPTIVE and HYPOTHESIS-GENERATING")
        print("- No statistical significance testing performed")
        print("- Focus on effect sizes and visualization")
        print("=" * 60)


def run_pfd_analysis(config_path: str = 'config.json') -> pd.DataFrame:
    """
    Main entry point for PFD analysis.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DataFrame with ROI features
    """
    config = Config(config_path)
    pipeline = PFDPipeline(config)
    return pipeline.run_analysis()
"""
ROI-specific test fixtures and utilities for IMC analysis testing.

Provides realistic ROI data patterns, edge cases, and validation utilities
for testing ROI-based analysis components.
"""

import numpy as np
import pytest
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .synthetic_data import SyntheticIMCDataGenerator, SyntheticDataConfig, ClusterPattern


class ROIQuality(Enum):
    """ROI quality levels for testing different data conditions."""
    EXCELLENT = "excellent"    # High-quality data, all metrics pass
    GOOD = "good"             # Good data with minor issues
    POOR = "poor"             # Low-quality data with significant issues
    CORRUPTED = "corrupted"   # Severely corrupted data


@dataclass
class ROIMetadata:
    """Metadata associated with ROI test data."""
    roi_id: str
    batch_id: str
    quality_level: ROIQuality
    tissue_type: str
    acquisition_date: str
    pixel_size_um: float
    n_channels: int
    expected_issues: List[str]
    notes: str


class ROITestDataGenerator:
    """Generate ROI test data with specific characteristics."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def create_excellent_roi(self, roi_id: str = "ROI_excellent") -> Tuple[Dict[str, Any], ROIMetadata]:
        """Create high-quality ROI data with no issues."""
        config = SyntheticDataConfig(
            n_points=2000,
            n_proteins=9,
            n_clusters=4,
            cluster_pattern=ClusterPattern.FOUR_CORNERS,
            noise_level=0.05,  # Low noise
            protein_expression_variance=0.2,  # Low variance
            random_seed=self.random_seed
        )
        
        generator = SyntheticIMCDataGenerator(config)
        roi_data = generator.generate_complete_dataset()
        
        metadata = ROIMetadata(
            roi_id=roi_id,
            batch_id="test_batch_01",
            quality_level=ROIQuality.EXCELLENT,
            tissue_type="kidney",
            acquisition_date="2025-09-29",
            pixel_size_um=1.0,
            n_channels=len(roi_data['protein_names']),
            expected_issues=[],
            notes="High-quality synthetic ROI for positive control testing"
        )
        
        return roi_data, metadata
    
    def create_good_roi(self, roi_id: str = "ROI_good") -> Tuple[Dict[str, Any], ROIMetadata]:
        """Create good-quality ROI data with minor issues."""
        config = SyntheticDataConfig(
            n_points=1500,
            n_proteins=8,
            n_clusters=3,
            cluster_pattern=ClusterPattern.GRADIENT,
            noise_level=0.15,  # Moderate noise
            protein_expression_variance=0.35,
            random_seed=self.random_seed
        )
        
        generator = SyntheticIMCDataGenerator(config)
        roi_data = generator.generate_complete_dataset()
        
        # Add some missing protein data (realistic issue)
        if 'CD206' in roi_data['ion_counts']:
            missing_indices = np.random.choice(
                roi_data['n_measurements'], 
                size=int(0.05 * roi_data['n_measurements']),
                replace=False
            )
            roi_data['ion_counts']['CD206'][missing_indices] = 0
        
        metadata = ROIMetadata(
            roi_id=roi_id,
            batch_id="test_batch_02",
            quality_level=ROIQuality.GOOD,
            tissue_type="liver",
            acquisition_date="2025-09-28",
            pixel_size_um=1.0,
            n_channels=len(roi_data['protein_names']),
            expected_issues=["minor_missing_data", "moderate_noise"],
            notes="Good quality ROI with minor data gaps"
        )
        
        return roi_data, metadata
    
    def create_poor_roi(self, roi_id: str = "ROI_poor") -> Tuple[Dict[str, Any], ROIMetadata]:
        """Create poor-quality ROI data with significant issues."""
        config = SyntheticDataConfig(
            n_points=800,  # Small size
            n_proteins=6,  # Few proteins
            n_clusters=2,  # Poor clustering
            cluster_pattern=ClusterPattern.RANDOM,  # No clear structure
            noise_level=0.4,  # High noise
            protein_expression_variance=0.6,
            random_seed=self.random_seed
        )
        
        generator = SyntheticIMCDataGenerator(config)
        roi_data = generator.generate_complete_dataset()
        
        # Add significant issues
        protein_names = roi_data['protein_names']
        
        # Add outliers
        for protein in protein_names:
            outlier_indices = np.random.choice(
                roi_data['n_measurements'],
                size=int(0.02 * roi_data['n_measurements']),
                replace=False
            )
            roi_data['ion_counts'][protein][outlier_indices] *= 100  # Extreme outliers
        
        # Add missing data
        missing_protein = protein_names[0]
        missing_indices = np.random.choice(
            roi_data['n_measurements'],
            size=int(0.2 * roi_data['n_measurements']),
            replace=False
        )
        roi_data['ion_counts'][missing_protein][missing_indices] = 0
        
        # Add coordinate drift (spatial artifacts)
        drift = np.random.normal(0, 5, (roi_data['n_measurements'], 2))
        roi_data['coords'] += drift
        
        metadata = ROIMetadata(
            roi_id=roi_id,
            batch_id="test_batch_03",
            quality_level=ROIQuality.POOR,
            tissue_type="spleen",
            acquisition_date="2025-09-27",
            pixel_size_um=1.2,  # Different pixel size
            n_channels=len(roi_data['protein_names']),
            expected_issues=[
                "high_noise", "outliers", "significant_missing_data", 
                "spatial_drift", "poor_clustering"
            ],
            notes="Poor quality ROI for testing robustness"
        )
        
        return roi_data, metadata
    
    def create_corrupted_roi(self, roi_id: str = "ROI_corrupted") -> Tuple[Dict[str, Any], ROIMetadata]:
        """Create severely corrupted ROI data for error handling tests."""
        config = SyntheticDataConfig(
            n_points=500,
            n_proteins=4,
            n_clusters=1,
            noise_level=0.8,
            random_seed=self.random_seed
        )
        
        generator = SyntheticIMCDataGenerator(config)
        roi_data = generator.generate_complete_dataset()
        
        # Add severe corruption
        protein_names = roi_data['protein_names']
        
        # Add NaN values
        for protein in protein_names[:2]:
            nan_indices = np.random.choice(
                roi_data['n_measurements'],
                size=int(0.1 * roi_data['n_measurements']),
                replace=False
            )
            roi_data['ion_counts'][protein][nan_indices] = np.nan
        
        # Add infinite values  
        if len(protein_names) > 2:
            inf_indices = np.random.choice(
                roi_data['n_measurements'],
                size=int(0.05 * roi_data['n_measurements']),
                replace=False
            )
            roi_data['ion_counts'][protein_names[2]][inf_indices] = np.inf
        
        # Corrupt coordinates
        coord_corruption = np.random.choice(
            roi_data['n_measurements'],
            size=int(0.15 * roi_data['n_measurements']),
            replace=False
        )
        roi_data['coords'][coord_corruption] = np.nan
        
        # Add negative ion counts
        if len(protein_names) > 3:
            negative_indices = np.random.choice(
                roi_data['n_measurements'],
                size=int(0.08 * roi_data['n_measurements']),
                replace=False
            )
            roi_data['ion_counts'][protein_names[3]][negative_indices] = -10
        
        metadata = ROIMetadata(
            roi_id=roi_id,
            batch_id="test_batch_04",
            quality_level=ROIQuality.CORRUPTED,
            tissue_type="unknown",
            acquisition_date="2025-09-26",
            pixel_size_um=1.0,
            n_channels=len(roi_data['protein_names']),
            expected_issues=[
                "nan_values", "infinite_values", "negative_counts",
                "corrupted_coordinates", "extreme_noise"
            ],
            notes="Severely corrupted ROI for error handling testing"
        )
        
        return roi_data, metadata
    
    def create_edge_case_rois(self) -> Dict[str, Tuple[Dict[str, Any], ROIMetadata]]:
        """Create various edge case ROIs."""
        edge_cases = {}
        
        # Empty ROI
        empty_roi = {
            'coords': np.array([]).reshape(0, 2),
            'ion_counts': {},
            'dna1_intensities': np.array([]),
            'dna2_intensities': np.array([]),
            'protein_names': [],
            'n_measurements': 0,
            'ground_truth_labels': np.array([]),
            'expected_clusters': 0
        }
        
        empty_metadata = ROIMetadata(
            roi_id="ROI_empty",
            batch_id="edge_cases",
            quality_level=ROIQuality.CORRUPTED,
            tissue_type="none",
            acquisition_date="2025-09-29",
            pixel_size_um=1.0,
            n_channels=0,
            expected_issues=["no_data"],
            notes="Empty ROI for boundary testing"
        )
        
        edge_cases['empty'] = (empty_roi, empty_metadata)
        
        # Single point ROI
        single_point_config = SyntheticDataConfig(n_points=1, n_proteins=3, random_seed=self.random_seed)
        single_gen = SyntheticIMCDataGenerator(single_point_config)
        single_roi = single_gen.generate_complete_dataset()
        
        single_metadata = ROIMetadata(
            roi_id="ROI_single_point",
            batch_id="edge_cases",
            quality_level=ROIQuality.POOR,
            tissue_type="minimal",
            acquisition_date="2025-09-29",
            pixel_size_um=1.0,
            n_channels=3,
            expected_issues=["insufficient_data"],
            notes="Single point ROI for minimum data testing"
        )
        
        edge_cases['single_point'] = (single_roi, single_metadata)
        
        # Extremely sparse ROI (mostly zeros)
        sparse_config = SyntheticDataConfig(n_points=100, n_proteins=5, random_seed=self.random_seed)
        sparse_gen = SyntheticIMCDataGenerator(sparse_config)
        sparse_roi = sparse_gen.generate_complete_dataset()
        
        # Make it very sparse
        for protein in sparse_roi['protein_names']:
            zero_indices = np.random.choice(
                sparse_roi['n_measurements'],
                size=int(0.9 * sparse_roi['n_measurements']),  # 90% zeros
                replace=False
            )
            sparse_roi['ion_counts'][protein][zero_indices] = 0
        
        sparse_metadata = ROIMetadata(
            roi_id="ROI_sparse",
            batch_id="edge_cases",
            quality_level=ROIQuality.POOR,
            tissue_type="sparse",
            acquisition_date="2025-09-29",
            pixel_size_um=1.0,
            n_channels=5,
            expected_issues=["extreme_sparsity"],
            notes="Extremely sparse ROI for low-signal testing"
        )
        
        edge_cases['sparse'] = (sparse_roi, sparse_metadata)
        
        return edge_cases


def create_roi_file(roi_data: Dict[str, Any], filepath: Path, 
                   metadata: Optional[ROIMetadata] = None) -> Path:
    """Create a realistic ROI file in IMC format."""
    # Create DataFrame in IMC format
    data_dict = {
        'X': roi_data['coords'][:, 0],
        'Y': roi_data['coords'][:, 1],
    }
    
    # Add ion counts with IMC-style column names
    for protein in roi_data['protein_names']:
        # IMC format: ProteinName(IsotopeLabel)
        if protein in ['DAPI1', 'DAPI2']:
            if protein == 'DAPI1':
                column_name = 'DNA1(Ir191Di)'
            else:
                column_name = 'DNA2(Ir193Di)'
        else:
            # Assign realistic isotope labels
            isotope_map = {
                'CD45': 'Sm149Di', 'CD31': 'Nd145Di', 'CD11b': 'Pr141Di',
                'CD68': 'Er168Di', 'CD3': 'Gd155Di', 'CD8': 'Dy164Di',
                'CD4': 'Gd160Di', 'CD206': 'Tm169Di', 'Vimentin': 'Er166Di'
            }
            isotope = isotope_map.get(protein, 'Nd150Di')
            column_name = f'{protein}({isotope})'
        
        data_dict[column_name] = roi_data['ion_counts'][protein]
    
    # Add DNA intensities
    if 'dna1_intensities' in roi_data:
        data_dict['DNA1(Ir191Di)'] = roi_data['dna1_intensities']
    if 'dna2_intensities' in roi_data:
        data_dict['DNA2(Ir193Di)'] = roi_data['dna2_intensities']
    
    # Create DataFrame and save
    df = pd.DataFrame(data_dict)
    df.to_csv(filepath, sep='\t', index=False)
    
    # Save metadata if provided
    if metadata:
        metadata_path = filepath.with_suffix('.metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                'roi_id': metadata.roi_id,
                'batch_id': metadata.batch_id,
                'quality_level': metadata.quality_level.value,
                'tissue_type': metadata.tissue_type,
                'acquisition_date': metadata.acquisition_date,
                'pixel_size_um': metadata.pixel_size_um,
                'n_channels': metadata.n_channels,
                'expected_issues': metadata.expected_issues,
                'notes': metadata.notes
            }, f, indent=2)
    
    return filepath


# Pytest fixtures
@pytest.fixture
def roi_generator():
    """ROI test data generator."""
    return ROITestDataGenerator()


@pytest.fixture
def excellent_roi(roi_generator):
    """High-quality ROI for positive control testing."""
    roi_data, metadata = roi_generator.create_excellent_roi()
    return roi_data, metadata


@pytest.fixture
def good_roi(roi_generator):
    """Good-quality ROI with minor issues."""
    roi_data, metadata = roi_generator.create_good_roi()
    return roi_data, metadata


@pytest.fixture
def poor_roi(roi_generator):
    """Poor-quality ROI for robustness testing."""
    roi_data, metadata = roi_generator.create_poor_roi()
    return roi_data, metadata


@pytest.fixture
def corrupted_roi(roi_generator):
    """Corrupted ROI for error handling testing."""
    roi_data, metadata = roi_generator.create_corrupted_roi()
    return roi_data, metadata


@pytest.fixture
def edge_case_rois(roi_generator):
    """Collection of edge case ROIs."""
    return roi_generator.create_edge_case_rois()


@pytest.fixture
def temp_roi_file(excellent_roi, tmp_path):
    """Create a temporary ROI file with realistic data."""
    roi_data, metadata = excellent_roi
    file_path = tmp_path / f"{metadata.roi_id}.txt"
    create_roi_file(roi_data, file_path, metadata)
    return file_path


@pytest.fixture
def multi_roi_files(tmp_path):
    """Create multiple ROI files for batch testing."""
    generator = ROITestDataGenerator()
    roi_files = []
    
    # Create different quality ROIs
    creators = [
        generator.create_excellent_roi,
        generator.create_good_roi, 
        generator.create_poor_roi
    ]
    
    for i, creator in enumerate(creators):
        roi_data, metadata = creator(f"ROI_batch_{i:03d}")
        file_path = tmp_path / f"{metadata.roi_id}.txt"
        create_roi_file(roi_data, file_path, metadata)
        roi_files.append(file_path)
    
    return roi_files


@pytest.fixture
def roi_quality_validation():
    """Utility for validating ROI data quality."""
    def _validate(roi_data: Dict[str, Any], metadata: ROIMetadata) -> Dict[str, bool]:
        """Validate ROI data against expected quality characteristics."""
        checks = {}
        
        # Basic structure checks
        checks['has_coords'] = 'coords' in roi_data and isinstance(roi_data['coords'], np.ndarray)
        checks['has_ion_counts'] = 'ion_counts' in roi_data and isinstance(roi_data['ion_counts'], dict)
        checks['has_protein_names'] = 'protein_names' in roi_data and isinstance(roi_data['protein_names'], list)
        
        if not all([checks['has_coords'], checks['has_ion_counts'], checks['has_protein_names']]):
            return checks
        
        # Data consistency checks
        coords = roi_data['coords']
        ion_counts = roi_data['ion_counts']
        protein_names = roi_data['protein_names']
        
        checks['coords_2d'] = coords.ndim == 2 and coords.shape[1] == 2
        checks['consistent_lengths'] = all(
            len(ion_counts[p]) == len(coords) for p in protein_names if p in ion_counts
        )
        checks['positive_coordinates'] = np.all(coords >= 0) if len(coords) > 0 else True
        
        # Quality-specific checks based on metadata
        if metadata.quality_level == ROIQuality.EXCELLENT:
            checks['no_nan_coords'] = not np.isnan(coords).any() if len(coords) > 0 else True
            checks['no_inf_values'] = all(
                not np.isinf(ion_counts[p]).any() for p in protein_names if p in ion_counts
            )
            checks['sufficient_data'] = len(coords) >= 1000
            
        elif metadata.quality_level == ROIQuality.CORRUPTED:
            # For corrupted data, we expect some of these to fail
            checks['expected_corruption'] = True
        
        return checks
    
    return _validate


# Utilities for testing
def get_roi_summary_stats(roi_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics for ROI data."""
    if not roi_data['ion_counts']:
        return {'n_points': 0, 'n_proteins': 0}
    
    protein_stats = {}
    for protein, counts in roi_data['ion_counts'].items():
        protein_stats[protein] = {
            'mean': float(np.mean(counts)),
            'std': float(np.std(counts)),
            'min': float(np.min(counts)),
            'max': float(np.max(counts)),
            'n_zeros': int(np.sum(counts == 0)),
            'n_nans': int(np.sum(np.isnan(counts))) if len(counts) > 0 else 0
        }
    
    coords = roi_data['coords']
    spatial_stats = {
        'x_range': [float(coords[:, 0].min()), float(coords[:, 0].max())] if len(coords) > 0 else [0, 0],
        'y_range': [float(coords[:, 1].min()), float(coords[:, 1].max())] if len(coords) > 0 else [0, 0],
        'spatial_extent': float(np.sqrt((coords[:, 0].max() - coords[:, 0].min())**2 + 
                                       (coords[:, 1].max() - coords[:, 1].min())**2)) if len(coords) > 0 else 0
    }
    
    return {
        'n_points': roi_data['n_measurements'],
        'n_proteins': len(roi_data['protein_names']),
        'proteins': protein_stats,
        'spatial': spatial_stats
    }


def create_batch_rois(n_rois: int = 5, quality_distribution: Optional[List[ROIQuality]] = None) -> List[Tuple[Dict[str, Any], ROIMetadata]]:
    """Create a batch of ROIs with specified quality distribution."""
    if quality_distribution is None:
        quality_distribution = [ROIQuality.EXCELLENT, ROIQuality.GOOD, ROIQuality.POOR]
    
    generator = ROITestDataGenerator()
    rois = []
    
    for i in range(n_rois):
        quality = quality_distribution[i % len(quality_distribution)]
        
        if quality == ROIQuality.EXCELLENT:
            roi_data, metadata = generator.create_excellent_roi(f"batch_roi_{i:03d}")
        elif quality == ROIQuality.GOOD:
            roi_data, metadata = generator.create_good_roi(f"batch_roi_{i:03d}")
        elif quality == ROIQuality.POOR:
            roi_data, metadata = generator.create_poor_roi(f"batch_roi_{i:03d}")
        else:  # CORRUPTED
            roi_data, metadata = generator.create_corrupted_roi(f"batch_roi_{i:03d}")
        
        rois.append((roi_data, metadata))
    
    return rois


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ROI fixtures...")
    
    generator = ROITestDataGenerator()
    
    # Test all quality levels
    quality_creators = [
        ("excellent", generator.create_excellent_roi),
        ("good", generator.create_good_roi),
        ("poor", generator.create_poor_roi),
        ("corrupted", generator.create_corrupted_roi)
    ]
    
    for quality_name, creator in quality_creators:
        roi_data, metadata = creator()
        stats = get_roi_summary_stats(roi_data)
        
        print(f"\n{quality_name.upper()} ROI ({metadata.roi_id}):")
        print(f"  Points: {stats['n_points']}, Proteins: {stats['n_proteins']}")
        print(f"  Quality: {metadata.quality_level.value}")
        print(f"  Expected issues: {metadata.expected_issues}")
        print(f"  Spatial extent: {stats['spatial']['spatial_extent']:.1f} μm")
    
    # Test edge cases
    edge_cases = generator.create_edge_case_rois()
    print(f"\nEdge cases created: {list(edge_cases.keys())}")
    
    # Test batch creation
    batch = create_batch_rois(3)
    print(f"\nBatch ROIs: {[metadata.roi_id for _, metadata in batch]}")
    
    print("ROI fixture testing complete!")
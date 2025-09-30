"""
Synthetic data generation utilities for IMC analysis testing.

Provides parameterizable synthetic datasets that mimic real IMC data patterns
while being fully controlled for testing purposes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class DatasetSize(Enum):
    """Predefined dataset sizes for different test scenarios."""
    TINY = 50       # For quick unit tests
    SMALL = 100     # For standard unit tests  
    MEDIUM = 1000   # For integration tests
    LARGE = 5000    # For performance tests
    XLARGE = 10000  # For stress tests


class ClusterPattern(Enum):
    """Different spatial clustering patterns for synthetic data."""
    RANDOM = "random"           # No spatial structure
    FOUR_CORNERS = "four_corners"   # 4 distinct spatial clusters
    GRADIENT = "gradient"       # Continuous spatial gradient
    RING = "ring"              # Ring-shaped structure
    HIERARCHICAL = "hierarchical"  # Nested cluster structure


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic dataset generation."""
    n_points: int = 1000
    n_proteins: int = 9
    n_clusters: int = 4
    cluster_pattern: ClusterPattern = ClusterPattern.FOUR_CORNERS
    spatial_extent: Tuple[float, float] = (100.0, 100.0)  # (width, height) in μm
    noise_level: float = 0.1
    protein_expression_variance: float = 0.3
    dna_base_intensity: Tuple[float, float] = (800, 600)  # (DAPI1, DAPI2)
    random_seed: int = 42


class SyntheticIMCDataGenerator:
    """Generate synthetic IMC datasets with controlled properties."""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        np.random.seed(config.random_seed)
        
        # Standard protein names for consistency
        self.default_proteins = [
            'CD45', 'CD31', 'CD11b', 'CD68', 'CD3',
            'CD8', 'CD4', 'DAPI1', 'DAPI2', 'CD206',
            'Vimentin', 'SMA', 'Collagen', 'Ki67', 'Caspase3'
        ]
    
    def generate_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate spatial coordinates with specified clustering pattern."""
        coords = np.zeros((self.config.n_points, 2))
        labels = np.zeros(self.config.n_points, dtype=int)
        
        if self.config.cluster_pattern == ClusterPattern.RANDOM:
            coords = np.random.uniform(
                0, self.config.spatial_extent,
                (self.config.n_points, 2)
            )
            labels = np.random.randint(0, self.config.n_clusters, self.config.n_points)
            
        elif self.config.cluster_pattern == ClusterPattern.FOUR_CORNERS:
            coords, labels = self._generate_four_corners()
            
        elif self.config.cluster_pattern == ClusterPattern.GRADIENT:
            coords, labels = self._generate_gradient()
            
        elif self.config.cluster_pattern == ClusterPattern.RING:
            coords, labels = self._generate_ring()
            
        elif self.config.cluster_pattern == ClusterPattern.HIERARCHICAL:
            coords, labels = self._generate_hierarchical()
        
        return coords, labels
    
    def _generate_four_corners(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 4 clusters in corners of spatial extent."""
        width, height = self.config.spatial_extent
        points_per_cluster = self.config.n_points // 4
        
        # Define corner centers
        centers = [
            (width * 0.25, height * 0.25),  # Bottom-left
            (width * 0.75, height * 0.25),  # Bottom-right
            (width * 0.25, height * 0.75),  # Top-left
            (width * 0.75, height * 0.75),  # Top-right
        ]
        
        coords_list = []
        labels_list = []
        
        for i, (cx, cy) in enumerate(centers):
            # Generate points around each center
            n_points = points_per_cluster
            if i == 0:  # Add remainder to first cluster
                n_points += self.config.n_points % 4
                
            cluster_coords = np.random.multivariate_normal(
                [cx, cy],
                [[width/16, 0], [0, height/16]],  # Cluster spread
                n_points
            )
            
            coords_list.append(cluster_coords)
            labels_list.extend([i] * n_points)
        
        coords = np.vstack(coords_list)
        labels = np.array(labels_list)
        
        return coords, labels
    
    def _generate_gradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate gradient pattern across spatial extent."""
        coords = np.random.uniform(
            0, self.config.spatial_extent,
            (self.config.n_points, 2)
        )
        
        # Create gradient-based labels
        width = self.config.spatial_extent[0]
        x_positions = coords[:, 0]
        labels = (x_positions / width * self.config.n_clusters).astype(int)
        labels = np.clip(labels, 0, self.config.n_clusters - 1)
        
        return coords, labels
    
    def _generate_ring(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ring-shaped structure."""
        center = (self.config.spatial_extent[0] / 2, self.config.spatial_extent[1] / 2)
        max_radius = min(self.config.spatial_extent) / 3
        
        # Generate polar coordinates
        angles = np.random.uniform(0, 2*np.pi, self.config.n_points)
        radii = np.random.uniform(max_radius*0.5, max_radius, self.config.n_points)
        
        # Convert to Cartesian
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        coords = np.column_stack([x, y])
        
        # Create ring-based labels
        labels = (angles / (2*np.pi) * self.config.n_clusters).astype(int)
        labels = np.clip(labels, 0, self.config.n_clusters - 1)
        
        return coords, labels
    
    def _generate_hierarchical(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate hierarchical cluster structure."""
        # Create main clusters, then subclusters within each
        main_clusters = 2
        sub_clusters_per_main = self.config.n_clusters // main_clusters
        
        coords_list = []
        labels_list = []
        label_counter = 0
        
        for main_idx in range(main_clusters):
            # Main cluster center
            main_center = (
                self.config.spatial_extent[0] * (0.25 + 0.5 * main_idx),
                self.config.spatial_extent[1] * 0.5
            )
            
            points_per_main = self.config.n_points // main_clusters
            
            for sub_idx in range(sub_clusters_per_main):
                # Sub-cluster center offset from main center
                angle = 2 * np.pi * sub_idx / sub_clusters_per_main
                offset = (
                    self.config.spatial_extent[0] * 0.1 * np.cos(angle),
                    self.config.spatial_extent[1] * 0.1 * np.sin(angle)
                )
                sub_center = (
                    main_center[0] + offset[0],
                    main_center[1] + offset[1]
                )
                
                points_per_sub = points_per_main // sub_clusters_per_main
                if main_idx == 0 and sub_idx == 0:  # Add remainder
                    points_per_sub += self.config.n_points % (main_clusters * sub_clusters_per_main)
                
                # Generate points around sub-cluster center
                sub_coords = np.random.multivariate_normal(
                    sub_center,
                    [[self.config.spatial_extent[0]/32, 0], 
                     [0, self.config.spatial_extent[1]/32]],
                    points_per_sub
                )
                
                coords_list.append(sub_coords)
                labels_list.extend([label_counter] * points_per_sub)
                label_counter += 1
        
        coords = np.vstack(coords_list)
        labels = np.array(labels_list)
        
        return coords, labels
    
    def generate_protein_expressions(self, cluster_labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate protein expression data with cluster-specific patterns."""
        n_proteins = min(self.config.n_proteins, len(self.default_proteins))
        protein_names = self.default_proteins[:n_proteins]
        
        # Define base expression levels per cluster per protein
        base_expressions = self._create_expression_patterns(protein_names, cluster_labels)
        
        ion_counts = {}
        
        for protein in protein_names:
            protein_values = []
            
            for point_idx, cluster_id in enumerate(cluster_labels):
                base_level = base_expressions[cluster_id][protein]
                
                # Add biological noise using negative binomial distribution
                # High variance for low expression, lower variance for high expression
                dispersion = max(1.0, base_level * self.config.protein_expression_variance)
                p_success = base_level / (base_level + dispersion)
                
                expression = np.random.negative_binomial(
                    n=max(1, int(dispersion)), 
                    p=min(0.99, max(0.01, p_success))
                )
                
                # Add measurement noise
                noise = np.random.normal(0, self.config.noise_level * np.sqrt(expression))
                final_expression = max(0, expression + noise)
                
                protein_values.append(float(final_expression))
            
            ion_counts[protein] = np.array(protein_values)
        
        return ion_counts
    
    def _create_expression_patterns(self, protein_names: List[str], cluster_labels: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Create realistic protein expression patterns for each cluster."""
        n_clusters = len(np.unique(cluster_labels))
        
        # Define protein expression profiles for different cell types
        expression_profiles = {
            # Immune cells (high CD45, CD11b)
            'immune': {'CD45': 15, 'CD31': 2, 'CD11b': 12, 'CD68': 8, 'CD3': 5, 'CD8': 3, 'CD4': 4, 'DAPI1': 20, 'DAPI2': 15},
            # Endothelial cells (high CD31)
            'endothelial': {'CD45': 3, 'CD31': 20, 'CD11b': 2, 'CD68': 1, 'CD3': 1, 'CD8': 1, 'CD4': 1, 'DAPI1': 18, 'DAPI2': 14},
            # Macrophages (high CD68, CD11b)
            'macrophage': {'CD45': 8, 'CD31': 3, 'CD11b': 18, 'CD68': 25, 'CD3': 2, 'CD8': 1, 'CD4': 1, 'DAPI1': 22, 'DAPI2': 16},
            # T cells (high CD3, CD8 or CD4)
            't_cell': {'CD45': 12, 'CD31': 2, 'CD11b': 3, 'CD68': 2, 'CD3': 20, 'CD8': 15, 'CD4': 8, 'DAPI1': 19, 'DAPI2': 15},
            # Stromal cells
            'stromal': {'CD45': 2, 'CD31': 5, 'CD11b': 2, 'CD68': 1, 'CD3': 1, 'CD8': 1, 'CD4': 1, 'DAPI1': 16, 'DAPI2': 12}
        }
        
        # Assign profiles to clusters
        profile_names = list(expression_profiles.keys())
        cluster_profiles = {}
        
        for cluster_id in range(n_clusters):
            profile_name = profile_names[cluster_id % len(profile_names)]
            base_profile = expression_profiles[profile_name].copy()
            
            # Fill in missing proteins with low expression
            for protein in protein_names:
                if protein not in base_profile:
                    base_profile[protein] = np.random.uniform(0.5, 3.0)
            
            # Add some cluster-specific variation
            for protein in protein_names:
                variation = np.random.uniform(0.7, 1.3)
                base_profile[protein] *= variation
            
            cluster_profiles[cluster_id] = base_profile
        
        return cluster_profiles
    
    def generate_dna_intensities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate DNA staining intensities (DAPI1, DAPI2)."""
        base_dapi1, base_dapi2 = self.config.dna_base_intensity
        
        # Generate with Poisson noise and some cell-to-cell variation
        dna1 = np.random.poisson(base_dapi1, self.config.n_points).astype(float)
        dna2 = np.random.poisson(base_dapi2, self.config.n_points).astype(float)
        
        # Add biological variation
        dna1 *= np.random.uniform(0.8, 1.2, self.config.n_points)
        dna2 *= np.random.uniform(0.8, 1.2, self.config.n_points)
        
        return dna1, dna2
    
    def generate_complete_dataset(self) -> Dict[str, Any]:
        """Generate complete synthetic IMC dataset."""
        coords, cluster_labels = self.generate_coordinates()
        ion_counts = self.generate_protein_expressions(cluster_labels)
        dna1_intensities, dna2_intensities = self.generate_dna_intensities()
        
        protein_names = list(ion_counts.keys())
        
        return {
            'coords': coords,
            'ion_counts': ion_counts,
            'dna1_intensities': dna1_intensities,
            'dna2_intensities': dna2_intensities,
            'protein_names': protein_names,
            'n_measurements': self.config.n_points,
            'ground_truth_labels': cluster_labels,
            'expected_clusters': len(np.unique(cluster_labels)),
            'config': self.config
        }


def create_dataset(size: DatasetSize, 
                  cluster_pattern: ClusterPattern = ClusterPattern.FOUR_CORNERS,
                  n_proteins: int = 9,
                  random_seed: int = 42) -> Dict[str, Any]:
    """Create a synthetic dataset with specified parameters."""
    config = SyntheticDataConfig(
        n_points=size.value,
        n_proteins=n_proteins,
        cluster_pattern=cluster_pattern,
        random_seed=random_seed
    )
    
    generator = SyntheticIMCDataGenerator(config)
    return generator.generate_complete_dataset()


def create_edge_case_datasets() -> Dict[str, Dict[str, Any]]:
    """Create datasets for edge case testing."""
    edge_cases = {}
    
    # Empty dataset
    empty_config = SyntheticDataConfig(n_points=0, n_proteins=0, random_seed=42)
    empty_gen = SyntheticIMCDataGenerator(empty_config)
    edge_cases['empty'] = {
        'coords': np.array([]).reshape(0, 2),
        'ion_counts': {},
        'dna1_intensities': np.array([]),
        'dna2_intensities': np.array([]),
        'protein_names': [],
        'n_measurements': 0,
        'ground_truth_labels': np.array([]),
        'expected_clusters': 0
    }
    
    # Single point
    single_config = SyntheticDataConfig(n_points=1, n_proteins=3, random_seed=42)
    single_gen = SyntheticIMCDataGenerator(single_config)
    edge_cases['single_point'] = single_gen.generate_complete_dataset()
    
    # Single protein
    single_protein_config = SyntheticDataConfig(n_points=100, n_proteins=1, random_seed=42)
    single_protein_gen = SyntheticIMCDataGenerator(single_protein_config)
    edge_cases['single_protein'] = single_protein_gen.generate_complete_dataset()
    
    # No clusters (all same label)
    no_cluster_config = SyntheticDataConfig(
        n_points=100, 
        n_proteins=5, 
        n_clusters=1,
        cluster_pattern=ClusterPattern.RANDOM,
        random_seed=42
    )
    no_cluster_gen = SyntheticIMCDataGenerator(no_cluster_config)
    edge_cases['no_clusters'] = no_cluster_gen.generate_complete_dataset()
    
    # High dimensional (many proteins)
    high_dim_config = SyntheticDataConfig(n_points=200, n_proteins=15, random_seed=42)
    high_dim_gen = SyntheticIMCDataGenerator(high_dim_config)
    edge_cases['high_dimensional'] = high_dim_gen.generate_complete_dataset()
    
    return edge_cases


def create_performance_datasets() -> Dict[str, Dict[str, Any]]:
    """Create datasets for performance testing."""
    performance_datasets = {}
    
    sizes = [DatasetSize.SMALL, DatasetSize.MEDIUM, DatasetSize.LARGE, DatasetSize.XLARGE]
    
    for size in sizes:
        dataset = create_dataset(
            size=size,
            cluster_pattern=ClusterPattern.FOUR_CORNERS,
            n_proteins=9,
            random_seed=42
        )
        performance_datasets[size.name.lower()] = dataset
    
    return performance_datasets


# Convenience functions for backward compatibility
def create_small_roi_data(random_seed: int = 42) -> Dict[str, Any]:
    """Create small ROI dataset for unit tests."""
    return create_dataset(DatasetSize.SMALL, random_seed=random_seed)


def create_medium_roi_data(random_seed: int = 42) -> Dict[str, Any]:
    """Create medium ROI dataset for integration tests."""  
    return create_dataset(DatasetSize.MEDIUM, random_seed=random_seed)


def create_large_roi_data(random_seed: int = 42) -> Dict[str, Any]:
    """Create large ROI dataset for performance tests."""
    return create_dataset(DatasetSize.LARGE, random_seed=random_seed)


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic IMC datasets...")
    
    # Create different sized datasets
    small_data = create_dataset(DatasetSize.SMALL)
    print(f"Small dataset: {small_data['n_measurements']} points, "
          f"{len(small_data['protein_names'])} proteins")
    
    # Create different clustering patterns
    patterns = [ClusterPattern.FOUR_CORNERS, ClusterPattern.GRADIENT, ClusterPattern.RING]
    for pattern in patterns:
        data = create_dataset(DatasetSize.SMALL, cluster_pattern=pattern)
        n_clusters = len(np.unique(data['ground_truth_labels']))
        print(f"{pattern.value} pattern: {n_clusters} clusters")
    
    # Create edge cases
    edge_cases = create_edge_case_datasets()
    print(f"Created {len(edge_cases)} edge case datasets")
    
    print("Dataset generation complete!")
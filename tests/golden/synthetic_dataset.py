"""
Fixed synthetic dataset for golden regression tests.

This dataset has known properties and expected outputs.
Any changes to analysis results indicate potential regressions.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


class SyntheticIMCDataset:
    """
    Fixed synthetic IMC dataset with known ground truth.
    
    Designed to test the "co-abundance revolution" and all pipeline stages.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize with fixed random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        
        # Known dataset properties
        self.n_points = 200
        self.n_proteins = 9
        self.expected_clusters = 4  # Ground truth
        self.expected_coabundance_features = 153  # 9 proteins -> 153 features
        
        # Protein names (matching real pipeline)
        self.protein_names = [
            'CD45', 'CD31', 'CD11b', 'CD68', 'CD3', 
            'CD8', 'CD4', 'DAPI1', 'DAPI2'
        ]
        
        # Generate fixed dataset
        self._generate_dataset()
    
    def _generate_dataset(self):
        """Generate the fixed synthetic dataset."""
        # Create 4 distinct clusters with known properties
        cluster_centers = np.array([
            [20, 20],   # Cluster 0: Bottom-left
            [80, 20],   # Cluster 1: Bottom-right  
            [20, 80],   # Cluster 2: Top-left
            [80, 80]    # Cluster 3: Top-right
        ])
        
        # Points per cluster
        points_per_cluster = self.n_points // 4
        
        coords = []
        true_labels = []
        
        # Generate coordinates for each cluster
        for i, center in enumerate(cluster_centers):
            # Add some noise around center
            cluster_coords = np.random.multivariate_normal(
                center, [[100, 0], [0, 100]], points_per_cluster
            )
            coords.append(cluster_coords)
            true_labels.extend([i] * points_per_cluster)
        
        self.coords = np.vstack(coords)
        self.true_labels = np.array(true_labels)
        
        # Generate ion counts with cluster-specific patterns
        self.ion_counts = self._generate_ion_counts()
        
        # Generate DNA intensities
        self.dna1_intensities = np.random.poisson(800, self.n_points).astype(float)
        self.dna2_intensities = np.random.poisson(600, self.n_points).astype(float)
    
    def _generate_ion_counts(self) -> Dict[str, np.ndarray]:
        """Generate ion counts with known cluster patterns."""
        ion_counts = {}
        
        # Define cluster-specific expression patterns
        cluster_patterns = {
            0: {'CD45': 10, 'CD31': 2, 'CD11b': 8, 'CD68': 3, 'CD3': 1, 'CD8': 1, 'CD4': 1, 'DAPI1': 15, 'DAPI2': 12},
            1: {'CD45': 2, 'CD31': 12, 'CD11b': 3, 'CD68': 2, 'CD3': 8, 'CD8': 6, 'CD4': 4, 'DAPI1': 14, 'DAPI2': 13},
            2: {'CD45': 8, 'CD31': 4, 'CD11b': 15, 'CD68': 12, 'CD3': 2, 'CD8': 1, 'CD4': 1, 'DAPI1': 16, 'DAPI2': 11},
            3: {'CD45': 3, 'CD31': 8, 'CD11b': 2, 'CD68': 1, 'CD3': 12, 'CD8': 8, 'CD4': 10, 'DAPI1': 13, 'DAPI2': 15}
        }
        
        for protein in self.protein_names:
            protein_values = []
            
            for point_idx in range(self.n_points):
                cluster_id = self.true_labels[point_idx]
                base_expression = cluster_patterns[cluster_id][protein]
                
                # Add Poisson noise
                expression = np.random.poisson(base_expression)
                protein_values.append(float(expression))
            
            ion_counts[protein] = np.array(protein_values)
        
        return ion_counts
    
    def get_roi_data(self) -> Dict[str, Any]:
        """Get data in ROI format expected by pipeline."""
        return {
            'coords': self.coords,
            'ion_counts': self.ion_counts,
            'dna1_intensities': self.dna1_intensities,
            'dna2_intensities': self.dna2_intensities,
            'protein_names': self.protein_names,
            'n_measurements': self.n_points,
            'ground_truth_labels': self.true_labels,
            'expected_clusters': self.expected_clusters,
            'expected_features': self.expected_coabundance_features
        }
    
    def save_to_file(self, filepath: Path):
        """Save dataset to file."""
        data = {
            'seed': self.seed,
            'n_points': self.n_points,
            'coords': self.coords.tolist(),
            'ion_counts': {k: v.tolist() for k, v in self.ion_counts.items()},
            'dna1_intensities': self.dna1_intensities.tolist(),
            'dna2_intensities': self.dna2_intensities.tolist(),
            'protein_names': self.protein_names,
            'true_labels': self.true_labels.tolist(),
            'expected_clusters': self.expected_clusters,
            'expected_coabundance_features': self.expected_coabundance_features
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: Path):
        """Load dataset from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        dataset = cls(seed=data['seed'])
        dataset.coords = np.array(data['coords'])
        dataset.ion_counts = {k: np.array(v) for k, v in data['ion_counts'].items()}
        dataset.dna1_intensities = np.array(data['dna1_intensities'])
        dataset.dna2_intensities = np.array(data['dna2_intensities'])
        dataset.true_labels = np.array(data['true_labels'])
        
        return dataset
    
    def validate_properties(self) -> Dict[str, bool]:
        """Validate that dataset has expected properties."""
        checks = {
            'correct_n_points': len(self.coords) == self.n_points,
            'correct_n_proteins': len(self.protein_names) == self.n_proteins,
            'coords_2d': self.coords.shape[1] == 2,
            'has_all_proteins': all(p in self.ion_counts for p in self.protein_names),
            'consistent_lengths': all(len(self.ion_counts[p]) == self.n_points 
                                    for p in self.protein_names),
            'positive_ion_counts': all(np.all(self.ion_counts[p] >= 0) 
                                     for p in self.protein_names),
            'correct_cluster_count': len(np.unique(self.true_labels)) == self.expected_clusters,
            'dna_positive': np.all(self.dna1_intensities > 0) and np.all(self.dna2_intensities > 0)
        }
        
        return checks


def create_golden_dataset() -> SyntheticIMCDataset:
    """Create the standard golden dataset."""
    return SyntheticIMCDataset(seed=42)


if __name__ == "__main__":
    # Generate and save the golden dataset
    dataset = create_golden_dataset()
    
    # Validate dataset
    checks = dataset.validate_properties()
    print("Dataset validation:")
    for check, passed in checks.items():
        print(f"  {check}: {'✓' if passed else '✗'}")
    
    # Save to file
    output_path = Path("tests/golden/golden_dataset.json")
    dataset.save_to_file(output_path)
    print(f"\nGolden dataset saved to {output_path}")
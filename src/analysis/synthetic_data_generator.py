"""
Synthetic Ground Truth Data Generator for IMC Pipeline Validation

Generates realistic synthetic IMC data with known spatial patterns, protein interactions,
and ground truth labels for quantitative validation of the analysis pipeline.

Key Features:
- Known spatial autocorrelation patterns (clustered, random, dispersed)
- Realistic protein interaction networks with co-localization
- Ground truth tissue regions and cell type distributions
- Common IMC artifacts: hot pixels, batch effects, spillover
- Compatible with existing pipeline data structures and validation framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy.spatial import cKDTree, distance_matrix
from scipy import stats, ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import existing modules for compatibility
from .spatial_stats import compute_ripleys_k, compute_spatial_correlation
from .coabundance_features import generate_coabundance_features
from ..validation.framework import ValidationRule, ValidationCategory, ValidationResult, ValidationSeverity


class SpatialPattern(Enum):
    """Types of spatial organization patterns."""
    CLUSTERED = "clustered"      # Spatially aggregated (Moran's I > 0)
    RANDOM = "random"           # Complete spatial randomness (Moran's I ≈ 0)
    DISPERSED = "dispersed"     # Spatially regular (Moran's I < 0)
    HIERARCHICAL = "hierarchical" # Multiple scales of organization


class TissueType(Enum):
    """Tissue region types with distinct characteristics."""
    EPITHELIAL = "epithelial"
    STROMAL = "stromal"
    IMMUNE_RICH = "immune_rich"
    VESSEL = "vessel"
    NECROTIC = "necrotic"
    TUMOR = "tumor"
    NORMAL = "normal"


@dataclass
class ProteinProperties:
    """Properties of individual protein markers."""
    name: str
    base_expression: float = 1.0          # Base expression level
    noise_level: float = 0.1              # Technical noise
    spatial_pattern: SpatialPattern = SpatialPattern.RANDOM
    tissue_specificity: Dict[TissueType, float] = field(default_factory=dict)
    interaction_partners: List[str] = field(default_factory=list)
    spillover_targets: Dict[str, float] = field(default_factory=dict)  # protein -> spillover fraction


@dataclass
class TissueRegion:
    """Definition of tissue regions with spatial and biological properties."""
    tissue_type: TissueType
    center: Tuple[float, float]
    radius: float
    cell_density: float = 1000.0          # cells per mm²
    protein_modifiers: Dict[str, float] = field(default_factory=dict)
    spatial_coherence: float = 0.8        # How spatially compact (0-1)


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    # Spatial parameters
    roi_size_um: Tuple[float, float] = (1000.0, 1000.0)
    pixel_size_um: float = 1.0
    n_cells_total: int = 10000
    
    # Protein panel
    protein_names: List[str] = field(default_factory=lambda: [
        'CD45', 'CD3', 'CD4', 'CD8', 'CD20', 'CD68', 'PanCK', 'Vimentin', 'DNA1'
    ])
    
    # Tissue organization
    tissue_regions: List[TissueRegion] = field(default_factory=list)
    background_density: float = 100.0     # Background cell density
    
    # Biological realism
    protein_interaction_strength: float = 0.3
    spatial_autocorr_range: float = 50.0  # Spatial correlation range in μm
    
    # Technical artifacts
    hot_pixel_probability: float = 0.001
    batch_effect_strength: float = 0.2
    spillover_strength: float = 0.05
    baseline_noise_level: float = 0.1
    
    # Validation parameters
    known_cluster_count: int = 5
    ground_truth_labels: Optional[np.ndarray] = None


class SyntheticDataGenerator:
    """
    Generate realistic synthetic IMC data with known ground truth.
    
    Creates spatially organized data with realistic protein interactions,
    technical artifacts, and known tissue patterns for pipeline validation.
    """
    
    def __init__(self, config: SyntheticDataConfig):
        """Initialize generator with configuration."""
        self.config = config
        self.random_state = np.random.RandomState(42)
        
        # Create default protein properties if not provided
        self.protein_properties = self._create_default_protein_properties()
        
        # Create default tissue regions if not provided
        if not self.config.tissue_regions:
            self.config.tissue_regions = self._create_default_tissue_regions()
        
        # Generated data cache
        self._generated_data = None
        self._ground_truth = None
    
    def _create_default_protein_properties(self) -> Dict[str, ProteinProperties]:
        """Create realistic protein properties for common IMC panel."""
        properties = {}
        
        # Pan-leukocyte marker
        properties['CD45'] = ProteinProperties(
            name='CD45',
            base_expression=2.0,
            spatial_pattern=SpatialPattern.CLUSTERED,
            tissue_specificity={
                TissueType.IMMUNE_RICH: 5.0,
                TissueType.STROMAL: 1.5,
                TissueType.EPITHELIAL: 0.2
            },
            interaction_partners=['CD3', 'CD20', 'CD68']
        )
        
        # T cell markers
        properties['CD3'] = ProteinProperties(
            name='CD3',
            base_expression=1.5,
            spatial_pattern=SpatialPattern.CLUSTERED,
            tissue_specificity={
                TissueType.IMMUNE_RICH: 3.0,
                TissueType.TUMOR: 2.0,
                TissueType.EPITHELIAL: 0.1
            },
            interaction_partners=['CD45', 'CD4', 'CD8']
        )
        
        properties['CD4'] = ProteinProperties(
            name='CD4',
            base_expression=1.0,
            spatial_pattern=SpatialPattern.CLUSTERED,
            tissue_specificity={
                TissueType.IMMUNE_RICH: 2.0,
                TissueType.STROMAL: 1.0
            },
            interaction_partners=['CD3', 'CD45'],
            spillover_targets={'CD8': 0.02}  # Slight spillover
        )
        
        properties['CD8'] = ProteinProperties(
            name='CD8',
            base_expression=0.8,
            spatial_pattern=SpatialPattern.CLUSTERED,
            tissue_specificity={
                TissueType.TUMOR: 2.5,
                TissueType.IMMUNE_RICH: 1.5
            },
            interaction_partners=['CD3', 'CD45'],
            spillover_targets={'CD4': 0.01}
        )
        
        # B cell marker
        properties['CD20'] = ProteinProperties(
            name='CD20',
            base_expression=1.2,
            spatial_pattern=SpatialPattern.CLUSTERED,
            tissue_specificity={
                TissueType.IMMUNE_RICH: 4.0,
                TissueType.STROMAL: 0.5
            },
            interaction_partners=['CD45']
        )
        
        # Macrophage marker
        properties['CD68'] = ProteinProperties(
            name='CD68',
            base_expression=1.8,
            spatial_pattern=SpatialPattern.CLUSTERED,
            tissue_specificity={
                TissueType.STROMAL: 3.0,
                TissueType.NECROTIC: 5.0,
                TissueType.VESSEL: 2.0
            },
            interaction_partners=['CD45', 'Vimentin']
        )
        
        # Epithelial marker
        properties['PanCK'] = ProteinProperties(
            name='PanCK',
            base_expression=2.5,
            spatial_pattern=SpatialPattern.CLUSTERED,
            tissue_specificity={
                TissueType.EPITHELIAL: 6.0,
                TissueType.TUMOR: 4.0,
                TissueType.STROMAL: 0.1
            },
            interaction_partners=[]
        )
        
        # Stromal marker
        properties['Vimentin'] = ProteinProperties(
            name='Vimentin',
            base_expression=2.0,
            spatial_pattern=SpatialPattern.DISPERSED,
            tissue_specificity={
                TissueType.STROMAL: 4.0,
                TissueType.VESSEL: 3.0,
                TissueType.EPITHELIAL: 0.2
            },
            interaction_partners=['CD68']
        )
        
        # DNA channel
        properties['DNA1'] = ProteinProperties(
            name='DNA1',
            base_expression=3.0,
            spatial_pattern=SpatialPattern.RANDOM,
            tissue_specificity={},  # Present in all nucleated cells
            noise_level=0.05  # DNA is usually cleaner
        )
        
        return properties
    
    def _create_default_tissue_regions(self) -> List[TissueRegion]:
        """Create default tissue organization pattern."""
        roi_width, roi_height = self.config.roi_size_um
        
        regions = [
            # Central tumor region
            TissueRegion(
                tissue_type=TissueType.TUMOR,
                center=(roi_width * 0.4, roi_height * 0.5),
                radius=150.0,
                cell_density=2000.0,
                protein_modifiers={'PanCK': 2.0, 'CD8': 1.5},
                spatial_coherence=0.9
            ),
            
            # Immune infiltration
            TissueRegion(
                tissue_type=TissueType.IMMUNE_RICH,
                center=(roi_width * 0.2, roi_height * 0.3),
                radius=100.0,
                cell_density=1800.0,
                protein_modifiers={'CD45': 2.0, 'CD3': 2.5, 'CD20': 3.0},
                spatial_coherence=0.7
            ),
            
            # Stromal region
            TissueRegion(
                tissue_type=TissueType.STROMAL,
                center=(roi_width * 0.7, roi_height * 0.6),
                radius=200.0,
                cell_density=800.0,
                protein_modifiers={'Vimentin': 2.5, 'CD68': 1.8},
                spatial_coherence=0.6
            ),
            
            # Normal epithelium
            TissueRegion(
                tissue_type=TissueType.EPITHELIAL,
                center=(roi_width * 0.8, roi_height * 0.2),
                radius=120.0,
                cell_density=1500.0,
                protein_modifiers={'PanCK': 3.0},
                spatial_coherence=0.85
            ),
            
            # Vessel
            TissueRegion(
                tissue_type=TissueType.VESSEL,
                center=(roi_width * 0.5, roi_height * 0.8),
                radius=50.0,
                cell_density=1200.0,
                protein_modifiers={'Vimentin': 2.0, 'CD68': 1.5},
                spatial_coherence=0.95
            )
        ]
        
        return regions
    
    def generate_spatial_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate spatial coordinates with realistic tissue organization.
        
        Returns:
            Tuple of (coordinates, tissue_labels)
        """
        roi_width, roi_height = self.config.roi_size_um
        
        # Generate background cells (uniform random)
        n_background = int(self.config.background_density * roi_width * roi_height / 1e6)
        background_coords = self.random_state.uniform(
            low=[0, 0], 
            high=[roi_width, roi_height], 
            size=(n_background, 2)
        )
        background_labels = np.full(n_background, -1, dtype=int)  # Background label
        
        all_coords = [background_coords]
        all_labels = [background_labels]
        
        # Generate cells for each tissue region
        for region_idx, region in enumerate(self.config.tissue_regions):
            # Calculate cells for this region
            region_area = np.pi * region.radius ** 2 / 1e6  # Convert to mm²
            n_region_cells = int(region.cell_density * region_area)
            
            if n_region_cells == 0:
                continue
            
            # Generate coordinates with spatial coherence
            region_coords = self._generate_region_coordinates(region, n_region_cells)
            region_labels = np.full(len(region_coords), region_idx, dtype=int)
            
            all_coords.append(region_coords)
            all_labels.append(region_labels)
        
        # Combine all coordinates
        coordinates = np.vstack(all_coords)
        tissue_labels = np.concatenate(all_labels)
        
        # Subsample to target cell count if needed
        if len(coordinates) > self.config.n_cells_total:
            indices = self.random_state.choice(
                len(coordinates), self.config.n_cells_total, replace=False
            )
            coordinates = coordinates[indices]
            tissue_labels = tissue_labels[indices]
        
        return coordinates, tissue_labels
    
    def _generate_region_coordinates(self, region: TissueRegion, n_cells: int) -> np.ndarray:
        """Generate coordinates for a specific tissue region."""
        coords = []
        n_generated = 0
        max_attempts = n_cells * 10
        
        while n_generated < n_cells and max_attempts > 0:
            # Generate candidate coordinates
            if region.spatial_coherence > 0.7:
                # High coherence: clustered pattern
                coords_batch = self._generate_clustered_points(
                    region.center, region.radius, min(1000, n_cells - n_generated)
                )
            elif region.spatial_coherence > 0.4:
                # Medium coherence: mixture of clustered and random
                n_clustered = int((n_cells - n_generated) * region.spatial_coherence)
                n_random = (n_cells - n_generated) - n_clustered
                
                clustered = self._generate_clustered_points(
                    region.center, region.radius * 0.8, n_clustered
                )
                random_coords = self._generate_uniform_circle(
                    region.center, region.radius, n_random
                )
                coords_batch = np.vstack([clustered, random_coords]) if len(clustered) > 0 and len(random_coords) > 0 else (clustered if len(clustered) > 0 else random_coords)
            else:
                # Low coherence: nearly uniform
                coords_batch = self._generate_uniform_circle(
                    region.center, region.radius, min(1000, n_cells - n_generated)
                )
            
            if len(coords_batch) > 0:
                coords.append(coords_batch)
                n_generated += len(coords_batch)
            
            max_attempts -= 1
        
        if coords:
            all_coords = np.vstack(coords)
            return all_coords[:n_cells]  # Exact count
        else:
            return np.array([]).reshape(0, 2)
    
    def _generate_clustered_points(self, center: Tuple[float, float], 
                                  radius: float, n_points: int) -> np.ndarray:
        """Generate spatially clustered points using multiple Gaussian clusters."""
        if n_points == 0:
            return np.array([]).reshape(0, 2)
        
        # Use 3-5 sub-clusters within the region
        n_subclusters = min(5, max(1, n_points // 50))
        points_per_cluster = n_points // n_subclusters
        
        coords = []
        for i in range(n_subclusters):
            # Random subcluster center within region
            angle = self.random_state.uniform(0, 2 * np.pi)
            dist = self.random_state.uniform(0, radius * 0.6)
            subcluster_center = (
                center[0] + dist * np.cos(angle),
                center[1] + dist * np.sin(angle)
            )
            
            # Generate points with Gaussian distribution
            cluster_radius = radius * 0.2
            cluster_coords = self.random_state.multivariate_normal(
                subcluster_center,
                [[cluster_radius**2, 0], [0, cluster_radius**2]],
                size=points_per_cluster if i < n_subclusters - 1 else n_points - len(coords)
            )
            
            # Keep points within region boundary
            distances = np.linalg.norm(cluster_coords - center, axis=1)
            valid_mask = distances <= radius
            coords.append(cluster_coords[valid_mask])
        
        if coords:
            return np.vstack(coords)
        else:
            return np.array([]).reshape(0, 2)
    
    def _generate_uniform_circle(self, center: Tuple[float, float], 
                                radius: float, n_points: int) -> np.ndarray:
        """Generate uniformly distributed points within a circle."""
        if n_points == 0:
            return np.array([]).reshape(0, 2)
        
        # Use rejection sampling for uniform distribution in circle
        coords = []
        while len(coords) < n_points:
            # Generate candidate points in square
            candidates = self.random_state.uniform(
                low=[center[0] - radius, center[1] - radius],
                high=[center[0] + radius, center[1] + radius],
                size=(n_points * 2, 2)
            )
            
            # Keep points within circle
            distances = np.linalg.norm(candidates - center, axis=1)
            valid_candidates = candidates[distances <= radius]
            coords.append(valid_candidates)
        
        all_coords = np.vstack(coords)
        return all_coords[:n_points]
    
    def generate_protein_expressions(self, coordinates: np.ndarray, 
                                   tissue_labels: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate realistic protein expression data with spatial patterns.
        
        Args:
            coordinates: Cell coordinates
            tissue_labels: Tissue region assignments
            
        Returns:
            Dictionary of protein expressions
        """
        n_cells = len(coordinates)
        expressions = {}
        
        for protein_name in self.config.protein_names:
            if protein_name not in self.protein_properties:
                # Create default properties for unknown proteins
                self.protein_properties[protein_name] = ProteinProperties(
                    name=protein_name,
                    base_expression=1.0,
                    spatial_pattern=SpatialPattern.RANDOM
                )
            
            props = self.protein_properties[protein_name]
            
            # Step 1: Base expression level
            base_expr = np.full(n_cells, props.base_expression)
            
            # Step 2: Tissue-specific modulation
            for cell_idx, tissue_label in enumerate(tissue_labels):
                if tissue_label >= 0 and tissue_label < len(self.config.tissue_regions):
                    tissue_type = self.config.tissue_regions[tissue_label].tissue_type
                    if tissue_type in props.tissue_specificity:
                        modifier = props.tissue_specificity[tissue_type]
                        base_expr[cell_idx] *= modifier
                    
                    # Region-specific modulation
                    region = self.config.tissue_regions[tissue_label]
                    if protein_name in region.protein_modifiers:
                        base_expr[cell_idx] *= region.protein_modifiers[protein_name]
            
            # Step 3: Spatial pattern overlay
            spatial_component = self._generate_spatial_pattern(
                coordinates, props.spatial_pattern
            )
            base_expr *= (1.0 + self.config.protein_interaction_strength * spatial_component)
            
            # Step 4: Protein interaction network effects
            interaction_component = self._compute_protein_interactions(
                coordinates, protein_name, expressions
            )
            base_expr *= (1.0 + interaction_component)
            
            # Step 5: Technical noise
            noise = self.random_state.normal(
                loc=0, scale=props.noise_level, size=n_cells
            )
            base_expr += noise
            
            # Step 6: Convert to ion counts (Poisson-distributed)
            # Ensure positive values before Poisson sampling
            base_expr = np.maximum(base_expr, 0.01)
            ion_counts = self.random_state.poisson(base_expr * 100)  # Scale for realistic counts
            
            expressions[protein_name] = ion_counts.astype(np.float64)
        
        # Step 7: Apply spillover effects
        expressions = self._apply_spillover_effects(expressions)
        
        # Step 8: Add technical artifacts
        expressions = self._add_technical_artifacts(expressions, coordinates)
        
        return expressions
    
    def _generate_spatial_pattern(self, coordinates: np.ndarray, 
                                 pattern: SpatialPattern) -> np.ndarray:
        """Generate spatial pattern overlay for protein expression."""
        n_cells = len(coordinates)
        
        if pattern == SpatialPattern.RANDOM:
            return self.random_state.normal(0, 0.1, n_cells)
        
        elif pattern == SpatialPattern.CLUSTERED:
            # Use Gaussian process with exponential kernel
            return self._gaussian_process_field(coordinates, correlation_length=50.0)
        
        elif pattern == SpatialPattern.DISPERSED:
            # Anti-correlated pattern (negative spatial autocorrelation)
            clustered = self._gaussian_process_field(coordinates, correlation_length=30.0)
            return -clustered
        
        elif pattern == SpatialPattern.HIERARCHICAL:
            # Multiple scale patterns
            large_scale = self._gaussian_process_field(coordinates, correlation_length=100.0)
            small_scale = self._gaussian_process_field(coordinates, correlation_length=25.0)
            return 0.7 * large_scale + 0.3 * small_scale
        
        else:
            return np.zeros(n_cells)
    
    def _gaussian_process_field(self, coordinates: np.ndarray, 
                               correlation_length: float) -> np.ndarray:
        """Generate correlated spatial field using Gaussian process."""
        n_cells = len(coordinates)
        
        # For computational efficiency, use grid-based approach for large datasets
        if n_cells > 5000:
            return self._grid_based_spatial_field(coordinates, correlation_length)
        
        # Direct covariance matrix approach for smaller datasets
        distances = distance_matrix(coordinates, coordinates)
        
        # Exponential covariance kernel
        covariance = np.exp(-distances / correlation_length)
        
        # Add small diagonal term for numerical stability
        covariance += 1e-6 * np.eye(n_cells)
        
        # Generate correlated random field
        try:
            L = np.linalg.cholesky(covariance)
            white_noise = self.random_state.normal(0, 1, n_cells)
            correlated_field = L @ white_noise
        except np.linalg.LinAlgError:
            # Fallback to simpler method if Cholesky fails
            warnings.warn("Cholesky decomposition failed, using fallback spatial pattern")
            correlated_field = self.random_state.normal(0, 0.2, n_cells)
        
        # Normalize to reasonable scale
        return correlated_field / np.std(correlated_field) * 0.2
    
    def _grid_based_spatial_field(self, coordinates: np.ndarray, 
                                 correlation_length: float) -> np.ndarray:
        """Efficient grid-based spatial field generation for large datasets."""
        roi_width, roi_height = self.config.roi_size_um
        
        # Create coarse grid
        grid_resolution = correlation_length / 2
        nx = int(roi_width / grid_resolution) + 1
        ny = int(roi_height / grid_resolution) + 1
        
        # Generate grid coordinates
        x_grid = np.linspace(0, roi_width, nx)
        y_grid = np.linspace(0, roi_height, ny)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Generate random field on grid
        grid_values = self.random_state.normal(0, 1, len(grid_coords))
        
        # Smooth the field
        grid_2d = grid_values.reshape(ny, nx)
        sigma = correlation_length / grid_resolution / 2
        smoothed_grid = ndimage.gaussian_filter(grid_2d, sigma=sigma, mode='reflect')
        
        # Interpolate to cell positions
        from scipy.interpolate import RegularGridInterpolator
        interpolator = RegularGridInterpolator(
            (y_grid, x_grid), smoothed_grid, 
            bounds_error=False, fill_value=0.0
        )
        
        cell_values = interpolator(coordinates[:, [1, 0]])  # Note: y, x order
        
        # Normalize
        return cell_values / np.std(cell_values) * 0.2
    
    def _compute_protein_interactions(self, coordinates: np.ndarray, 
                                    protein_name: str, 
                                    existing_expressions: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute protein interaction effects."""
        n_cells = len(coordinates)
        interaction_effect = np.zeros(n_cells)
        
        if protein_name not in self.protein_properties:
            return interaction_effect
        
        props = self.protein_properties[protein_name]
        
        # Local neighborhood interactions
        if len(existing_expressions) > 0 and props.interaction_partners:
            tree = cKDTree(coordinates)
            
            for partner in props.interaction_partners:
                if partner in existing_expressions:
                    partner_expr = existing_expressions[partner]
                    
                    # For each cell, compute interaction with neighbors
                    neighbor_radius = 20.0  # μm
                    for i in range(n_cells):
                        neighbors = tree.query_ball_point(coordinates[i], neighbor_radius)
                        if len(neighbors) > 1:  # Exclude self
                            neighbor_expr = np.mean([partner_expr[j] for j in neighbors if j != i])
                            # Positive interaction (co-localization)
                            interaction_effect[i] += 0.1 * (neighbor_expr / 100.0)  # Scale down
        
        return np.clip(interaction_effect, -0.5, 0.5)
    
    def _apply_spillover_effects(self, expressions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply spectral spillover between channels."""
        corrected_expressions = expressions.copy()
        
        for source_protein, source_expr in expressions.items():
            if source_protein in self.protein_properties:
                props = self.protein_properties[source_protein]
                
                for target_protein, spillover_fraction in props.spillover_targets.items():
                    if target_protein in corrected_expressions:
                        spillover_amount = source_expr * spillover_fraction * self.config.spillover_strength
                        corrected_expressions[target_protein] += spillover_amount
        
        return corrected_expressions
    
    def _add_technical_artifacts(self, expressions: Dict[str, np.ndarray], 
                               coordinates: np.ndarray) -> Dict[str, np.ndarray]:
        """Add realistic technical artifacts."""
        artifacted_expressions = expressions.copy()
        n_cells = len(coordinates)
        
        # 1. Hot pixels
        n_hot_pixels = int(n_cells * self.config.hot_pixel_probability)
        if n_hot_pixels > 0:
            hot_indices = self.random_state.choice(n_cells, n_hot_pixels, replace=False)
            for protein_name in artifacted_expressions.keys():
                # Hot pixels have 10-100x normal expression
                multiplier = self.random_state.uniform(10, 100, len(hot_indices))
                artifacted_expressions[protein_name][hot_indices] *= multiplier
        
        # 2. Batch effects (systematic bias)
        batch_effect = self.random_state.normal(
            1.0, self.config.batch_effect_strength, n_cells
        )
        for protein_name in artifacted_expressions.keys():
            artifacted_expressions[protein_name] *= batch_effect
        
        # 3. Detector noise
        for protein_name in artifacted_expressions.keys():
            noise = self.random_state.normal(
                0, self.config.baseline_noise_level * np.sqrt(np.maximum(artifacted_expressions[protein_name], 1)), 
                n_cells
            )
            artifacted_expressions[protein_name] += noise
            
            # Ensure non-negative
            artifacted_expressions[protein_name] = np.maximum(
                artifacted_expressions[protein_name], 0
            )
        
        return artifacted_expressions
    
    def generate_ground_truth_clusters(self, coordinates: np.ndarray, 
                                     expressions: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate ground truth cluster labels based on known patterns."""
        # Use tissue regions as basis but refine with expression patterns
        tissue_labels = self._assign_tissue_labels(coordinates)
        
        # Create feature matrix for clustering refinement
        feature_list = []
        protein_names = []
        for protein_name, expr in expressions.items():
            if protein_name != 'DNA1':  # Exclude DNA from clustering
                feature_list.append(expr.reshape(-1, 1))
                protein_names.append(protein_name)
        
        if feature_list:
            feature_matrix = np.hstack(feature_list)
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_matrix)
            
            # K-means clustering to refine tissue labels
            kmeans = KMeans(
                n_clusters=self.config.known_cluster_count,
                random_state=42,
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            return cluster_labels
        else:
            # Fallback to tissue labels
            unique_tissues = np.unique(tissue_labels[tissue_labels >= 0])
            cluster_map = {tissue: i for i, tissue in enumerate(unique_tissues)}
            cluster_map[-1] = -1  # Background
            
            return np.array([cluster_map[label] for label in tissue_labels])
    
    def _assign_tissue_labels(self, coordinates: np.ndarray) -> np.ndarray:
        """Assign tissue labels based on spatial regions."""
        n_cells = len(coordinates)
        tissue_labels = np.full(n_cells, -1, dtype=int)  # Background by default
        
        for cell_idx, coord in enumerate(coordinates):
            for region_idx, region in enumerate(self.config.tissue_regions):
                distance = np.linalg.norm(coord - region.center)
                if distance <= region.radius:
                    tissue_labels[cell_idx] = region_idx
                    break  # First match wins
        
        return tissue_labels
    
    def generate_complete_dataset(self) -> Dict[str, Any]:
        """
        Generate complete synthetic IMC dataset with ground truth.
        
        Returns:
            Dictionary with all data components and ground truth
        """
        # Generate spatial coordinates and tissue organization
        coordinates, tissue_labels = self.generate_spatial_coordinates()
        
        # Generate protein expressions
        expressions = self.generate_protein_expressions(coordinates, tissue_labels)
        
        # Generate ground truth cluster labels
        cluster_labels = self.generate_ground_truth_clusters(coordinates, expressions)
        
        # Extract DNA channels for segmentation
        dna1_intensities = expressions.get('DNA1', np.ones(len(coordinates)))
        dna2_intensities = expressions.get('DNA2', dna1_intensities * 0.8)  # Correlated
        
        # Prepare ion counts dictionary (exclude DNA from protein analysis)
        ion_counts = {name: expr for name, expr in expressions.items() 
                     if name not in ['DNA1', 'DNA2']}
        
        # Compute validation metrics
        validation_metrics = self._compute_validation_metrics(
            coordinates, expressions, cluster_labels
        )
        
        # Package complete dataset
        dataset = {
            # Core data
            'coordinates': coordinates,
            'ion_counts': ion_counts,
            'dna1_intensities': dna1_intensities,
            'dna2_intensities': dna2_intensities,
            
            # Ground truth
            'ground_truth_clusters': cluster_labels,
            'tissue_labels': tissue_labels,
            'protein_expressions': expressions,
            
            # Metadata
            'protein_names': list(ion_counts.keys()),
            'n_cells': len(coordinates),
            'roi_bounds': (0, self.config.roi_size_um[0], 0, self.config.roi_size_um[1]),
            'tissue_regions': self.config.tissue_regions,
            'protein_properties': self.protein_properties,
            
            # Validation
            'validation_metrics': validation_metrics,
            'config': self.config
        }
        
        # Cache for reuse
        self._generated_data = dataset
        
        return dataset
    
    def _compute_validation_metrics(self, coordinates: np.ndarray, 
                                  expressions: Dict[str, np.ndarray],
                                  cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Compute validation metrics for the synthetic dataset."""
        metrics = {}
        
        # Spatial statistics
        try:
            # Ripley's K for spatial pattern validation
            distances, k_values = compute_ripleys_k(
                coordinates, max_distance=100, n_bins=20
            )
            metrics['ripleys_k'] = {
                'distances': distances.tolist(),
                'k_values': k_values.tolist()
            }
        except:
            metrics['ripleys_k'] = None
        
        # Protein correlations
        protein_corrs = {}
        protein_names = list(expressions.keys())
        for i, protein1 in enumerate(protein_names):
            for protein2 in protein_names[i+1:]:
                try:
                    corr = compute_spatial_correlation(
                        expressions[protein1].reshape(-1, 1),
                        expressions[protein2].reshape(-1, 1)
                    )
                    protein_corrs[f"{protein1}_{protein2}"] = float(corr)
                except:
                    protein_corrs[f"{protein1}_{protein2}"] = 0.0
        
        metrics['protein_correlations'] = protein_corrs
        
        # Cluster statistics
        unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
        metrics['cluster_stats'] = {
            'n_clusters': len(unique_clusters),
            'cluster_sizes': [np.sum(cluster_labels == c) for c in unique_clusters],
            'silhouette_score': self._compute_silhouette_score(expressions, cluster_labels)
        }
        
        # Spatial coherence of clusters
        try:
            from .spatial_clustering import compute_spatial_coherence
            spatial_coherence = compute_spatial_coherence(cluster_labels, coordinates)
            metrics['spatial_coherence'] = float(spatial_coherence)
        except:
            metrics['spatial_coherence'] = None
        
        return metrics
    
    def _compute_silhouette_score(self, expressions: Dict[str, np.ndarray], 
                                 cluster_labels: np.ndarray) -> float:
        """Compute silhouette score for clustering quality."""
        try:
            from sklearn.metrics import silhouette_score
            
            # Create feature matrix
            feature_list = [expr.reshape(-1, 1) for expr in expressions.values()]
            if feature_list:
                feature_matrix = np.hstack(feature_list)
                
                # Only compute for labeled points
                labeled_mask = cluster_labels >= 0
                if np.sum(labeled_mask) > 10 and len(np.unique(cluster_labels[labeled_mask])) > 1:
                    score = silhouette_score(
                        feature_matrix[labeled_mask],
                        cluster_labels[labeled_mask]
                    )
                    return float(score)
        except:
            pass
        
        return 0.0
    
    def create_validation_dataset(self, noise_levels: List[float] = None) -> Dict[str, Any]:
        """
        Create dataset specifically designed for validation testing.
        
        Includes multiple noise levels and known artifacts for robustness testing.
        """
        if noise_levels is None:
            noise_levels = [0.05, 0.1, 0.2, 0.4]
        
        base_dataset = self.generate_complete_dataset()
        validation_dataset = {'base': base_dataset}
        
        # Generate variants with different noise levels
        for noise_level in noise_levels:
            # Modify config for this noise level
            modified_config = SyntheticDataConfig(
                roi_size_um=self.config.roi_size_um,
                pixel_size_um=self.config.pixel_size_um,
                n_cells_total=self.config.n_cells_total,
                protein_names=self.config.protein_names,
                tissue_regions=self.config.tissue_regions,
                baseline_noise_level=noise_level,
                hot_pixel_probability=self.config.hot_pixel_probability * (1 + noise_level),
                batch_effect_strength=self.config.batch_effect_strength * (1 + noise_level)
            )
            
            # Generate noisy variant
            noisy_generator = SyntheticDataGenerator(modified_config)
            noisy_generator.protein_properties = self.protein_properties
            noisy_dataset = noisy_generator.generate_complete_dataset()
            
            validation_dataset[f'noise_{noise_level:.2f}'] = noisy_dataset
        
        return validation_dataset


class SyntheticDataValidator(ValidationRule):
    """Validation rule for synthetic data quality."""
    
    def __init__(self):
        super().__init__(
            name="SyntheticDataValidator",
            category=ValidationCategory.SCIENTIFIC_QUALITY
        )
    
    def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Validate synthetic dataset quality."""
        try:
            # Check required components
            required_keys = ['coordinates', 'ion_counts', 'ground_truth_clusters']
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                return self._create_result(
                    ValidationSeverity.CRITICAL,
                    f"Missing required components: {missing_keys}",
                    quality_score=0.0
                )
            
            # Validate data consistency
            n_cells = len(data['coordinates'])
            cluster_labels = data['ground_truth_clusters']
            
            if len(cluster_labels) != n_cells:
                return self._create_result(
                    ValidationSeverity.CRITICAL,
                    f"Size mismatch: coordinates ({n_cells}) vs clusters ({len(cluster_labels)})",
                    quality_score=0.0
                )
            
            # Check protein expression distributions
            ion_counts = data['ion_counts']
            quality_issues = []
            protein_quality_scores = []
            
            for protein_name, counts in ion_counts.items():
                if len(counts) != n_cells:
                    quality_issues.append(f"{protein_name}: size mismatch")
                    continue
                
                # Check for realistic distribution
                if np.all(counts == 0):
                    quality_issues.append(f"{protein_name}: all zeros")
                    protein_quality_scores.append(0.0)
                elif np.std(counts) == 0:
                    quality_issues.append(f"{protein_name}: no variation")
                    protein_quality_scores.append(0.2)
                else:
                    # Check if distribution is reasonable for ion counts
                    mean_val = np.mean(counts)
                    var_val = np.var(counts)
                    cv = np.sqrt(var_val) / mean_val if mean_val > 0 else float('inf')
                    
                    # For Poisson-like data, CV should be reasonable
                    if cv < 0.1:
                        protein_quality_scores.append(0.5)  # Too uniform
                    elif cv > 10:
                        protein_quality_scores.append(0.7)  # Very noisy but realistic
                    else:
                        protein_quality_scores.append(1.0)  # Good
            
            overall_quality = np.mean(protein_quality_scores) if protein_quality_scores else 0.0
            
            # Check cluster validity
            unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
            n_clusters = len(unique_clusters)
            
            if n_clusters < 2:
                quality_issues.append("Too few clusters for meaningful analysis")
                overall_quality *= 0.5
            elif n_clusters > 20:
                quality_issues.append("Too many clusters may indicate over-segmentation")
                overall_quality *= 0.8
            
            # Determine severity
            if overall_quality >= 0.8:
                severity = ValidationSeverity.PASS
                message = "Synthetic dataset validation passed"
            elif overall_quality >= 0.6:
                severity = ValidationSeverity.WARNING
                message = f"Synthetic dataset has quality issues: {'; '.join(quality_issues)}"
            else:
                severity = ValidationSeverity.CRITICAL
                message = f"Synthetic dataset failed validation: {'; '.join(quality_issues)}"
            
            return self._create_result(
                severity,
                message,
                quality_score=overall_quality,
                context={
                    'n_cells': n_cells,
                    'n_proteins': len(ion_counts),
                    'n_clusters': n_clusters,
                    'quality_issues': quality_issues
                }
            )
            
        except Exception as e:
            return self._create_result(
                ValidationSeverity.CRITICAL,
                f"Validation failed with exception: {str(e)}",
                quality_score=0.0
            )


def create_example_datasets() -> Dict[str, Dict[str, Any]]:
    """Create example synthetic datasets for different scenarios."""
    datasets = {}
    
    # 1. Simple validation dataset
    simple_config = SyntheticDataConfig(
        roi_size_um=(500.0, 500.0),
        n_cells_total=5000,
        protein_names=['CD45', 'CD3', 'CD20', 'PanCK', 'DNA1'],
        known_cluster_count=3,
        baseline_noise_level=0.1
    )
    simple_generator = SyntheticDataGenerator(simple_config)
    datasets['simple'] = simple_generator.generate_complete_dataset()
    
    # 2. Complex tissue architecture
    complex_config = SyntheticDataConfig(
        roi_size_um=(1000.0, 1000.0),
        n_cells_total=15000,
        protein_names=['CD45', 'CD3', 'CD4', 'CD8', 'CD20', 'CD68', 'PanCK', 'Vimentin', 'DNA1'],
        known_cluster_count=7,
        baseline_noise_level=0.15,
        hot_pixel_probability=0.002,
        spillover_strength=0.08
    )
    complex_generator = SyntheticDataGenerator(complex_config)
    datasets['complex'] = complex_generator.generate_complete_dataset()
    
    # 3. High noise scenario
    noisy_config = SyntheticDataConfig(
        roi_size_um=(800.0, 800.0),
        n_cells_total=8000,
        protein_names=['CD45', 'CD3', 'CD20', 'PanCK', 'Vimentin', 'DNA1'],
        known_cluster_count=4,
        baseline_noise_level=0.3,
        hot_pixel_probability=0.005,
        batch_effect_strength=0.4,
        spillover_strength=0.1
    )
    noisy_generator = SyntheticDataGenerator(noisy_config)
    datasets['high_noise'] = noisy_generator.generate_complete_dataset()
    
    return datasets


def validate_synthetic_dataset(dataset: Dict[str, Any]) -> ValidationResult:
    """Convenience function to validate a synthetic dataset."""
    validator = SyntheticDataValidator()
    return validator.validate(dataset)


if __name__ == "__main__":
    # Example usage
    print("Generating example synthetic datasets...")
    
    # Create datasets
    datasets = create_example_datasets()
    
    # Validate each dataset
    for name, dataset in datasets.items():
        print(f"\nValidating {name} dataset:")
        result = validate_synthetic_dataset(dataset)
        print(f"  Status: {result.severity.value}")
        print(f"  Quality Score: {result.quality_score:.2f}")
        print(f"  Message: {result.message}")
        
        # Print dataset stats
        print(f"  Cells: {len(dataset['coordinates'])}")
        print(f"  Proteins: {len(dataset['ion_counts'])}")
        print(f"  Clusters: {len(np.unique(dataset['ground_truth_clusters']))}")
        
        # Spatial statistics
        if 'validation_metrics' in dataset and dataset['validation_metrics'].get('spatial_coherence'):
            print(f"  Spatial Coherence: {dataset['validation_metrics']['spatial_coherence']:.3f}")
    
    print("\nSynthetic data generation complete!")
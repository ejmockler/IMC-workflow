"""
Pixel-Level Spatial Statistics for IMC Analysis
Statistical methods for analyzing spatial patterns without cell segmentation
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from scipy.spatial import cKDTree, distance_matrix
from scipy import stats, spatial
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SpatialStatisticsResult:
    """Container for spatial statistics results"""
    morans_i: float  # Global Moran's I
    morans_p: float  # P-value for Moran's I
    gearys_c: float  # Geary's C
    gearys_p: float  # P-value for Geary's C
    local_morans: np.ndarray  # Local Moran's I for each point
    getis_ord: np.ndarray  # Getis-Ord Gi* statistics
    variogram: Dict[str, np.ndarray]  # Variogram analysis
    ripley_k: Dict[str, np.ndarray]  # Ripley's K function
    protein: Optional[str]  # Protein name if single-channel


class SpatialStatistics:
    """Computes spatial statistics for protein expression patterns"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize spatial statistics calculator
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.bandwidth = self.config.get('bandwidth', 50)
        self.permutations = self.config.get('permutations', 999)
        self.methods = self.config.get('methods', 
                                       ['morans_i', 'gearys_c', 'getis_ord'])
    
    def analyze(self, coords: np.ndarray,
               values: np.ndarray,
               protein_idx: Optional[int] = None) -> SpatialStatisticsResult:
        """
        Compute spatial statistics for expression data
        
        Args:
            coords: Spatial coordinates (n_points, 2)
            values: Expression values (n_points, n_proteins)
            protein_idx: Index of specific protein (None = mean)
            
        Returns:
            SpatialStatisticsResult
        """
        # Select expression channel
        if protein_idx is not None:
            expression = values[:, protein_idx]
            protein_name = f"Protein_{protein_idx}"
        else:
            expression = values.mean(axis=1)
            protein_name = "Mean_Expression"
        
        # Build spatial weights matrix
        W = self._build_weights_matrix(coords)
        
        # Compute global statistics
        morans_i, morans_p = self._morans_i(expression, W) if 'morans_i' in self.methods else (0, 1)
        gearys_c, gearys_p = self._gearys_c(expression, W) if 'gearys_c' in self.methods else (0, 1)
        
        # Compute local statistics
        local_morans = self._local_morans_i(expression, W) if 'morans_i' in self.methods else np.zeros(len(coords))
        getis_ord = self._getis_ord(expression, coords) if 'getis_ord' in self.methods else np.zeros(len(coords))
        
        # Compute variogram
        variogram = self._compute_variogram(coords, expression) if 'variogram' in self.methods else {}
        
        # Compute Ripley's K
        ripley_k = self._ripleys_k(coords, expression) if 'ripley_k' in self.methods else {}
        
        return SpatialStatisticsResult(
            morans_i=morans_i,
            morans_p=morans_p,
            gearys_c=gearys_c,
            gearys_p=gearys_p,
            local_morans=local_morans,
            getis_ord=getis_ord,
            variogram=variogram,
            ripley_k=ripley_k,
            protein=protein_name
        )
    
    def _build_weights_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Build spatial weights matrix"""
        n = len(coords)
        
        # Compute distance matrix
        dist_matrix = distance_matrix(coords, coords)
        
        # Create weights based on distance decay
        W = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = dist_matrix[i, j]
                    if d <= self.bandwidth:
                        # Gaussian kernel
                        W[i, j] = np.exp(-(d ** 2) / (2 * (self.bandwidth / 2) ** 2))
        
        # Row-standardize
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        W = W / row_sums[:, np.newaxis]
        
        return W
    
    def _morans_i(self, values: np.ndarray, W: np.ndarray) -> Tuple[float, float]:
        """
        Compute global Moran's I statistic
        
        Moran's I measures spatial autocorrelation:
        - I > 0: positive autocorrelation (clustering)
        - I = 0: random pattern
        - I < 0: negative autocorrelation (dispersion)
        """
        n = len(values)
        
        # Center values
        values_centered = values - values.mean()
        
        # Compute Moran's I
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * values_centered[i] * values_centered[j]
        
        denominator = np.sum(values_centered ** 2)
        
        if denominator == 0:
            return 0.0, 1.0
        
        S0 = np.sum(W)  # Sum of all weights
        I = (n / S0) * (numerator / denominator)
        
        # Compute expected value and variance under null hypothesis
        E_I = -1 / (n - 1)
        
        # Simplified variance calculation
        b2 = np.sum(values_centered ** 4) / n / (np.sum(values_centered ** 2) / n) ** 2
        
        S1 = 0.5 * np.sum((W + W.T) ** 2)
        S2 = np.sum(np.sum(W + W.T, axis=1) ** 2)
        S3 = (np.sum(values_centered ** 4) / n) / (np.sum(values_centered ** 2) / n) ** 2
        S4 = (n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2
        S5 = (n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2
        
        var_I = ((n * S4 - S3 * S5) / ((n - 1) * (n - 2) * (n - 3) * S0 ** 2) - E_I ** 2)
        
        if var_I <= 0:
            return I, 1.0
        
        # Z-score
        z = (I - E_I) / np.sqrt(var_I)
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return I, p_value
    
    def _gearys_c(self, values: np.ndarray, W: np.ndarray) -> Tuple[float, float]:
        """
        Compute Geary's C statistic
        
        Geary's C also measures spatial autocorrelation:
        - C < 1: positive autocorrelation
        - C = 1: random pattern
        - C > 1: negative autocorrelation
        """
        n = len(values)
        
        # Center values
        values_centered = values - values.mean()
        
        # Compute Geary's C
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * (values[i] - values[j]) ** 2
        
        denominator = np.sum(values_centered ** 2)
        
        if denominator == 0:
            return 1.0, 1.0
        
        S0 = np.sum(W)
        C = ((n - 1) / (2 * S0)) * (numerator / denominator)
        
        # Expected value
        E_C = 1.0
        
        # Simplified variance (approximation)
        var_C = ((2 * S0 + np.sum(W ** 2)) * (n - 1) - 4 * S0 ** 2) / (2 * (n + 1) * S0 ** 2)
        
        if var_C <= 0:
            return C, 1.0
        
        # Z-score
        z = (C - E_C) / np.sqrt(var_C)
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return C, p_value
    
    def _local_morans_i(self, values: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute Local Moran's I (LISA) for each point
        
        Identifies local clusters and outliers:
        - High-High: hot spots
        - Low-Low: cold spots
        - High-Low: spatial outliers
        - Low-High: spatial outliers
        """
        n = len(values)
        local_i = np.zeros(n)
        
        # Standardize values
        values_std = (values - values.mean()) / values.std()
        
        for i in range(n):
            # Local Moran's I for point i
            weighted_sum = 0
            for j in range(n):
                if i != j:
                    weighted_sum += W[i, j] * values_std[j]
            
            local_i[i] = values_std[i] * weighted_sum
        
        return local_i
    
    def _getis_ord(self, values: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """
        Compute Getis-Ord Gi* statistic for hot spot detection
        
        Gi* > 0: hot spot (high values cluster)
        Gi* < 0: cold spot (low values cluster)
        Gi* ≈ 0: no significant clustering
        """
        n = len(values)
        gi_star = np.zeros(n)
        
        # Build KDTree for neighbor search
        tree = cKDTree(coords)
        
        for i in range(n):
            # Find neighbors within bandwidth
            neighbors = tree.query_ball_point(coords[i], self.bandwidth)
            
            if len(neighbors) <= 1:
                continue
            
            # Include point i itself (Gi*)
            x_neighbors = values[neighbors]
            
            # Compute weighted sum
            distances = np.array([np.linalg.norm(coords[i] - coords[j]) 
                                 for j in neighbors])
            weights = np.exp(-(distances ** 2) / (2 * (self.bandwidth / 2) ** 2))
            weights[distances == 0] = 1  # Self-weight
            
            weighted_sum = np.sum(weights * x_neighbors)
            sum_weights = np.sum(weights)
            sum_weights_sq = np.sum(weights ** 2)
            
            # Global statistics
            x_mean = values.mean()
            x_std = values.std()
            
            if x_std == 0 or sum_weights == 0:
                continue
            
            # Compute Gi*
            numerator = weighted_sum - x_mean * sum_weights
            denominator = x_std * np.sqrt((n * sum_weights_sq - sum_weights ** 2) / (n - 1))
            
            if denominator > 0:
                gi_star[i] = numerator / denominator
        
        return gi_star
    
    def _compute_variogram(self, coords: np.ndarray, 
                          values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute empirical variogram for spatial structure analysis
        
        Variogram shows how variance changes with distance
        """
        # Define lag distances
        max_dist = np.sqrt(np.sum((coords.max(axis=0) - coords.min(axis=0)) ** 2)) / 3
        n_lags = 20
        lag_size = max_dist / n_lags
        lags = np.arange(lag_size, max_dist + lag_size, lag_size)
        
        # Compute pairwise differences and distances
        n = len(values)
        semivariance = []
        lag_centers = []
        n_pairs = []
        
        for lag in lags:
            lag_min = lag - lag_size / 2
            lag_max = lag + lag_size / 2
            
            variance_sum = 0
            pair_count = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    
                    if lag_min <= dist < lag_max:
                        variance_sum += (values[i] - values[j]) ** 2
                        pair_count += 1
            
            if pair_count > 0:
                semivariance.append(variance_sum / (2 * pair_count))
                lag_centers.append(lag)
                n_pairs.append(pair_count)
        
        return {
            'distance': np.array(lag_centers),
            'semivariance': np.array(semivariance),
            'n_pairs': np.array(n_pairs)
        }
    
    def _ripleys_k(self, coords: np.ndarray, 
                   values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute Ripley's K function adapted for continuous values
        
        Measures clustering at different spatial scales
        """
        # Define radii to test
        max_radius = np.sqrt(np.sum((coords.max(axis=0) - coords.min(axis=0)) ** 2)) / 4
        radii = np.linspace(0, max_radius, 50)
        
        # Study area
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        area = x_range * y_range
        
        n = len(coords)
        intensity = n / area
        
        # Weight points by expression values
        weights = values / values.sum()
        
        # Compute K function
        k_values = []
        
        for r in radii:
            k_sum = 0
            
            for i in range(n):
                # Find neighbors within radius r
                distances = np.linalg.norm(coords - coords[i], axis=1)
                within_radius = distances <= r
                within_radius[i] = False  # Exclude self
                
                # Weight by expression values
                k_sum += weights[i] * np.sum(weights[within_radius])
            
            k_values.append(k_sum * area)
        
        # Compute L function (variance-stabilized K)
        l_values = np.sqrt(np.array(k_values) / np.pi) - radii
        
        # Expected K under CSR (complete spatial randomness)
        expected_k = np.pi * radii ** 2
        
        return {
            'radius': radii,
            'K': np.array(k_values),
            'L': l_values,
            'expected_K': expected_k
        }


class MultiProteinSpatialStatistics(SpatialStatistics):
    """Spatial statistics for multiple proteins with cross-correlation"""
    
    def analyze_all_proteins(self, coords: np.ndarray,
                            values: np.ndarray,
                            protein_names: List[str]) -> Dict[str, SpatialStatisticsResult]:
        """
        Analyze spatial statistics for all proteins
        
        Args:
            coords: Spatial coordinates
            values: Expression values (n_points, n_proteins)
            protein_names: List of protein names
            
        Returns:
            Dictionary mapping protein names to results
        """
        results = {}
        
        for i, protein_name in enumerate(protein_names):
            result = self.analyze(coords, values, protein_idx=i)
            result.protein = protein_name
            results[protein_name] = result
        
        # Also compute for mean expression
        mean_result = self.analyze(coords, values, protein_idx=None)
        mean_result.protein = "Mean_Expression"
        results["Mean_Expression"] = mean_result
        
        return results
    
    def bivariate_morans_i(self, coords: np.ndarray,
                           values1: np.ndarray,
                           values2: np.ndarray) -> Tuple[float, float]:
        """
        Compute bivariate Moran's I for spatial cross-correlation
        
        Args:
            coords: Spatial coordinates
            values1: First variable values
            values2: Second variable values
            
        Returns:
            Bivariate Moran's I and p-value
        """
        W = self._build_weights_matrix(coords)
        n = len(values1)
        
        # Standardize values
        z1 = (values1 - values1.mean()) / values1.std()
        z2 = (values2 - values2.mean()) / values2.std()
        
        # Compute bivariate Moran's I
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * z1[i] * z2[j]
        
        I_bivariate = numerator / n
        
        # Permutation test for significance
        I_permuted = []
        for _ in range(self.permutations):
            z2_perm = np.random.permutation(z2)
            num_perm = 0
            for i in range(n):
                for j in range(n):
                    num_perm += W[i, j] * z1[i] * z2_perm[j]
            I_permuted.append(num_perm / n)
        
        I_permuted = np.array(I_permuted)
        p_value = np.mean(np.abs(I_permuted) >= np.abs(I_bivariate))
        
        return I_bivariate, p_value


def identify_spatial_patterns(coords: np.ndarray,
                             values: np.ndarray,
                             pattern_types: List[str] = ['hotspots', 'gradients']) -> Dict[str, np.ndarray]:
    """
    Identify different types of spatial patterns
    
    Args:
        coords: Spatial coordinates
        values: Expression values
        pattern_types: Types of patterns to identify
        
    Returns:
        Dictionary mapping pattern type to labels/values
    """
    analyzer = SpatialStatistics()
    results = {}
    
    if 'hotspots' in pattern_types:
        # Use Getis-Ord Gi* for hotspot detection
        stats_result = analyzer.analyze(coords, values)
        gi_star = stats_result.getis_ord
        
        # Classify into hot/cold spots
        hotspots = np.zeros(len(coords), dtype=int)
        hotspots[gi_star > 1.96] = 1  # Hot spots (p < 0.05)
        hotspots[gi_star < -1.96] = -1  # Cold spots
        
        results['hotspots'] = hotspots
    
    if 'gradients' in pattern_types:
        # Detect spatial gradients
        from scipy.interpolate import griddata
        
        # Create regular grid
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        grid_x, grid_y = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
        
        # Interpolate values to grid
        if values.ndim == 1:
            grid_values = griddata(coords, values, (grid_x, grid_y), method='cubic')
        else:
            grid_values = griddata(coords, values.mean(axis=1), (grid_x, grid_y), method='cubic')
        
        # Compute gradient
        gy, gx = np.gradient(grid_values)
        gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
        
        # Map back to original coordinates
        gradient_at_points = griddata(
            np.column_stack([grid_x.ravel(), grid_y.ravel()]),
            gradient_magnitude.ravel(),
            coords,
            method='nearest'
        )
        
        results['gradients'] = gradient_at_points
    
    return results
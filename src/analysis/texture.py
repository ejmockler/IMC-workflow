"""
Spatial Texture Analysis for IMC Data
Quantifies spatial patterns without requiring cell segmentation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from scipy import ndimage
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TextureFeatures:
    """Container for texture analysis results"""
    haralick: Dict[str, np.ndarray]  # Haralick texture features
    lbp: np.ndarray  # Local binary pattern histogram
    glcm: np.ndarray  # Gray-level co-occurrence matrix
    statistics: Dict[str, float]  # Basic spatial statistics
    scale: int  # Analysis scale (window size)
    protein: Optional[str]  # Protein name if single-channel


class TextureAnalyzer:
    """Analyzes spatial texture patterns in protein expression"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize texture analyzer
        
        Args:
            config: Analysis configuration
        """
        self.config = config or {}
        self.window_sizes = self.config.get('window_sizes', [10, 50, 100])
        self.n_gray_levels = self.config.get('n_gray_levels', 32)
        self.features_to_compute = self.config.get('features', 
                                                   ['haralick', 'lbp', 'glcm'])
    
    def analyze(self, coords: np.ndarray,
               values: np.ndarray,
               protein_idx: Optional[int] = None) -> Dict[int, TextureFeatures]:
        """
        Analyze texture at multiple scales
        
        Args:
            coords: Spatial coordinates (n_pixels, 2)
            values: Expression values (n_pixels, n_proteins)
            protein_idx: Index of specific protein to analyze (None = mean)
            
        Returns:
            Dictionary mapping scale to TextureFeatures
        """
        # Select protein channel
        if protein_idx is not None:
            expression = values[:, protein_idx]
            protein_name = f"Protein_{protein_idx}"
        else:
            expression = values.mean(axis=1)
            protein_name = "Mean_Expression"
        
        # Create image representation
        image = self._create_image(coords, expression)
        
        # Analyze at each scale
        results = {}
        for window_size in self.window_sizes:
            features = self._analyze_scale(image, window_size)
            features.protein = protein_name
            results[window_size] = features
        
        return results
    
    def _create_image(self, coords: np.ndarray, 
                     expression: np.ndarray) -> np.ndarray:
        """Create 2D image from scattered coordinates"""
        # Determine image bounds
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        # Create image grid
        resolution = self.config.get('resolution', 1.0)
        height = int((y_max - y_min) * resolution) + 1
        width = int((x_max - x_min) * resolution) + 1
        
        image = np.zeros((height, width))
        
        # Map coordinates to image indices
        x_idx = ((coords[:, 0] - x_min) / (x_max - x_min) * (width - 1)).astype(int)
        y_idx = ((coords[:, 1] - y_min) / (y_max - y_min) * (height - 1)).astype(int)
        
        # Fill image
        for i, (x, y) in enumerate(zip(x_idx, y_idx)):
            image[y, x] = expression[i]
        
        # Interpolate missing values
        mask = image == 0
        if mask.any():
            indices = ndimage.distance_transform_edt(
                mask, return_distances=False, return_indices=True
            )
            image = image[tuple(indices)]
        
        return image
    
    def _analyze_scale(self, image: np.ndarray, 
                      window_size: int) -> TextureFeatures:
        """Analyze texture at specific scale"""
        features = {}
        
        # Quantize image for texture analysis
        quantized = self._quantize_image(image)
        
        # Compute requested features
        if 'haralick' in self.features_to_compute:
            features['haralick'] = self._compute_haralick(quantized, window_size)
        
        if 'lbp' in self.features_to_compute:
            features['lbp'] = self._compute_lbp(image, window_size)
        
        if 'glcm' in self.features_to_compute:
            features['glcm'] = self._compute_glcm(quantized, window_size)
        
        # Basic statistics
        features['statistics'] = self._compute_statistics(image, window_size)
        
        return TextureFeatures(
            haralick=features.get('haralick', {}),
            lbp=features.get('lbp', np.array([])),
            glcm=features.get('glcm', np.array([])),
            statistics=features.get('statistics', {}),
            scale=window_size,
            protein=None
        )
    
    def _quantize_image(self, image: np.ndarray) -> np.ndarray:
        """Quantize image to specified gray levels"""
        # Normalize to [0, 1]
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
        
        # Quantize to n_gray_levels
        quantized = (img_norm * (self.n_gray_levels - 1)).astype(np.uint8)
        
        return quantized
    
    def _compute_haralick(self, image: np.ndarray, 
                         window_size: int) -> Dict[str, np.ndarray]:
        """Compute Haralick texture features"""
        features = {}
        
        # Define distances and angles for GLCM
        distances = [1, 2, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Compute GLCM
        glcm = graycomatrix(image, distances=distances, angles=angles,
                          levels=self.n_gray_levels, symmetric=True,
                          normed=True)
        
        # Extract Haralick features
        feature_names = ['contrast', 'dissimilarity', 'homogeneity',
                        'energy', 'correlation', 'ASM']
        
        for feature_name in feature_names:
            features[feature_name] = graycoprops(glcm, feature_name).flatten()
        
        # Custom features
        features['entropy'] = self._compute_glcm_entropy(glcm)
        features['variance'] = self._compute_glcm_variance(glcm)
        
        return features
    
    def _compute_glcm_entropy(self, glcm: np.ndarray) -> np.ndarray:
        """Compute entropy from GLCM"""
        eps = 1e-10
        entropy_vals = []
        
        for d in range(glcm.shape[2]):
            for a in range(glcm.shape[3]):
                p = glcm[:, :, d, a].flatten()
                p = p[p > 0]  # Remove zeros
                if len(p) > 0:
                    entropy_vals.append(-np.sum(p * np.log(p + eps)))
                else:
                    entropy_vals.append(0)
        
        return np.array(entropy_vals)
    
    def _compute_glcm_variance(self, glcm: np.ndarray) -> np.ndarray:
        """Compute variance from GLCM"""
        variance_vals = []
        
        for d in range(glcm.shape[2]):
            for a in range(glcm.shape[3]):
                matrix = glcm[:, :, d, a]
                i, j = np.meshgrid(range(matrix.shape[0]), 
                                  range(matrix.shape[1]), 
                                  indexing='ij')
                mean_i = np.sum(i * matrix)
                variance = np.sum(matrix * (i - mean_i) ** 2)
                variance_vals.append(variance)
        
        return np.array(variance_vals)
    
    def _compute_lbp(self, image: np.ndarray, 
                    window_size: int) -> np.ndarray:
        """Compute Local Binary Pattern features"""
        # LBP parameters
        radius = min(3, window_size // 10)
        n_points = 8 * radius
        
        # Compute LBP
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        # Compute histogram
        n_bins = n_points + 2  # Number of uniform patterns + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, 
                              range=(0, n_bins), density=True)
        
        return hist
    
    def _compute_glcm(self, image: np.ndarray, 
                     window_size: int) -> np.ndarray:
        """Compute raw GLCM for detailed analysis"""
        # Simple GLCM at distance 1, angle 0
        glcm = graycomatrix(image, distances=[1], angles=[0],
                          levels=self.n_gray_levels,
                          symmetric=True, normed=True)
        
        return glcm[:, :, 0, 0]  # Return 2D matrix
    
    def _compute_statistics(self, image: np.ndarray, 
                           window_size: int) -> Dict[str, float]:
        """Compute basic spatial statistics"""
        from scipy import stats
        
        # Create windowed view
        kernel_size = min(window_size, min(image.shape))
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        # Local statistics
        local_mean = ndimage.convolve(image, kernel, mode='reflect')
        local_var = ndimage.generic_filter(image, np.var, 
                                          size=kernel_size, mode='reflect')
        
        statistics = {
            'global_mean': float(np.mean(image)),
            'global_std': float(np.std(image)),
            'global_skewness': float(stats.skew(image.ravel())),
            'global_kurtosis': float(stats.kurtosis(image.ravel())),
            'local_mean_std': float(np.std(local_mean)),
            'local_var_mean': float(np.mean(local_var)),
            'entropy': float(entropy(np.histogram(image, bins=self.n_gray_levels)[0] + 1e-10))
        }
        
        return statistics


class MultiProteinTextureAnalyzer(TextureAnalyzer):
    """Analyzes texture patterns across multiple proteins"""
    
    def analyze_all_proteins(self, coords: np.ndarray,
                            values: np.ndarray,
                            protein_names: List[str]) -> Dict[str, Dict[int, TextureFeatures]]:
        """
        Analyze texture for all proteins
        
        Args:
            coords: Spatial coordinates
            values: Expression values (n_pixels, n_proteins)
            protein_names: List of protein names
            
        Returns:
            Nested dict: protein -> scale -> features
        """
        results = {}
        
        for i, protein_name in enumerate(protein_names):
            protein_features = self.analyze(coords, values, protein_idx=i)
            
            # Update protein name in features
            for scale_features in protein_features.values():
                scale_features.protein = protein_name
            
            results[protein_name] = protein_features
        
        # Also compute combined texture
        combined_features = self.analyze(coords, values, protein_idx=None)
        for scale_features in combined_features.values():
            scale_features.protein = "Combined"
        results["Combined"] = combined_features
        
        return results
    
    def compute_texture_similarity(self, features1: TextureFeatures,
                                  features2: TextureFeatures) -> float:
        """
        Compute similarity between two texture patterns
        
        Args:
            features1: First texture features
            features2: Second texture features
            
        Returns:
            Similarity score [0, 1]
        """
        similarities = []
        
        # Compare Haralick features
        if features1.haralick and features2.haralick:
            for key in features1.haralick:
                if key in features2.haralick:
                    f1 = features1.haralick[key]
                    f2 = features2.haralick[key]
                    # Correlation coefficient
                    if len(f1) > 0 and len(f2) > 0:
                        corr = np.corrcoef(f1, f2)[0, 1]
                        similarities.append(corr)
        
        # Compare LBP histograms
        if len(features1.lbp) > 0 and len(features2.lbp) > 0:
            # Chi-square distance converted to similarity
            chi2 = np.sum((features1.lbp - features2.lbp) ** 2 / 
                         (features1.lbp + features2.lbp + 1e-10))
            lbp_sim = np.exp(-chi2)
            similarities.append(lbp_sim)
        
        # Compare statistics
        if features1.statistics and features2.statistics:
            stat_diffs = []
            for key in features1.statistics:
                if key in features2.statistics:
                    v1 = features1.statistics[key]
                    v2 = features2.statistics[key]
                    # Normalized difference
                    diff = abs(v1 - v2) / (abs(v1) + abs(v2) + 1e-10)
                    stat_diffs.append(diff)
            
            if stat_diffs:
                stat_sim = 1 - np.mean(stat_diffs)
                similarities.append(stat_sim)
        
        return np.mean(similarities) if similarities else 0.0


def classify_tissue_texture(coords: np.ndarray,
                           values: np.ndarray,
                           n_classes: int = 5) -> np.ndarray:
    """
    Classify tissue regions based on texture patterns
    
    Args:
        coords: Spatial coordinates
        values: Expression values
        n_classes: Number of texture classes
        
    Returns:
        Texture class labels for each pixel
    """
    from sklearn.cluster import KMeans
    
    # Analyze texture at medium scale
    analyzer = TextureAnalyzer({'window_sizes': [50]})
    features = analyzer.analyze(coords, values)
    
    # Extract feature vector
    texture_features = features[50]
    
    # Flatten Haralick features
    feature_vector = []
    for feat_array in texture_features.haralick.values():
        feature_vector.extend(feat_array)
    
    # Add LBP histogram
    feature_vector.extend(texture_features.lbp)
    
    # Add statistics
    feature_vector.extend(texture_features.statistics.values())
    
    # Reshape for clustering (treat as single sample for now)
    # In practice, would compute local texture features
    feature_vector = np.array(feature_vector).reshape(1, -1)
    
    # For actual pixel-level classification, would need sliding window
    # This is simplified version
    labels = np.zeros(len(coords), dtype=int)
    
    return labels
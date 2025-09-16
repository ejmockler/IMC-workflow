"""Metadata-driven analysis framework for IMC data without ground truth.

This module implements scientifically valid analyses that can be performed
using only experimental metadata, without pathologist annotations or ground truth.

CRITICAL: This is discovery analysis, not validation. All findings must be
interpreted as hypothesis-generating, not definitive biological truth.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

from src.config import Config


class MetadataDrivenAnalysis:
    """Analysis framework that properly leverages experimental metadata."""
    
    def __init__(self, config: Config):
        """Initialize metadata-driven analysis.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
    def analyze_differential_expression(self, 
                                       values: np.ndarray,
                                       metadata: pd.DataFrame,
                                       group_column: str,
                                       protein_names: List[str]) -> Dict[str, Any]:
        """Analyze differential protein expression between metadata-defined groups.
        
        This answers: "Does marker expression differ between experimental conditions?"
        NOT: "What cell types are present?"
        
        Args:
            values: Protein expression values (N pixels/cells, M proteins)
            metadata: DataFrame with experimental metadata
            group_column: Column name for grouping (e.g., 'condition', 'timepoint')
            protein_names: List of protein marker names
            
        Returns:
            Dictionary with differential expression results
        """
        results = {
            'group_column': group_column,
            'groups': [],
            'differential_markers': {},
            'statistical_tests': {}
        }
        
        # Get unique groups
        groups = metadata[group_column].unique()
        results['groups'] = list(groups)
        
        if len(groups) < 2:
            results['error'] = f"Need at least 2 groups for differential analysis, found {len(groups)}"
            return results
        
        # Calculate mean expression per group
        group_means = {}
        group_stds = {}
        
        for group in groups:
            group_mask = metadata[group_column] == group
            group_values = values[group_mask]
            
            if len(group_values) == 0:
                continue
                
            group_means[group] = np.mean(group_values, axis=0)
            group_stds[group] = np.std(group_values, axis=0)
        
        results['group_means'] = {str(g): means.tolist() for g, means in group_means.items()}
        results['group_stds'] = {str(g): stds.tolist() for g, stds in group_stds.items()}
        
        # Perform statistical tests for each protein
        for i, protein in enumerate(protein_names):
            protein_results = {
                'protein': protein,
                'tests': {}
            }
            
            # Pairwise comparisons
            if len(groups) == 2:
                # Two-group comparison (t-test or Mann-Whitney U)
                group1_vals = values[metadata[group_column] == groups[0], i]
                group2_vals = values[metadata[group_column] == groups[1], i]
                
                # Check normality
                _, p_norm1 = stats.shapiro(group1_vals[:min(5000, len(group1_vals))])
                _, p_norm2 = stats.shapiro(group2_vals[:min(5000, len(group2_vals))])
                
                if p_norm1 > 0.05 and p_norm2 > 0.05:
                    # Use t-test
                    t_stat, p_value = stats.ttest_ind(group1_vals, group2_vals)
                    test_name = 't-test'
                    test_stat = t_stat
                else:
                    # Use Mann-Whitney U
                    u_stat, p_value = stats.mannwhitneyu(group1_vals, group2_vals)
                    test_name = 'Mann-Whitney U'
                    test_stat = u_stat
                
                protein_results['tests'][f"{groups[0]}_vs_{groups[1]}"] = {
                    'test': test_name,
                    'statistic': float(test_stat),
                    'p_value': float(p_value),
                    'fold_change': float(group_means[groups[1]][i] / (group_means[groups[0]][i] + 1e-10))
                }
                
            else:
                # Multiple group comparison (ANOVA or Kruskal-Wallis)
                group_values = [values[metadata[group_column] == g, i] for g in groups]
                
                # Check if we can use ANOVA
                normality_ok = all(stats.shapiro(gv[:min(5000, len(gv))])[1] > 0.05 
                                  for gv in group_values if len(gv) > 3)
                
                if normality_ok:
                    f_stat, p_value = stats.f_oneway(*group_values)
                    test_name = 'ANOVA'
                    test_stat = f_stat
                else:
                    h_stat, p_value = stats.kruskal(*group_values)
                    test_name = 'Kruskal-Wallis'
                    test_stat = h_stat
                
                protein_results['tests']['overall'] = {
                    'test': test_name,
                    'statistic': float(test_stat),
                    'p_value': float(p_value)
                }
            
            results['differential_markers'][protein] = protein_results
        
        # Multiple testing correction
        all_p_values = []
        test_labels = []
        
        for protein, protein_data in results['differential_markers'].items():
            for test_label, test_data in protein_data['tests'].items():
                all_p_values.append(test_data['p_value'])
                test_labels.append(f"{protein}_{test_label}")
        
        if all_p_values:
            from statsmodels.stats.multitest import multipletests
            _, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
            
            # Add corrected p-values
            idx = 0
            for protein, protein_data in results['differential_markers'].items():
                for test_label in protein_data['tests']:
                    protein_data['tests'][test_label]['p_value_corrected'] = float(corrected_p_values[idx])
                    idx += 1
        
        # Summary of significant markers
        significant_markers = []
        for protein, protein_data in results['differential_markers'].items():
            for test_label, test_data in protein_data['tests'].items():
                if test_data.get('p_value_corrected', test_data['p_value']) < 0.05:
                    significant_markers.append({
                        'protein': protein,
                        'comparison': test_label,
                        'p_value_corrected': test_data.get('p_value_corrected', test_data['p_value']),
                        'fold_change': test_data.get('fold_change', None)
                    })
        
        results['significant_markers'] = significant_markers
        results['interpretation'] = self._generate_differential_interpretation(results)
        
        return results
    
    def analyze_temporal_dynamics(self,
                                 values: np.ndarray,
                                 coordinates: np.ndarray,
                                 metadata: pd.DataFrame,
                                 time_column: str,
                                 protein_names: List[str]) -> Dict[str, Any]:
        """Analyze how protein expression changes over time.
        
        This answers: "How do markers change temporally?"
        NOT: "What is the mechanism of change?"
        
        Args:
            values: Protein expression values
            coordinates: Spatial coordinates
            metadata: DataFrame with timepoint information
            time_column: Column name for timepoints
            protein_names: List of protein names
            
        Returns:
            Temporal dynamics analysis results
        """
        results = {
            'time_column': time_column,
            'timepoints': [],
            'temporal_trends': {},
            'spatial_evolution': {}
        }
        
        # Get unique timepoints
        timepoints = sorted(metadata[time_column].unique())
        results['timepoints'] = list(timepoints)
        
        if len(timepoints) < 2:
            results['error'] = f"Need at least 2 timepoints, found {len(timepoints)}"
            return results
        
        # Analyze each protein's temporal trend
        for i, protein in enumerate(protein_names):
            protein_trend = {
                'protein': protein,
                'mean_expression_by_time': {},
                'variance_by_time': {},
                'trend_test': {}
            }
            
            means = []
            variances = []
            
            for timepoint in timepoints:
                time_mask = metadata[time_column] == timepoint
                time_values = values[time_mask, i]
                
                if len(time_values) > 0:
                    mean_val = float(np.mean(time_values))
                    var_val = float(np.var(time_values))
                    
                    protein_trend['mean_expression_by_time'][str(timepoint)] = mean_val
                    protein_trend['variance_by_time'][str(timepoint)] = var_val
                    
                    means.append(mean_val)
                    variances.append(var_val)
            
            # Test for temporal trend (Spearman correlation with time)
            if len(means) >= 3:
                time_numeric = np.arange(len(means))
                rho, p_value = stats.spearmanr(time_numeric, means)
                
                protein_trend['trend_test'] = {
                    'method': 'Spearman correlation',
                    'rho': float(rho),
                    'p_value': float(p_value),
                    'interpretation': self._interpret_trend(rho, p_value)
                }
            
            results['temporal_trends'][protein] = protein_trend
        
        # Analyze spatial pattern evolution
        results['spatial_evolution'] = self._analyze_spatial_evolution(
            values, coordinates, metadata, time_column, timepoints
        )
        
        results['interpretation'] = self._generate_temporal_interpretation(results)
        
        return results
    
    def discover_phenotypes(self,
                          values: np.ndarray,
                          protein_names: List[str],
                          n_phenotypes: Optional[int] = None) -> Dict[str, Any]:
        """Discover cellular/pixel phenotypes through unsupervised clustering.
        
        CRITICAL: These are mathematical clusters, not validated cell types.
        
        Args:
            values: Protein expression values
            protein_names: List of protein names
            n_phenotypes: Number of phenotypes to find (auto if None)
            
        Returns:
            Discovered phenotype information
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        results = {
            'method': 'unsupervised_clustering',
            'phenotypes': {},
            'assignments': None,
            'quality_metrics': {}
        }
        
        # Standardize features
        scaler = StandardScaler()
        values_scaled = scaler.fit_transform(values)
        
        # Determine optimal number of clusters if not specified
        if n_phenotypes is None:
            silhouette_scores = []
            k_range = range(2, min(15, len(values) // 100))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(values_scaled)
                score = silhouette_score(values_scaled, labels, sample_size=min(10000, len(values)))
                silhouette_scores.append(score)
            
            if silhouette_scores:
                n_phenotypes = list(k_range)[np.argmax(silhouette_scores)]
            else:
                n_phenotypes = 5  # Default
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_phenotypes, random_state=42, n_init=20)
        phenotype_labels = kmeans.fit_predict(values_scaled)
        
        results['assignments'] = phenotype_labels
        results['n_phenotypes'] = n_phenotypes
        
        # Characterize each phenotype
        for phenotype_id in range(n_phenotypes):
            mask = phenotype_labels == phenotype_id
            phenotype_values = values[mask]
            
            if len(phenotype_values) == 0:
                continue
            
            # Calculate phenotype signature
            mean_expression = np.mean(phenotype_values, axis=0)
            std_expression = np.std(phenotype_values, axis=0)
            
            # Z-score relative to population
            population_mean = np.mean(values, axis=0)
            population_std = np.std(values, axis=0)
            z_scores = (mean_expression - population_mean) / (population_std + 1e-10)
            
            # Identify defining markers (high absolute z-score)
            defining_markers = []
            for i, (protein, z_score) in enumerate(zip(protein_names, z_scores)):
                if abs(z_score) > 1.5:  # Threshold for "defining"
                    defining_markers.append({
                        'protein': protein,
                        'z_score': float(z_score),
                        'mean_expression': float(mean_expression[i]),
                        'direction': 'high' if z_score > 0 else 'low'
                    })
            
            results['phenotypes'][f'phenotype_{phenotype_id}'] = {
                'size': int(np.sum(mask)),
                'frequency': float(np.sum(mask) / len(phenotype_labels)),
                'mean_expression': mean_expression.tolist(),
                'std_expression': std_expression.tolist(),
                'z_scores': z_scores.tolist(),
                'defining_markers': sorted(defining_markers, key=lambda x: abs(x['z_score']), reverse=True),
                'description': self._generate_phenotype_description(defining_markers)
            }
        
        # Calculate quality metrics
        results['quality_metrics'] = {
            'silhouette_score': float(silhouette_score(values_scaled, phenotype_labels, 
                                                      sample_size=min(10000, len(values)))),
            'inertia': float(kmeans.inertia_),
            'interpretation': 'Higher silhouette score (closer to 1) indicates better separation'
        }
        
        results['interpretation'] = self._generate_phenotype_interpretation(results)
        
        return results
    
    def _analyze_spatial_evolution(self, values, coordinates, metadata, 
                                  time_column, timepoints) -> Dict[str, Any]:
        """Analyze how spatial patterns evolve over time."""
        spatial_evolution = {
            'spatial_heterogeneity_by_time': {},
            'clustering_by_time': {}
        }
        
        for timepoint in timepoints:
            time_mask = metadata[time_column] == timepoint
            
            if np.sum(time_mask) < 10:
                continue
            
            time_coords = coordinates[time_mask]
            time_values = values[time_mask]
            
            # Calculate spatial heterogeneity (coefficient of variation in local neighborhoods)
            from sklearn.neighbors import NearestNeighbors
            
            nbrs = NearestNeighbors(n_neighbors=min(10, len(time_coords)))
            nbrs.fit(time_coords)
            _, indices = nbrs.kneighbors(time_coords)
            
            local_cvs = []
            for i, neighbors in enumerate(indices):
                neighbor_values = time_values[neighbors]
                # Calculate CV for each protein
                for p in range(time_values.shape[1]):
                    protein_values = neighbor_values[:, p]
                    if np.mean(protein_values) > 0:
                        cv = np.std(protein_values) / np.mean(protein_values)
                        local_cvs.append(cv)
            
            spatial_evolution['spatial_heterogeneity_by_time'][str(timepoint)] = {
                'mean_local_cv': float(np.mean(local_cvs)) if local_cvs else 0,
                'interpretation': 'Higher CV indicates more spatial heterogeneity'
            }
        
        return spatial_evolution
    
    def _generate_differential_interpretation(self, results: Dict) -> str:
        """Generate interpretation text for differential expression results."""
        n_significant = len(results.get('significant_markers', []))
        
        if n_significant == 0:
            return ("No statistically significant differences in marker expression were observed "
                   f"between {results['group_column']} groups after multiple testing correction. "
                   "This suggests similar protein expression profiles across conditions.")
        else:
            interpretation = (f"Found {n_significant} significant marker-comparison pairs "
                            f"between {results['group_column']} groups. ")
            
            # Highlight top changes
            if results['significant_markers']:
                top_marker = results['significant_markers'][0]
                interpretation += (f"The strongest change was in {top_marker['protein']} "
                                 f"(adjusted p={top_marker['p_value_corrected']:.3e}). ")
            
            interpretation += ("These differences suggest phenotypic changes between conditions. "
                             "However, biological interpretation requires validation against "
                             "known markers and functional studies.")
            
            return interpretation
    
    def _generate_temporal_interpretation(self, results: Dict) -> str:
        """Generate interpretation text for temporal dynamics."""
        significant_trends = []
        
        for protein, trend_data in results['temporal_trends'].items():
            if 'trend_test' in trend_data:
                if trend_data['trend_test'].get('p_value', 1) < 0.05:
                    significant_trends.append(protein)
        
        if not significant_trends:
            return ("No significant temporal trends detected in marker expression. "
                   "Expression levels appear stable across timepoints.")
        else:
            return (f"Detected significant temporal trends in {len(significant_trends)} markers: "
                   f"{', '.join(significant_trends[:3])}{'...' if len(significant_trends) > 3 else ''}. "
                   "These temporal patterns may reflect biological processes but require "
                   "validation through orthogonal methods.")
    
    def _generate_phenotype_description(self, defining_markers: List[Dict]) -> str:
        """Generate description of phenotype based on markers."""
        if not defining_markers:
            return "No strongly defining markers"
        
        high_markers = [m['protein'] for m in defining_markers if m['direction'] == 'high'][:3]
        low_markers = [m['protein'] for m in defining_markers if m['direction'] == 'low'][:2]
        
        description = ""
        if high_markers:
            description += f"{', '.join(high_markers)}+"
        if low_markers:
            description += f" {', '.join(low_markers)}-"
        
        return description.strip()
    
    def _generate_phenotype_interpretation(self, results: Dict) -> str:
        """Generate interpretation for discovered phenotypes."""
        n_phenotypes = results.get('n_phenotypes', 0)
        silhouette = results['quality_metrics'].get('silhouette_score', 0)
        
        interpretation = (f"Identified {n_phenotypes} distinct phenotypes based on "
                        f"{len(results.get('phenotypes', {}).get('phenotype_0', {}).get('mean_expression', []))} markers. ")
        
        if silhouette > 0.5:
            interpretation += "Good separation between phenotypes (silhouette > 0.5). "
        elif silhouette > 0.25:
            interpretation += "Moderate separation between phenotypes. "
        else:
            interpretation += "Weak separation - phenotypes may not be distinct. "
        
        interpretation += ("CRITICAL: These are mathematical clusters, not validated cell types. "
                         "Biological interpretation requires comparison with published markers "
                         "and functional validation.")
        
        return interpretation
    
    def _interpret_trend(self, rho: float, p_value: float) -> str:
        """Interpret temporal trend statistics."""
        if p_value >= 0.05:
            return "No significant temporal trend"
        elif rho > 0.5:
            return "Strong increasing trend"
        elif rho > 0:
            return "Moderate increasing trend"
        elif rho < -0.5:
            return "Strong decreasing trend"
        else:
            return "Moderate decreasing trend"


class MetadataValidator:
    """Validates analysis results using only metadata constraints."""
    
    @staticmethod
    def validate_consistency(results: Dict[str, Any], 
                           metadata: pd.DataFrame) -> Dict[str, Any]:
        """Validate internal consistency of results.
        
        This is NOT biological validation, just checking that results
        make statistical and logical sense given the metadata.
        """
        validation = {
            'consistency_checks': [],
            'warnings': [],
            'passed': True
        }
        
        # Check sample sizes
        if 'groups' in results:
            for group in results['groups']:
                group_size = np.sum(metadata[results['group_column']] == group)
                if group_size < 30:
                    validation['warnings'].append(
                        f"Small sample size for group '{group}' (n={group_size}). "
                        "Statistical tests may be underpowered."
                    )
        
        # Check for batch effects if batch info available
        if 'batch' in metadata.columns:
            validation['warnings'].append(
                "Batch information present but not corrected for. "
                "Consider batch effect correction."
            )
        
        # Check multiple testing burden
        if 'differential_markers' in results:
            n_tests = sum(len(m['tests']) for m in results['differential_markers'].values())
            if n_tests > 100:
                validation['warnings'].append(
                    f"Large number of tests ({n_tests}). "
                    "Multiple testing correction is critical."
                )
        
        validation['passed'] = len(validation['warnings']) == 0
        
        return validation
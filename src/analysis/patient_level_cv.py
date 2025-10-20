"""
Patient-Level Cross-Validation for IMC Analysis

Ensures proper cross-validation that treats Subject as the unit of analysis,
preventing data leakage and pseudoreplication in model evaluation.

Implements stratified subject-level splitting with spatial awareness.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterator, Union, Any
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

from .mixed_effects_models import HierarchicalDataStructure


@dataclass
class CVConfig:
    """Configuration for patient-level cross-validation."""
    n_splits: int = 5
    stratify_by: Optional[str] = None  # Column to stratify by (e.g., 'Condition')
    min_subjects_per_fold: int = 2
    spatial_block_size: Optional[float] = None  # For spatial block CV
    random_state: int = 42
    ensure_balance: bool = True  # Ensure balanced representation in folds


class SubjectLevelSplitter(BaseCrossValidator):
    """
    Cross-validation splitter that ensures subjects never appear in both train and test.
    
    Addresses pseudoreplication by treating subjects as the unit of analysis.
    """
    
    def __init__(self, config: CVConfig):
        self.config = config
        self.random_state = config.random_state
        
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations."""
        return self.config.n_splits
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets.
        
        Args:
            X: Feature matrix (can be None, we use groups for splitting)
            y: Target vector (used for stratification if config.stratify_by is set)
            groups: Subject identifiers for each sample
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if groups is None:
            raise ValueError("groups (subject identifiers) must be provided")
        
        # Convert to array for consistency
        groups = np.array(groups)
        unique_subjects = np.unique(groups)
        n_subjects = len(unique_subjects)
        
        # Check if we have enough subjects for meaningful CV
        if n_subjects < self.config.n_splits:
            warnings.warn(f"Only {n_subjects} subjects available for {self.config.n_splits}-fold CV. "
                         f"Reducing to {n_subjects}-fold (leave-one-out).")
            actual_n_splits = n_subjects
        else:
            actual_n_splits = self.config.n_splits
        
        # Check minimum subjects per fold
        subjects_per_fold = n_subjects // actual_n_splits
        if subjects_per_fold < self.config.min_subjects_per_fold:
            warnings.warn(f"Each fold will have ~{subjects_per_fold} subjects, "
                         f"which is less than minimum {self.config.min_subjects_per_fold}")
        
        # Stratify subjects if requested
        if self.config.stratify_by is not None and y is not None:
            subject_stratification = self._get_subject_stratification(groups, y)
            subject_splits = self._stratified_subject_split(
                unique_subjects, subject_stratification, actual_n_splits
            )
        else:
            # Simple random split of subjects
            np.random.seed(self.random_state)
            np.random.shuffle(unique_subjects)
            subject_splits = np.array_split(unique_subjects, actual_n_splits)
        
        # Generate train/test indices for each fold
        for i in range(actual_n_splits):
            test_subjects = set(subject_splits[i])
            train_subjects = set(unique_subjects) - test_subjects
            
            # Map subjects back to sample indices
            test_indices = np.where(np.isin(groups, list(test_subjects)))[0]
            train_indices = np.where(np.isin(groups, list(train_subjects)))[0]
            
            yield train_indices, test_indices
    
    def _get_subject_stratification(self, groups: np.ndarray, y: np.ndarray) -> Dict[str, str]:
        """Get stratification variable for each subject."""
        df = pd.DataFrame({'subject': groups, 'strat': y})
        
        # For each subject, use the most common stratification value
        subject_strat = df.groupby('subject')['strat'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        ).to_dict()
        
        return subject_strat
    
    def _stratified_subject_split(self, 
                                 subjects: np.ndarray,
                                 subject_stratification: Dict[str, str],
                                 n_splits: int) -> List[np.ndarray]:
        """Split subjects while maintaining stratification balance."""
        
        # Group subjects by stratification variable
        strat_groups = {}
        for subject in subjects:
            strat_val = subject_stratification.get(subject, 'unknown')
            if strat_val not in strat_groups:
                strat_groups[strat_val] = []
            strat_groups[strat_val].append(subject)
        
        # Initialize splits
        splits = [[] for _ in range(n_splits)]
        
        # Distribute subjects from each stratification group
        np.random.seed(self.random_state)
        for strat_val, strat_subjects in strat_groups.items():
            np.random.shuffle(strat_subjects)
            
            # Distribute as evenly as possible across splits
            for i, subject in enumerate(strat_subjects):
                split_idx = i % n_splits
                splits[split_idx].append(subject)
        
        return [np.array(split) for split in splits]


class StratifiedSubjectCV:
    """
    Stratified cross-validation at the subject level with balance checking.
    """
    
    def __init__(self, config: CVConfig, hierarchy: HierarchicalDataStructure):
        self.config = config
        self.hierarchy = hierarchy
        self.splitter = SubjectLevelSplitter(config)
        
    def validate_model(self,
                      data: pd.DataFrame,
                      target_column: str,
                      feature_columns: List[str],
                      model_class,
                      model_params: Dict = None) -> Dict[str, float]:
        """
        Perform cross-validation with proper subject-level splitting.
        
        Args:
            data: Full dataset with hierarchical structure
            target_column: Name of target variable column
            feature_columns: List of feature column names
            model_class: Model class to fit (should have fit/predict methods)
            model_params: Parameters to pass to model constructor
            
        Returns:
            Dictionary with cross-validation metrics
        """
        if model_params is None:
            model_params = {}
        
        # Prepare data
        X = data[feature_columns].values
        y = data[target_column].values
        groups = data[self.hierarchy.subject_column].values
        
        # Stratification target if configured
        stratify_target = None
        if self.config.stratify_by and self.config.stratify_by in data.columns:
            stratify_target = data[self.config.stratify_by].values
        
        # Perform cross-validation
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'n_train_subjects': [],
            'n_test_subjects': []
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(
            self.splitter.split(X, stratify_target, groups)
        ):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]
            groups_test = groups[test_idx]
            
            # Track subject counts
            n_train_subjects = len(np.unique(groups_train))
            n_test_subjects = len(np.unique(groups_test))
            cv_scores['n_train_subjects'].append(n_train_subjects)
            cv_scores['n_test_subjects'].append(n_test_subjects)
            
            # Fit model
            try:
                model = model_class(**model_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted', zero_division=0
                )
                
                cv_scores['accuracy'].append(accuracy)
                cv_scores['precision'].append(precision)
                cv_scores['recall'].append(recall)
                cv_scores['f1'].append(f1)
                
            except Exception as e:
                warnings.warn(f"Fold {fold_idx} failed: {e}")
                # Add NaN for failed folds
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    cv_scores[metric].append(np.nan)
        
        # Calculate summary statistics
        summary_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = np.array(cv_scores[metric])
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) > 0:
                summary_metrics[f'{metric}_mean'] = float(np.mean(valid_values))
                summary_metrics[f'{metric}_std'] = float(np.std(valid_values))
                summary_metrics[f'{metric}_ci_lower'] = float(np.percentile(valid_values, 2.5))
                summary_metrics[f'{metric}_ci_upper'] = float(np.percentile(valid_values, 97.5))
            else:
                summary_metrics[f'{metric}_mean'] = np.nan
                summary_metrics[f'{metric}_std'] = np.nan
                summary_metrics[f'{metric}_ci_lower'] = np.nan
                summary_metrics[f'{metric}_ci_upper'] = np.nan
        
        # Add subject count statistics
        summary_metrics['mean_train_subjects'] = float(np.mean(cv_scores['n_train_subjects']))
        summary_metrics['mean_test_subjects'] = float(np.mean(cv_scores['n_test_subjects']))
        summary_metrics['total_subjects'] = len(self.hierarchy.subjects)
        
        return summary_metrics


class SpatialBlockCV:
    """
    Spatial block cross-validation for spatially correlated data.
    
    Creates spatial blocks and ensures blocks are kept together in train/test splits.
    """
    
    def __init__(self, config: CVConfig):
        self.config = config
        if config.spatial_block_size is None:
            raise ValueError("spatial_block_size must be specified for SpatialBlockCV")
        
    def create_spatial_blocks(self,
                             spatial_coords: np.ndarray,
                             block_size: float) -> np.ndarray:
        """
        Create spatial blocks for cross-validation.
        
        Args:
            spatial_coords: N x 2 array of spatial coordinates
            block_size: Size of spatial blocks in coordinate units
            
        Returns:
            Array of block identifiers for each point
        """
        # Create grid of blocks
        x_coords = spatial_coords[:, 0]
        y_coords = spatial_coords[:, 1]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Create block boundaries
        x_blocks = np.arange(x_min, x_max + block_size, block_size)
        y_blocks = np.arange(y_min, y_max + block_size, block_size)
        
        # Assign each point to a block
        x_block_idx = np.digitize(x_coords, x_blocks) - 1
        y_block_idx = np.digitize(y_coords, y_blocks) - 1
        
        # Create unique block identifiers
        n_x_blocks = len(x_blocks) - 1
        block_ids = y_block_idx * n_x_blocks + x_block_idx
        
        return block_ids
    
    def split(self,
              spatial_coords: np.ndarray,
              subject_ids: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate spatial block cross-validation splits.
        
        Args:
            spatial_coords: N x 2 spatial coordinates
            subject_ids: Subject identifier for each point
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        # Create spatial blocks
        block_ids = self.create_spatial_blocks(
            spatial_coords, self.config.spatial_block_size
        )
        
        # Group blocks by subject to maintain subject-level splitting
        subject_blocks = {}
        for i, (subject, block) in enumerate(zip(subject_ids, block_ids)):
            if subject not in subject_blocks:
                subject_blocks[subject] = set()
            subject_blocks[subject].add(block)
        
        # Get unique subjects and their associated blocks
        subjects = list(subject_blocks.keys())
        np.random.seed(self.config.random_state)
        np.random.shuffle(subjects)
        
        # Create splits by assigning subjects (and their blocks) to folds
        n_splits = self.config.n_splits
        subject_folds = np.array_split(subjects, n_splits)
        
        for fold_idx in range(n_splits):
            test_subjects = set(subject_folds[fold_idx])
            train_subjects = set(subjects) - test_subjects
            
            # Get blocks for train and test subjects
            test_blocks = set()
            train_blocks = set()
            
            for subject in test_subjects:
                test_blocks.update(subject_blocks[subject])
            
            for subject in train_subjects:
                train_blocks.update(subject_blocks[subject])
            
            # Map blocks back to indices
            test_indices = np.where(np.isin(block_ids, list(test_blocks)))[0]
            train_indices = np.where(np.isin(block_ids, list(train_blocks)))[0]
            
            yield train_indices, test_indices


class ValidationMetrics:
    """
    Validation metrics that account for nested data structure.
    """
    
    @staticmethod
    def effective_sample_size(n_observations: int,
                             intraclass_correlation: float,
                             cluster_size: float) -> float:
        """
        Calculate effective sample size for clustered data.
        
        Args:
            n_observations: Total number of observations
            intraclass_correlation: ICC from mixed-effects model
            cluster_size: Average cluster size
            
        Returns:
            Effective sample size adjusted for clustering
        """
        design_effect = 1 + (cluster_size - 1) * intraclass_correlation
        return n_observations / design_effect
    
    @staticmethod
    def cluster_adjusted_confidence_interval(effect_size: float,
                                           standard_error: float,
                                           effective_n: float,
                                           confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval adjusted for clustering.
        
        Args:
            effect_size: Estimated effect size
            standard_error: Standard error of effect size
            effective_n: Effective sample size
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        from scipy import stats
        
        # Use t-distribution with effective degrees of freedom
        df = max(1, effective_n - 1)
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        margin_of_error = t_critical * standard_error
        
        return (
            effect_size - margin_of_error,
            effect_size + margin_of_error
        )
    
    @staticmethod
    def calculate_power(effect_size: float,
                       n_subjects: int,
                       alpha: float = 0.05,
                       icc: float = 0.1,
                       cluster_size: float = 10) -> float:
        """
        Calculate statistical power for nested design.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            n_subjects: Number of subjects (clusters)
            alpha: Type I error rate
            icc: Intraclass correlation
            cluster_size: Average cluster size
            
        Returns:
            Statistical power (0-1)
        """
        from scipy import stats
        
        # Effective sample size
        effective_n = ValidationMetrics.effective_sample_size(
            n_subjects * cluster_size, icc, cluster_size
        )
        
        # Calculate non-centrality parameter
        ncp = effect_size * np.sqrt(effective_n / 4)  # For two-group comparison
        
        # Critical value for two-tailed test
        df = max(1, effective_n - 2)
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Power calculation
        power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)
        
        return float(np.clip(power, 0, 1))


def perform_nested_cv_analysis(data: pd.DataFrame,
                              hierarchy: HierarchicalDataStructure,
                              config: CVConfig) -> Dict[str, Any]:
    """
    Comprehensive nested cross-validation analysis.
    
    Args:
        data: Dataset with hierarchical structure
        hierarchy: Hierarchical data structure definition
        config: Cross-validation configuration
        
    Returns:
        Dictionary with comprehensive CV analysis results
    """
    
    # Initialize CV splitter
    cv_splitter = StratifiedSubjectCV(config, hierarchy)
    
    # Basic validation statistics
    n_subjects = len(hierarchy.subjects)
    n_total_obs = len(data)
    
    results = {
        'n_subjects': n_subjects,
        'n_total_observations': n_total_obs,
        'avg_observations_per_subject': n_total_obs / n_subjects if n_subjects > 0 else 0
    }
    
    # Check if we have sufficient subjects for meaningful CV
    min_subjects_needed = config.n_splits * config.min_subjects_per_fold
    
    if n_subjects < min_subjects_needed:
        warnings.warn(f"Only {n_subjects} subjects available, but need at least "
                     f"{min_subjects_needed} for {config.n_splits}-fold CV with "
                     f"{config.min_subjects_per_fold} subjects per fold.")
        results['cv_feasible'] = False
        results['recommended_n_splits'] = max(1, n_subjects // config.min_subjects_per_fold)
    else:
        results['cv_feasible'] = True
        results['recommended_n_splits'] = config.n_splits
    
    # Analyze balance if stratification requested
    if config.stratify_by and config.stratify_by in data.columns:
        stratification_analysis = _analyze_stratification_balance(
            data, hierarchy, config.stratify_by
        )
        results['stratification_analysis'] = stratification_analysis
    
    return results


def _analyze_stratification_balance(data: pd.DataFrame,
                                   hierarchy: HierarchicalDataStructure,
                                   stratify_column: str) -> Dict[str, Any]:
    """Analyze balance of stratification variable across subjects."""
    
    # Subject-level stratification counts
    subject_strat = data.groupby(hierarchy.subject_column)[stratify_column].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    )
    
    strat_counts = subject_strat.value_counts().to_dict()
    strat_proportions = subject_strat.value_counts(normalize=True).to_dict()
    
    # Check balance
    max_prop = max(strat_proportions.values())
    min_prop = min(strat_proportions.values())
    balance_ratio = min_prop / max_prop if max_prop > 0 else 0
    
    return {
        'stratification_counts': strat_counts,
        'stratification_proportions': strat_proportions,
        'balance_ratio': float(balance_ratio),
        'is_balanced': balance_ratio > 0.5,  # Reasonable balance threshold
        'n_categories': len(strat_counts)
    }
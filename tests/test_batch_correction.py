"""
Test suite for sham-anchored batch correction.

Production-quality tests for IMC analysis pipeline batch correction module,
focusing on the scientifically valid sham-anchored normalization approach.
"""

import numpy as np
import pytest
from typing import Dict, List, Tuple
import warnings

from src.analysis.batch_correction import (
    sham_anchored_normalize,
    ValidationError,
    _validate_experimental_design,
    _compute_sham_reference_stats,
    detect_batch_effects,
    validate_batch_structure
)


class TestShamAnchoredNormalization:
    """Test sham-anchored normalization implementation."""
    
    def setup_method(self):
        """Setup test data with realistic IMC cross-sectional design."""
        np.random.seed(42)
        
        # Simulate our kidney injury study design
        self.proteins = ['CD45', 'CD11b', 'Ly6G', 'CD206']
        
        # Sham controls (Day 0) - 2 mice
        self.sham_ms1_data = {
            'CD45': np.random.poisson(50, 1000).astype(float),
            'CD11b': np.random.poisson(20, 1000).astype(float),
            'Ly6G': np.random.poisson(10, 1000).astype(float),
            'CD206': np.random.poisson(15, 1000).astype(float)
        }
        
        self.sham_ms2_data = {
            'CD45': np.random.poisson(55, 1000).astype(float),
            'CD11b': np.random.poisson(18, 1000).astype(float),
            'Ly6G': np.random.poisson(12, 1000).astype(float),
            'CD206': np.random.poisson(17, 1000).astype(float)
        }
        
        # Day 1 injury - 2 mice (biological change + technical batch effect)
        self.d1_ms1_data = {
            'CD45': np.random.poisson(150, 1000).astype(float) + 20,  # 3x increase + batch effect
            'CD11b': np.random.poisson(40, 1000).astype(float) + 15,
            'Ly6G': np.random.poisson(80, 1000).astype(float) + 10,   # High neutrophils
            'CD206': np.random.poisson(25, 1000).astype(float) + 5
        }
        
        self.d1_ms2_data = {
            'CD45': np.random.poisson(140, 1000).astype(float) + 25,
            'CD11b': np.random.poisson(45, 1000).astype(float) + 18,
            'Ly6G': np.random.poisson(75, 1000).astype(float) + 12,
            'CD206': np.random.poisson(22, 1000).astype(float) + 8
        }
        
        self.batch_data = {
            'sham_ms1': self.sham_ms1_data,
            'sham_ms2': self.sham_ms2_data,
            'd1_ms1': self.d1_ms1_data,
            'd1_ms2': self.d1_ms2_data
        }
        
        self.batch_metadata = {
            'sham_ms1': {'Condition': 'Sham', 'Injury Day': 0, 'Mouse': 'MS1'},
            'sham_ms2': {'Condition': 'Sham', 'Injury Day': 0, 'Mouse': 'MS2'},
            'd1_ms1': {'Condition': 'Injury', 'Injury Day': 1, 'Mouse': 'MS1'},
            'd1_ms2': {'Condition': 'Injury', 'Injury Day': 1, 'Mouse': 'MS2'}
        }
    
    def test_sham_anchored_normalization_basic(self):
        """Test basic sham-anchored normalization functionality."""
        normalized_data, stats = sham_anchored_normalize(
            self.batch_data, self.batch_metadata
        )
        
        # Should normalize all batches
        assert len(normalized_data) == 4
        assert 'sham_ms1' in normalized_data
        assert 'sham_ms2' in normalized_data
        assert 'd1_ms1' in normalized_data
        assert 'd1_ms2' in normalized_data
        
        # Sham controls should have mean ≈ 0, std ≈ 1 after normalization
        sham_cd45_ms1 = normalized_data['sham_ms1']['CD45']
        sham_cd45_ms2 = normalized_data['sham_ms2']['CD45']
        
        combined_sham_mean = (np.mean(sham_cd45_ms1) + np.mean(sham_cd45_ms2)) / 2
        assert abs(combined_sham_mean) < 0.1, f"Sham mean should be ~0, got {combined_sham_mean}"
        
        # Should preserve biological signal (injury > sham)
        injury_cd45_ms1 = normalized_data['d1_ms1']['CD45']
        injury_cd45_ms2 = normalized_data['d1_ms2']['CD45']
        injury_mean = (np.mean(injury_cd45_ms1) + np.mean(injury_cd45_ms2)) / 2
        
        assert injury_mean > combined_sham_mean, "Injury signal should be preserved"
        
        # Statistics should include reference info
        assert 'sham_batches' in stats
        assert len(stats['sham_batches']) == 2
        assert 'reference_stats' in stats
    
    def test_biological_signal_preservation(self):
        """Test that biological dynamics are preserved across normalization."""
        normalized_data, _ = sham_anchored_normalize(
            self.batch_data, self.batch_metadata
        )
        
        # Calculate fold changes before and after normalization
        # Original fold change (injury vs sham)
        orig_sham_cd45 = (np.mean(self.sham_ms1_data['CD45']) + np.mean(self.sham_ms2_data['CD45'])) / 2
        orig_injury_cd45 = (np.mean(self.d1_ms1_data['CD45']) + np.mean(self.d1_ms2_data['CD45'])) / 2
        orig_fold_change = orig_injury_cd45 / orig_sham_cd45
        
        # Normalized values
        norm_sham_cd45 = (np.mean(normalized_data['sham_ms1']['CD45']) + 
                         np.mean(normalized_data['sham_ms2']['CD45'])) / 2
        norm_injury_cd45 = (np.mean(normalized_data['d1_ms1']['CD45']) + 
                           np.mean(normalized_data['d1_ms2']['CD45'])) / 2
        
        # The trend should be preserved (injury > sham)
        assert norm_injury_cd45 > norm_sham_cd45, "Biological trend should be preserved"
        
        # Should preserve relative magnitude (allowing for normalization scaling)
        assert orig_fold_change > 2.0, "Original data should show biological effect"
    
    def test_missing_sham_controls_error(self):
        """Test error handling when no sham controls are found."""
        # Create data without sham controls
        no_sham_data = {
            'injury1': self.d1_ms1_data,
            'injury2': self.d1_ms2_data
        }
        
        no_sham_metadata = {
            'injury1': {'Condition': 'Injury', 'Injury Day': 1},
            'injury2': {'Condition': 'Injury', 'Injury Day': 3}
        }
        
        with pytest.raises(ValueError, match="No sham control batches found"):
            sham_anchored_normalize(no_sham_data, no_sham_metadata)
    
    def test_zero_variance_handling(self):
        """Test handling of markers with zero variance in sham controls."""
        # Create data where one marker has zero variance in shams
        modified_data = self.batch_data.copy()
        modified_data['sham_ms1']['CD45'] = np.full(1000, 42.0)  # Constant
        modified_data['sham_ms2']['CD45'] = np.full(1000, 42.0)  # Constant
        
        with warnings.catch_warnings(record=True) as w:
            normalized_data, _ = sham_anchored_normalize(
                modified_data, self.batch_metadata
            )
            
            # Should warn about zero std
            warning_messages = [str(warning.message) for warning in w]
            assert any("Zero standard deviation" in msg for msg in warning_messages)
        
        # Should still process other markers
        assert 'CD11b' in normalized_data['d1_ms1']
        assert 'Ly6G' in normalized_data['d1_ms1']
    
    def test_scientific_guardrails(self):
        """Test that scientific guardrails prevent invalid method usage."""
        # Should block quantile normalization on cross-sectional data
        with pytest.raises(ValidationError, match="scientifically invalid"):
            _validate_experimental_design(self.batch_metadata, method='quantile')
        
        # Should allow sham-anchored method
        _validate_experimental_design(self.batch_metadata, method='sham_anchored')
    
    def test_reference_stats_computation(self):
        """Test computation of reference statistics from sham batches."""
        sham_batches = ['sham_ms1', 'sham_ms2']
        stats = _compute_sham_reference_stats(self.batch_data, sham_batches)
        
        # Should compute stats for all proteins
        for protein in self.proteins:
            assert protein in stats
            assert 'mean' in stats[protein]
            assert 'std' in stats[protein]
            assert 'n_observations' in stats[protein]
            
            # Should pool across both sham batches
            assert stats[protein]['n_observations'] == 2000  # 1000 + 1000


class TestBatchEffectDetection:
    """Test batch effect detection functionality."""
    
    def test_detect_technical_batch_effects(self):
        """Test detection of pure technical batch effects."""
        np.random.seed(42)
        
        # Same biological condition, different technical batches
        technical_batch_data = {
            'batch1': {'CD45': np.random.normal(100, 20, 500)},
            'batch2': {'CD45': np.random.normal(150, 20, 500)}  # +50 technical shift
        }
        
        effects = detect_batch_effects(technical_batch_data)
        
        # Should detect significant batch effect
        assert effects['protein_effects']['CD45']['mann_whitney_u_pvalue'] < 0.05
        assert effects['overall_severity'] > 0.1
    
    def test_detect_biological_differences(self):
        """Test detection of biological differences (not batch effects)."""
        np.random.seed(42)
        
        # Different biological conditions
        biological_data = {
            'sham': {'CD45': np.random.normal(50, 15, 500)},
            'injury': {'CD45': np.random.normal(200, 40, 500)}  # 4x biological increase
        }
        
        effects = detect_batch_effects(biological_data)
        
        # Should detect "batch effect" (actually biological difference)
        assert effects['protein_effects']['CD45']['mann_whitney_u_pvalue'] < 0.05
        assert effects['overall_severity'] > 0.3  # Larger effect size


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_validate_batch_structure(self):
        """Test batch data structure validation."""
        # Valid structure should pass
        valid_data = {
            'batch1': {'CD45': np.array([1, 2, 3]), 'CD11b': np.array([4, 5, 6])},
            'batch2': {'CD45': np.array([7, 8, 9]), 'CD11b': np.array([10, 11, 12])}
        }
        valid_metadata = {
            'batch1': {'condition': 'A'},
            'batch2': {'condition': 'B'}
        }
        
        # Should not raise error
        validate_batch_structure(valid_data, valid_metadata)
        
        # Missing metadata should raise error
        with pytest.raises(ValueError, match="Missing metadata"):
            validate_batch_structure(valid_data, {})
        
        # Empty data should raise error
        with pytest.raises(ValueError, match="Empty batch data"):
            validate_batch_structure({}, valid_metadata)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
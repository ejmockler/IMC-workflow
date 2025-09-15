#!/usr/bin/env python3
"""
Consolidated Validation Test Suite
Tests for clustering validation framework
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from src.utils.data_loader import load_roi_as_imc_data, subsample_data
from src.config import Config
from src.utils.helpers import find_roi_files
from src.analysis.validation import (
    ValidationSuite,
    ConsistencyValidator,
    SilhouetteValidator,
    SpatialCoherenceValidator,
    ValidationResult
)
from src.analysis.clustering import ClustererFactory


class ValidationTestSuite:
    """Organized validation testing"""
    
    def __init__(self):
        self.config = Config('config.json')
        
    def create_test_data(self, n_samples=500, n_features=5, n_clusters=3):
        """Create synthetic clustered data"""
        np.random.seed(42)
        
        data = []
        coords = []
        labels = []
        
        for i in range(n_clusters):
            cluster_size = n_samples // n_clusters
            # Feature space cluster
            center = np.random.randn(n_features) * 3
            cluster_data = np.random.randn(cluster_size, n_features) * 0.5 + center
            # Spatial cluster
            spatial_center = np.array([i * 50, 0])
            cluster_coords = np.random.randn(cluster_size, 2) * 10 + spatial_center
            
            data.append(cluster_data)
            coords.append(cluster_coords)
            labels.extend([i] * cluster_size)
        
        return np.vstack(data), np.vstack(coords), np.array(labels)
    
    def test_consistency_validator(self):
        """Test clustering consistency validation"""
        print("\n" + "="*60)
        print("TEST 1: Consistency Validator")
        print("="*60)
        
        data, coords, true_labels = self.create_test_data(n_samples=300)
        
        validator = ConsistencyValidator(n_runs=3)
        result = validator.validate(data, true_labels)
        
        print(f"   Consistency score: {result.score:.3f}")
        print(f"   Mean ARI: {result.details['mean_ari']:.3f}")
        print(f"   Interpretation: {result.interpretation}")
        
        assert isinstance(result, ValidationResult), "Invalid result type"
        assert 0 <= result.score <= 1, "Score out of range"
        assert result.is_acceptable(threshold=0.5), "Consistency too low for synthetic data"
        
        print("   ✓ Consistency validator passed")
        return True
    
    def test_silhouette_validator(self):
        """Test silhouette score validation"""
        print("\n" + "="*60)
        print("TEST 2: Silhouette Validator")
        print("="*60)
        
        data, coords, true_labels = self.create_test_data(n_samples=300)
        
        validator = SilhouetteValidator()
        result = validator.validate(data, true_labels, coords=coords)
        
        print(f"   Silhouette score: {result.score:.3f}")
        print(f"   N clusters: {result.details['n_clusters']}")
        print(f"   Interpretation: {result.interpretation}")
        
        assert isinstance(result, ValidationResult), "Invalid result type"
        assert -1 <= result.score <= 1, "Score out of range"
        
        # Check per-cluster scores
        if 'per_cluster_scores' in result.details:
            n_cluster_scores = len(result.details['per_cluster_scores'])
            print(f"   Per-cluster scores computed: {n_cluster_scores}")
        
        print("   ✓ Silhouette validator passed")
        return True
    
    def test_spatial_coherence_validator(self):
        """Test spatial coherence validation"""
        print("\n" + "="*60)
        print("TEST 3: Spatial Coherence Validator")
        print("="*60)
        
        data, coords, true_labels = self.create_test_data(n_samples=300)
        
        validator = SpatialCoherenceValidator()
        result = validator.validate(data, true_labels, coords=coords)
        
        print(f"   Spatial coherence: {result.score:.3f}")
        print(f"   Moran's I: {result.details.get('morans_i', 0):.3f}")
        print(f"   Fragmentation: {result.details.get('fragmentation', 0):.3f}")
        print(f"   Interpretation: {result.interpretation}")
        
        assert isinstance(result, ValidationResult), "Invalid result type"
        assert 0 <= result.score <= 1, "Score out of range"
        
        print("   ✓ Spatial coherence validator passed")
        return True
    
    def test_validation_suite(self):
        """Test complete validation suite"""
        print("\n" + "="*60)
        print("TEST 4: Validation Suite")
        print("="*60)
        
        data, coords, _ = self.create_test_data(n_samples=500)
        
        # Cluster the data
        clusterer = ClustererFactory.create('kmeans')
        cluster_result = clusterer.fit_predict(data, n_clusters=3)
        labels = cluster_result.labels
        
        # Run validation suite
        suite = ValidationSuite()
        results = suite.validate_all(data, labels, coords=coords)
        
        print(f"   Validators run: {list(results.keys())}")
        
        # Check each validator result
        for name, result in results.items():
            print(f"   {name}: {result.score:.3f} - {result.interpretation[:30]}...")
            assert isinstance(result, ValidationResult), f"Invalid result for {name}"
        
        # Test summary score
        summary = suite.get_summary_score(results)
        print(f"   Summary score: {summary:.3f}")
        assert 0 <= summary <= 1, "Summary score out of range"
        
        # Test report formatting
        report = suite.format_report(results)
        assert "Clustering Validation Report" in report, "Report missing header"
        assert "Overall Score" in report, "Report missing summary"
        
        print("   ✓ Validation suite passed")
        return True
    
    def test_with_real_data(self):
        """Test validation with real ROI data"""
        print("\n" + "="*60)
        print("TEST 5: Real Data Validation")
        print("="*60)
        
        roi_files = find_roi_files(self.config.data_dir)
        
        if not roi_files:
            print("   No ROI files found - skipping")
            return True
        
        # Load and subsample ROI
        data = load_roi_as_imc_data(roi_files[0], 'config.json')
        if data.metadata['n_pixels'] > 1000:
            data = subsample_data(data, 1000)
        
        print(f"   Testing with {data.roi_id}: {len(data.coords)} pixels")
        
        # Cluster
        clusterer = ClustererFactory.create('minibatch')
        result = clusterer.fit_predict(data.values, n_clusters=10)
        
        # Validate
        validator = SilhouetteValidator()
        val_result = validator.validate(data.values, result.labels, coords=data.coords)
        
        print(f"   Silhouette score: {val_result.score:.3f}")
        print(f"   Quality assessment: {val_result.interpretation}")
        
        print("   ✓ Real data validation passed")
        return True
    
    def test_edge_cases(self):
        """Test validation edge cases"""
        print("\n" + "="*60)
        print("TEST 6: Edge Cases")
        print("="*60)
        
        # Test with single cluster
        data = np.random.randn(100, 5)
        labels = np.zeros(100, dtype=int)
        coords = np.random.randn(100, 2)
        
        validator = SilhouetteValidator()
        result = validator.validate(data, labels, coords=coords)
        
        print(f"   Single cluster silhouette: {result.score:.3f}")
        assert result.score == 0.0, "Single cluster should have score 0"
        
        # Test with all different clusters
        labels = np.arange(100)
        result = validator.validate(data, labels, coords=coords)
        print(f"   All singleton clusters: {result.score:.3f}")
        
        # Test with no coordinates for spatial validator
        spatial_val = SpatialCoherenceValidator()
        result = spatial_val.validate(data, labels)  # No coords
        assert result.score == 0.0, "Should handle missing coordinates"
        print(f"   No coordinates handled: {result.interpretation[:40]}...")
        
        print("   ✓ Edge cases handled correctly")
        return True
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "#"*60)
        print("#" + " "*17 + "VALIDATION TEST SUITE" + " "*18 + "#")
        print("#"*60)
        
        all_passed = True
        
        try:
            self.test_consistency_validator()
            self.test_silhouette_validator()
            self.test_spatial_coherence_validator()
            self.test_validation_suite()
            self.test_with_real_data()
            self.test_edge_cases()
            
        except AssertionError as e:
            print(f"\n✗ Test failed: {e}")
            all_passed = False
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
        
        if all_passed:
            print("\n" + "#"*60)
            print("#" + " "*19 + "ALL TESTS PASSED" + " "*20 + "#")
            print("#"*60)
            print("\n✓ Validation framework fully tested!\n")
        
        return 0 if all_passed else 1


def main():
    """Main test runner"""
    suite = ValidationTestSuite()
    return suite.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
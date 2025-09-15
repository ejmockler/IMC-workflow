#!/usr/bin/env python3
"""
Consolidated Benchmark Test Suite
Combines all benchmark testing into a single, well-organized file
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from src.utils.data_loader import load_roi_data, subsample_data, load_roi_as_imc_data
from src.config import Config
from src.utils.helpers import find_roi_files
from src.analysis.benchmark import (
    BenchmarkSuite, 
    ClusteringComparator, 
    KernelComparator,
    IntegratedComparator,
    ReportGenerator
)


class BenchmarkTestSuite:
    """Organized benchmark testing"""
    
    def __init__(self):
        self.config = Config('config.json')
        self.test_data = None
        self.test_coords = None
        
    def setup_synthetic_data(self, n_samples=500, n_features=5, n_clusters=3):
        """Create synthetic test data"""
        np.random.seed(42)
        
        # Create clustered data
        data = []
        coords = []
        
        for i in range(n_clusters):
            cluster_size = n_samples // n_clusters
            # Create cluster in feature space
            center = np.random.randn(n_features) * 3
            cluster_data = np.random.randn(cluster_size, n_features) * 0.5 + center
            # Create cluster in spatial coordinates
            spatial_center = np.array([i * 100, 0])
            cluster_coords = np.random.randn(cluster_size, 2) * 20 + spatial_center
            
            data.append(cluster_data)
            coords.append(cluster_coords)
        
        self.test_data = np.vstack(data)
        self.test_coords = np.vstack(coords)
        
        return self.test_data, self.test_coords
    
    def test_basic_functionality(self):
        """Test basic benchmark functionality"""
        print("\n" + "="*60)
        print("TEST 1: Basic Functionality")
        print("="*60)
        
        # Setup small test data
        data, coords = self.setup_synthetic_data(n_samples=300)
        print(f"Test data: {data.shape[0]} samples, {data.shape[1]} features")
        
        # Test ClusteringComparator
        print("\n1. ClusteringComparator:")
        comparator = ClusteringComparator(['kmeans', 'minibatch'])
        result = comparator.compare(data, coords, n_clusters=3)
        
        print(f"   Best method: {result.best_method}")
        print(f"   Best score: {result.best_score:.3f}")
        
        assert hasattr(result, 'method_scores'), "Result missing method_scores"
        assert result.best_method in ['kmeans', 'minibatch'], "Invalid best method"
        print("   ✓ ClusteringComparator passed")
        
        # Test KernelComparator
        print("\n2. KernelComparator:")
        kernel_comp = KernelComparator(['gaussian'])
        kernel_result = kernel_comp.compare(data, coords, n_clusters=3)
        
        print(f"   Best kernel: {kernel_result.best_method}")
        print(f"   Score: {kernel_result.best_score:.3f}")
        
        assert hasattr(kernel_result, 'method_scores'), "Result missing method_scores"
        print("   ✓ KernelComparator passed")
        
        return True
    
    def test_benchmark_suite(self):
        """Test full benchmark suite"""
        print("\n" + "="*60)
        print("TEST 2: Benchmark Suite")
        print("="*60)
        
        data, coords = self.setup_synthetic_data(n_samples=500)
        
        # Configure minimal suite for testing
        suite = BenchmarkSuite()
        suite.clustering_algorithms = ['kmeans', 'minibatch']
        suite.kernel_types = ['gaussian']
        suite.subset_sizes = [200, 400]
        
        print("Running benchmark suite...")
        report = suite.run_full_benchmark(data, coords)
        
        # Verify report structure
        assert hasattr(report, 'timestamp'), "Report missing timestamp"
        assert hasattr(report, 'comparisons'), "Report missing comparisons"
        assert 'clustering' in report.comparisons, "Missing clustering comparison"
        
        print(f"\nReport generated at: {report.timestamp}")
        print(f"Comparisons: {list(report.comparisons.keys())}")
        
        if report.summary_statistics:
            print(f"Mean quality: {report.summary_statistics.get('mean_quality', 0):.3f}")
        
        print("   ✓ Benchmark suite passed")
        
        return report
    
    def test_report_generation(self, report=None):
        """Test report generation"""
        print("\n" + "="*60)
        print("TEST 3: Report Generation")
        print("="*60)
        
        if report is None:
            # Generate a simple report
            data, coords = self.setup_synthetic_data(n_samples=200)
            suite = BenchmarkSuite()
            suite.clustering_algorithms = ['kmeans']
            suite.kernel_types = ['gaussian']
            suite.subset_sizes = [100]
            report = suite.run_full_benchmark(data, coords)
        
        generator = ReportGenerator()
        output_dir = Path('benchmark_reports')
        output_dir.mkdir(exist_ok=True)
        
        # Test Markdown generation
        md_path = output_dir / 'test_report.md'
        generator.generate_markdown(report, str(md_path))
        assert md_path.exists(), "Markdown report not generated"
        print(f"   ✓ Markdown report: {md_path}")
        
        # Test HTML generation
        html_path = output_dir / 'test_report.html'
        generator.generate_html(report, str(html_path))
        assert html_path.exists(), "HTML report not generated"
        print(f"   ✓ HTML report: {html_path}")
        
        return True
    
    def test_with_real_data(self):
        """Test with actual ROI data if available"""
        print("\n" + "="*60)
        print("TEST 4: Real ROI Data (if available)")
        print("="*60)
        
        roi_files = find_roi_files(self.config.data_dir)
        
        if not roi_files:
            print("   No ROI files found - skipping")
            return True
        
        # Load first ROI
        roi_file = roi_files[0]
        print(f"Testing with: {roi_file.name}")
        
        # Use new centralized loader
        data = load_roi_as_imc_data(roi_file, 'config.json')
        
        # Subsample for speed
        if data.metadata['n_pixels'] > 2000:
            data = subsample_data(data, 2000)
            print(f"   Subsampled to {len(data.coords)} pixels")
        
        # Quick benchmark
        comparator = ClusteringComparator(['kmeans'])
        result = comparator.compare(data.values, data.coords, n_clusters=10)
        
        print(f"   Clustering score: {result.best_score:.3f}")
        print("   ✓ Real data test passed")
        
        return True
    
    def test_integrated_comparison(self):
        """Test integrated kernel+clustering comparison"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Comparison")
        print("="*60)
        
        data, coords = self.setup_synthetic_data(n_samples=300)
        
        comparator = IntegratedComparator(
            clustering_algorithms=['kmeans'],
            kernel_types=['gaussian'],
            lambda_values=[0.5]
        )
        
        print("Running integrated comparison...")
        result = comparator.compare(data, coords, n_clusters=3)
        
        print(f"   Best combination: {result.best_method}")
        print(f"   Score: {result.best_score:.3f}")
        
        assert hasattr(result, 'rankings'), "Result missing rankings"
        print("   ✓ Integrated comparison passed")
        
        return True
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "#"*60)
        print("#" + " "*18 + "BENCHMARK TEST SUITE" + " "*19 + "#")
        print("#"*60)
        
        all_passed = True
        
        try:
            # Run tests in sequence
            self.test_basic_functionality()
            report = self.test_benchmark_suite()
            self.test_report_generation(report)
            self.test_with_real_data()
            self.test_integrated_comparison()
            
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
            print("\n✓ Benchmark system fully tested and working!\n")
        
        return 0 if all_passed else 1


def main():
    """Main test runner"""
    suite = BenchmarkTestSuite()
    return suite.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
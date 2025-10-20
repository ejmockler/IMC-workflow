"""
Simple test for result comparison utilities (no external dependencies).
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test basic imports - bypass the analysis package __init__.py
try:
    print("Attempting to import result comparison module directly...")
    
    # Direct import to avoid analysis package dependencies
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "result_comparison", 
        str(Path(__file__).parent / "src" / "analysis" / "result_comparison.py")
    )
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    
    print("✓ Successfully imported result comparison module")
    
    # Extract classes
    ResultComparer = rc.ResultComparer
    ToleranceProfile = rc.ToleranceProfile
    ComparisonSeverity = rc.ComparisonSeverity
    ComparisonResult = rc.ComparisonResult
    DiffReport = rc.DiffReport
    
    print("✓ Successfully imported result comparison classes")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test basic functionality
def test_tolerance_profiles():
    """Test tolerance profile creation."""
    print("\nTesting tolerance profiles...")
    
    # Standard profile
    standard = ToleranceProfile()
    assert standard.cluster_assignment_tolerance == 0.15
    assert standard.expression_rtol == 0.03
    print("✓ Standard tolerance profile created")
    
    # Strict profile
    strict = ToleranceProfile.create_strict()
    assert strict.cluster_assignment_tolerance == 0.05
    assert strict.expression_rtol == 0.01
    print("✓ Strict tolerance profile created")
    
    # Permissive profile
    permissive = ToleranceProfile.create_permissive()
    assert permissive.cluster_assignment_tolerance == 0.25
    assert permissive.expression_rtol == 0.05
    print("✓ Permissive tolerance profile created")

def test_comparison_result():
    """Test comparison result creation."""
    print("\nTesting comparison results...")
    
    result = ComparisonResult(
        field_path="test.field",
        severity=ComparisonSeverity.EQUIVALENT,
        message="Test message",
        difference_metric=0.02,
        tolerance_used=0.05
    )
    
    assert result.field_path == "test.field"
    assert result.severity == ComparisonSeverity.EQUIVALENT
    assert result.difference_metric == 0.02
    print("✓ ComparisonResult created successfully")
    
    # Test serialization
    result_dict = result.to_dict()
    assert result_dict['field_path'] == "test.field"
    assert result_dict['severity'] == "equivalent"
    print("✓ ComparisonResult serialization works")

def test_diff_report():
    """Test diff report creation."""
    print("\nTesting diff report...")
    
    report = DiffReport(
        comparison_id="test_001",
        timestamp="2025-01-01T00:00:00",
        total_comparisons=10,
        identical_count=3,
        equivalent_count=5,
        different_count=2,
        incomparable_count=0
    )
    
    assert report.total_comparisons == 10
    assert report.is_scientifically_equivalent == False  # Has 2 different
    print("✓ DiffReport created successfully")
    
    # Test summary stats
    stats = report.summary_stats
    assert stats['different_percent'] == 20.0
    assert stats['scientific_equivalence_percent'] == 80.0
    print("✓ DiffReport summary statistics work")

def test_result_comparer():
    """Test result comparer instantiation."""
    print("\nTesting result comparer...")
    
    # Default comparer
    comparer = ResultComparer()
    assert comparer.tolerance_profile is not None
    assert hasattr(comparer, 'comparison_methods')
    print("✓ ResultComparer created with default profile")
    
    # Custom profile comparer
    custom_profile = ToleranceProfile(expression_rtol=0.10)
    comparer_custom = ResultComparer(custom_profile)
    assert comparer_custom.tolerance_profile.expression_rtol == 0.10
    print("✓ ResultComparer created with custom profile")

def test_comparison_methods_exist():
    """Test that comparison methods are defined."""
    print("\nTesting comparison methods...")
    
    comparer = ResultComparer()
    expected_methods = [
        '_compare_arrays', '_compare_scalars', '_compare_clustering_results',
        '_compare_spatial_statistics', '_compare_metadata', '_compare_hierarchical_results'
    ]
    
    for method_name in expected_methods:
        assert hasattr(comparer, method_name), f"Missing method: {method_name}"
    print("✓ All comparison methods are defined")

def main():
    """Run all tests."""
    print("Testing Result Comparison Utilities")
    print("===================================")
    
    try:
        test_tolerance_profiles()
        test_comparison_result() 
        test_diff_report()
        test_result_comparer()
        test_comparison_methods_exist()
        
        print("\n" + "="*50)
        print("✓ All tests passed! Result comparison utility is ready.")
        print("\nNext steps:")
        print("1. Install numpy/pandas to run full examples")
        print("2. Use compare_results() for quick comparisons")
        print("3. Use ResultComparer class for detailed analysis")
        print("4. Check examples/result_comparison_examples.py for usage")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
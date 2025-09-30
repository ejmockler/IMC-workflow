# IMC Analysis Pipeline - Testing Documentation

## Overview

This document provides comprehensive guidance for testing the IMC (Imaging Mass Cytometry) analysis pipeline. The test suite has evolved from basic "assert it didn't explode" tests to robust validation that catches real issues and prevents regressions.

## Directory Structure

```
tests/
├── README.md                    # This documentation
├── conftest.py                  # Shared fixtures and configuration
├── pytest.ini                  # Pytest configuration (in project root)
├── fixtures/                   # Reusable test fixtures
│   ├── synthetic_data.py        # Synthetic dataset generation
│   ├── config_fixtures.py       # Standard test configurations
│   └── roi_fixtures.py          # ROI test data patterns
├── golden/                     # Golden regression test data
│   ├── synthetic_dataset.py     # Fixed golden dataset
│   ├── golden_dataset.json      # Serialized golden data
│   └── dataset_checksum.txt     # Data integrity checksum
├── baselines/                  # Performance baselines
│   ├── performance_metrics.json # Expected performance characteristics
│   └── complexity_benchmarks.py # Complexity analysis utilities
└── test_*.py                   # Individual test modules
```

## Test Categories

### 1. Unit Tests
**Purpose**: Test individual functions and classes in isolation  
**Marker**: `@pytest.mark.unit`  
**Examples**: `test_ion_count_processing.py`, `test_slic_segmentation.py`

```python
@pytest.mark.unit
def test_apply_arcsinh_transform():
    """Test arcsinh transformation of ion count data."""
    ion_counts = {'CD45': np.array([0, 5, 10, 50])}
    transformed, cofactors = apply_arcsinh_transform(ion_counts)
    
    assert 'CD45' in transformed
    assert np.all(transformed['CD45'] >= 0)  # Arcsinh preserves positivity
```

### 2. Integration Tests
**Purpose**: Test component interactions and data flow  
**Marker**: `@pytest.mark.integration`  
**Examples**: `test_pipeline_integration.py`, `test_multiscale_analysis.py`

```python
@pytest.mark.integration
def test_pipeline_end_to_end(sample_roi_data, mock_config):
    """Test complete pipeline with real data flow."""
    pipeline = IMCAnalysisPipeline(config=mock_config)
    results = pipeline.analyze_single_roi(**sample_roi_data)
    
    # Validate complete result structure
    assert 'cluster_labels' in results
    assert 'feature_matrix' in results
    assert len(results['cluster_labels']) == sample_roi_data['n_measurements']
```

### 3. Performance Tests
**Purpose**: Prevent performance regressions and verify complexity  
**Marker**: `@pytest.mark.performance`  
**Examples**: `test_performance_regression.py`, `test_memory_profiling.py`

```python
@pytest.mark.performance
def test_linear_scaling_behavior():
    """Test that processing scales linearly, not quadratically."""
    data_sizes = [1000, 5000, 10000]
    times = []
    
    for size in data_sizes:
        data = generate_test_data(size)
        start_time = time.time()
        process_data(data)
        times.append(time.time() - start_time)
    
    # Verify roughly linear scaling
    for i in range(1, len(times)):
        size_ratio = data_sizes[i] / data_sizes[i-1]
        time_ratio = times[i] / times[i-1]
        assert time_ratio <= size_ratio * 2.0  # Allow some overhead
```

### 4. Security Tests
**Purpose**: Verify security fixes and prevent vulnerabilities  
**Marker**: `@pytest.mark.security`  
**Examples**: `test_security.py`

```python
@pytest.mark.security
def test_no_pickle_rce_vulnerability():
    """Test that pickle RCE vulnerability is fixed."""
    # Verify no dangerous pickle imports in codebase
    dangerous_patterns = [r"import\s+pickle", r"pickle\.loads?\("]
    violations = scan_codebase_for_patterns(dangerous_patterns)
    assert not violations, f"Dangerous pickle usage found: {violations}"
```

### 5. Golden Regression Tests
**Purpose**: Catch silent failures where code runs but produces wrong results  
**Marker**: `@pytest.mark.regression`  
**Examples**: `test_golden_regression.py`

```python
@pytest.mark.regression
def test_coabundance_feature_count_invariant():
    """Test that co-abundance features always generate exactly 153 features."""
    golden_dataset = create_golden_dataset()
    roi_data = golden_dataset.get_roi_data()
    
    features, feature_names = generate_coabundance_features(
        ion_counts_array, roi_data['protein_names'], roi_data['coords']
    )
    
    # Critical invariant - exactly 153 features for 9 proteins
    assert features.shape[1] == 153
    assert len(feature_names) == 153
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m "integration and not slow"  # Integration tests, skip slow ones
pytest -m performance            # Performance tests only
pytest -m security              # Security tests only
```

### Test Filtering

```bash
# Run tests matching pattern
pytest -k "test_ion_count"       # All ion count tests
pytest -k "not slow"             # Skip slow tests
pytest tests/test_security.py    # Specific file

# Run failed tests from last run
pytest --lf

# Run tests that changed since last commit
pytest --testmon
```

### Performance Testing

```bash
# Run performance tests (marked as slow)
pytest -m "performance" --durations=10

# Generate detailed performance report
pytest -m performance --benchmark-autosave

# Skip performance tests in CI
pytest -m "not performance"
```

## Test Data Patterns

### 1. Synthetic Data Generation

The test suite uses deterministic synthetic data generation to ensure reproducible results:

```python
# From conftest.py
@pytest.fixture
def small_roi_data(random_seed):
    """Generate small ROI dataset for unit tests."""
    n_points = 100
    return {
        'coords': np.random.uniform(0, 50, (n_points, 2)),
        'ion_counts': {
            'CD45': np.random.negative_binomial(5, 0.3, n_points).astype(float),
            'CD31': np.random.negative_binomial(10, 0.5, n_points).astype(float),
        },
        'dna1_intensities': np.random.poisson(800, n_points).astype(float),
        'dna2_intensities': np.random.poisson(600, n_points).astype(float),
        'protein_names': ['CD45', 'CD31'],
        'n_measurements': n_points
    }
```

### 2. Golden Dataset for Regression Testing

The golden dataset provides fixed synthetic data with known ground truth:

```python
# From tests/golden/synthetic_dataset.py
golden_dataset = SyntheticIMCDataset(seed=42)
roi_data = golden_dataset.get_roi_data()

# Known properties
assert golden_dataset.n_points == 200
assert golden_dataset.expected_clusters == 4
assert golden_dataset.expected_coabundance_features == 153
```

### 3. Configuration Fixtures

Standardized test configurations prevent duplication:

```python
@pytest.fixture
def mock_config():
    """Standard test configuration using SimpleNamespace."""
    return SimpleNamespace(
        multiscale=SimpleNamespace(scales_um=[10.0, 20.0, 40.0]),
        clustering=SimpleNamespace(method="leiden", resolution=1.0),
        # ... other config sections
    )
```

## Testing Best Practices

### 1. Test Structure and Naming

- **Test classes**: Group related tests with `Test*` prefix
- **Test methods**: Descriptive names with `test_` prefix
- **Fixtures**: Reusable data/objects with clear scope
- **Markers**: Categorize tests for selective running

```python
class TestCoabundanceFeatures:
    """Test co-abundance feature generation."""
    
    def test_feature_count_invariant_for_nine_proteins(self):
        """Test that 9 proteins generate exactly 153 co-abundance features."""
        # Specific, descriptive test name explains expected behavior
```

### 2. Assertion Patterns

**Good**: Specific assertions with clear failure messages
```python
assert features.shape[1] == 153, \
    f"Expected 153 features, got {features.shape[1]}"
```

**Bad**: Generic assertions without context
```python
assert features is not None  # What should features be?
```

### 3. Error Testing

Test both success and failure cases:

```python
def test_invalid_input_handling():
    """Test graceful handling of invalid input data."""
    with pytest.raises(ValueError, match="Ion counts cannot be negative"):
        process_ion_counts({'CD45': np.array([-1, -2, -3])})
```

### 4. Deterministic Testing

Use fixed random seeds for reproducible results:

```python
@pytest.fixture
def random_seed():
    """Ensure deterministic tests."""
    np.random.seed(42)
```

### 5. Resource Management

Clean up resources properly:

```python
@pytest.fixture
def temp_directory():
    """Create and cleanup temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
    # Automatic cleanup
```

## Golden Regression Testing

### Concept

Golden regression tests use fixed synthetic datasets with known expected outputs. Any change in results indicates potential regressions.

### Implementation

1. **Fixed Dataset**: `SyntheticIMCDataset` with seed=42
2. **Known Properties**: Expected cluster count, feature count, etc.
3. **Checksum Validation**: Detect dataset changes
4. **Result Validation**: Compare against expected outputs

### Example

```python
def test_spatial_clustering_ground_truth_recovery():
    """Test that spatial clustering recovers known ground truth."""
    golden_dataset = create_golden_dataset()
    roi_data = golden_dataset.get_roi_data()
    
    # Perform clustering
    cluster_labels = perform_spatial_clustering(roi_data)
    
    # Compare with ground truth using ARI score
    ari_score = adjusted_rand_score(
        roi_data['ground_truth_labels'], 
        cluster_labels
    )
    
    # Should achieve good performance on synthetic data
    assert ari_score > 0.3, f"Poor clustering performance: ARI={ari_score:.3f}"
```

## Complexity Analysis

Performance tests use complexity analysis rather than brittle timing assertions:

### Scaling Tests

```python
def test_linear_scaling():
    """Test that function scales linearly with input size."""
    data_sizes = [1000, 5000, 10000]
    times = []
    
    for size in data_sizes:
        data = generate_data(size)
        time_taken = measure_processing_time(data)
        times.append(time_taken)
    
    # Check scaling behavior
    for i in range(1, len(times)):
        size_ratio = data_sizes[i] / data_sizes[i-1]
        time_ratio = times[i] / times[i-1]
        
        # Linear: time_ratio ≈ size_ratio
        # Quadratic: time_ratio ≈ size_ratio²
        assert time_ratio <= size_ratio * 1.5  # Allow overhead
```

### Memory Usage Tests

```python
def test_memory_efficiency():
    """Test that processing doesn't consume excessive memory."""
    import tracemalloc
    
    tracemalloc.start()
    process_large_dataset()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert peak / (1024**2) <= 100  # Max 100MB
```

## Continuous Integration

### Test Organization

```yaml
# Example CI pipeline
stages:
  - unit_tests:      pytest -m unit
  - integration:     pytest -m integration  
  - security:        pytest -m security
  - performance:     pytest -m "performance and not slow"
  - regression:      pytest tests/test_golden_regression.py
```

### Coverage Requirements

- **Overall Coverage**: ≥40% (current threshold in pytest.ini)
- **Critical Modules**: Aim for ≥80% coverage
- **Performance**: Use complexity analysis, not timing thresholds

### Failure Analysis

1. **Unit Test Failures**: Usually indicate breaking changes
2. **Integration Failures**: Often reveal API mismatches  
3. **Security Failures**: Potential vulnerability regressions
4. **Performance Failures**: Algorithm efficiency regressions
5. **Golden Failures**: Silent correctness regressions

## Common Testing Patterns

### Mock Usage

```python
# Mock external dependencies
@patch('src.analysis.external_library.expensive_function')
def test_function_with_mocked_dependency(mock_expensive):
    mock_expensive.return_value = "mocked_result"
    result = function_using_dependency()
    assert result == "expected_based_on_mock"
    mock_expensive.assert_called_once()
```

### Parametrized Tests

```python
@pytest.mark.parametrize("protein_count,expected_features", [
    (3, 21),   # 3 proteins → 21 co-abundance features
    (5, 65),   # 5 proteins → 65 co-abundance features  
    (9, 153),  # 9 proteins → 153 co-abundance features
])
def test_coabundance_feature_scaling(protein_count, expected_features):
    """Test feature count scaling with protein count."""
    proteins = [f'protein_{i}' for i in range(protein_count)]
    features = generate_coabundance_features(proteins)
    assert len(features) == expected_features
```

### Fixture Scoping

```python
# Module-scoped for expensive setup
@pytest.fixture(scope="module")
def large_dataset():
    return expensive_data_generation()

# Function-scoped for test isolation  
@pytest.fixture(scope="function")
def clean_temp_dir():
    return tempfile.mkdtemp()
```

## Debugging Test Failures

### Verbose Output

```bash
pytest -v --tb=long  # Detailed traceback
pytest -s           # Show print statements
pytest --pdb        # Drop into debugger on failure
```

### Selective Running

```bash
pytest tests/test_specific.py::TestClass::test_method
pytest -k "coabundance and not slow"
pytest --lf --tb=no  # Re-run failures only, minimal output
```

### Coverage Analysis

```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html to see detailed coverage
```

## Future Improvements

1. **Property-Based Testing**: Use hypothesis for generative testing
2. **Visual Regression Tests**: Compare generated plots/figures
3. **Database Testing**: Test with actual IMC databases
4. **Stress Testing**: Large-scale data processing validation
5. **Contract Testing**: API compatibility between components

## Summary

This testing framework has evolved from basic smoke tests to comprehensive validation that:

- **Prevents regressions** through golden dataset testing
- **Ensures correctness** with property-based validation  
- **Maintains performance** via complexity analysis
- **Enforces security** through vulnerability scanning
- **Supports refactoring** with reliable test coverage

The key insight: tests should validate correctness and properties, not just "didn't crash."
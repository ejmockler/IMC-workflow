# IMC Analysis Pipeline - Testing Best Practices Guide

## Overview

This guide documents the evolution of the IMC analysis pipeline testing approach from basic "assert it didn't explode" tests to comprehensive validation that catches real issues and prevents regressions.

## The Testing Evolution Journey

### Phase 1: Basic Smoke Tests (BEFORE)
```python
# Old approach - minimal testing
def test_function_runs():
    """Test that function doesn't crash."""
    result = some_function(test_data)
    assert result is not None  # Just check it didn't explode
```

**Problems with this approach:**
- Silent failures where code runs but produces wrong results
- No validation of correctness or expected properties
- Brittle tests that break on implementation changes
- No detection of performance regressions
- Security vulnerabilities go unnoticed

### Phase 2: Property-Based Validation (AFTER)
```python  
# New approach - comprehensive validation
def test_coabundance_feature_generation():
    """Test co-abundance feature generation with property validation."""
    golden_dataset = create_golden_dataset()
    roi_data = golden_dataset.get_roi_data()
    
    features, feature_names = generate_coabundance_features(
        ion_counts_array, roi_data['protein_names'], roi_data['coords']
    )
    
    # Validate mathematical properties
    assert features.shape[1] == 153, f"Expected 153 features for 9 proteins, got {features.shape[1]}"
    assert len(feature_names) == 153
    assert np.all(np.isfinite(features)), "Features contain NaN or infinite values"
    
    # Validate feature composition (products, ratios, covariances)  
    product_count = sum(1 for name in feature_names if '*' in name)
    ratio_count = sum(1 for name in feature_names if '/' in name)
    assert product_count == 36, f"Expected 36 products for 9 proteins, got {product_count}"
    assert ratio_count == 72, f"Expected 72 ratios for 9 proteins, got {ratio_count}"
    
    # Validate determinism
    features2, _ = generate_coabundance_features(
        ion_counts_array, roi_data['protein_names'], roi_data['coords']
    )
    np.testing.assert_array_almost_equal(features, features2, decimal=10)
```

## Core Testing Principles

### 1. Test Properties, Not Implementation

**Good**: Test mathematical invariants and expected properties
```python
def test_clustering_produces_valid_labels():
    """Test clustering output properties."""
    labels = perform_clustering(data)
    
    # Property validation
    assert len(labels) == len(data), "Label count mismatch"
    assert np.all(labels >= -1), "Invalid cluster labels (should be >= -1)"
    assert len(np.unique(labels[labels >= 0])) <= len(data), "More clusters than points"
```

**Bad**: Test implementation details
```python
def test_clustering_uses_leiden_algorithm():
    """Test implementation detail - brittle."""
    with patch('some_internal_function') as mock:
        perform_clustering(data)
        mock.assert_called()  # Breaks when implementation changes
```

### 2. Use Golden Datasets for Regression Testing

**Concept**: Fixed synthetic datasets with known ground truth detect silent failures.

```python
def test_spatial_clustering_ground_truth_recovery():
    """Test clustering against known ground truth."""
    golden_dataset = create_golden_dataset()  # Fixed seed=42
    roi_data = golden_dataset.get_roi_data()
    
    cluster_labels = perform_spatial_clustering(roi_data)
    true_labels = roi_data['ground_truth_labels']
    
    # Measure clustering quality
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    
    # Should achieve good performance on synthetic data
    assert ari_score > 0.3, f"Poor clustering performance: ARI={ari_score:.3f}"
    
    # For golden dataset, expect excellent performance
    if ari_score > 0.7:
        print(f"Excellent clustering: ARI={ari_score:.3f}")
```

### 3. Complexity Analysis Over Timing Tests

**Good**: Test algorithmic complexity
```python
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
        # Linear scaling: time_ratio ≈ size_ratio
        assert time_ratio <= size_ratio * 2.0, f"Non-linear scaling detected: {time_ratio:.2f}"
```

**Bad**: Brittle absolute timing tests
```python
def test_function_is_fast():
    """Brittle timing test."""
    start = time.time()
    process_data(test_data)
    elapsed = time.time() - start
    assert elapsed < 1.0  # Fails on slow CI, different hardware
```

### 4. Comprehensive Error Testing

**Test both success and failure modes:**

```python
def test_invalid_input_handling():
    """Test graceful handling of invalid input data."""
    
    # Test various invalid inputs
    invalid_inputs = [
        {'coords': np.array([]), 'ion_counts': {}},  # Empty data
        {'coords': np.array([[np.nan, 1]]), 'ion_counts': {'p1': [5]}},  # NaN coords
        {'coords': np.array([[1, 2]]), 'ion_counts': {'p1': [np.inf]}},  # Infinite values
        {'coords': np.array([[1, 2]]), 'ion_counts': {'p1': [-1]}},  # Negative counts
    ]
    
    for invalid_input in invalid_inputs:
        with pytest.raises((ValueError, TypeError), match="Invalid input"):
            process_roi_data(invalid_input)
```

### 5. Security-First Testing

**Test for vulnerabilities proactively:**

```python
@pytest.mark.security
def test_no_pickle_rce_vulnerability():
    """Test that pickle RCE vulnerability is fixed."""
    
    # Scan codebase for dangerous patterns
    dangerous_patterns = [r"import\s+pickle", r"pickle\.loads?\("]
    violations = scan_codebase_for_patterns(dangerous_patterns)
    
    # Allow only safe usage in specific files
    allowed_files = ["data_storage.py"]  # Only for test file creation
    filtered_violations = [v for v in violations 
                          if not any(allowed in v for allowed in allowed_files)]
    
    assert not filtered_violations, f"Dangerous pickle usage: {filtered_violations}"

def test_roi_id_path_traversal_protection():
    """Test protection against path traversal attacks."""
    dangerous_ids = ["../../../etc/passwd", "roi_id; rm -rf /", "a" * 1000]
    
    storage = create_storage_backend({'format': 'json'}, temp_dir)
    
    for dangerous_id in dangerous_ids:
        try:
            storage.save_roi_analysis(dangerous_id, {'test': 'data'})
            # Verify no files created outside temp_dir
            all_files = list(Path(temp_dir).rglob("*"))
            for file_path in all_files:
                assert str(temp_dir) in str(file_path.resolve())
        except (ValueError, OSError):
            pass  # Expected - dangerous IDs should be rejected
```

## Test Categories and Markers

### Unit Tests (`@pytest.mark.unit`)
- Test individual functions in isolation
- Fast execution (< 5 seconds each)
- Mock external dependencies
- Focus on correctness and edge cases

```python
@pytest.mark.unit
def test_arcsinh_transform_properties():
    """Test arcsinh transformation mathematical properties."""
    ion_counts = {'CD45': np.array([0, 1, 10, 100, 1000])}
    transformed, cofactors = apply_arcsinh_transform(ion_counts)
    
    # Mathematical properties
    assert np.all(transformed['CD45'] >= 0)  # Preserves non-negativity
    assert transformed['CD45'][0] == 0  # arcsinh(0) = 0
    assert np.all(np.diff(transformed['CD45']) > 0)  # Monotonically increasing
```

### Integration Tests (`@pytest.mark.integration`)
- Test component interactions
- Real data flow between modules
- Longer execution time acceptable
- Validate complete workflows

```python
@pytest.mark.integration  
def test_pipeline_end_to_end():
    """Test complete analysis pipeline."""
    pipeline = IMCAnalysisPipeline(config=test_config)
    
    # Real data flow through entire pipeline
    results = pipeline.analyze_single_roi(
        coords=roi_data['coords'],
        ion_counts=roi_data['ion_counts'],
        # ... other parameters
    )
    
    # Validate complete result structure
    assert 'cluster_labels' in results
    assert 'feature_matrix' in results
    assert len(results['cluster_labels']) == len(roi_data['coords'])
    assert results['feature_matrix'].shape[1] == 153  # Co-abundance features
```

### Performance Tests (`@pytest.mark.performance`)
- Complexity analysis and regression detection
- Memory usage validation
- May be skipped in fast CI runs
- Focus on scaling behavior

```python
@pytest.mark.performance
def test_memory_usage_linear_scaling():
    """Test that memory usage scales predictably."""
    import tracemalloc
    
    tracemalloc.start()
    
    # Test different data sizes
    for size in [1000, 5000, 10000]:
        initial_memory = tracemalloc.get_traced_memory()[0]
        
        # Process data
        large_data = generate_test_data(size)
        process_data(large_data)
        
        peak_memory = tracemalloc.get_traced_memory()[1]
        memory_used = (peak_memory - initial_memory) / (1024**2)  # MB
        
        # Memory should scale reasonably with data size
        memory_per_point = memory_used / size
        assert memory_per_point < 0.1, f"Excessive memory per point: {memory_per_point:.3f}MB"
    
    tracemalloc.stop()
```

### Security Tests (`@pytest.mark.security`)
- Vulnerability prevention and detection
- Input validation and sanitization
- Safe file handling practices

### Regression Tests (`@pytest.mark.regression`)
- Golden dataset validation
- Prevent silent correctness failures
- Detect changes in algorithm outputs

## Data Management Best Practices

### 1. Deterministic Test Data
```python
@pytest.fixture
def random_seed():
    """Ensure deterministic test results."""
    np.random.seed(42)

@pytest.fixture  
def reproducible_roi_data(random_seed):
    """Generate reproducible test data."""
    # Fixed seed ensures same data across runs
    return generate_synthetic_roi(n_points=1000, seed=42)
```

### 2. Comprehensive Fixtures
```python
# Size-based fixtures for different test scenarios
@pytest.fixture
def small_roi_data():    # 100 points - unit tests
@pytest.fixture  
def medium_roi_data():   # 1000 points - integration tests
@pytest.fixture
def large_roi_data():    # 5000 points - performance tests

# Quality-based fixtures for robustness testing
@pytest.fixture
def excellent_roi():     # High quality, no issues
@pytest.fixture
def poor_roi():         # Poor quality, multiple issues
@pytest.fixture
def corrupted_roi():    # Corrupted data for error handling
```

### 3. Configuration Management
```python
# Profile-based configurations
@pytest.fixture
def unit_test_config():      # Fast, minimal features
@pytest.fixture
def integration_test_config(): # Full features, realistic
@pytest.fixture  
def performance_test_config():  # Optimized for speed
@pytest.fixture
def security_test_config():     # Safe settings, no pickle
```

## Assertion Patterns

### Good Assertions
```python
# Specific with clear failure messages
assert features.shape[1] == 153, f"Expected 153 features, got {features.shape[1]}"

# Test properties, not exact values
assert 0.3 < ari_score < 1.0, f"ARI score {ari_score} outside valid range"

# Validate data structure
assert isinstance(results, dict), f"Expected dict, got {type(results)}"
assert all(key in results for key in required_keys), f"Missing keys: {missing_keys}"

# Mathematical properties
assert np.all(labels >= -1), "Cluster labels must be >= -1 (outliers)"
assert len(np.unique(labels)) <= len(data), "More clusters than data points"
```

### Bad Assertions
```python
# Vague, unhelpful when failing
assert result is not None
assert len(result) > 0

# Too specific, brittle
assert result == [1, 2, 3, 4]  # Exact match fragile to changes

# Implementation details
assert 'leiden' in str(clustering_method)  # Tests implementation, not behavior
```

## Error Testing Patterns

### Comprehensive Error Coverage
```python
def test_error_handling_comprehensive():
    """Test various error conditions systematically."""
    
    # Invalid data types
    with pytest.raises(TypeError, match="Expected numpy array"):
        process_data("not an array")
    
    # Invalid data shapes
    with pytest.raises(ValueError, match="Coordinates must be 2D"):
        process_data(np.array([1, 2, 3]))  # 1D instead of 2D
    
    # Invalid data values
    with pytest.raises(ValueError, match="Ion counts cannot be negative"):
        process_data({'coords': valid_coords, 'ion_counts': {'p1': [-1, -2]}})
    
    # Empty data
    with pytest.raises(ValueError, match="Cannot process empty dataset"):
        process_data({'coords': np.array([]), 'ion_counts': {}})
    
    # Corrupted data
    with pytest.raises(ValueError, match="NaN values detected"):
        process_data({'coords': [[np.nan, 1]], 'ion_counts': {'p1': [5]}})
```

## Resource Management

### Memory and Cleanup
```python
@pytest.fixture
def temp_directory():
    """Create and automatically clean up temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
    # Automatic cleanup

def test_memory_leak_detection():
    """Test for memory leaks in processing."""
    import gc
    import psutil
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Run processing multiple times
    for _ in range(10):
        data = generate_test_data(1000)
        result = process_data(data)
        del data, result  # Explicit cleanup
        gc.collect()
    
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / (1024**2)  # MB
    
    assert memory_increase < 10, f"Memory leak detected: {memory_increase}MB increase"
```

## Continuous Integration Best Practices

### Test Organization for CI
```python
# pytest.ini configuration
[tool:pytest]
markers =
    unit: Fast unit tests (always run)
    integration: Integration tests (run in full CI) 
    performance: Performance tests (run nightly)
    slow: Slow tests (run nightly)
    security: Security tests (always run)

# CI pipeline stages
stages:
  - fast:        pytest -m "unit and not slow"
  - standard:    pytest -m "integration and not slow"  
  - security:    pytest -m security
  - nightly:     pytest -m "performance or slow"
```

### Performance Monitoring
```python
# Track performance metrics over time
def save_performance_metrics(test_results):
    """Save performance metrics for trend analysis."""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'commit_hash': get_git_commit(),
        'timing_results': test_results,
        'memory_usage': get_memory_stats(),
        'test_environment': get_env_info()
    }
    
    with open('performance_history.jsonl', 'a') as f:
        json.dump(metrics, f)
        f.write('\n')
```

## Debugging Test Failures

### Useful pytest Options
```bash
# Detailed output
pytest -v --tb=long

# Show print statements  
pytest -s

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Re-run only failed tests
pytest --lf

# Show test durations
pytest --durations=10
```

### Diagnostic Information
```python
def test_with_diagnostics():
    """Example of diagnostic information in tests."""
    roi_data = create_test_roi()
    
    print(f"Test data: {len(roi_data['coords'])} points, {len(roi_data['protein_names'])} proteins")
    
    try:
        result = process_roi(roi_data)
    except Exception as e:
        # Add diagnostic information to failures
        print(f"ROI data summary: {get_roi_summary_stats(roi_data)}")
        print(f"Error occurred with config: {test_config}")
        raise
    
    # Validate with informative messages
    assert result['n_clusters'] > 0, f"No clusters found. Data stats: {get_roi_summary_stats(roi_data)}"
```

## Common Anti-Patterns to Avoid

### 1. Testing Implementation Details
```python
# BAD - tests internal implementation
def test_uses_specific_library():
    with patch('internal_module.specific_function') as mock:
        process_data(test_data)
        mock.assert_called()

# GOOD - tests behavior and properties  
def test_produces_valid_output():
    result = process_data(test_data)
    assert isinstance(result, dict)
    assert 'clusters' in result
    assert len(result['clusters']) <= len(test_data)
```

### 2. Overly Broad Exception Catching
```python
# BAD - catches everything, hides real issues
def test_function_handles_errors():
    try:
        result = risky_function(bad_data)
        assert True  # Just didn't crash
    except:
        pass

# GOOD - specific exception testing
def test_function_validates_input():
    with pytest.raises(ValueError, match="Invalid input format"):
        risky_function(malformed_data)
```

### 3. Non-Deterministic Tests
```python
# BAD - random behavior makes tests flaky
def test_clustering():
    data = generate_random_data()  # Different each run
    clusters = cluster_data(data)
    assert len(clusters) > 0  # Sometimes fails randomly

# GOOD - deterministic with fixed seed
def test_clustering_reproducible():
    np.random.seed(42)  # Fixed seed
    data = generate_test_data(seed=42)
    clusters = cluster_data(data)
    assert len(clusters) == 4  # Known expected result
```

### 4. Testing Too Many Things At Once
```python
# BAD - monolithic test, hard to debug failures
def test_entire_pipeline():
    # Tests loading, processing, clustering, saving, visualization
    # If it fails, unclear which component broke
    
# GOOD - focused tests
def test_data_loading():
    # Only tests data loading
    
def test_clustering_algorithm():
    # Only tests clustering with known input
    
def test_result_saving():
    # Only tests saving with known data
```

## Measuring Test Quality

### Coverage Analysis
```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# Focus on critical modules
pytest --cov=src/analysis --cov-report=term-missing
```

### Test Effectiveness Metrics
1. **Mutation Testing**: Do tests catch intentional bugs?
2. **Regression Detection**: Do tests catch when outputs change?
3. **Property Validation**: Do tests verify mathematical correctness?
4. **Error Coverage**: Do tests cover all error paths?

### Test Maintenance
- **Green Tests**: All tests should pass on main branch
- **Fast Feedback**: Unit tests should run in < 1 minute
- **Deterministic**: Tests should never be flaky
- **Independent**: Tests shouldn't depend on each other
- **Clear Names**: Test names should explain what they validate

## Summary

The evolution from "assert it didn't explode" to comprehensive validation represents a fundamental shift in testing philosophy:

**Before**: Tests checked that code ran without crashing
**After**: Tests validate correctness, performance, security, and properties

**Key Improvements:**
1. **Property-based validation** catches silent failures
2. **Golden regression testing** prevents correctness regressions  
3. **Complexity analysis** prevents performance regressions
4. **Security testing** prevents vulnerability regressions
5. **Comprehensive fixtures** enable thorough testing
6. **Deterministic data** ensures reproducible results

This approach transforms tests from a debugging aid to a comprehensive validation system that enables confident refactoring and ensures long-term code quality.
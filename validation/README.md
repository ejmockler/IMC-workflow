# Validation Directory

This directory contains scientific validation analyses, separate from software verification tests.

## Purpose

Unlike the `tests/` directory which verifies that code works correctly, this directory validates that the **scientific outputs** are meaningful and biologically coherent.

## Contents

- **Jupyter notebooks**: Interactive analysis of algorithm changes
- **Validation reports**: PDF/HTML reports showing impact of major changes
- **Benchmark datasets**: Small, real datasets for consistent validation
- **Scientific assessment**: Expert review of clustering biological relevance

## Workflow

1. **Algorithm Change**: Developer modifies clustering/co-abundance algorithms
2. **Run Validation**: Execute validation notebooks on benchmark data
3. **Generate Report**: Create PDF/HTML showing before/after comparison
4. **Expert Review**: Scientist reviews biological coherence of changes
5. **Commit Report**: Add validation report to git history

## Key Distinction

- **`tests/`**: Fast, automated, pass/fail software verification (pytest)
- **`validation/`**: Slow, manual, expert-reviewed scientific validation

## Files

- `validate_coabundance_biology.ipynb`: Check that protein interactions make biological sense
- `validate_clustering_coherence.ipynb`: Assess spatial clustering quality
- `validate_scale_relationships.ipynb`: Verify hierarchical scale relationships
- `benchmark_datasets/`: Small, real IMC datasets for consistent validation
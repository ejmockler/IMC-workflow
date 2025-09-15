# IMC Analysis Benchmark Report

Generated: 2025-09-15T03:09:42.777065

## Data Description

| Metric | Value |
|--------|-------|
| Samples | 498 |
| Features | 5 |
| Memory (MB) | 0.02 |

## Clustering Comparison

**Best Method:** kmeans (Score: 0.440)

| Method | Quality | Time (s) | Memory (MB) |
|--------|---------|----------|-------------|
| **kmeans** | 0.440 | 0.08 | -0.1 |
| minibatch | 0.390 | 0.07 | -0.2 |

## Kernels Comparison

**Best Method:** gaussian (Score: 0.320)

| Method | Quality | Time (s) | Memory (MB) |
|--------|---------|----------|-------------|
| **gaussian** | 0.320 | 0.12 | -1.5 |

## Integrated Comparison

**Best Method:** kmeans+gaussian(λ=0.7) (Score: 0.724)

| Method | Quality | Time (s) | Memory (MB) |
|--------|---------|----------|-------------|
| kmeans+gaussian(λ=0.3) | 0.376 | 0.07 | 0.0 |
| **kmeans+gaussian(λ=0.7)** | 0.724 | 0.01 | 0.0 |
| minibatch+gaussian(λ=0.3) | 0.332 | 0.07 | 0.0 |
| minibatch+gaussian(λ=0.7) | 0.568 | 0.07 | 0.0 |

## Summary Statistics

| Metric | Value |
|--------|-------|
| best_overall_method | integrated_best |
| best_overall_score | 0.724 |
| mean_quality | 0.450 |
| std_quality | 0.136 |
| max_quality | 0.724 |

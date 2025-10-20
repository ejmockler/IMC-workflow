#!/usr/bin/env python3
"""
Test visualization with quality metrics in title.
"""

import numpy as np
from pathlib import Path

print("Testing visualization with quality metrics...")

# Create more realistic test data
np.random.seed(42)

# Create synthetic DNA composite with tissue structure
composite_dna = np.zeros((100, 100))
# Add some tissue regions (simulate sparse tissue)
for _ in range(5):
    x, y = np.random.randint(10, 90, 2)
    size = np.random.randint(10, 30)
    composite_dna[max(0, x-size):min(100, x+size), 
                  max(0, y-size):min(100, y+size)] = np.random.rand() * 100

# Create segmentation with variable superpixel sizes
labels = np.full((100, 100), -1, dtype=int)  # Start with background
superpixel_id = 0

# Create superpixels only in tissue regions
for i in range(0, 100, 10):
    for j in range(0, 100, 10):
        region = composite_dna[i:i+10, j:j+10]
        if region.mean() > 10:  # Only segment tissue areas
            # Variable superpixel size for heterogeneity
            size = np.random.randint(5, 15)
            labels[i:min(i+size, 100), j:min(j+size, 100)] = superpixel_id
            superpixel_id += 1

# Calculate metrics
tissue_pixels = np.sum(labels >= 0)
total_pixels = labels.size
tissue_coverage = (tissue_pixels / total_pixels) * 100

unique_superpixels = np.unique(labels[labels >= 0])
n_superpixels = len(unique_superpixels)
mean_superpixel_size = tissue_pixels / n_superpixels if n_superpixels > 0 else 0

# Calculate heterogeneity
superpixel_sizes = [np.sum(labels == sp) for sp in unique_superpixels]
size_cv = np.std(superpixel_sizes) / np.mean(superpixel_sizes) if np.mean(superpixel_sizes) > 0 else 0

print(f"\nQuality Metrics:")
print(f"  Tissue Coverage: {tissue_coverage:.1f}%")
print(f"  Number of Segments: {n_superpixels}")
print(f"  Mean Segment Size: {mean_superpixel_size:.0f} pixels")
print(f"  Size Heterogeneity (CV): {size_cv:.2f}")

# Create bounds
bounds = (0, 100, 0, 100)

# Create synthetic protein data
protein_names = ['CD45', 'CD11b', 'CD31', 'CD140a', 'CD140b', 'CD206', 'CD44', 'CD34', 'Ly6G']
transformed_arrays = {
    protein: np.random.rand(n_superpixels) * 10 for protein in protein_names
}

# Create cofactors
cofactors_used = {
    protein: np.random.uniform(0.5, 5.0) for protein in protein_names
}

# Create minimal config
from types import SimpleNamespace
config = SimpleNamespace(
    visualization={'validation_plots': {
        'primary_markers': {'immune_markers': 'CD45', 'vascular_markers': 'CD31'},
        'colormaps': {'immune_markers': 'Reds', 'vascular_markers': 'Blues', 'default': 'viridis'},
        'always_include': ['CD206', 'CD44'],
        'max_additional_channels': 5,
        'layout': {'figsize': [20, 12]}
    }},
    channel_groups={
        'immune_markers': {'pan_leukocyte': ['CD45']},
        'vascular_markers': ['CD31', 'CD34'],
        'stromal_markers': ['CD140a', 'CD140b']
    },
    dna_processing={'arcsinh_transform': {'enabled': True}}
)

# Generate plot with quality metrics in title
try:
    from src.viz_utils.plotting import plot_segmentation_overlay
    
    # Ensure output directory exists
    plots_dir = Path("plots/validation")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create title with metrics
    title = (f"Test_ROI | 20μm scale | "
            f"Coverage: {tissue_coverage:.1f}% | "
            f"Segments: {n_superpixels} ({mean_superpixel_size:.0f}px) | "
            f"Heterogeneity: CV={size_cv:.2f}")
    
    fig = plot_segmentation_overlay(
        image=composite_dna,
        labels=labels,
        bounds=bounds,
        transformed_arrays=transformed_arrays,
        cofactors_used=cofactors_used,
        config=config,
        title=title
    )
    
    # Save plot
    output_file = plots_dir / "test_quality_metrics.png"
    fig.savefig(output_file, dpi=100, bbox_inches='tight')
    
    print(f"\n✓ Plot saved to {output_file}")
    print(f"✓ Title shows: {title}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
#!/usr/bin/env python3
"""
Test visualization functionality with minimal dependencies.
"""

import numpy as np
from pathlib import Path

print("Testing visualization setup...")

# Check if plots can be generated
try:
    from src.viz_utils.plotting import plot_segmentation_overlay
    print("✓ Visualization function imported successfully")
except ImportError as e:
    print(f"✗ Could not import visualization function: {e}")
    
# Create synthetic test data
print("\nCreating synthetic test data...")
np.random.seed(42)

# Create synthetic DNA composite (100x100)
composite_dna = np.random.rand(100, 100) * 100

# Create synthetic segmentation labels  
labels = np.zeros((100, 100), dtype=int)
for i in range(10):
    for j in range(10):
        labels[i*10:(i+1)*10, j*10:(j+1)*10] = i * 10 + j

# Create bounds
bounds = (0, 100, 0, 100)

# Create synthetic protein data (100 superpixels, 9 proteins)
protein_names = ['CD45', 'CD11b', 'CD31', 'CD140a', 'CD140b', 'CD206', 'CD44', 'CD34', 'Ly6G']
transformed_arrays = {
    protein: np.random.rand(100) * 10 for protein in protein_names
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

print("Test data created successfully")

# Try to generate plot
print("\nAttempting to generate validation plot...")
try:
    from src.viz_utils.plotting import plot_segmentation_overlay
    
    # Ensure output directory exists
    plots_dir = Path("plots/validation")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plot_segmentation_overlay(
        image=composite_dna,
        labels=labels,
        bounds=bounds,
        transformed_arrays=transformed_arrays,
        cofactors_used=cofactors_used,
        config=config,
        title="Test Visualization - Scale 10μm"
    )
    
    # Save plot
    output_file = plots_dir / "test_visualization.png"
    fig.savefig(output_file, dpi=100, bbox_inches='tight')
    
    print(f"✓ Plot saved successfully to {output_file}")
    
    # Check file exists
    if output_file.exists():
        size_kb = output_file.stat().st_size / 1024
        print(f"✓ File size: {size_kb:.1f} KB")
    
except Exception as e:
    print(f"✗ Failed to generate plot: {e}")
    import traceback
    traceback.print_exc()

print("\nVisualization test complete.")
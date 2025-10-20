#!/usr/bin/env python3
"""
Demo: Parameter Profiles Integration

Shows how parameter profiles integrate with existing Config system.
Works without external dependencies for demonstration.
"""

import json
from pathlib import Path


def mock_numpy_arrays():
    """Create mock data without numpy dependency for demo."""
    # Mock coordinate data (1000 points)
    coords = [[i % 50, i // 50] for i in range(1000)]
    
    # Mock DNA intensities
    dna = [abs(hash(f"dna_{i}")) % 100 / 10.0 for i in range(1000)]
    
    return coords, dna


def demo_tissue_profiles():
    """Demonstrate tissue-specific profiles."""
    print("=== Tissue-Specific Parameter Profiles ===")
    
    # Read the profiles directly from the code
    profiles_content = Path('src/analysis/parameter_profiles.py').read_text()
    
    # Extract and show tissue profiles (simplified)
    print("Available tissue profiles:")
    print("- kidney: Optimized for tubular/glomerular structures")
    print("- brain: Optimized for neuronal/glial structures")  
    print("- tumor: Optimized for heterogeneous environments")
    print("- default: General-purpose parameters")
    
    # Show example kidney parameters
    print("\nKidney profile parameters:")
    print("  scales_um: [10.0, 20.0, 75.0]  # Capillary, tubular, architectural")
    print("  slic_params: {compactness: 15.0, sigma: 1.2}  # Higher for elongated tubules")
    print("  clustering: {resolution_range: [0.3, 1.5]}  # Conservative for clear structures")


def demo_data_adaptation():
    """Demonstrate data-driven parameter adaptation."""
    print("\n=== Data-Driven Parameter Adaptation ===")
    
    coords, dna = mock_numpy_arrays()
    
    print("Mock data characteristics:")
    print(f"  Number of measurements: {len(coords)}")
    print(f"  Spatial range: {max(c[0] for c in coords)} x {max(c[1] for c in coords)} μm")
    print(f"  DNA signal range: {min(dna):.1f} - {max(dna):.1f}")
    
    # Estimate characteristics (simplified)
    x_range = max(c[0] for c in coords) - min(c[0] for c in coords)
    y_range = max(c[1] for c in coords) - min(c[1] for c in coords)
    density = len(coords) / (x_range * y_range)
    
    non_zero_dna = [d for d in dna if d > 0]
    sparsity = 1.0 - (len(non_zero_dna) / len(dna))
    
    print(f"\nEstimated characteristics:")
    print(f"  Density: {density:.1f} pixels/μm²")
    print(f"  Sparsity: {sparsity:.2f}")
    
    # Show adaptation logic
    print("\nAdaptation rules:")
    if density > 100:
        print("  High density → Tighter clustering resolution [0.3, 1.2]")
    elif density < 10:
        print("  Low density → Wider clustering resolution [0.8, 3.0]")
    else:
        print("  Medium density → Standard clustering resolution [0.5, 2.0]")
    
    if sparsity > 0.7:
        print("  Sparse data → Lower QC thresholds (-30%)")
    else:
        print("  Dense data → Standard QC thresholds")


def demo_config_integration():
    """Demonstrate integration with existing Config."""
    print("\n=== Config Integration ===")
    
    # Show current config structure
    if Path('config.json').exists():
        with open('config.json') as f:
            config = json.load(f)
        
        print("Current config scales:", config.get('segmentation', {}).get('scales_um', 'Not found'))
        print("Current SLIC params:", config.get('segmentation', {}).get('slic_params', 'Not found'))
    
    print("\nParameter profiles integration approach:")
    print("1. Config object remains unchanged")
    print("2. Profiles generate override parameters")
    print("3. Overrides passed to pipeline analysis functions")
    print("4. Original config preserved for other uses")
    
    print("\nExample usage in pipeline:")
    print("```python")
    print("# Load normal config")
    print("config = Config('config.json')")
    print("")
    print("# Get kidney-specific parameters") 
    print("kidney_params = create_adaptive_config(")
    print("    config=config,")
    print("    coords=roi_data['coords'],")
    print("    dna_intensities=roi_data['dna1_intensities'],")
    print("    tissue_type='kidney'")
    print(")")
    print("")
    print("# Run analysis with adaptive parameters")
    print("result = pipeline.analyze_single_roi(")
    print("    roi_data=roi_data,")
    print("    override_config=kidney_params")
    print(")")
    print("```")


def demo_resolution_scaling():
    """Demonstrate resolution-based parameter scaling."""
    print("\n=== Resolution-Based Parameter Scaling ===")
    
    print("Parameter scaling for different resolutions:")
    
    resolutions = [0.5, 1.0, 2.0]  # μm per pixel
    base_scale = 20.0  # μm
    
    for res in resolutions:
        pixels = max(1, round(base_scale / res))
        print(f"  {res} μm/pixel: {base_scale}μm = {pixels} pixels")
        
        # Sigma adjustment
        if res < 0.5:
            sigma_mult = 0.7
            note = "(less smoothing for high-res)"
        elif res > 2.0:
            sigma_mult = 1.5
            note = "(more smoothing for low-res)"
        else:
            sigma_mult = 1.0
            note = "(standard)"
        
        print(f"    Sigma multiplier: {sigma_mult} {note}")


if __name__ == "__main__":
    print("Parameter Profiles Demo")
    print("=" * 50)
    
    try:
        demo_tissue_profiles()
        demo_data_adaptation()
        demo_config_integration()
        demo_resolution_scaling()
        
        print("\n" + "=" * 50)
        print("✓ Demo completed successfully!")
        print("\nKey benefits:")
        print("- Tissue-specific parameter optimization")
        print("- Data-driven threshold adaptation")
        print("- Clean integration with existing Config")
        print("- No breaking changes to current pipeline")
        print("- Simple, obvious parameter selection logic")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
"""
Test Script for Watershed DNA Segmentation

Validates watershed segmentation implementation with synthetic and real IMC data.
Tests nucleus detection, cell boundary inference, and quality metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from typing import Dict, Tuple

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from analysis.watershed_segmentation import (
        prepare_dna_for_nucleus_detection,
        detect_nucleus_seeds,
        perform_watershed_segmentation,
        aggregate_to_cells,
        compute_cell_properties,
        assess_watershed_quality,
        watershed_pipeline
    )
    from analysis.slic_segmentation import slic_pipeline
    from config import Config
    print("✓ Successfully imported watershed segmentation modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Check dependencies
try:
    from skimage.segmentation import watershed
    from skimage.measure import regionprops
    from scipy import ndimage
    print("✓ Required dependencies available")
except ImportError as e:
    print(f"✗ Missing dependencies: {e}")
    print("Install with: pip install scikit-image scipy")
    sys.exit(1)


def create_synthetic_tissue_data(
    n_cells: int = 50,
    image_size: int = 200,
    cell_radius_range: Tuple = (3, 8),
    noise_level: float = 0.1
) -> Dict:
    """
    Create synthetic IMC tissue data with realistic nucleus and cell structures.
    
    Args:
        n_cells: Number of cells to simulate
        image_size: Size of synthetic image
        cell_radius_range: Range of cell radii in pixels
        noise_level: Background noise level
        
    Returns:
        Dictionary with synthetic data
    """
    print(f"Creating synthetic tissue with {n_cells} cells...")
    
    # Create coordinate grid
    y, x = np.mgrid[0:image_size, 0:image_size]
    coords = np.column_stack([x.ravel(), y.ravel()])
    
    # Generate random cell centers
    np.random.seed(42)  # Reproducible results
    cell_centers = np.random.uniform(
        low=max(cell_radius_range), 
        high=image_size - max(cell_radius_range), 
        size=(n_cells, 2)
    )
    
    # Create DNA channels with realistic nucleus structure
    dna1_field = np.zeros((image_size, image_size))
    dna2_field = np.zeros((image_size, image_size))
    
    # Create protein expression fields
    cd45_field = np.zeros((image_size, image_size))
    cd31_field = np.zeros((image_size, image_size))
    
    for i, (cx, cy) in enumerate(cell_centers):
        # Random cell radius
        cell_radius = np.random.uniform(*cell_radius_range)
        nucleus_radius = cell_radius * 0.4  # Nucleus is ~40% of cell
        
        # Create distance maps
        cell_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        nucleus_mask = cell_dist <= nucleus_radius
        cell_mask = cell_dist <= cell_radius
        
        # Add DNA signal (stronger in nucleus)
        nucleus_intensity = np.random.uniform(50, 100)
        cytoplasm_intensity = nucleus_intensity * 0.2
        
        # DNA1 and DNA2 with slight variations
        dna1_field[nucleus_mask] += nucleus_intensity * np.random.uniform(0.8, 1.2)
        dna1_field[cell_mask & ~nucleus_mask] += cytoplasm_intensity * np.random.uniform(0.5, 1.0)
        
        dna2_field[nucleus_mask] += nucleus_intensity * np.random.uniform(0.9, 1.1)
        dna2_field[cell_mask & ~nucleus_mask] += cytoplasm_intensity * np.random.uniform(0.6, 0.9)
        
        # Add protein expression with cell type variation
        if i % 3 == 0:  # Immune cells
            cd45_field[cell_mask] += np.random.uniform(30, 80)
        elif i % 5 == 0:  # Endothelial cells
            cd31_field[cell_mask] += np.random.uniform(40, 90)
    
    # Add background noise
    dna1_field += np.random.exponential(noise_level, dna1_field.shape)
    dna2_field += np.random.exponential(noise_level, dna2_field.shape)
    cd45_field += np.random.exponential(noise_level, cd45_field.shape)
    cd31_field += np.random.exponential(noise_level, cd31_field.shape)
    
    # Extract values at coordinate positions
    dna1_intensities = dna1_field.ravel()
    dna2_intensities = dna2_field.ravel()
    
    ion_counts = {
        'CD45': cd45_field.ravel(),
        'CD31': cd31_field.ravel()
    }
    
    return {
        'coords': coords,
        'dna1_intensities': dna1_intensities,
        'dna2_intensities': dna2_intensities,
        'ion_counts': ion_counts,
        'true_cell_centers': cell_centers,
        'image_shape': (image_size, image_size),
        'n_true_cells': n_cells
    }


def test_nucleus_detection():
    """Test nucleus detection functionality."""
    print("\n" + "="*60)
    print("TESTING NUCLEUS DETECTION")
    print("="*60)
    
    # Create synthetic data
    synthetic_data = create_synthetic_tissue_data(n_cells=30, image_size=150)
    
    # Prepare DNA composite
    dna_image, bounds = prepare_dna_for_nucleus_detection(
        synthetic_data['coords'],
        synthetic_data['dna1_intensities'],
        synthetic_data['dna2_intensities'],
        resolution_um=1.0
    )
    
    print(f"DNA composite shape: {dna_image.shape}")
    print(f"Spatial bounds: {bounds}")
    print(f"DNA signal range: {dna_image.min():.3f} - {dna_image.max():.3f}")
    
    # Detect nucleus seeds
    nucleus_seeds, detection_stats = detect_nucleus_seeds(
        dna_image,
        min_nucleus_size_um2=9.0,  # ~3x3 pixels
        max_nucleus_size_um2=100.0,
        resolution_um=1.0
    )
    
    print(f"\nNucleus Detection Results:")
    print(f"  Seeds detected: {detection_stats['n_nucleus_seeds']}")
    print(f"  True cells: {synthetic_data['n_true_cells']}")
    print(f"  Detection rate: {detection_stats['n_nucleus_seeds'] / synthetic_data['n_true_cells']:.2f}")
    print(f"  Tissue coverage: {detection_stats['binary_mask_coverage']:.3f}")
    print(f"  Nucleus density: {detection_stats['nucleus_density_per_um2']:.6f} per μm²")
    
    # Test quality assessment
    if detection_stats['n_nucleus_seeds'] > 0:
        print("✓ Nucleus detection successful")
        return True, dna_image, nucleus_seeds, detection_stats
    else:
        print("✗ No nuclei detected")
        return False, dna_image, nucleus_seeds, detection_stats


def test_watershed_segmentation():
    """Test watershed cell segmentation."""
    print("\n" + "="*60)
    print("TESTING WATERSHED SEGMENTATION")
    print("="*60)
    
    # Get nucleus detection results
    success, dna_image, nucleus_seeds, detection_stats = test_nucleus_detection()
    
    if not success:
        print("✗ Cannot test watershed without nucleus seeds")
        return False, None, None
    
    # Perform watershed segmentation
    cell_labels, segmentation_stats = perform_watershed_segmentation(
        dna_image,
        nucleus_seeds,
        max_cell_radius_um=12.0,
        resolution_um=1.0
    )
    
    print(f"\nWatershed Segmentation Results:")
    print(f"  Cells segmented: {segmentation_stats['n_cells_detected']}")
    print(f"  Mean cell area: {segmentation_stats['mean_cell_area_um2']:.1f} μm²")
    print(f"  Cell area std: {segmentation_stats['std_cell_area_um2']:.1f} μm²")
    print(f"  Mean eccentricity: {segmentation_stats['mean_eccentricity']:.3f}")
    print(f"  Segmentation coverage: {segmentation_stats['segmentation_coverage']:.3f}")
    
    if segmentation_stats['n_cells_detected'] > 0:
        print("✓ Watershed segmentation successful")
        return True, cell_labels, segmentation_stats
    else:
        print("✗ No cells segmented")
        return False, cell_labels, segmentation_stats


def test_quality_assessment():
    """Test quality assessment functionality."""
    print("\n" + "="*60)
    print("TESTING QUALITY ASSESSMENT")
    print("="*60)
    
    # Get segmentation results
    nucleus_success, dna_image, nucleus_seeds, detection_stats = test_nucleus_detection()
    watershed_success, cell_labels, segmentation_stats = test_watershed_segmentation()
    
    if not (nucleus_success and watershed_success):
        print("✗ Cannot test quality assessment without successful segmentation")
        return False
    
    # Assess quality
    quality_assessment = assess_watershed_quality(
        cell_labels,
        dna_image,
        nucleus_seeds,
        detection_stats,
        segmentation_stats
    )
    
    print(f"\nQuality Assessment Results:")
    print(f"  Overall quality score: {quality_assessment['overall_quality_score']:.3f}")
    print(f"  Seed-to-cell ratio: {quality_assessment['metrics']['seed_to_cell_ratio']:.3f}")
    print(f"  Mean eccentricity: {quality_assessment['metrics']['mean_eccentricity']:.3f}")
    print(f"  Mean solidity: {quality_assessment['metrics']['mean_solidity']:.3f}")
    print(f"  Area CV: {quality_assessment['metrics']['area_cv']:.3f}")
    
    print(f"\nQuality Flags:")
    for flag, value in quality_assessment['quality_flags'].items():
        status = "✓" if value else "✗"
        print(f"  {status} {flag}")
    
    if quality_assessment['overall_quality_score'] > 0.5:
        print("✓ Quality assessment passed")
        return True
    else:
        print("✗ Quality assessment failed")
        return False


def test_full_pipeline():
    """Test complete watershed pipeline."""
    print("\n" + "="*60)
    print("TESTING COMPLETE WATERSHED PIPELINE")
    print("="*60)
    
    # Create synthetic data
    synthetic_data = create_synthetic_tissue_data(n_cells=25, image_size=120)
    
    # Load configuration
    try:
        config = Config('config.json')
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"⚠ Could not load config: {e}, using defaults")
        config = None
    
    # Run complete pipeline
    try:
        results = watershed_pipeline(
            coords=synthetic_data['coords'],
            ion_counts=synthetic_data['ion_counts'],
            dna1_intensities=synthetic_data['dna1_intensities'],
            dna2_intensities=synthetic_data['dna2_intensities'],
            target_scale_um=20.0,
            resolution_um=1.0,
            config=config
        )
        
        print(f"\nPipeline Results:")
        print(f"  Method: {results['method']}")
        print(f"  Cells detected: {results['n_segments_used']}")
        print(f"  Proteins analyzed: {len(results['cell_counts'])}")
        print(f"  Quality score: {results['quality_assessment']['overall_quality_score']:.3f}")
        
        # Check interface compatibility
        required_keys = [
            'superpixel_counts', 'superpixel_coords', 'superpixel_labels',
            'cell_counts', 'cell_coords', 'cell_labels', 'transformed_arrays'
        ]
        
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            print(f"✗ Missing required keys: {missing_keys}")
            return False
        
        print("✓ Interface compatibility check passed")
        print("✓ Complete pipeline test successful")
        return True, results
        
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_comparison_with_slic():
    """Compare watershed with SLIC segmentation."""
    print("\n" + "="*60)
    print("TESTING WATERSHED vs SLIC COMPARISON")
    print("="*60)
    
    # Create synthetic data
    synthetic_data = create_synthetic_tissue_data(n_cells=20, image_size=100)
    
    try:
        config = Config('config.json')
    except:
        config = None
    
    # Test watershed
    print("Running watershed segmentation...")
    watershed_results = watershed_pipeline(
        coords=synthetic_data['coords'],
        ion_counts=synthetic_data['ion_counts'],
        dna1_intensities=synthetic_data['dna1_intensities'],
        dna2_intensities=synthetic_data['dna2_intensities'],
        target_scale_um=20.0,
        config=config
    )
    
    # Test SLIC
    print("Running SLIC segmentation...")
    slic_results = slic_pipeline(
        coords=synthetic_data['coords'],
        ion_counts=synthetic_data['ion_counts'],
        dna1_intensities=synthetic_data['dna1_intensities'],
        dna2_intensities=synthetic_data['dna2_intensities'],
        target_scale_um=20.0,
        config=config
    )
    
    print(f"\nComparison Results:")
    print(f"  Watershed segments: {watershed_results['n_segments_used']}")
    print(f"  SLIC segments: {slic_results['n_segments_used']}")
    print(f"  True cells: {synthetic_data['n_true_cells']}")
    
    # Compare detection rates
    watershed_rate = watershed_results['n_segments_used'] / synthetic_data['n_true_cells']
    slic_rate = slic_results['n_segments_used'] / synthetic_data['n_true_cells']
    
    print(f"  Watershed detection rate: {watershed_rate:.2f}")
    print(f"  SLIC detection rate: {slic_rate:.2f}")
    
    print("✓ Comparison test completed")
    return True


def create_visualization(results, synthetic_data, output_dir="test_results"):
    """Create visualization of segmentation results."""
    print(f"\nCreating visualizations in {output_dir}/...")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Get image shape
        image_shape = synthetic_data['image_shape']
        
        # Reshape data for visualization
        dna_composite = results['composite_dna']
        cell_labels = results['cell_labels']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # DNA composite
        im1 = axes[0, 0].imshow(dna_composite, cmap='viridis')
        axes[0, 0].set_title('DNA Composite')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Cell segmentation
        im2 = axes[0, 1].imshow(cell_labels, cmap='tab20')
        axes[0, 1].set_title(f'Cell Segmentation ({results["n_segments_used"]} cells)')
        axes[0, 1].axis('off')
        
        # Quality metrics
        quality = results['quality_assessment']
        axes[1, 0].bar(range(len(quality['quality_flags'])), 
                      [1 if v else 0 for v in quality['quality_flags'].values()])
        axes[1, 0].set_xticks(range(len(quality['quality_flags'])))
        axes[1, 0].set_xticklabels(quality['quality_flags'].keys(), rotation=45)
        axes[1, 0].set_title('Quality Flags')
        axes[1, 0].set_ylabel('Pass (1) / Fail (0)')
        
        # Cell size distribution
        cell_props = results['cell_props']
        if cell_props:
            areas = [props['area_um2'] for props in cell_props.values()]
            axes[1, 1].hist(areas, bins=10, alpha=0.7)
            axes[1, 1].set_title('Cell Area Distribution')
            axes[1, 1].set_xlabel('Area (μm²)')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/watershed_test_results.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to {output_dir}/watershed_test_results.png")
        return True
        
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")
        return False


def main():
    """Run all watershed segmentation tests."""
    print("WATERSHED DNA SEGMENTATION TEST SUITE")
    print("=" * 80)
    
    test_results = {}
    
    # Run tests
    test_results['nucleus_detection'] = test_nucleus_detection()[0]
    test_results['watershed_segmentation'] = test_watershed_segmentation()[0]
    test_results['quality_assessment'] = test_quality_assessment()
    
    pipeline_success, pipeline_results = test_full_pipeline()
    test_results['full_pipeline'] = pipeline_success
    
    test_results['slic_comparison'] = test_comparison_with_slic()
    
    # Create visualization if pipeline succeeded
    if pipeline_success and pipeline_results:
        synthetic_data = create_synthetic_tissue_data(n_cells=25, image_size=120)
        test_results['visualization'] = create_visualization(pipeline_results, synthetic_data)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:.<50} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Watershed segmentation implementation is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Review implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
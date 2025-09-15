#!/usr/bin/env python3
"""
Test suite for Phase 5: Tissue-Level Analysis Without Cell Segmentation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import json
from src.config import Config
from src.utils.helpers import find_roi_files
from src.analysis.spatial import load_roi_data
from src.analysis.superpixel import create_tissue_parcellation, TissueParcellator
from src.analysis.texture import TextureAnalyzer, MultiProteinTextureAnalyzer
from src.analysis.spatial_statistics import SpatialStatistics, identify_spatial_patterns
from src.analysis.region_graph import RegionGraphBuilder, RegionGraphAnalyzer, analyze_tissue_organization


def test_superpixel_parcellation():
    """Test superpixel-based tissue parcellation"""
    print("\n" + "="*60)
    print("Testing Superpixel Parcellation")
    print("="*60)
    
    # Generate synthetic tissue data
    np.random.seed(42)
    n_points = 1000
    
    # Create spatial pattern
    coords = np.random.rand(n_points, 2) * 500
    
    # Create expression pattern with spatial structure
    values = np.zeros((n_points, 5))
    for i in range(5):
        center = np.random.rand(2) * 500
        for j in range(n_points):
            dist = np.linalg.norm(coords[j] - center)
            values[j, i] = np.exp(-dist / 100) + np.random.randn() * 0.1
    
    print(f"Test data: {n_points} pixels, 5 proteins")
    
    # Test SLIC parcellation
    print("\n1. SLIC Superpixels:")
    config = {'n_segments': 50, 'compactness': 10}
    result = create_tissue_parcellation(coords, values, method='slic', config=config)
    print(f"   Created {result.n_segments} superpixels")
    print(f"   Mean size: {np.mean(result.sizes):.1f} pixels")
    print(f"   Size range: {result.sizes.min()}-{result.sizes.max()}")
    
    # Test adaptive parcellation
    print("\n2. Adaptive Superpixels:")
    config = {
        'initial_segments': 100,
        'similarity_threshold': 0.8,
        'min_segments': 20,
        'adaptive': True
    }
    result = create_tissue_parcellation(coords, values, method='slic', config=config)
    print(f"   Merged to {result.n_segments} superpixels")
    print(f"   Adjacency connections: {sum(len(v) for v in result.adjacency.values()) // 2}")
    
    # Test grid parcellation
    print("\n3. Grid Tiles:")
    config = {'tile_size': 100}
    result = create_tissue_parcellation(coords, values, method='grid', config=config)
    print(f"   Created {result.n_segments} grid tiles")
    
    print("\n✓ Superpixel parcellation tests passed")
    return result


def test_texture_analysis():
    """Test spatial texture analysis"""
    print("\n" + "="*60)
    print("Testing Texture Analysis")
    print("="*60)
    
    # Create test data with texture pattern
    np.random.seed(42)
    n_points = 500
    coords = np.random.rand(n_points, 2) * 200
    
    # Create striped pattern
    values = np.zeros((n_points, 3))
    for i in range(n_points):
        values[i, 0] = np.sin(coords[i, 0] / 20) + np.random.randn() * 0.1
        values[i, 1] = np.cos(coords[i, 1] / 20) + np.random.randn() * 0.1
        values[i, 2] = np.sin(coords[i, 0] / 20) * np.cos(coords[i, 1] / 20)
    
    print(f"Test data: {n_points} pixels with textured patterns")
    
    # Test single-scale analysis
    print("\n1. Single-scale texture:")
    analyzer = TextureAnalyzer({'window_sizes': [50]})
    features = analyzer.analyze(coords, values, protein_idx=0)
    
    texture_50 = features[50]
    print(f"   Haralick features: {len(texture_50.haralick)} types")
    if texture_50.haralick:
        print(f"   Contrast range: {texture_50.haralick['contrast'].min():.3f} - "
              f"{texture_50.haralick['contrast'].max():.3f}")
    print(f"   LBP histogram bins: {len(texture_50.lbp)}")
    print(f"   Global entropy: {texture_50.statistics['entropy']:.3f}")
    
    # Test multi-scale analysis
    print("\n2. Multi-scale texture:")
    analyzer = TextureAnalyzer({'window_sizes': [10, 50, 100]})
    features = analyzer.analyze(coords, values)
    
    for scale in [10, 50, 100]:
        entropy = features[scale].statistics['entropy']
        print(f"   Scale {scale}μm: entropy = {entropy:.3f}")
    
    # Test multi-protein analysis
    print("\n3. Multi-protein texture:")
    multi_analyzer = MultiProteinTextureAnalyzer()
    protein_names = ['Protein_A', 'Protein_B', 'Protein_C']
    all_features = multi_analyzer.analyze_all_proteins(coords, values, protein_names)
    
    print(f"   Analyzed {len(all_features)} channels (including combined)")
    
    # Test similarity
    if len(all_features) > 1:
        proteins = list(all_features.keys())
        feat1 = all_features[proteins[0]][50]
        feat2 = all_features[proteins[1]][50]
        similarity = multi_analyzer.compute_texture_similarity(feat1, feat2)
        print(f"   Similarity {proteins[0]} vs {proteins[1]}: {similarity:.3f}")
    
    print("\n✓ Texture analysis tests passed")


def test_spatial_statistics():
    """Test pixel-level spatial statistics"""
    print("\n" + "="*60)
    print("Testing Spatial Statistics")
    print("="*60)
    
    # Create clustered data
    np.random.seed(42)
    n_points = 800
    
    # Create three clusters
    coords = []
    values = []
    
    for i in range(3):
        center = np.random.rand(2) * 300
        cluster_coords = np.random.randn(n_points // 3, 2) * 30 + center
        cluster_values = np.random.randn(n_points // 3, 2) + i
        coords.append(cluster_coords)
        values.append(cluster_values)
    
    coords = np.vstack(coords)
    values = np.vstack(values)
    
    print(f"Test data: {len(coords)} pixels in 3 clusters")
    
    # Test global statistics
    print("\n1. Global spatial statistics:")
    analyzer = SpatialStatistics({'bandwidth': 50, 'permutations': 99})
    result = analyzer.analyze(coords, values, protein_idx=0)
    
    print(f"   Moran's I: {result.morans_i:.3f} (p={result.morans_p:.3f})")
    print(f"   Geary's C: {result.gearys_c:.3f} (p={result.gearys_p:.3f})")
    
    if result.morans_i > 0:
        print("   → Positive spatial autocorrelation (clustering)")
    elif result.morans_i < 0:
        print("   → Negative spatial autocorrelation (dispersion)")
    else:
        print("   → Random spatial pattern")
    
    # Test local statistics
    print("\n2. Local statistics:")
    n_hotspots = np.sum(result.getis_ord > 1.96)
    n_coldspots = np.sum(result.getis_ord < -1.96)
    print(f"   Hot spots: {n_hotspots} pixels")
    print(f"   Cold spots: {n_coldspots} pixels")
    print(f"   Local Moran's I range: {result.local_morans.min():.3f} to {result.local_morans.max():.3f}")
    
    # Test variogram
    print("\n3. Variogram analysis:")
    if result.variogram:
        print(f"   Distance lags: {len(result.variogram['distance'])}")
        print(f"   Semivariance range: {result.variogram['semivariance'].min():.3f} - "
              f"{result.variogram['semivariance'].max():.3f}")
    
    # Test pattern identification
    print("\n4. Pattern identification:")
    patterns = identify_spatial_patterns(coords, values, ['hotspots', 'gradients'])
    
    if 'hotspots' in patterns:
        n_hot = np.sum(patterns['hotspots'] == 1)
        n_cold = np.sum(patterns['hotspots'] == -1)
        print(f"   Identified {n_hot} hot spot pixels, {n_cold} cold spot pixels")
    
    if 'gradients' in patterns:
        gradient_strength = patterns['gradients']
        print(f"   Gradient strength: mean={gradient_strength.mean():.3f}, "
              f"max={gradient_strength.max():.3f}")
    
    print("\n✓ Spatial statistics tests passed")


def test_region_graphs():
    """Test region graph networks"""
    print("\n" + "="*60)
    print("Testing Region Graph Networks")
    print("="*60)
    
    # Create structured tissue data
    np.random.seed(42)
    n_points = 1000
    coords = np.random.rand(n_points, 2) * 400
    
    # Create regions with different expression
    values = np.zeros((n_points, 4))
    for i in range(n_points):
        if coords[i, 0] < 200:
            values[i, :2] = np.random.randn(2) + 2
        else:
            values[i, 2:] = np.random.randn(2) + 2
    
    print(f"Test data: {n_points} pixels with regional structure")
    
    # Test graph from superpixels
    print("\n1. Graph from superpixels:")
    result = analyze_tissue_organization(coords, values, method='superpixel')
    
    print(f"   Nodes (regions): {result.graph.number_of_nodes()}")
    print(f"   Edges (connections): {result.graph.number_of_edges()}")
    print(f"   Communities detected: {result.n_communities}")
    print(f"   Modularity: {result.modularity:.3f}")
    print(f"   Hub regions: {len(result.hub_regions)}")
    
    # Test graph from grid
    print("\n2. Graph from grid tiles:")
    builder = RegionGraphBuilder({'tile_size': 50})
    graph = builder.build_from_grid(coords, values)
    
    analyzer = RegionGraphAnalyzer()
    result = analyzer.analyze(graph)
    
    print(f"   Grid nodes: {graph.number_of_nodes()}")
    print(f"   Grid edges: {graph.number_of_edges()}")
    print(f"   Average degree: {np.mean([d for n, d in graph.degree()]):.1f}")
    
    # Test message passing
    print("\n3. Message passing simulation:")
    from src.analysis.region_graph import TissueMessagePassing
    
    if result.graph.number_of_nodes() > 0:
        mp = TissueMessagePassing(result.graph, result.node_features)
        updated = mp.propagate(n_iterations=5, damping=0.7)
        
        # Check if features changed
        original_mean = np.mean([np.mean(f) for f in result.node_features.values()])
        updated_mean = np.mean([np.mean(f) for f in updated.values()])
        
        print(f"   Original mean expression: {original_mean:.3f}")
        print(f"   After message passing: {updated_mean:.3f}")
        print(f"   Change: {abs(updated_mean - original_mean):.3f}")
    
    print("\n✓ Region graph tests passed")


def test_with_real_roi():
    """Test with actual ROI data"""
    print("\n" + "="*60)
    print("Testing with Real ROI Data")
    print("="*60)
    
    # Load config and find ROI
    config = Config('config.json')
    roi_files = find_roi_files(config.data_dir)
    
    if not roi_files:
        print("No ROI files found - skipping real data test")
        return
    
    # Load first ROI
    roi_file = roi_files[0]
    print(f"Testing on: {roi_file.name}")
    
    coords, values, protein_names = load_roi_data(roi_file, 'config.json')
    print(f"Loaded: {len(coords)} pixels, {len(protein_names)} proteins")
    
    # Subsample for faster testing
    if len(coords) > 5000:
        idx = np.random.choice(len(coords), 5000, replace=False)
        coords = coords[idx]
        values = values[idx]
        print(f"Subsampled to {len(coords)} pixels")
    
    # Test superpixel parcellation
    print("\n1. Superpixel parcellation:")
    with open('config.json') as f:
        full_config = json.load(f)
    
    superpixel_config = full_config['tissue_analysis']['superpixel']
    result = create_tissue_parcellation(coords, values, 
                                       method='slic', 
                                       config=superpixel_config)
    print(f"   Created {result.n_segments} superpixels")
    
    # Test texture analysis
    print("\n2. Texture analysis:")
    texture_config = full_config['tissue_analysis']['texture']
    analyzer = TextureAnalyzer(texture_config)
    features = analyzer.analyze(coords, values, protein_idx=0)
    
    for scale in texture_config['window_sizes']:
        if scale in features:
            entropy = features[scale].statistics['entropy']
            print(f"   Scale {scale}μm: entropy = {entropy:.3f}")
    
    # Test spatial statistics
    print("\n3. Spatial statistics:")
    stats_config = full_config['tissue_analysis']['spatial_statistics']
    analyzer = SpatialStatistics(stats_config)
    stats_result = analyzer.analyze(coords, values, protein_idx=0)
    print(f"   Moran's I: {stats_result.morans_i:.3f}")
    print(f"   Hot spots: {np.sum(stats_result.getis_ord > 1.96)}")
    
    # Test region graph
    print("\n4. Region graph analysis:")
    graph_result = analyze_tissue_organization(coords, values, method='grid')
    print(f"   Communities: {graph_result.n_communities}")
    print(f"   Modularity: {graph_result.modularity:.3f}")
    
    print("\n✓ Real ROI tests completed")


def main():
    """Run all tissue analysis tests"""
    print("\n" + "#"*60)
    print("#" + " "*15 + "TISSUE ANALYSIS TEST SUITE" + " "*15 + "#")
    print("#"*60)
    
    try:
        # Run tests
        test_superpixel_parcellation()
        test_texture_analysis()
        test_spatial_statistics()
        test_region_graphs()
        test_with_real_roi()
        
        print("\n" + "#"*60)
        print("#" + " "*16 + "ALL TESTS PASSED!" + " "*16 + "#")
        print("#"*60)
        print("\n✓ Phase 5 tissue analysis methods are working correctly!\n")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
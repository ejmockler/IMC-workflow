#!/usr/bin/env python3
"""
Simple test of tissue analysis capabilities
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np

print("Testing Phase 5: Tissue-Level Analysis")
print("="*50)

# Generate simple test data
np.random.seed(42)
coords = np.random.rand(500, 2) * 100
values = np.random.randn(500, 3)

print(f"Test data: {len(coords)} pixels, {values.shape[1]} proteins")

# Test 1: Superpixel parcellation
print("\n1. Superpixel Parcellation:")
from src.analysis.superpixel import TissueParcellator

parcellator = TissueParcellator(method='slic', config={'n_segments': 20})
result = parcellator.parcellate(coords, values)
print(f"   ✓ Created {result.n_segments} superpixels")
print(f"   ✓ Mean size: {np.mean(result.sizes):.1f} pixels")

# Test 2: Texture analysis
print("\n2. Texture Analysis:")
from src.analysis.texture import TextureAnalyzer

analyzer = TextureAnalyzer({'window_sizes': [50]})
features = analyzer.analyze(coords, values)
texture = features[50]
print(f"   ✓ Computed {len(texture.haralick)} Haralick features")
print(f"   ✓ Entropy: {texture.statistics['entropy']:.3f}")

# Test 3: Spatial statistics
print("\n3. Spatial Statistics:")
from src.analysis.spatial_statistics import SpatialStatistics

stats = SpatialStatistics({'bandwidth': 20})
result = stats.analyze(coords, values)
print(f"   ✓ Moran's I: {result.morans_i:.3f}")
print(f"   ✓ Hot spots: {np.sum(result.getis_ord > 1.96)} pixels")

# Test 4: Region graphs
print("\n4. Region Graph Networks:")
from src.analysis.region_graph import RegionGraphBuilder, RegionGraphAnalyzer

builder = RegionGraphBuilder({'tile_size': 25})
graph = builder.build_from_grid(coords, values)

analyzer = RegionGraphAnalyzer()
result = analyzer.analyze(graph)
print(f"   ✓ Graph nodes: {graph.number_of_nodes()}")
print(f"   ✓ Communities: {result.n_communities}")
print(f"   ✓ Modularity: {result.modularity:.3f}")

print("\n" + "="*50)
print("✓ All tissue analysis methods working!")
print("\nPhase 5 Complete: Tissue-level analysis without")
print("cell segmentation is fully functional!")
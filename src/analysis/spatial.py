#!/usr/bin/env python3
"""
Spatial Blob Analysis - Tissue Region Organization
Identifies coherent expression regions and their spatial relationships

Updated to use robust spatial analysis engine with superpixel parcellation
and established spatial statistics libraries.
"""

import numpy as np
import json
from pathlib import Path
from scipy.spatial import cKDTree
from scipy import ndimage
from collections import defaultdict, Counter
from math import comb
import warnings

# Import centralized data loader
from src.utils.data_loader import load_roi_data

# Import new robust spatial analysis
try:
    from .spatial_engine_final import analyze_spatial_organization_robust
    ROBUST_SPATIAL_AVAILABLE = True
except ImportError:
    ROBUST_SPATIAL_AVAILABLE = False
    warnings.warn("Robust spatial analysis not available - using legacy methods")

def calculate_optimal_k(n_proteins, config_path='config.json'):
    """
    Calculate optimal K for KMeans based on protein panel combinatorics.
    
    Uses theoretical maximum of expression patterns:
    - Dual combinations: C(n,2) 
    - Single dominant: n
    - Background/low expression: ~2-3 states
    
    For 7 proteins: C(7,2) + 7 + 2 = 21 + 7 + 2 = 30
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    analysis_config = config.get('analysis', {})
    k_strategy = analysis_config.get('k_strategy', 'dual_combinations')
    
    if k_strategy == 'dual_combinations':
        # Primary approach: dual protein combinations + singles + background
        dual_combos = comb(n_proteins, 2) if n_proteins >= 2 else 0
        single_dominant = n_proteins
        background_states = 2  # low/negative expression states
        optimal_k = dual_combos + single_dominant + background_states
    elif k_strategy == 'conservative':
        # Conservative approach: fewer clusters for stability
        optimal_k = min(20, max(10, n_proteins * 2))
    else:
        # Fallback to reasonable default
        optimal_k = min(25, max(15, n_proteins * 3))
    
    # Ensure reasonable bounds
    optimal_k = max(5, min(50, optimal_k))
    
    print(f"      Calculated optimal K: {optimal_k} (strategy: {k_strategy}, proteins: {n_proteins})")
    return optimal_k


def identify_expression_blobs(coords, values, protein_names, min_blob_size=5, config_path='config.json', validate=True):
    """Identify coherent expression regions (blobs) with optional validation"""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 1. Calculate optimal K based on protein panel
    n_blob_types = calculate_optimal_k(len(protein_names), config_path)
    
    # 2. Select clustering algorithm
    clustering_config = config.get('clustering', {})
    algorithm = clustering_config.get('algorithm', 'kmeans')
    
    # Check if we should auto-select
    if clustering_config.get('auto_select', False):
        from src.analysis.clustering import auto_select_clusterer
        algorithm = auto_select_clusterer(values)
        print(f"      Auto-selected clustering: {algorithm}")
    
    # 3. Cluster pixels by expression signature
    if algorithm == 'kmeans':
        # Legacy direct implementation for backward compatibility
        kmeans = KMeans(n_clusters=n_blob_types, random_state=42)
        blob_labels = kmeans.fit_predict(values)
    else:
        # Use advanced clustering
        from src.analysis.clustering import ClustererFactory
        
        clusterer = ClustererFactory.create(
            algorithm, 
            clustering_config.get('algorithms', {})
        )
        
        # Some algorithms determine their own n_clusters
        if algorithm in ['phenograph', 'leiden']:
            result = clusterer.fit_predict(values)
        else:
            result = clusterer.fit_predict(values, n_clusters=n_blob_types)
        
        blob_labels = result.labels
        actual_n_clusters = result.n_clusters
        
        if actual_n_clusters != n_blob_types:
            print(f"      {algorithm} found {actual_n_clusters} clusters (requested: {n_blob_types})")
    
    # Optional benchmarking
    if clustering_config.get('benchmark_algorithms', False):
        from src.analysis.clustering import benchmark_clusterers
        
        print("\n      Running clustering benchmark...")
        benchmark_algos = ['kmeans', 'minibatch', 'flowsom']
        results = benchmark_clusterers(values, benchmark_algos, n_blob_types)
        
        print("      Benchmark Results:")
        for algo, metrics in results.items():
            if 'error' in metrics:
                print(f"        {algo}: {metrics['error']}")
            else:
                print(f"        {algo}: {metrics['time_seconds']:.3f}s, "
                      f"silhouette: {metrics['silhouette_score']:.3f}")
        print()
    
    # 4. Optional validation of clustering quality
    validation_results = None
    if validate:
        if config.get('analysis', {}).get('validation', {}).get('enabled', False):
            from src.analysis.validation import ValidationSuite
            
            suite = ValidationSuite()
            validation_results = suite.validate_all(
                data=values,
                labels=blob_labels,
                coords=coords
            )
            
            # Print summary if in detailed mode
            if config.get('analysis', {}).get('validation', {}).get('report_format') == 'detailed':
                print("      " + suite.format_report(validation_results).replace('\n', '\n      '))
    
    # 5. Get blob type characteristics
    blob_signatures = {}
    
    for blob_id in range(n_blob_types):
        mask = blob_labels == blob_id
        if np.sum(mask) >= min_blob_size:  # Include smaller blobs to show more types
            blob_signatures[blob_id] = {
                'mean_expression': values[mask].mean(axis=0),
                'dominant_proteins': [],
                'size': np.sum(mask),
                'coords': coords[mask]
            }
            
            # Find dominant proteins (top 2)
            mean_expr = values[mask].mean(axis=0)
            top_indices = np.argsort(mean_expr)[-2:][::-1]
            blob_signatures[blob_id]['dominant_proteins'] = [protein_names[i] for i in top_indices]
    
    # Merge blobs with identical protein signatures
    merged_blobs = {}
    blob_type_mapping = {}
    
    for blob_id, sig in blob_signatures.items():
        # Canonicalize by sorting the top-2 marker names so order doesn't matter
        canonical_pair = '+'.join(sorted(sig['dominant_proteins'][:2]))
        
        if canonical_pair not in merged_blobs:
            # First time seeing this protein combination
            merged_blobs[canonical_pair] = {
                'mean_expression': sig['mean_expression'].copy(),
                'dominant_proteins': sorted(sig['dominant_proteins'][:2]),
                'size': sig['size'],
                'coords': sig['coords']
            }
            blob_type_mapping[blob_id] = canonical_pair
        else:
            # Merge with existing blob of same type
            existing = merged_blobs[canonical_pair]
            total_size = existing['size'] + sig['size']
            
            # Weighted average of expression
            weight1 = existing['size'] / total_size
            weight2 = sig['size'] / total_size
            merged_blobs[canonical_pair]['mean_expression'] = (
                existing['mean_expression'] * weight1 + 
                sig['mean_expression'] * weight2
            )
            merged_blobs[canonical_pair]['size'] = total_size
            merged_blobs[canonical_pair]['coords'] = np.vstack([existing['coords'], sig['coords']])
            blob_type_mapping[blob_id] = canonical_pair
    
    print(f"      Found {len(blob_signatures)} spatial clusters → {len(merged_blobs)} distinct blob types")
    for blob_name, sig in merged_blobs.items():
        pct = (sig['size'] / len(coords)) * 100
        print(f"         {blob_name}: {pct:.1f}% of tissue")
    
    # Return validation results if computed
    if validation_results is not None:
        return blob_labels, merged_blobs, blob_type_mapping, validation_results
    
    return blob_labels, merged_blobs, blob_type_mapping

def analyze_blob_spatial_relationships(coords, blob_labels, blob_signatures, blob_type_mapping):
    """Analyze which blob types are spatial neighbors - optimized"""
    
    # Build spatial tree
    tree = cKDTree(coords)
    
    # For each blob type, find its spatial neighbors
    blob_type_neighbors = defaultdict(lambda: defaultdict(int))
    blob_type_contacts = defaultdict(int)
    
    # Sample pixels for performance (every 10th pixel)
    sample_indices = np.arange(0, len(coords), 10)
    
    for sample_idx in sample_indices:
        current_cluster_id = blob_labels[sample_idx]
        current_blob_type = blob_type_mapping.get(current_cluster_id)
        
        if current_blob_type:
            # Check neighbors at 15μm
            neighbors = tree.query_ball_point(coords[sample_idx], 15.0)
            
            for neighbor_idx in neighbors[:20]:  # Limit neighbors checked
                if neighbor_idx != sample_idx:
                    neighbor_cluster_id = blob_labels[neighbor_idx]
                    neighbor_blob_type = blob_type_mapping.get(neighbor_cluster_id)
                    
                    if neighbor_blob_type and neighbor_blob_type != current_blob_type:
                        blob_type_neighbors[current_blob_type][neighbor_blob_type] += 1
                        blob_type_contacts[current_blob_type] += 1
    
    # Normalize to get contact frequencies
    blob_contact_freq = {}
    for blob_type in blob_signatures.keys():
        total_contacts = blob_type_contacts[blob_type]
        if total_contacts > 0:
            blob_contact_freq[blob_type] = {
                neighbor: count / total_contacts 
                for neighbor, count in blob_type_neighbors[blob_type].items()
            }
    
    return blob_contact_freq


def identify_tissue_neighborhoods(coords, blob_labels, blob_signatures, blob_type_mapping, config_path='config.json', scale=None):
    """
    Identify tissue neighborhoods using simple distance-based clustering of blob centers.
    
    This approach groups nearby blobs into neighborhoods based on spatial proximity
    and composition similarity, providing interpretable tissue organization analysis.
    
    Args:
        scale: Specific scale name from config, or None to use default_scale
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    nbhd_config = config.get('neighborhood_analysis', {})
    
    # Handle multi-scale configuration
    scales = nbhd_config.get('scales', {})
    if scales:
        # New multi-scale config
        if scale is None:
            scale = nbhd_config.get('default_scale', 'microenvironment')
        if scale in scales:
            clustering_radius = scales[scale]['radius']
            scale_desc = scales[scale].get('description', '')
        else:
            # Fallback to first scale
            first_scale = list(scales.keys())[0]
            clustering_radius = scales[first_scale]['radius']
            scale_desc = scales[first_scale].get('description', '')
    else:
        # Legacy single-scale config
        clustering_radius = nbhd_config.get('clustering_radius', 40)
        scale_desc = ''
    
    min_blob_size = nbhd_config.get('min_blob_size', 100)  # pixels
    min_neighborhood_size = nbhd_config.get('min_neighborhood_size', 500)  # pixels
    
    if scale_desc:
        print(f"      Analyzing tissue neighborhoods at {scale} scale (radius: {clustering_radius}μm) - {scale_desc}")
    else:
        print(f"      Analyzing tissue neighborhoods (clustering radius: {clustering_radius}μm)")
    
    # Extract blob centers and filter by size
    valid_blobs = []
    for blob_id, signature in blob_signatures.items():
        if signature['size'] >= min_blob_size:
            # Calculate center of mass for this blob
            blob_mask = np.array([blob_type_mapping.get(label) == blob_id for label in blob_labels])
            if np.any(blob_mask):
                blob_coords = coords[blob_mask]
                center = np.mean(blob_coords, axis=0)
                valid_blobs.append({
                    'id': blob_id,
                    'center': center,
                    'size': signature['size'],
                    'dominant_proteins': signature['dominant_proteins'][:2],
                    'coords': blob_coords
                })
    
    if len(valid_blobs) < 2:
        print(f"      Too few valid blobs ({len(valid_blobs)}) for neighborhood detection")
        return None
    
    print(f"      Found {len(valid_blobs)} valid blobs for neighborhood analysis")
    
    # Build spatial connectivity between blob centers
    blob_centers = np.array([blob['center'] for blob in valid_blobs])
    tree = cKDTree(blob_centers)
    
    # Find connected components using distance threshold
    connected_groups = []
    visited = set()
    
    for i, blob in enumerate(valid_blobs):
        if i in visited:
            continue
            
        # Start new neighborhood group
        group = [i]
        queue = [i]
        visited.add(i)
        
        while queue:
            current_idx = queue.pop(0)
            
            # Find nearby blobs
            neighbors = tree.query_ball_point(blob_centers[current_idx], clustering_radius)
            
            for neighbor_idx in neighbors:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    group.append(neighbor_idx)
                    queue.append(neighbor_idx)
        
        if len(group) >= 2:  # Neighborhood must have at least 2 blobs
            connected_groups.append(group)
    
    if not connected_groups:
        print(f"      No connected blob groups found")
        return None
    
    print(f"      Found {len(connected_groups)} potential neighborhoods")
    
    # Build neighborhoods from connected groups
    neighborhoods = {}
    pixel_assignments = np.full(len(coords), -1, dtype=int)
    final_neighborhoods = 0
    
    for group_id, blob_indices in enumerate(connected_groups):
        # Collect all pixels from blobs in this group
        neighborhood_pixels = []
        group_composition = Counter()
        
        for blob_idx in blob_indices:
            blob = valid_blobs[blob_idx]
            blob_id = blob['id']
            
            # Find all pixels belonging to this blob
            blob_mask = np.array([blob_type_mapping.get(label) == blob_id for label in blob_labels])
            pixel_indices = np.where(blob_mask)[0]
            
            neighborhood_pixels.extend(pixel_indices)
            group_composition['+'.join(blob['dominant_proteins'])] += blob['size']
        
        # Check minimum neighborhood size
        if len(neighborhood_pixels) < min_neighborhood_size:
            continue
        
        # Assign pixels to this neighborhood
        for pixel_idx in neighborhood_pixels:
            pixel_assignments[pixel_idx] = final_neighborhoods
        
        # Calculate neighborhood statistics
        total_size = sum(group_composition.values())
        dominant_pairs = sorted(group_composition.items(), key=lambda x: x[1], reverse=True)[:3]
        
        neighborhoods[final_neighborhoods] = {
            'size': len(neighborhood_pixels),
            'blob_count': len(blob_indices),
            'composition': group_composition,
            'dominant_pairs': [pair for pair, _ in dominant_pairs],
            'coverage': len(neighborhood_pixels) / len(coords)
        }
        
        final_neighborhoods += 1
    
    if final_neighborhoods == 0:
        print(f"      No neighborhoods met minimum size requirement")
        return None
    
    # Calculate overall statistics
    assigned_pixels = np.sum(pixel_assignments >= 0)
    coverage = assigned_pixels / len(coords)
    
    print(f"      Found {final_neighborhoods} neighborhoods covering {coverage:.1%} of tissue")
    
    return {
        'neighborhoods': neighborhoods,
        'pixel_assignments': pixel_assignments,
        'n_neighborhoods': final_neighborhoods,
        'coverage': coverage,
        'clustering_radius': clustering_radius
    }


def calculate_neighborhood_entropy_map(coords, blob_labels, blob_type_mapping, window_size=25, config_path='config.json'):
    """
    Simplified entropy calculation for boundary identification (optional feature).
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    nbhd_config = config.get('neighborhood_analysis', {})
    # Simple grid-based entropy calculation
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
    
    entropy_points = []
    x_centers = [min_x + (max_x - min_x) * 0.25, min_x + (max_x - min_x) * 0.75]
    y_centers = [min_y + (max_y - min_y) * 0.25, min_y + (max_y - min_y) * 0.75]
    
    for x_center in x_centers:
        for y_center in y_centers:
            # Count blob types in this region
            distances = np.sqrt((coords[:, 0] - x_center)**2 + (coords[:, 1] - y_center)**2)
            nearby_indices = distances < window_size
            
            if np.sum(nearby_indices) > 10:
                nearby_labels = blob_labels[nearby_indices]
                blob_types = [blob_type_mapping.get(label) for label in nearby_labels]
                blob_types = [bt for bt in blob_types if bt is not None]
                
                if blob_types:
                    type_counts = Counter(blob_types)
                    total = sum(type_counts.values())
                    probs = [count/total for count in type_counts.values()]
                    local_entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
                    entropy_points.append((x_center, y_center, local_entropy))
    
    if not entropy_points:
        return None
    
    # Return as tuple format: (x_coords, y_coords, entropy_values) 
    x_coords, y_coords, entropy_values = zip(*entropy_points)
    return (list(x_coords), list(y_coords), list(entropy_values))


def analyze_multiscale_neighborhoods(coords, blob_labels, blob_signatures, blob_type_mapping, config_path='config.json'):
    """
    Analyze neighborhoods at multiple spatial scales defined in config.
    
    Returns:
        Dict mapping scale names to neighborhood analysis results
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    nbhd_config = config.get('neighborhood_analysis', {})
    scales = nbhd_config.get('scales', {})
    
    if not scales:
        # Fallback to single-scale analysis
        print("      No multi-scale configuration found, using default")
        result = identify_tissue_neighborhoods(
            coords, blob_labels, blob_signatures, blob_type_mapping, config_path
        )
        return {'default': result}
    
    multiscale_results = {}
    
    for scale_name, scale_params in scales.items():
        print(f"\n    Scale: {scale_name}")
        neighborhoods = identify_tissue_neighborhoods(
            coords, blob_labels, blob_signatures, blob_type_mapping, 
            config_path, scale=scale_name
        )
        
        if neighborhoods:
            # Add scale-specific metadata
            neighborhoods['scale_name'] = scale_name
            neighborhoods['scale_radius'] = scale_params['radius']
            neighborhoods['scale_description'] = scale_params.get('description', '')
            
            multiscale_results[scale_name] = neighborhoods
    
    return multiscale_results


def analyze_roi_spatial_organization(coords, values, protein_names, config_path='config.json', 
                                   use_robust_analysis=True):
    """
    Enhanced spatial analysis that combines blob detection with robust spatial statistics.
    
    Args:
        coords: Pixel coordinates (N x 2)
        values: Protein expression values (N x P) 
        protein_names: List of protein names
        config_path: Configuration file path
        use_robust_analysis: Whether to use new robust spatial engine
        
    Returns:
        Combined analysis results with both blob analysis and spatial statistics
    """
    print("   🔬 Spatial Analysis: Starting comprehensive analysis")
    
    # 1. Traditional blob analysis for tissue organization
    print("   📍 Step 1: Identifying expression blobs...")
    result = identify_expression_blobs(coords, values, protein_names, validate=True)
    
    if len(result) == 4:
        blob_labels, blob_signatures, blob_type_mapping, validation_results = result
    else:
        blob_labels, blob_signatures, blob_type_mapping = result
        validation_results = None
    
    # 2. Blob spatial relationships
    print("   🤝 Step 2: Analyzing blob spatial relationships...")
    blob_contacts = analyze_blob_spatial_relationships(
        coords, blob_labels, blob_signatures, blob_type_mapping
    )
    
    # 3. Multi-scale neighborhood analysis
    print("   🏘️  Step 3: Multi-scale neighborhood analysis...")
    multiscale_neighborhoods = analyze_multiscale_neighborhoods(
        coords, blob_labels, blob_signatures, blob_type_mapping, config_path
    )
    
    # 4. Enhanced spatial statistics (new robust analysis)
    spatial_statistics = {}
    if use_robust_analysis and ROBUST_SPATIAL_AVAILABLE:
        print("   📊 Step 4: Enhanced spatial statistics (superpixel + squidpy)...")
        try:
            spatial_statistics = analyze_spatial_organization_robust(
                coords, values, protein_names, config_path
            )
        except Exception as e:
            print(f"   ⚠️  Robust spatial analysis failed: {e}")
            print("   🔄 Falling back to legacy analysis...")
            spatial_statistics = {}
    else:
        if not ROBUST_SPATIAL_AVAILABLE:
            print("   ⚠️  Robust spatial analysis not available (install squidpy/scanpy)")
        else:
            print("   📊 Step 4: Skipping enhanced spatial statistics (disabled)")
    
    # 5. Entropy map for tissue boundaries
    print("   🗺️  Step 5: Calculating entropy map...")
    entropy_map = calculate_neighborhood_entropy_map(
        coords, blob_labels, blob_type_mapping, config_path=config_path
    )
    
    # Combine all results
    comprehensive_results = {
        # Traditional blob analysis
        'blob_labels': blob_labels,
        'blob_signatures': blob_signatures,
        'blob_type_mapping': blob_type_mapping,
        'blob_contacts': blob_contacts,
        'validation_results': validation_results,
        
        # Multi-scale analysis
        'multiscale_neighborhoods': multiscale_neighborhoods,
        'entropy_map': entropy_map,
        
        # Enhanced spatial statistics
        'spatial_statistics': spatial_statistics,
        
        # Analysis metadata
        'analysis_method': 'comprehensive_spatial',
        'robust_analysis_used': use_robust_analysis and ROBUST_SPATIAL_AVAILABLE,
        'total_pixels': len(coords),
        'n_proteins': len(protein_names)
    }
    
    print(f"   ✅ Spatial analysis complete: {len(blob_signatures)} blob types, "
          f"{len(blob_contacts)} contacts, enhanced statistics: {bool(spatial_statistics)}")
    
    return comprehensive_results


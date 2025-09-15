#!/usr/bin/env python3
"""
IMC Analysis Pipeline
"""

import numpy as np
import pandas as pd
import json
import time
import re
import os
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class IMCData:
    coords: np.ndarray
    values: np.ndarray
    protein_names: List[str]
    roi_id: str

@dataclass
class SpatialStructure:
    neighbors_by_distance: Dict[int, List[List[int]]]
    tree: cKDTree

@dataclass
class AnalysisResults:
    spatial_autocorrelation: Dict  # For visualization compatibility
    spatial_organization: Dict     # Raw spatial data
    colocalization: Dict
    metadata: Dict

class DataLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> IMCData:
        pass

class IMCLoader(DataLoader):
    def __init__(self, config_path: str = 'config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def load(self, file_path: str) -> IMCData:
        df = pd.read_csv(file_path, sep='\t')
        
        # Priority 1: Use all functional groups if specified
        if 'functional_groups' in self.config['proteins']:
            selected_names = set()
            # Collect proteins from all functional groups except structural controls
            for group_name, proteins in self.config['proteins']['functional_groups'].items():
                if group_name != 'structural_controls':  # Skip DNA channels
                    selected_names.update(proteins)
            
            selected_names = list(selected_names)
            available = []
            
            # Map protein names to full column names with isotope tags
            for protein_name in selected_names:
                found = False
                for col in df.columns:
                    # Match by prefix (e.g., "CD45" matches "CD45(Y89Di)")
                    if col.startswith(protein_name + '('):
                        available.append(col)
                        found = True
                        break
                if not found:
                    print(f"⚠️  Warning: Protein {protein_name} not found in data")
        
        # Priority 2: Use explicit channels list
        elif 'channels' in self.config['proteins']:
            protein_channels = self.config['proteins']['channels']
            available = [p for p in protein_channels if p in df.columns]
        
        # Priority 3: Auto-detect
        else:
            exclude = (self.config['proteins'].get('background_channels', []) + 
                      self.config['proteins'].get('control_channels', []) +
                      self.config['proteins'].get('dna_channels', []) +
                      ['X', 'Y', 'Z', 'Time', 'Event_length', 'Center_X', 'Center_Y'])
            available = [col for col in df.columns if col not in exclude and '(' in col]
        
        if not available:
            raise ValueError(f"No protein channels found in {file_path}")
            
        coord_cols = ['X', 'Y']  # Standard IMC coordinate columns
        coords = df[coord_cols].values
        values = np.arcsinh(df[available].values / 5.0)
        
        return IMCData(coords, values, available, file_path)

# Pipeline Builder
class AnalysisPipelineBuilder:
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.data_loader = None
        self.max_pixels = config['analysis']['max_pixels_per_roi']
        self.distances = config['analysis']['distance_measurements']
        self.coloc_distance = config['analysis'].get('colocalization_distance', 15)
        self.max_clusters = config['analysis'].get('max_clusters', 8)
        
    def with_data_loader(self, loader: DataLoader):
        self.data_loader = loader
        return self
        
    def with_precision(self, max_pixels: int):
        self.max_pixels = max_pixels
        return self
        
    def with_distances(self, distances: List[int]):
        self.distances = distances
        return self
        
    def with_colocalization_distance(self, distance: int):
        self.coloc_distance = distance
        return self
        
    def with_max_clusters(self, max_clusters: int):
        self.max_clusters = max_clusters
        return self
        
    def build(self) -> 'AnalysisPipeline':
        if not self.data_loader:
            self.data_loader = IMCLoader(self.config_path)
        
        return AnalysisPipeline(
            data_loader=self.data_loader,
            max_pixels=self.max_pixels,
            distances=self.distances,
            coloc_distance=self.coloc_distance,
            max_clusters=self.max_clusters,
            config_path=self.config_path
        )

# Analysis Functions
class SpatialAnalyzer:
    @staticmethod
    def build_spatial_structure(coords: np.ndarray, distances: List[int]) -> SpatialStructure:
        tree = cKDTree(coords)
        neighbors = {}
        
        print(f"      Building spatial structure for {len(coords)} pixels...")
        
        for d in distances:
            print(f"      → Distance {d}μm...", end='', flush=True)
            start = time.time()
            
            # Use distance bands instead of exact filtering
            inner_radius = max(0, d - 2.0)
            outer_radius = d + 2.0
            
            # Get all neighbors in outer radius
            outer_pairs = tree.query_ball_tree(tree, outer_radius)
            
            # Filter out inner radius if needed
            if inner_radius > 0:
                inner_pairs = tree.query_ball_tree(tree, inner_radius)
                filtered = []
                for i, (outer, inner) in enumerate(zip(outer_pairs, inner_pairs)):
                    # Keep neighbors in distance band only
                    band_neighbors = [n for n in outer if n not in inner and n != i]
                    filtered.append(band_neighbors)
            else:
                # Just remove self
                filtered = [[n for n in pairs if n != i] for i, pairs in enumerate(outer_pairs)]
            
            neighbors[d] = filtered
            print(f" {time.time() - start:.1f}s")
        
        return SpatialStructure(neighbors, tree)
    
    @staticmethod
    def compute_correlation(values: np.ndarray, neighbors: List[List[int]]) -> float:
        # Sample pairs for performance
        pairs = []
        for i, neighbor_list in enumerate(neighbors):
            # Limit neighbors for performance
            sampled_neighbors = neighbor_list[:3] if len(neighbor_list) > 3 else neighbor_list
            pairs.extend([(i, j) for j in sampled_neighbors if i < j])
        
        # Sample pairs for large datasets
        if len(pairs) > 1000:
            indices = np.random.choice(len(pairs), 1000, replace=False)
            pairs = [pairs[i] for i in indices]
        
        if len(pairs) < 10:
            return 0.0
        
        vals_i = values[[p[0] for p in pairs]]
        vals_j = values[[p[1] for p in pairs]]
        
        if np.std(vals_i) > 1e-6 and np.std(vals_j) > 1e-6:
            corr = np.corrcoef(vals_i, vals_j)[0, 1]
            return 0.0 if np.isnan(corr) else corr
        return 0.0

class OrganizationAnalyzer:
    @staticmethod
    def analyze_protein(values: np.ndarray, spatial_structure: SpatialStructure) -> Tuple[Dict, int]:
        results = {}
        for distance, neighbors in spatial_structure.neighbors_by_distance.items():
            results[distance] = SpatialAnalyzer.compute_correlation(values, neighbors)
        
        # Organization scale
        correlations = list(results.values())
        max_corr = max(correlations) if correlations else 0
        threshold = 0.5 * max_corr
        
        scale = 10
        for dist, corr in results.items():
            if corr < threshold:
                scale = dist
                break
        
        return results, scale

class ColocalizationAnalyzer:
    @staticmethod
    def analyze_proteins(data: IMCData, spatial_structure: SpatialStructure, distance: int, threshold: float = 0.0) -> Dict:
        # Use closest available distance
        best_d = min(spatial_structure.neighbors_by_distance.keys(), key=lambda x: abs(x - distance))
        neighbors = spatial_structure.neighbors_by_distance[best_d]
        
        coloc = {}
        n_proteins = len(data.protein_names)
        
        # Compute ALL possible protein pairs - no filtering
        for i in range(n_proteins):
            for j in range(i + 1, n_proteins):
                # Compute Pearson correlation for spatial colocalization
                local_scores = []
                
                for pixel_idx, neighbor_list in enumerate(neighbors):
                    if len(neighbor_list) < 3:
                        continue
                    
                    # Get neighborhood values including center pixel
                    neighbor_indices = [pixel_idx] + neighbor_list[:10]  # Limit for performance
                    vals_i = data.values[neighbor_indices, i]
                    vals_j = data.values[neighbor_indices, j]
                    
                    # Simple Pearson correlation - much more robust for IMC data
                    if len(vals_i) > 1 and vals_i.std() > 0 and vals_j.std() > 0:
                        correlation = np.corrcoef(vals_i, vals_j)[0, 1]
                        if not np.isnan(correlation):
                            local_scores.append(abs(correlation))  # Use absolute value
                    else:
                        local_scores.append(0)
                
                # Aggregate correlation scores
                if local_scores:
                    # Use robust statistics - median correlation
                    score = float(np.median(local_scores))
                    score_std = float(np.std(local_scores))
                else:
                    score = 0.0
                    score_std = 0.0
                    
                # Determine significance based on configurable thresholds
                base_thr = float(threshold) if threshold is not None else 0.2
                thr_weak = base_thr
                thr_moderate = min(base_thr + 0.2, 0.999)
                thr_strong = min(base_thr + 0.4, 0.999)
                if score > thr_strong:
                    significance = 'strong'
                elif score > thr_moderate:
                    significance = 'moderate'
                elif score > thr_weak:
                    significance = 'weak'
                else:
                    significance = 'negligible'
                    
                pair_name = f"{data.protein_names[i]}_{data.protein_names[j]}"
                coloc[pair_name] = {
                    'colocalization_score': score,
                    'score_std': score_std,
                    'protein_1': data.protein_names[i],
                    'protein_2': data.protein_names[j],
                    'n_measurements': len(local_scores),
                    'significance': significance
                }
                print(f"      {pair_name}: {score:.3f} ({significance})")
        
        print(f"      📊 Complete analysis: {len(coloc)} protein pairs captured (ALL combinations)")
        return coloc

# Removed PhenotypeAnalyzer - phenotypes not interpretable for spatial protein expression

class AnalysisPipeline:
    def __init__(self, data_loader: DataLoader, max_pixels: int, distances: List[int], 
                 coloc_distance: int, max_clusters: int, config_path: str = 'config.json'):
        self.data_loader = data_loader
        self.max_pixels = max_pixels
        self.distances = distances
        self.coloc_distance = coloc_distance
        self.max_clusters = max_clusters
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def _subsample(self, data: IMCData) -> IMCData:
        """Precision control"""
        if len(data.coords) <= self.max_pixels:
            return data
        
        indices = np.random.choice(len(data.coords), self.max_pixels, replace=False)
        return IMCData(
            data.coords[indices],
            data.values[indices],
            data.protein_names,
            data.roi_id
        )
    
    def analyze(self, file_path: str) -> AnalysisResults:
        """Execute complete analysis pipeline"""
        start_time = time.time()
        print(f"🧬 IMC Pipeline: {file_path}")
        
        # 1. Load and subsample
        data = self.data_loader.load(file_path)
        data = self._subsample(data)
        print(f"   📊 {len(data.coords)} pixels, {len(data.protein_names)} proteins")
        
        # 2. Enhanced spatial analysis (new robust method)
        use_robust = self.config.get('spatial_analysis', {}).get('scale_selection', {}).get('method') == 'adaptive_data_driven'
        
        if use_robust:
            print(f"   🔬 Using robust spatial analysis (superpixel + squidpy)")
            try:
                from .spatial import analyze_roi_spatial_organization
                comprehensive_results = analyze_roi_spatial_organization(
                    data.coords, data.values, data.protein_names, 
                    self.config_path, use_robust_analysis=True
                )
                
                # Extract spatial statistics for pipeline compatibility
                spatial_stats = comprehensive_results.get('spatial_statistics', {})
                spatial_autocorr = spatial_stats.get('spatial_autocorrelation', {})
                coloc = spatial_stats.get('colocalization', {})
                spatial_org = {}
                
                # Convert spatial autocorr to legacy format for visualization compatibility
                for protein, stats in spatial_autocorr.items():
                    protein_clean = protein.split('(')[0]
                    spatial_org[protein_clean] = {
                        'morans_i': stats.get('morans_i', 0.0),
                        'p_value': stats.get('p_value_corrected', 1.0),
                        'significant': stats.get('significant_corrected', False),
                        'organization_scale': spatial_stats.get('spatial_parameters', {}).get('functional_radius', 75.0)
                    }
                
                print(f"   📊 Enhanced analysis: {len(spatial_autocorr)} proteins, {len(coloc)} pairs")
                
            except Exception as e:
                print(f"   ⚠️  Robust analysis failed: {e}, falling back to legacy")
                use_robust = False
        
        if not use_robust:
            print(f"   🔬 Using legacy spatial analysis")
            # Legacy spatial analysis
            spatial_structure = SpatialAnalyzer.build_spatial_structure(data.coords, self.distances)
            print(f"   🕸️  Spatial structure computed")
            
            # 3. Analyze organization
            spatial_org = {}
            for i, protein in enumerate(data.protein_names):
                org_results, scale = OrganizationAnalyzer.analyze_protein(data.values[:, i], spatial_structure)
                spatial_org[protein] = {
                    'correlations': org_results,
                    'organization_scale': scale
                }
                print(f"   🔍 {protein}: {scale}μm")
            
            # 4. Colocalization
            threshold = self.config['analysis'].get('correlation_threshold', 0.05)
            coloc = ColocalizationAnalyzer.analyze_proteins(data, spatial_structure, self.coloc_distance, threshold)
            print(f"   🤝 {len(coloc)} colocalization pairs")
        
        # 5. Phenotypes
        # Removed phenotype analysis - focus on spatial protein patterns
        
        # 6. Results - fix data structure for visualization compatibility  
        spatial_autocorr = {}
        for i, protein in enumerate(data.protein_names):
            protein_clean = protein.split('(')[0]
            # Compute per-protein spatial autocorrelation across all distances
            protein_values = data.values[:, i:i+1]  # Keep 2D shape for compatibility
            protein_autocorr = {}
            for distance, neighbors in spatial_structure.neighbors_by_distance.items():
                # Use the existing correlation function but for single protein
                corr = SpatialAnalyzer.compute_correlation(protein_values, neighbors)
                if corr == 0.0:  # Fallback: use variance as proxy
                    pairs = []
                    for j, neighbor_list in enumerate(neighbors):
                        sampled_neighbors = neighbor_list[:3] if len(neighbor_list) > 3 else neighbor_list
                        pairs.extend([(j, k) for k in sampled_neighbors])
                    
                    if pairs:
                        vals = protein_values[[p[0] for p in pairs], 0]
                        neighbor_vals = protein_values[[p[1] for p in pairs], 0]
                        if len(vals) > 1 and vals.std() > 0:
                            corr = np.corrcoef(vals, neighbor_vals)[0,1] 
                            corr = 0.0 if np.isnan(corr) else abs(corr)
                
                protein_autocorr[distance] = max(0.0, corr)  # Ensure non-negative
            spatial_autocorr[protein_clean] = protein_autocorr
            
        results = AnalysisResults(
            spatial_autocorrelation=spatial_autocorr,  # Fix key name
            spatial_organization=spatial_org,  # Keep for compatibility
            colocalization=coloc,
            metadata={
                'roi_id': data.roi_id,
                'n_pixels': len(data.coords),
                'protein_names': data.protein_names,
                'analysis_time': time.time() - start_time
            }
        )
        
        # Save
        output_file = file_path.replace('.txt', '_pipeline_analysis.json')
        results_dict = {
            'spatial_autocorrelation': results.spatial_autocorrelation,  # Add for visualization
            'spatial_organization': results.spatial_organization,
            'colocalization': results.colocalization,
            'metadata': results.metadata  # Keep metadata separate
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"✅ Pipeline complete: {results.metadata['analysis_time']:.1f}s → {output_file}")
        return results

def parse_roi_metadata(filename, config_path: str = 'config.json'):
    """Extract experimental metadata from CSV file and filename patterns"""
    import pandas as pd
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    exp_config = config['experimental']
    metadata = {}
    
    # Load metadata CSV file
    metadata_file = exp_config.get('metadata_file')
    csv_metadata = {}
    
    if metadata_file and os.path.exists(metadata_file):
        try:
            df = pd.read_csv(metadata_file)
            # Match filename to CSV entry
            base_filename = filename.replace('.txt', '')
            
            # Find matching row in CSV
            matching_row = df[df['File Name'] == base_filename]
            if not matching_row.empty:
                row = matching_row.iloc[0]
                csv_metadata = {
                    'condition': row['Condition'] if 'Condition' in row else '',
                    'timepoint': row['Injury Day'] if 'Injury Day' in row else None,
                    'mouse_id': row['Mouse'] if 'Mouse' in row else '',
                    'region': row['Details '] if 'Details ' in row else '',  # This contains Cortex/Medulla (note trailing space)
                    'roi_id': row['ROI (Region of Interest)'] if 'ROI (Region of Interest)' in row else '',
                    'file_code': row['File Code'] if 'File Code' in row else ''
                }
                print(f"   📊 CSV metadata found for {filename}: {csv_metadata}")
            else:
                print(f"   ⚠️  No CSV metadata found for {filename}")
        except Exception as e:
            print(f"   ❌ Error reading metadata CSV: {e}")
    
    # Parse from filename patterns as fallback
    # Format: IMC_241218_Alun_ROI_D7_M2_03_26.txt
    if 'D1' in filename:
        metadata['timepoint'] = 1
        metadata['condition'] = 'Injury'
    elif 'D3' in filename:
        metadata['timepoint'] = 3
        metadata['condition'] = 'Injury'
    elif 'D7' in filename:
        metadata['timepoint'] = 7
        metadata['condition'] = 'Injury'
    elif 'Sham' in filename or 'D0' in filename or 'Sam' in filename:
        metadata['timepoint'] = 0
        metadata['condition'] = 'Sham'
    elif 'Test' in filename:
        metadata['timepoint'] = None
        metadata['condition'] = 'Test'
    
    # Extract mouse ID (biological replicate)
    if 'M1' in filename:
        metadata['mouse_id'] = 'M1'
        metadata['subject'] = f"Mouse1_T{metadata.get('timepoint', 'X')}"
    elif 'M2' in filename:
        metadata['mouse_id'] = 'M2' 
        metadata['subject'] = f"Mouse2_T{metadata.get('timepoint', 'X')}"
    elif 'Sam1' in filename:
        metadata['mouse_id'] = 'Sam1'
        metadata['subject'] = f"Sham1_T0"
    elif 'Sam2' in filename:
        metadata['mouse_id'] = 'Sam2'
        metadata['subject'] = f"Sham2_T0"
    elif 'Test' in filename:
        metadata['mouse_id'] = 'Test'
        metadata['subject'] = f"Test_Sample"
    
    # Extract ROI number if present
    import re
    roi_match = re.search(r'ROI.*?(\d+)_(\d+)', filename)
    if roi_match:
        metadata['roi_number'] = f"{roi_match.group(1)}_{roi_match.group(2)}"
    
    # Merge CSV metadata (higher priority) with filename metadata
    for key, value in csv_metadata.items():
        if value and str(value).strip() and str(value) != 'nan':
            metadata[key] = value
    
    # Ensure region is set for both field names
    region = metadata.get('region', 'Unknown')
    metadata['region'] = region
    metadata['tissue_region'] = region  # For visualizer compatibility
    
    return metadata

def batch_experimental_analysis(data_dir, config_path: str = 'config.json'):
    """Analyze all ROIs with experimental grouping"""
    data_dir = Path(data_dir)
    roi_files = list(data_dir.glob("*ROI*.txt"))
    
    print(f"🧪 Batch Analysis: {len(roi_files)} ROIs")
    
    pipeline = create_standard_pipeline(config_path)
    results = []
    
    for roi_file in roi_files:
        metadata = parse_roi_metadata(roi_file.name, config_path)
        # Create display string from available metadata
        key_fields = ['condition', 'region', 'timepoint', 'subject']
        display_parts = [str(metadata.get(field, 'Unknown')) for field in key_fields if metadata.get(field)]
        display = '_'.join(display_parts) if display_parts else 'Unknown'
        print(f"   📍 {display}")
        
        roi_result = pipeline.analyze(str(roi_file))
        roi_result.metadata.update(metadata)
        results.append(roi_result)
    
    # Save batch results
    batch_output = {
        'batch_results': [
            {
                'metadata': r.metadata,
                'spatial_organization': r.spatial_organization,
                'colocalization': r.colocalization,
            } for r in results
        ],
        'experimental_design': {
            field: [r.metadata.get(field) for r in results if r.metadata.get(field)]
            for field in set().union(*(r.metadata.keys() for r in results))
        }
    }
    
    with open('batch_experimental_results.json', 'w') as f:
        json.dump(batch_output, f, indent=2, default=str)
    
    print(f"✅ Batch complete → batch_experimental_results.json")
    return results

def create_standard_pipeline(config_path: str = 'config.json') -> AnalysisPipeline:
    """Create pipeline from config"""
    return AnalysisPipelineBuilder(config_path).build()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if path in ['--help', '-h']:
            print("Usage: python imc_pipeline.py <roi_file.txt|data_directory>")
            print("Example: python imc_pipeline.py data/241218_IMC_Alun/")
            sys.exit(0)
        if Path(path).is_dir():
            batch_experimental_analysis(path)
        else:
            pipeline = create_standard_pipeline()
            pipeline.analyze(path)
    else:
        print("Usage: python imc_pipeline.py <roi_file.txt|data_directory>")
        print("Example: python imc_pipeline.py data/241218_IMC_Alun/")
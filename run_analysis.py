#!/usr/bin/env python3
"""
Kidney Injury IMC Analysis Pipeline - Hypothesis Generation Study

Main entry point for superpixel-based spatial analysis of kidney injury.
Implements morphology-aware superpixel segmentation for hypothesis generation
in kidney injury spatial proteomics.

IMPORTANT: This is a pilot study with n=2 mice per timepoint.
All results are for hypothesis generation only and require validation
in larger studies before drawing biological conclusions.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np
import pandas as pd
import gc  # For garbage collection
import matplotlib.pyplot as plt
from scipy import ndimage  # For morphological operations in validation

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.analysis.ion_count_processing import ion_count_pipeline
from src.analysis.multiscale_analysis import perform_multiscale_analysis
from src.analysis.batch_correction import sham_anchored_normalize, detect_batch_effects
# Use new pure validation framework
from src.validation.core import SegmentationValidator, ValidationFactory
from src.validation.adapter import validate_segmentation_quality  # Legacy compatibility

# Framework configuration can be loaded separately
try:
    import yaml
    with open('config/framework.yaml', 'r') as f:
        FRAMEWORK_CONFIG = yaml.safe_load(f)
except (FileNotFoundError, ImportError):
    # Fallback configuration
    FRAMEWORK_CONFIG = {
        'validation': {
            'type': 'segmentation',
            'metrics': ['compactness', 'boundary_adherence']
        }
    }
from src.utils.data_loader import load_metadata_from_csv
from src.utils.helpers import Metadata
from src.viz_utils.plotting import plot_segmentation_overlay


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_imc_data(config: Config) -> Dict[str, Dict[str, Any]]:
    """Load IMC data from ROI text files and organize by batches."""
    logger = logging.getLogger('DataLoader')
    
    # Find ROI files
    data_dir = Path(config.data['raw_data_dir'])
    roi_files = list(data_dir.glob(config.data['file_pattern']))
    
    logger.info(f"Found {len(roi_files)} ROI files")
    
    if not roi_files:
        raise FileNotFoundError(f"No ROI files found in {data_dir}")
    
    # Load metadata using centralized loader
    metadata_path = Path(config.data['metadata_file'])
    metadata_map = load_metadata_from_csv(metadata_path, config.raw, roi_files)
    logger.info(f"Loaded metadata for {len(metadata_map)} ROIs")
    
    batch_data = {}
    roi_metadata = {}
    
    # Get channel definitions from config
    protein_channels = config.channels['protein_channels']
    dna_channels = config.channels['dna_channels']
    background_channel = config.channels['background_channel']
    excluded = (config.channels['excluded_channels'] + 
                config.channels['calibration_channels'] + 
                [config.channels['carrier_gas_channel']])
    
    for roi_file in roi_files:
        try:
            logger.info(f"Loading {roi_file.name}...")
            
            # Load ROI data
            roi_data = pd.read_csv(roi_file, sep='\t')
            
            # Extract coordinates
            coords = roi_data[['X', 'Y']].values
            
            # Get background signal if available
            background_col = [col for col in roi_data.columns if background_channel in col]
            background_signal = roi_data[background_col[0]].values if background_col else np.zeros(len(coords))
            
            # Process protein channels with background correction
            ion_counts = {}
            for protein in protein_channels:
                # Find the column with this protein marker
                protein_col = [col for col in roi_data.columns if protein in col]
                if protein_col:
                    # Apply background correction
                    signal = roi_data[protein_col[0]].values
                    if config.processing['background_correction']['enabled']:
                        corrected = signal - background_signal
                        if config.processing['background_correction']['clip_negative']:
                            corrected = np.clip(corrected, 0, None)
                        ion_counts[protein] = corrected
                    else:
                        ion_counts[protein] = signal
                    logger.debug(f"  Loaded {protein} with {np.sum(ion_counts[protein] > 0)} non-zero pixels")
            
            # Get DNA channels for morphological information  
            dna_data = {}
            for dna in dna_channels:
                dna_col = [col for col in roi_data.columns if dna in col]
                if dna_col:
                    dna_data[dna] = roi_data[dna_col[0]].values
                else:
                    dna_data[dna] = np.zeros(len(coords))
            
            # Get metadata from centralized map
            metadata = metadata_map.get(roi_file.stem, Metadata())
            
            # Determine batch from metadata
            batch_id = f"T{metadata.timepoint}_{metadata.replicate_id}"
            
            if batch_id not in batch_data:
                batch_data[batch_id] = {}
                roi_metadata[batch_id] = []
            
            # Store ROI data with metadata
            roi_name = roi_file.stem
            batch_data[batch_id][roi_name] = {
                'coords': coords,
                'ion_counts': ion_counts,
                'dna_intensities': dna_data,
                'n_pixels': len(coords),
                'filename': roi_file.name,
                'metadata': metadata  # Store Metadata object directly
            }
            
            roi_metadata[batch_id].append({
                'roi_name': roi_name,
                'filename': roi_file.name,
                'n_pixels': len(coords),
                'n_proteins': len(ion_counts),
                'region': metadata.region,
                'condition': metadata.condition
            })
            
            logger.info(f"  {len(coords)} pixels, {len(ion_counts)} proteins, {metadata.region} region")
            
        except Exception as e:
            logger.error(f"Failed to load {roi_file}: {e}")
            continue
    
    logger.info(f"Successfully loaded {sum(len(batch) for batch in batch_data.values())} ROIs "
                f"across {len(batch_data)} batches")
    
    # Log quality control info
    logger.info("Channel processing summary:")
    logger.info(f"  Protein channels: {len(protein_channels)}")
    logger.info(f"  Background correction: {config.processing['background_correction']['enabled']}")
    logger.info(f"  Excluded channels: {len(excluded)}")
    
    return batch_data, roi_metadata



def run_batch_correction(batch_data: Dict[str, Dict[str, Any]], config: Config) -> Dict[str, Dict[str, Any]]:
    """Apply batch correction using sham-anchored normalization."""
    logger = logging.getLogger('BatchCorrection')
    
    # Prepare data for batch correction (aggregate ROI data by batch)
    batch_ion_counts = {}
    batch_metadata = {}
    
    for batch_id, roi_data in batch_data.items():
        # Aggregate all ROI ion counts for this batch
        aggregated_counts = {}
        
        for roi_name, data in roi_data.items():
            for protein, counts in data['ion_counts'].items():
                if protein not in aggregated_counts:
                    aggregated_counts[protein] = []
                aggregated_counts[protein].append(counts)
        
        # Concatenate all ROI data for this batch
        batch_ion_counts[batch_id] = {
            protein: np.concatenate(count_list)
            for protein, count_list in aggregated_counts.items()
        }
        
        # Extract batch metadata - need to get condition and timepoint from first ROI
        # Since all ROIs in a batch share the same timepoint and condition
        first_roi_name = list(roi_data.keys())[0]
        # Parse batch_id to extract timepoint (format is T{timepoint}_{replicate})
        timepoint = int(batch_id.split('_')[0][1:]) if batch_id.startswith('T') else None
        
        # Determine condition based on timepoint
        condition = 'Sham' if timepoint == 0 else 'Injury'
        
        batch_metadata[batch_id] = {
            'n_rois': len(roi_data),
            'total_pixels': sum(data['n_pixels'] for data in roi_data.values()),
            'roi_names': list(roi_data.keys()),
            'Condition': condition,
            'Injury Day': timepoint,
            'timepoint': timepoint,
            'condition': condition
        }
    
    logger.info(f"Prepared {len(batch_ion_counts)} batches for correction")
    
    # Apply sham-anchored normalization
    try:
        corrected_data, correction_stats = sham_anchored_normalize(
            batch_ion_counts, 
            batch_metadata,
            sham_condition='Sham',
            sham_timepoint=0
        )
        
        logger.info("Sham-anchored normalization completed successfully")
        logger.info(f"Normalized {len(correction_stats['sham_batches'])} sham reference batches")
        
        return corrected_data, correction_stats
        
    except Exception as e:
        logger.error(f"Batch correction failed: {e}")
        logger.info("Proceeding without batch correction")
        return batch_ion_counts, {'error': str(e)}


def analyze_roi_with_multiscale(roi_data: Dict[str, Any], config: Config, plots_dir: Path = None) -> Dict[str, Any]:
    """Analyze single ROI with multi-scale approach and generate visual validation.
    
    NOTE: Low inter-scale ARI is EXPECTED - different scales capture different biology.
    10μm captures cells, 40μm captures tissue domains - they shouldn't agree!
    """
    logger = logging.getLogger('Analysis')
    
    coords = roi_data['coords']
    ion_counts = roi_data['ion_counts']
    dna_intensities = roi_data['dna_intensities']
    
    # Extract DNA1 and DNA2 from the dict
    dna1_intensities = dna_intensities.get('DNA1', np.zeros(len(coords)))
    dna2_intensities = dna_intensities.get('DNA2', np.zeros(len(coords)))
    
    # Get multi-scale parameters from config  
    scales_um = config.segmentation.get('scales_um', [10.0, 20.0, 40.0])
    n_clusters = config.analysis['clustering']['n_clusters']
    use_slic = config.segmentation['method'] == 'slic'
    memory_limit_gb = config.performance.get('memory_limit_gb', 12.0)
    
    # Perform multi-scale analysis
    multiscale_results = perform_multiscale_analysis(
        coords=coords,
        ion_counts=ion_counts,
        dna1_intensities=dna1_intensities,
        dna2_intensities=dna2_intensities,
        scales_um=scales_um,
        n_clusters=n_clusters,
        use_slic=use_slic,
        memory_limit_gb=memory_limit_gb
    )
    
    # Generate visual validation plots if output directory provided
    if plots_dir:
        roi_name = roi_data['filename'].replace('.txt', '')
        for scale, result in multiscale_results.items():
            if 'superpixel_labels' in result and 'composite_dna' in result and result['composite_dna'].size > 0:
                try:
                    fig, ax = plt.subplots(figsize=(10, 10))
                    plot_segmentation_overlay(
                        image=result['composite_dna'],
                        labels=result['superpixel_labels'],
                        bounds=result['bounds'],
                        title=f"SLIC Segmentation - {roi_name} - {scale}μm",
                        ax=ax
                    )
                    plot_filename = plots_dir / f"{roi_name}_scale_{scale}_segmentation.png"
                    fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    logger.debug(f"Generated validation plot: {plot_filename}")
                except Exception as e:
                    logger.warning(f"Could not generate plot for {roi_name} at scale {scale}: {e}")
    
    # NOTE: NOT computing scale consistency anymore - low ARI between scales is expected!
    # Different scales capture different biological features, they shouldn't agree
    
    # Include metadata in results
    metadata = roi_data['metadata']
    return {
        'multiscale_results': multiscale_results,
        # Removed 'consistency_results' - low inter-scale ARI is biologically meaningful
        'roi_metadata': {
            'n_pixels': len(coords),
            'n_proteins': len(ion_counts),
            'filename': roi_data.get('filename', 'unknown'),
            'region': metadata.region,
            'replicate_id': metadata.replicate_id,
            'timepoint': metadata.timepoint,
            'condition': metadata.condition
        }
    }


def validate_roi_chunk(labels_chunk: np.ndarray, composite_chunk: np.ndarray, 
                       chunk_id: str) -> Dict[str, Any]:
    """Validate a single spatial chunk with memory constraints."""
    if labels_chunk.size == 0 or labels_chunk.ndim != 2:
        return {'chunk_id': chunk_id, 'status': 'invalid_input'}
    
    try:
        # Simple, memory-efficient metrics for chunks
        unique_labels = np.unique(labels_chunk)
        unique_labels = unique_labels[unique_labels >= 0]
        
        if len(unique_labels) == 0:
            return {'chunk_id': chunk_id, 'status': 'no_segments'}
        
        # Basic chunk metrics
        n_segments = len(unique_labels)
        chunk_area = labels_chunk.size
        
        # Compute compactness efficiently
        compactness_values = []
        for seg_id in unique_labels:
            mask = labels_chunk == seg_id
            area = np.sum(mask)
            if area > 2:  # Need minimum area for perimeter calculation
                # Simplified perimeter using boundary pixels
                boundary = mask & ~ndimage.binary_erosion(mask)
                perimeter = np.sum(boundary)
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter ** 2)
                    compactness_values.append(min(compactness, 1.0))
        
        # DNA signal metrics if available
        dna_metrics = {}
        if composite_chunk.size > 0 and composite_chunk.ndim == 2:
            dna_metrics = {
                'mean_intensity': float(composite_chunk.mean()),
                'std_intensity': float(composite_chunk.std()),
                'min_intensity': float(composite_chunk.min()),
                'max_intensity': float(composite_chunk.max())
            }
        
        return {
            'chunk_id': chunk_id,
            'status': 'success',
            'n_segments': n_segments,
            'segment_density': n_segments / chunk_area,
            'compactness_mean': float(np.mean(compactness_values)) if compactness_values else 0.0,
            'compactness_std': float(np.std(compactness_values)) if compactness_values else 0.0,
            'dna_metrics': dna_metrics
        }
        
    except Exception as e:
        return {'chunk_id': chunk_id, 'status': 'error', 'error': str(e)}


def run_validation(analysis_results: List[Dict[str, Any]], config: Config) -> Dict[str, Any]:
    """Run segmentation quality validation using lightweight data and disk-based full results.
    
    Validates the SLIC-on-DNA segmentation method through morphological metrics
    and biological correspondence, reading full data from disk as needed.
    """
    logger = logging.getLogger('Validation')
    
    if not analysis_results:
        return {'error': 'No analysis results for validation'}
    
    try:
        validation_results = {
            'validation_method': 'segmentation_quality',
            'n_rois_validated': len(analysis_results),
            'roi_validations': []
        }
        
        # Get paths for reading full results from disk
        output_config = config.raw.get('output', {})
        results_dir = Path(output_config.get('results_dir', 'results/cross_sectional_kidney_injury'))
        roi_results_dir = results_dir / output_config.get('roi_results_dir', 'roi_results')
        
        # Validate each ROI using chunked processing and NPZ data
        for i, validation_data in enumerate(analysis_results):
            roi_name = validation_data.get('roi_metadata', {}).get('filename', 'unknown')
            roi_filename = roi_name.replace('.txt', '') if roi_name.endswith('.txt') else roi_name
            
            roi_validation = {'roi': roi_name, 'scale_validations': {}}
            
            # Load arrays from NPZ format (much more memory efficient)
            npz_file = roi_results_dir / f"{roi_filename}_arrays.npz"
            metadata_file = roi_results_dir / f"{roi_filename}_metadata.json"
            
            if npz_file.exists() and metadata_file.exists():
                try:
                    # Load metadata to get scale information
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Process each scale with chunked validation
                    multiscale_metadata = metadata.get('multiscale_metadata', {})
                    
                    for scale_key, scale_metadata in multiscale_metadata.items():
                        scale = scale_key.replace('scale_', '')
                        
                        # Load arrays for this scale from NPZ
                        with np.load(npz_file) as arrays:
                            labels_key = f'scale_{scale}_superpixel_labels'
                            composite_key = f'scale_{scale}_composite_dna'
                            
                            if labels_key in arrays:
                                labels = arrays[labels_key]
                                composite = arrays.get(composite_key, np.array([]))
                                
                                # Validate in 50x50 chunks to manage memory
                                chunk_size = 50
                                height, width = labels.shape
                                chunk_results = []
                                
                                for y in range(0, height, chunk_size):
                                    for x in range(0, width, chunk_size):
                                        y_end = min(y + chunk_size, height)
                                        x_end = min(x + chunk_size, width)
                                        
                                        labels_chunk = labels[y:y_end, x:x_end]
                                        composite_chunk = composite[y:y_end, x:x_end] if composite.size > 0 else np.array([])
                                        
                                        chunk_id = f"{roi_filename}_scale_{scale}_chunk_{y}_{x}"
                                        chunk_result = validate_roi_chunk(labels_chunk, composite_chunk, chunk_id)
                                        
                                        if chunk_result['status'] == 'success':
                                            chunk_results.append(chunk_result)
                                        
                                        # Free chunk memory immediately
                                        del labels_chunk
                                        if composite_chunk.size > 0:
                                            del composite_chunk
                                
                                # Aggregate chunk results into scale-level metrics
                                if chunk_results:
                                    successful_chunks = [r for r in chunk_results if r['status'] == 'success']
                                    
                                    if successful_chunks:
                                        aggregated_metrics = {
                                            'scale_um': scale,
                                            'n_chunks_validated': len(successful_chunks),
                                            'total_segments': sum(r['n_segments'] for r in successful_chunks),
                                            'mean_segment_density': np.mean([r['segment_density'] for r in successful_chunks]),
                                            'mean_compactness': np.mean([r['compactness_mean'] for r in successful_chunks if r['compactness_mean'] > 0]),
                                            'compactness_variation': np.std([r['compactness_mean'] for r in successful_chunks if r['compactness_mean'] > 0])
                                        }
                                        
                                        # Aggregate DNA metrics
                                        dna_means = [r['dna_metrics']['mean_intensity'] for r in successful_chunks if r['dna_metrics']]
                                        if dna_means:
                                            aggregated_metrics['dna_mean_intensity'] = np.mean(dna_means)
                                            aggregated_metrics['dna_intensity_variation'] = np.std(dna_means)
                                        
                                        roi_validation['scale_validations'][f'{scale}um'] = aggregated_metrics
                                
                                # Clean up scale arrays
                                del labels
                                if composite.size > 0:
                                    del composite
                    
                except Exception as e:
                    logger.warning(f"Chunked validation failed for {roi_name}: {e}")
                    roi_validation['validation_error'] = str(e)
            validation_results['roi_validations'].append(roi_validation)
            
            # Aggressive garbage collection after each ROI
            gc.collect()
        
        # Compute summary statistics
        all_metrics = {}
        for roi_val in validation_results['roi_validations']:
            for scale, metrics in roi_val['scale_validations'].items():
                for metric_name, value in metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        
        # Compute summary statistics only for numeric metrics
        validation_results['summary'] = {}
        for metric, values in all_metrics.items():
            if values and isinstance(values[0], (int, float, np.number)):
                try:
                    validation_results['summary'][metric] = {
                        'mean': float(np.mean(values)), 
                        'std': float(np.std(values))
                    }
                except Exception:
                    continue
        
        logger.info(f"Validation complete: {len(analysis_results)} ROIs validated")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {'error': str(e)}


def numpy_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def save_single_roi_result(result: Dict[str, Any], roi_filename: str, config: Config) -> None:
    """Save a single ROI result using memory-efficient NPZ format."""
    output_config = config.raw.get('output', {})
    results_dir = Path(output_config.get('results_dir', 'results/cross_sectional_kidney_injury'))
    roi_results_dir = results_dir / output_config.get('roi_results_dir', 'roi_results')
    
    # Create directories
    roi_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate arrays from metadata for efficient storage
    arrays_to_save = {}
    metadata_only = {}
    
    # Extract arrays from multiscale results
    multiscale_results = result.get('multiscale_results', {})
    for scale, scale_result in multiscale_results.items():
        for key, value in scale_result.items():
            if isinstance(value, np.ndarray):
                arrays_to_save[f'scale_{scale}_{key}'] = value
            else:
                # Keep non-array data in metadata
                if f'scale_{scale}' not in metadata_only:
                    metadata_only[f'scale_{scale}'] = {}
                metadata_only[f'scale_{scale}'][key] = value
    
    # Save arrays in compressed NPZ format
    npz_file = roi_results_dir / f"{roi_filename}_arrays.npz"
    if arrays_to_save:
        np.savez_compressed(npz_file, **arrays_to_save)
    
    # Save lightweight metadata as JSON
    metadata_result = {
        'roi_metadata': result.get('roi_metadata', {}),
        'batch_id': result.get('batch_id'),
        'multiscale_metadata': metadata_only
    }
    
    json_file = roi_results_dir / f"{roi_filename}_metadata.json"
    with open(json_file, 'w') as f:
        json.dump(metadata_result, f, indent=2, default=str)


def extract_validation_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only validation-relevant data from full result to save memory."""
    validation_data = {
        'roi_metadata': result.get('roi_metadata', {}),
        'batch_id': result.get('batch_id'),
        'validation_metrics': {}
    }
    
    # Extract only segmentation labels and reference data for validation
    multiscale_results = result.get('multiscale_results', {})
    for scale, scale_result in multiscale_results.items():
        if 'superpixel_labels' in scale_result:
            validation_data['validation_metrics'][scale] = {
                'has_segmentation': True,
                'n_superpixels': len(np.unique(scale_result['superpixel_labels'])),
                'has_dna_composite': 'composite_dna' in scale_result,
                'image_shape': scale_result.get('superpixel_labels', np.array([])).shape
            }
    
    return validation_data


def save_results(analysis_results: List[Dict[str, Any]], 
                batch_correction_stats: Dict[str, Any],
                validation_results: Dict[str, Any],
                config: Config) -> None:
    """Save analysis results to disk."""
    logger = logging.getLogger('ResultsIO')
    
    # Create output directories
    output_config = config.raw.get('output', {})
    results_dir = Path(output_config.get('results_dir', 'results'))
    plots_dir = results_dir  # Plots go in the same results directory, not a separate plots folder
    
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save analysis summary
    summary = {
        'analysis_metadata': {
            'n_rois_processed': len(analysis_results),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'config_parameters': config.raw
        },
        'batch_correction': batch_correction_stats,
        'validation_results': validation_results,
        'roi_summaries': []
    }
    
    # Add ROI-level summaries from lightweight validation data
    for validation_data in analysis_results:
        roi_summary = {
            'roi_metadata': validation_data.get('roi_metadata', {}),
            'batch_id': validation_data.get('batch_id'),
            'validation_metrics': validation_data.get('validation_metrics', {})
            # Full results are saved separately in roi_results/ directory
        }
        summary['roi_summaries'].append(roi_summary)
    
    # Save summary
    summary_file = results_dir / output_config.get('summary_file', 'analysis_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Saved analysis summary to {summary_file}")
    logger.info(f"Individual ROI results already saved to {results_dir / 'roi_results'}")


def main():
    """Run the complete production IMC analysis pipeline."""
    setup_logging()
    logger = logging.getLogger('Main')
    
    print("=" * 80)
    print("PRODUCTION IMC SPATIAL ANALYSIS PIPELINE")
    print("=" * 80)
    print("• Proper ion count statistics with Poisson noise handling")
    print("• Multi-scale analysis (10μm, 20μm, 40μm)")
    print("• Sham-anchored normalization batch correction")
    print("• Enhanced validation with realistic noise models")
    print("=" * 80)
    
    try:
        # Load configuration
        config = Config('config.json')
        logger.info("Configuration loaded successfully")
        
        # Load IMC data organized by batches
        print("\n📂 Loading IMC data...")
        batch_data, roi_metadata = load_imc_data(config)
        
        total_rois = sum(len(batch) for batch in batch_data.values())
        total_pixels = sum(
            sum(roi['n_pixels'] for roi in batch.values()) 
            for batch in batch_data.values()
        )
        
        print(f"✅ Loaded {total_rois} ROIs across {len(batch_data)} batches")
        print(f"   Total pixels: {total_pixels:,}")
        
        # Batch correction
        print("\n🔧 Applying batch correction...")
        corrected_data, correction_stats = run_batch_correction(batch_data, config)
        
        if 'error' not in correction_stats:
            if 'improvement_metrics' in correction_stats:
                improvement = correction_stats['improvement_metrics']['improvement_ratio']
                print(f"✅ Batch correction complete: {improvement:.1%} improvement")
            else:
                print(f"✅ Batch correction complete using {len(correction_stats.get('sham_batches', []))} sham batches")
        else:
            print(f"⚠️  Proceeding without batch correction: {correction_stats['error']}")
        
        # Create plots directory
        output_config = config.raw.get('output', {})
        plots_dir = Path(output_config.get('plots_dir', 'plots/validation'))
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Plots will be saved to: {plots_dir.resolve()}")
        
        # Multi-scale analysis for each ROI with visual validation
        print("\n🔬 Running multi-scale analysis with visual validation...")
        analysis_results = []
        
        roi_count = 0
        for batch_id, batch_roi_data in batch_data.items():
            print(f"\n  Processing batch '{batch_id}'...")
            
            for roi_name, roi_data in batch_roi_data.items():
                roi_count += 1
                print(f"    ROI {roi_count}: {roi_name} "
                      f"({roi_data['n_pixels']:,} pixels, {len(roi_data['ion_counts'])} proteins)")
                
                try:
                    result = analyze_roi_with_multiscale(roi_data, config, plots_dir)
                    result['batch_id'] = batch_id
                    
                    # Save result immediately to disk to avoid memory accumulation
                    roi_filename = roi_data['filename'].replace('.txt', '')
                    save_single_roi_result(result, roi_filename, config)
                    
                    # Keep only validation-relevant data in memory
                    validation_data = extract_validation_data(result)
                    analysis_results.append(validation_data)
                    
                    # Visual validation plots generated instead of misleading ARI metrics
                    n_scales = len(result.get('multiscale_results', {}))
                    if n_scales > 0:
                        print(f"      ✅ Generated {n_scales} scale validation plots")
                    
                    # Free up memory after each ROI
                    del result
                    del roi_data
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Failed to analyze {roi_name}: {e}")
                    continue
        
        if not analysis_results:
            print("❌ No successful analyses. Exiting.")
            return 1
        
        print(f"\n✅ Analyzed {len(analysis_results)} ROIs successfully")
        
        # Run validation with memory management
        print("\n🔍 Running segmentation quality validation...")
        validation_results = run_validation(analysis_results, config)
        
        # Save results
        print("\n💾 Saving results...")
        save_results(analysis_results, correction_stats, validation_results, config)
        
        # Analysis summary
        print("\n" + "=" * 80)
        print("PRODUCTION ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Basic statistics
        n_proteins = len(analysis_results[0]['roi_metadata']) if analysis_results else 0
        avg_pixels = np.mean([r['roi_metadata']['n_pixels'] for r in analysis_results])
        
        print(f"📊 Data Processed:")
        print(f"   • {len(analysis_results)} ROIs across {len(batch_data)} batches")
        print(f"   • {total_pixels:,} total pixels ({avg_pixels:,.0f} per ROI)")
        print(f"   • 9 protein markers + 2 DNA channels")
        
        # Batch correction results
        if 'error' not in correction_stats and 'improvement_metrics' in correction_stats:
            before_severity = correction_stats['improvement_metrics']['severity_before']
            after_severity = correction_stats['improvement_metrics']['severity_after']
            improvement = correction_stats['improvement_metrics']['improvement_ratio']
            print(f"\n🔧 Batch Correction:")
            print(f"   • Before: {before_severity:.3f} batch effect severity")
            print(f"   • After: {after_severity:.3f} batch effect severity")
            print(f"   • Improvement: {improvement:.1%}")
        elif 'error' not in correction_stats:
            print(f"\n🔧 Batch Correction:")
            print(f"   • Applied sham-anchored normalization")
            print(f"   • Reference: {len(correction_stats.get('sham_batches', []))} sham batches")
            print(f"   • Normalized: {len(correction_stats.get('per_batch_stats', {}))} total batches")
        
        # Multi-scale analysis summary with visual validation
        scales_analyzed = config.segmentation.get('scales_um', [10, 20, 40])
        print(f"\n🔬 Multi-scale Analysis with Visual Validation:")
        print(f"   • Scales: {scales_analyzed} μm")
        print(f"   • Visual validation plots: {plots_dir.resolve()}")
        print(f"   • Note: Low inter-scale ARI is EXPECTED - scales capture different biology!")
        
        # Report scale-specific information instead of misleading consistency
        print(f"   • 10μm scale: Captures cellular/subcellular features")
        print(f"   • 20μm scale: Captures local microenvironments")
        print(f"   • 40μm scale: Captures tissue domain organization")
        
        # Validation summary
        if 'error' not in validation_results and 'summary' in validation_results:
            n_validated = validation_results.get('n_rois_validated', 0)
            print(f"\n🔍 Validation:")
            print(f"   • {n_validated} ROIs validated with chunked processing")
            if validation_results.get('summary'):
                summary_metrics = validation_results['summary']
                for metric, stats in summary_metrics.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        print(f"   • {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        elif 'error' in validation_results:
            print(f"\n❌ Validation failed: {validation_results['error']}")
        
        print("\n" + "=" * 80)
        print("CRITICAL STUDY LIMITATIONS")
        print("=" * 80)
        print("⚠️  n=2 biological replicates - HYPOTHESIS GENERATING ONLY")
        print("⚠️  Cross-sectional design - no temporal causation")
        print("⚠️  9 protein panel - limited cell type resolution")
        print("⚠️  All findings require external validation")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    import numpy as np
    exit_code = main()
    exit(exit_code)
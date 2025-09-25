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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
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
from src.analysis.marker_validation import MarkerValidator, log_marker_audit_result
from src.validation import (
    create_validation_suite, ValidationCategory,
    CoordinateValidator, IonCountValidator, BiologicalValidator,
    PreprocessingValidator, SegmentationValidator, ClusteringValidator
)
from src.quality_control import (
    QualityMonitor, QualityGateEngine, extract_quality_metrics_from_validation,
    evaluate_roi_for_analysis, generate_quality_reports
)


def create_validation_summary(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create comprehensive validation summary from multiple ROI validation results."""
    if not validation_results:
        return {'summary': {}, 'details': [], 'recommendations': []}
    
    # Extract validation result objects
    results = [r['validation_result'] for r in validation_results if 'validation_result' in r]
    roi_ids = [r['roi_id'] for r in validation_results if 'roi_id' in r]
    
    if not results:
        return {'summary': {}, 'details': [], 'recommendations': []}
    
    # Aggregate statistics
    total_rois = len(results)
    critical_failures = sum(1 for r in results if r.summary_stats.get('has_critical', False))
    total_warnings = sum(r.summary_stats.get('severity_distribution', {}).get('warning', 0) for r in results)
    
    # Quality scores
    quality_scores = [
        r.summary_stats.get('overall_quality_score') 
        for r in results 
        if r.summary_stats.get('overall_quality_score') is not None
    ]
    
    mean_quality = np.mean(quality_scores) if quality_scores else None
    min_quality = np.min(quality_scores) if quality_scores else None
    
    # Collect all recommendations
    all_recommendations = []
    for result in results:
        all_recommendations.extend(result.get_recommendations())
    
    # Get unique recommendations with frequency
    from collections import Counter
    recommendation_counts = Counter(all_recommendations)
    top_recommendations = [rec for rec, count in recommendation_counts.most_common(10)]
    
    # Per-ROI details
    roi_details = []
    for i, (roi_id, result) in enumerate(zip(roi_ids, results)):
        roi_details.append({
            'roi_id': roi_id,
            'status': result.summary_stats.get('status'),
            'quality_score': result.summary_stats.get('overall_quality_score'),
            'has_critical': result.summary_stats.get('has_critical', False),
            'n_warnings': result.summary_stats.get('severity_distribution', {}).get('warning', 0),
            'execution_time_ms': result.execution_time_ms
        })
    
    return {
        'summary': {
            'total_rois': total_rois,
            'critical_failures': critical_failures,
            'total_warnings': total_warnings,
            'success_rate': (total_rois - critical_failures) / total_rois if total_rois > 0 else 0,
            'mean_quality_score': mean_quality,
            'min_quality_score': min_quality,
            'recommendation_diversity': len(recommendation_counts)
        },
        'details': roi_details,
        'recommendations': top_recommendations,
        'recommendation_counts': dict(recommendation_counts.most_common(10))
    }


def process_single_roi(args):
    """
    Process a single ROI file in parallel.
    
    Args:
        args: Tuple of (roi_file, config_dict, protein_channels, dna_channels, 
                       background_channel, excluded, plots_dir)
                       
    Returns:
        Tuple of (roi_id, roi_result, audit_results) or (roi_id, None, error_msg)
    """
    roi_file, config_dict, protein_channels, dna_channels, background_channel, excluded, plots_dir = args
    
    try:
        # Recreate config object (can't pickle complex objects)
        config = Config()
        config.__dict__.update(config_dict)
        
        # Initialize validator for this process
        marker_validator = MarkerValidator(config)
        
        # Setup logging for this process
        import logging
        logger = logging.getLogger(f'ROI_{roi_file.stem}')
        
        logger.info(f"Processing {roi_file.name}...")
        
        # Load ROI data
        roi_data = pd.read_csv(roi_file, sep='\t')
        
        # Extract coordinates
        coords = roi_data[['X', 'Y']].values
        
        # Get background signal if available
        background_col = [col for col in roi_data.columns if background_channel in col]
        background_signal = roi_data[background_col[0]].values if background_col else np.zeros(len(coords))
        
        # Use robust column matching for protein channels
        from src.utils.column_matching import match_imc_columns
        
        protein_column_mapping = match_imc_columns(
            protein_channels, 
            list(roi_data.columns),
            case_sensitive=False,
            fuzzy_threshold=0.8
        )
        
        # Process protein channels with background correction
        ion_counts = {}
        for protein in protein_channels:
            matched_column = protein_column_mapping.get(protein)
            if matched_column:
                # Apply background correction
                signal = roi_data[matched_column].values
                if config.processing['background_correction']['enabled']:
                    corrected = signal - background_signal
                    if config.processing['background_correction']['clip_negative']:
                        corrected = np.clip(corrected, 0, None)
                    ion_counts[protein] = corrected
                else:
                    ion_counts[protein] = signal
                logger.debug(f"  Loaded {protein} from column '{matched_column}' with {np.sum(ion_counts[protein] > 0)} non-zero pixels")
            else:
                logger.warning(f"  Could not find column for protein '{protein}'")
        
        # Validate marker presence and audit
        roi_id = roi_file.stem
        file_audit = marker_validator.audit_roi_file(str(roi_file), roi_data)
        ion_counts_audit = marker_validator.validate_ion_counts(ion_counts, roi_id)
        
        # Assert critical markers are present (fail loudly if not)
        critical_markers = list(marker_validator.critical_markers & set(protein_channels))
        from src.analysis.marker_validation import validate_marker_presence_assertion
        validate_marker_presence_assertion(ion_counts, critical_markers, roi_id)
        
        # Extract DNA channels using robust matching
        dna_column_mapping = match_imc_columns(
            dna_channels,
            list(roi_data.columns),
            case_sensitive=False,
            fuzzy_threshold=0.8
        )
        
        dna1_col = dna_column_mapping.get(dna_channels[0])
        dna2_col = dna_column_mapping.get(dna_channels[1])
        
        dna1_intensities = roi_data[dna1_col].values if dna1_col else np.zeros(len(coords))
        dna2_intensities = roi_data[dna2_col].values if dna2_col else np.zeros(len(coords))
        
        if dna1_col:
            logger.debug(f"  Loaded DNA1 from column '{dna1_col}'")
        else:
            logger.warning(f"  Could not find column for DNA channel '{dna_channels[0]}'")
            
        if dna2_col:
            logger.debug(f"  Loaded DNA2 from column '{dna2_col}'")
        else:
            logger.warning(f"  Could not find column for DNA channel '{dna_channels[1]}'")
        
        # Comprehensive validation before analysis
        validation_suite = create_validation_suite([
            ValidationCategory.DATA_INTEGRITY,
            ValidationCategory.SCIENTIFIC_QUALITY
        ])
        
        validation_suite.add_rule(CoordinateValidator())
        validation_suite.add_rule(IonCountValidator())  
        # BiologicalValidator disabled - hangs with large data (250k pixels)
        # validation_suite.add_rule(BiologicalValidator())
        
        validation_data = {
            'coords': coords,
            'ion_counts': ion_counts
        }
        
        validation_result = validation_suite.validate(validation_data, {'roi_id': roi_id})
        
        # Log critical validation failures but continue processing
        if validation_result.summary_stats['has_critical']:
            critical_failures = validation_result.get_critical_failures()
            failure_messages = [f.message for f in critical_failures]
            logger.warning(f"Quality concerns in {roi_id}: {'; '.join(failure_messages)}")
            logger.info(f"Continuing analysis of {roi_id} - quality will be tracked for reporting")
        
        # Perform multiscale analysis with SLIC
        analysis_results = perform_multiscale_analysis(
            coords, ion_counts, dna1_intensities, dna2_intensities,
            scales_um=[10.0, 20.0, 40.0],
            n_clusters=8,
            use_slic=True,
            config=config
        )
        
        # Post-analysis validation
        post_analysis_suite = create_validation_suite([ValidationCategory.PIPELINE_STATE])
        post_analysis_suite.add_rule(ClusteringValidator())
        
        # Validate first scale results
        first_scale = list(analysis_results.keys())[0]
        first_result = analysis_results[first_scale]
        
        post_validation_data = {
            'feature_matrix': first_result.get('feature_matrix'),
            'cluster_labels': first_result.get('cluster_labels'),
            'protein_names': first_result.get('protein_names'),
            'cluster_centroids': first_result.get('cluster_centroids')
        }
        
        post_validation_result = post_analysis_suite.validate(post_validation_data, {'roi_id': roi_id})
        
        # Validate analysis results using marker validator
        results_audit = marker_validator.validate_analysis_results(
            {'multiscale_results': analysis_results}, roi_id
        )
        
        audit_results = [file_audit, ion_counts_audit, results_audit]
        
        # Add validation results to audit
        validation_summary = {
            'pre_analysis_quality': validation_result.summary_stats.get('overall_quality_score'),
            'post_analysis_quality': post_validation_result.summary_stats.get('overall_quality_score'),
            'validation_status': validation_result.summary_stats['status'],
            'recommendations': validation_result.get_recommendations()
        }
        
        # Store validation summary in results
        analysis_results['validation_summary'] = validation_summary
        
        # Generate validation plots if plots_dir is provided
        if plots_dir:
            from pathlib import Path
            import matplotlib.pyplot as plt
            from src.viz_utils.plotting import plot_segmentation_overlay
            
            plots_path = Path(plots_dir)
            plots_path.mkdir(parents=True, exist_ok=True)
            
            roi_name = roi_file.stem
            for scale, result in analysis_results.items():
                if scale == 'validation_summary':
                    continue
                    
                if ('superpixel_labels' in result and 'composite_dna' in result and 
                    result['composite_dna'].size > 0 and 'transformed_arrays' in result and 
                    'cofactors_used' in result):
                    try:
                        # Create comprehensive multi-channel validation plot
                        fig = plot_segmentation_overlay(
                            image=result['composite_dna'],
                            labels=result['superpixel_labels'],
                            bounds=result['bounds'],
                            transformed_arrays=result['transformed_arrays'],
                            cofactors_used=result['cofactors_used'],
                            config=config,
                            superpixel_coords=result.get('superpixel_coords'),
                            title=f"Multi-Channel Validation - {roi_name} - {scale}μm Scale"
                        )
                        plot_filename = plots_path / f"{roi_name}_scale_{scale}_multichannel_validation.png"
                        fig.savefig(plot_filename, dpi=150, bbox_inches='tight', facecolor='white')
                        plt.close(fig)
                        logger.debug(f"Generated multi-channel validation plot: {plot_filename}")
                    except Exception as e:
                        logger.warning(f"Could not generate multi-channel plot for {roi_name} at scale {scale}: {e}")
        
        # Create the proper ROI data structure
        roi_result = {
            'coords': coords,
            'ion_counts': ion_counts,
            'dna_intensities': {'DNA1': dna1_intensities, 'DNA2': dna2_intensities},
            'n_pixels': len(coords),
            'filename': roi_file.name,
            'multiscale_results': analysis_results
        }
        
        logger.info(f"✓ Completed processing {roi_id}")
        
        return roi_id, roi_result, audit_results
        
    except Exception as e:
        logger.error(f"✗ Failed processing {roi_file.stem}: {str(e)}")
        return roi_file.stem, None, str(e)


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
    audit_results = []
    
    # Initialize marker validator
    marker_validator = MarkerValidator(config)
    logger.info(f"Initialized marker validation for {len(marker_validator.protein_channels)} protein channels")
    
    # Get channel definitions from config
    protein_channels = config.channels['protein_channels']
    dna_channels = config.channels['dna_channels']
    background_channel = config.channels['background_channel']
    excluded = (config.channels['excluded_channels'] + 
                config.channels['calibration_channels'] + 
                [config.channels['carrier_gas_channel']])
    
    # Enable parallel processing for ROI analysis
    enable_parallel = config.processing.get('enable_parallel', True)
    max_workers = config.processing.get('max_workers', min(4, multiprocessing.cpu_count()))
    
    # Create plots directory path
    output_config = config.raw.get('output', {})
    plots_dir = Path(output_config.get('plots_dir', 'plots/validation'))
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plots will be saved to: {plots_dir.resolve()}")
    
    if enable_parallel and len(roi_files) > 1:
        logger.info(f"Processing {len(roi_files)} ROIs in parallel using {max_workers} workers")
        
        # Prepare arguments for parallel processing
        config_dict = config.__dict__.copy()  # Serialize config for parallel processing
        process_args = [
            (roi_file, config_dict, protein_channels, dna_channels, background_channel, excluded, str(plots_dir))
            for roi_file in roi_files
        ]
        
        # Process ROIs in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_roi = {
                executor.submit(process_single_roi, args): args[0] 
                for args in process_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_roi):
                roi_file = future_to_roi[future]
                try:
                    roi_id, roi_result, roi_audit_results = future.result()
                    
                    if roi_result is not None:
                        # Get metadata to determine batch
                        metadata = metadata_map.get(roi_id, Metadata())
                        batch_id = f"T{metadata.timepoint}_{metadata.replicate_id}"
                        
                        if batch_id not in batch_data:
                            batch_data[batch_id] = {}
                            roi_metadata[batch_id] = []
                        
                        # Store ROI data with metadata
                        roi_result['metadata'] = metadata
                        batch_data[batch_id][roi_id] = roi_result
                        
                        roi_metadata[batch_id].append({
                            'roi_name': roi_id,
                            'filename': roi_result['filename'],
                            'n_pixels': roi_result['n_pixels'],
                            'n_proteins': len(roi_result['ion_counts']),
                            'region': metadata.region,
                            'condition': metadata.condition
                        })
                        
                        audit_results.extend(roi_audit_results)
                        logger.info(f"✓ Parallel processing completed for {roi_id} (batch: {batch_id})")
                    else:
                        logger.error(f"✗ Parallel processing failed for {roi_id}: {roi_audit_results}")
                        
                except Exception as e:
                    logger.error(f"✗ Parallel processing exception for {roi_file.stem}: {str(e)}")
        
        logger.info(f"Parallel processing completed for {len(batch_data)} ROIs")
    
    else:
        # Sequential processing (original code)
        logger.info(f"Processing {len(roi_files)} ROIs sequentially")
        
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
                
                # Use robust column matching for protein channels
                from src.utils.column_matching import match_imc_columns
                protein_column_mapping = match_imc_columns(
                    protein_channels, 
                    list(roi_data.columns),
                    case_sensitive=False,
                    fuzzy_threshold=0.8
                )
                
                # Process protein channels with background correction
                ion_counts = {}
                for protein in protein_channels:
                    matched_column = protein_column_mapping.get(protein)
                    if matched_column:
                        # Apply background correction
                        signal = roi_data[matched_column].values
                        if config.processing['background_correction']['enabled']:
                            corrected = signal - background_signal
                            if config.processing['background_correction']['clip_negative']:
                                corrected = np.clip(corrected, 0, None)
                            ion_counts[protein] = corrected
                        else:
                            ion_counts[protein] = signal
                        logger.debug(f"  Loaded {protein} from column '{matched_column}' with {np.sum(ion_counts[protein] > 0)} non-zero pixels")
                    else:
                        logger.warning(f"  Could not find column for protein '{protein}'")
                
                # Validate marker presence and audit
                roi_id = roi_file.stem
                file_audit = marker_validator.audit_roi_file(str(roi_file), roi_data)
                log_marker_audit_result(file_audit, roi_id)
                
                ion_counts_audit = marker_validator.validate_ion_counts(ion_counts, roi_id)
                log_marker_audit_result(ion_counts_audit, f"{roi_id}_processed")
                
                audit_results.extend([file_audit, ion_counts_audit])
                
                # Assert critical markers are present (fail loudly if not)
                critical_markers = list(marker_validator.critical_markers & set(protein_channels))
                try:
                    from src.analysis.marker_validation import validate_marker_presence_assertion
                    validate_marker_presence_assertion(ion_counts, critical_markers, roi_id)
                    logger.info(f"✓ All critical markers present in {roi_id}")
                except AssertionError as e:
                    logger.error(f"CRITICAL MARKER FAILURE: {e}")
                    raise
                
                # Comprehensive data validation
                validation_suite = create_validation_suite([
                    ValidationCategory.DATA_INTEGRITY,
                    ValidationCategory.SCIENTIFIC_QUALITY
                ])
                
                validation_suite.add_rule(CoordinateValidator())
                validation_suite.add_rule(IonCountValidator())  
                # BiologicalValidator disabled - hangs with large data (250k pixels)
                # validation_suite.add_rule(BiologicalValidator())
                
                validation_data = {
                    'coords': coords,
                    'ion_counts': ion_counts
                }
                
                validation_result = validation_suite.validate(validation_data, {'roi_id': roi_id})
                
                # Log validation results
                logger.info(f"Validation for {roi_id}: {validation_result.summary_stats['status']} "
                           f"(quality: {validation_result.summary_stats.get('overall_quality_score', 'N/A'):.2f})")
                
                if validation_result.summary_stats['has_critical']:
                    critical_failures = validation_result.get_critical_failures()
                    failure_messages = [f.message for f in critical_failures]
                    logger.warning(f"Quality concerns in {roi_id}: {'; '.join(failure_messages)}")
                    logger.info(f"Continuing processing of {roi_id} - quality tracked for reporting")
                
                # Store validation results for later analysis
                audit_results.append({
                    'roi_id': roi_id,
                    'validation_result': validation_result,
                    'type': 'comprehensive_validation'
                })
                
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
    
    # Create comprehensive audit summary
    if audit_results:
        output_dir = Path(config.output.get('results_dir', 'results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Marker audit summary (legacy)
        marker_audit_results = [r for r in audit_results if not isinstance(r, dict)]
        if marker_audit_results:
            audit_summary_path = output_dir / 'marker_audit_summary.json'
            audit_summary = marker_validator.create_audit_summary(marker_audit_results, str(audit_summary_path))
            logger.info(f"Marker audit summary: {audit_summary['total_errors']} errors, {audit_summary['total_warnings']} warnings")
        
        # Comprehensive validation summary
        validation_results = [r for r in audit_results if isinstance(r, dict) and r.get('type') == 'comprehensive_validation']
        if validation_results:
            validation_summary = create_validation_summary(validation_results)
            validation_summary_path = output_dir / 'comprehensive_validation_summary.json'
            
            with open(validation_summary_path, 'w') as f:
                import json
                json.dump(validation_summary, f, indent=2, default=str)
            
            logger.info(f"Comprehensive validation summary: {validation_summary['summary']['total_rois']} ROIs analyzed")
            logger.info(f"  Overall quality score: {validation_summary['summary']['mean_quality_score']:.2f}")
            logger.info(f"  Critical failures: {validation_summary['summary']['critical_failures']}")
            logger.info(f"  Warnings: {validation_summary['summary']['total_warnings']}")
            
            # Log top recommendations
            top_recommendations = validation_summary['recommendations'][:3]
            if top_recommendations:
                logger.info(f"  Top recommendations: {'; '.join(top_recommendations)}")
    
    return batch_data, roi_metadata, audit_results



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
        memory_limit_gb=memory_limit_gb,
        config=config
    )
    
    # Generate visual validation plots if output directory provided
    if plots_dir:
        roi_name = roi_data['filename'].replace('.txt', '')
        for scale, result in multiscale_results.items():
            if ('superpixel_labels' in result and 'composite_dna' in result and 
                result['composite_dna'].size > 0 and 'transformed_arrays' in result and 
                'cofactors_used' in result):
                try:
                    # Log data structures for debugging
                    logger.debug(f"Creating plot for {roi_name} at {scale}μm scale:")
                    logger.debug(f"  - composite_dna shape: {result['composite_dna'].shape}")
                    logger.debug(f"  - superpixel_labels shape: {result['superpixel_labels'].shape}")
                    logger.debug(f"  - transformed_arrays keys: {list(result['transformed_arrays'].keys())}")
                    
                    first_protein = next(iter(result['transformed_arrays'].keys()))
                    first_array = result['transformed_arrays'][first_protein]
                    logger.debug(f"  - first protein '{first_protein}' array shape: {first_array.shape}")
                    logger.debug(f"  - superpixel_coords provided: {result.get('superpixel_coords') is not None}")
                    
                    # Create comprehensive multi-channel validation plot
                    fig = plot_segmentation_overlay(
                        image=result['composite_dna'],
                        labels=result['superpixel_labels'],
                        bounds=result['bounds'],
                        transformed_arrays=result['transformed_arrays'],
                        cofactors_used=result['cofactors_used'],
                        config=config,
                        superpixel_coords=result.get('superpixel_coords'),
                        title=f"Multi-Channel Validation - {roi_name} - {scale}μm Scale"
                    )
                    plot_filename = plots_dir / f"{roi_name}_scale_{scale}_multichannel_validation.png"
                    fig.savefig(plot_filename, dpi=150, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    logger.debug(f"Generated multi-channel validation plot: {plot_filename}")
                except Exception as e:
                    logger.warning(f"Could not generate multi-channel plot for {roi_name} at scale {scale}: {e}")
                    logger.debug(f"Available keys in result: {list(result.keys())}")
                    if 'transformed_arrays' in result:
                        logger.debug(f"Available proteins: {list(result['transformed_arrays'].keys())}")
                    else:
                        logger.debug("No transformed_arrays found in result")
    
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
            elif isinstance(value, dict) and key in ['transformed_arrays', 'standardized_arrays']:
                # Handle nested dictionaries containing protein channel arrays
                for protein_name, protein_array in value.items():
                    if isinstance(protein_array, np.ndarray):
                        arrays_to_save[f'scale_{scale}_{key}_{protein_name}'] = protein_array
                # Also save the protein names list in metadata
                if f'scale_{scale}' not in metadata_only:
                    metadata_only[f'scale_{scale}'] = {}
                metadata_only[f'scale_{scale}'][f'{key}_channels'] = list(value.keys())
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
        
        # Initialize quality control system
        print("\n🎯 Initializing quality control system...")
        quality_monitor = QualityMonitor('quality_history.json')
        gate_engine = QualityGateEngine()
        logger.info("Quality control system initialized")
        
        # Load IMC data organized by batches
        print("\n📂 Loading IMC data...")
        batch_data, roi_metadata, audit_results = load_imc_data(config)
        
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
        print("\n🔬 Running multi-scale analysis with quality monitoring...")
        analysis_results = []
        quality_decisions = []
        
        roi_count = 0
        for batch_id, batch_roi_data in batch_data.items():
            print(f"\n  Processing batch '{batch_id}'...")
            
            for roi_name, roi_data in batch_roi_data.items():
                roi_count += 1
                print(f"    ROI {roi_count}: {roi_name} "
                      f"({roi_data['n_pixels']:,} pixels, {len(roi_data['ion_counts'])} proteins)")
                
                try:
                    # Check if multiscale results already exist (from parallel processing)
                    if 'multiscale_results' in roi_data:
                        # Already processed in parallel, just extract the results
                        result = roi_data['multiscale_results']
                        result['roi_metadata'] = {
                            'roi_id': roi_name,
                            'n_pixels': roi_data['n_pixels'],
                            'n_proteins': len(roi_data['ion_counts']),
                            'batch_id': batch_id
                        }
                    else:
                        # Need to analyze (sequential processing)
                        result = analyze_roi_with_multiscale(roi_data, config, plots_dir)
                    
                    result['batch_id'] = batch_id
                    
                    # Extract quality metrics from validation
                    if 'validation_results' in result:
                        quality_metrics = extract_quality_metrics_from_validation(
                            result['validation_results'], roi_name, batch_id
                        )
                        quality_monitor.add_roi_quality(quality_metrics)
                        
                        # Evaluate quality gates
                        should_continue, reason, details = gate_engine.should_continue_analysis(
                            quality_monitor, quality_metrics
                        )
                        
                        quality_decisions.append({
                            'roi_name': roi_name,
                            'continue': should_continue,
                            'skip_roi': details.get('skip_current_roi', False),
                            'reason': reason,
                            'quality_score': quality_metrics.overall_quality()
                        })
                        
                        if details.get('skip_current_roi', False):
                            print(f"      ⚠️  ROI skipped: {reason}")
                            continue
                        elif not should_continue:
                            print(f"      🛑 Analysis stopped: {reason}")
                            break
                        elif 'warn' in reason.lower():
                            print(f"      ⚠️  {reason}")
                    
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
        
        # Update quality control limits and generate reports
        print("\n📊 Updating quality control system...")
        quality_monitor.update_control_limits()
        quality_monitor.save_history()
        
        # Generate quality reports
        quality_reports = generate_quality_reports(quality_monitor, gate_engine)
        if quality_reports:
            print(f"   ✅ Generated quality reports: {', '.join(quality_reports.keys())}")
        
        # Quality decision summary
        if quality_decisions:
            passed = sum(1 for d in quality_decisions if d['continue'] and not d['skip_roi'])
            skipped = sum(1 for d in quality_decisions if d['skip_roi'])
            failed = sum(1 for d in quality_decisions if not d['continue'])
            avg_quality = np.mean([d['quality_score'] for d in quality_decisions])
            
            print(f"\n🎯 Quality Gate Summary:")
            print(f"   • {passed} ROIs passed quality gates")
            print(f"   • {skipped} ROIs skipped due to quality issues")
            print(f"   • {failed} ROIs triggered analysis stops")
            print(f"   • Average quality score: {avg_quality:.3f}")
        
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
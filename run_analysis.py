#!/usr/bin/env python3
"""
IMC Spatial Analysis Pipeline
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.core.data_structures import IMCData
from src.core.pipeline import AnalysisPipelineBuilder
from src.processing.preprocessing import DataLoader, QualityControl, Normalizer
from src.analysis import (
    TissueSegmenter, FeatureExtractor, BiologicalAnnotator,
    SegmentClusterer, SpatialAnalyzer, ValidationSuite
)
from src.visualization import IMCPlotFactory
from src.utils.helpers import find_roi_files


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_roi_data(config: Config) -> List[IMCData]:
    """Load all ROI data from the data directory."""
    logger = logging.getLogger('DataLoader')
    
    # Find ROI files
    data_dir = Path('data/241218_IMC_Alun')
    roi_files = find_roi_files(data_dir)
    
    logger.info(f"Found {len(roi_files)} ROI files")
    
    # Load data
    loader = DataLoader(config.to_dict())
    roi_data = []
    
    for roi_file in roi_files:
        try:
            imc_data = loader.load_roi(roi_file)
            roi_data.append(imc_data)
            logger.info(f"Loaded {roi_file.name}: {imc_data.n_pixels} pixels, {imc_data.n_markers} markers")
        except Exception as e:
            logger.warning(f"Failed to load {roi_file}: {e}")
    
    logger.info(f"Successfully loaded {len(roi_data)} ROIs")
    return roi_data


def analyze_single_roi(imc_data: IMCData, config: Config) -> Dict[str, Any]:
    """Analyze a single ROI using the modular pipeline."""
    logger = logging.getLogger('ROIAnalysis')
    
    # Preprocessing
    normalizer = Normalizer(config.to_dict())
    imc_data_norm = normalizer.arcsinh_transform(imc_data, cofactor=5.0)
    
    # Segmentation
    segmenter = TissueSegmenter(config.tissue_analysis['superpixel'])
    seg_result = segmenter.segment_tissue(imc_data_norm)
    logger.info(f"Segmentation: {seg_result.n_segments} segments")
    
    # Feature extraction
    feature_extractor = FeatureExtractor({'feature_types': ['statistical', 'morphological', 'spatial']})
    feature_result = feature_extractor.extract_features(imc_data_norm)
    logger.info(f"Feature extraction: {feature_result.n_segments} segments, {len(feature_result.features.columns)} features")
    
    # Biological annotation
    annotator = BiologicalAnnotator(config.to_dict())
    annotated_features = annotator.annotate_segments(feature_result.features, imc_data.marker_names)
    
    # Clustering
    clusterer = SegmentClusterer({'method': 'kmeans', 'n_clusters': 'auto'})
    cluster_result = clusterer.cluster_segments(annotated_features, imc_data.marker_names)
    logger.info(f"Clustering: {cluster_result.n_clusters} domains identified")
    
    # Spatial analysis
    spatial_analyzer = SpatialAnalyzer({'n_permutations': 1000})
    spatial_result = spatial_analyzer.analyze_spatial_organization(imc_data_norm)
    
    # Package results
    return {
        'metadata': imc_data.metadata,
        'segmentation': {
            'n_segments': seg_result.n_segments,
            'method': seg_result.method,
            'parameters': seg_result.parameters
        },
        'features': annotated_features,
        'clustering': {
            'labels': cluster_result.labels,
            'n_clusters': cluster_result.n_clusters,
            'cluster_names': cluster_result.cluster_names,
            'method': cluster_result.method
        },
        'spatial_analysis': {
            'spatial_autocorrelation': spatial_result.spatial_autocorrelation,
            'p_values': spatial_result.p_values,
            'test_method': spatial_result.test_method
        },
        'imc_data': imc_data_norm  # For visualization
    }


def run_validation(roi_data: List[IMCData], config: Config) -> Dict[str, Any]:
    """Run validation suite."""
    logger = logging.getLogger('Validation')
    
    if not roi_data:
        return {'error': 'No ROI data for validation'}
    
    # Use first ROI for parameter validation
    test_roi = roi_data[0]
    
    # Prepare ROI data list with metadata for cross-validation
    roi_data_with_metadata = [(roi, roi.metadata) for roi in roi_data]
    
    # Run validation
    validator = ValidationSuite(config.to_dict())
    validation_results = validator.run_full_validation(test_roi, roi_data_with_metadata)
    
    logger.info(f"Validation complete: {validation_results['summary']['n_passed']}/{validation_results['summary']['n_tests']} tests passed")
    
    return validation_results


def create_visualizations(analysis_results: List[Dict[str, Any]], 
                         validation_results: Dict[str, Any],
                         config: Config) -> List[Path]:
    """Create all visualization figures."""
    logger = logging.getLogger('Visualization')
    
    plot_factory = IMCPlotFactory(config.to_dict())
    saved_figures = []
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For each ROI, create basic visualizations
    for i, result in enumerate(analysis_results[:3]):  # Limit to first 3 ROIs for demo
        imc_data = result['imc_data']
        features = result['features']
        cluster_labels = result['clustering']['labels']
        cluster_names = result['clustering']['cluster_names']
        spatial_analysis = result['spatial_analysis']
        
        roi_name = result['metadata'].get('filename', f'roi_{i}')
        
        # Segmentation overview
        fig = plot_factory.create_segmentation_overview(imc_data)
        fig_path = plot_factory.save_figure(fig, f'{roi_name}_segmentation.png', output_dir)
        saved_figures.append(fig_path)
        plt.close(fig)
        
        # Feature heatmap
        fig = plot_factory.create_feature_heatmap(features, imc_data.marker_names)
        fig_path = plot_factory.save_figure(fig, f'{roi_name}_features.png', output_dir)
        saved_figures.append(fig_path)
        plt.close(fig)
        
        # Clustering results
        fig = plot_factory.create_clustering_results(features, cluster_labels, 
                                                    cluster_names, imc_data.marker_names)
        fig_path = plot_factory.save_figure(fig, f'{roi_name}_clustering.png', output_dir)
        saved_figures.append(fig_path)
        plt.close(fig)
        
        # Spatial analysis
        fig = plot_factory.create_spatial_analysis_results(spatial_analysis)
        fig_path = plot_factory.save_figure(fig, f'{roi_name}_spatial.png', output_dir)
        saved_figures.append(fig_path)
        plt.close(fig)
    
    # Validation summary
    if validation_results and 'validation_results' in validation_results:
        fig = plot_factory.create_validation_summary(validation_results)
        fig_path = plot_factory.save_figure(fig, 'validation_summary.png', output_dir)
        saved_figures.append(fig_path)
        plt.close(fig)
    
    logger.info(f"Created {len(saved_figures)} visualization figures")
    return saved_figures


def main():
    """Run the complete IMC analysis pipeline."""
    setup_logging()
    logger = logging.getLogger('Main')
    
    print("=" * 60)
    print("IMC Spatial Analysis Pipeline")
    print("=" * 60)
    
    # Load configuration
    config = Config('config.json')
    logger.info("Configuration loaded")
    
    # Load ROI data
    print("\n📂 Loading ROI data...")
    roi_data = load_roi_data(config)
    
    if not roi_data:
        print("❌ No ROI data loaded. Exiting.")
        return
    
    print(f"✅ Loaded {len(roi_data)} ROIs")
    
    # Analyze ROIs
    print("\n🔬 Running analysis pipeline...")
    analysis_results = []
    
    for i, imc_data in enumerate(roi_data[:3]):  # Analyze first 3 ROIs for demo
        print(f"  Analyzing ROI {i+1}/{min(3, len(roi_data))}...")
        try:
            result = analyze_single_roi(imc_data, config)
            analysis_results.append(result)
        except Exception as e:
            logger.error(f"Failed to analyze ROI {i+1}: {e}")
            continue
    
    if not analysis_results:
        print("❌ No successful analyses. Exiting.")
        return
    
    print(f"✅ Analyzed {len(analysis_results)} ROIs successfully")
    
    # Run validation
    print("\n🔍 Running validation...")
    validation_results = run_validation(roi_data, config)
    
    if 'error' not in validation_results:
        overall_score = validation_results['summary']['overall_score']
        print(f"✅ Validation complete: Overall score {overall_score:.3f}")
    else:
        print(f"⚠️  Validation limited: {validation_results['error']}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    saved_figures = create_visualizations(analysis_results, validation_results, config)
    print(f"✅ Created {len(saved_figures)} figures in {config.output_dir}/")
    
    # Summary
    print("\n" + "=" * 60)
    print("Analysis Summary:")
    print(f"  • ROIs processed: {len(analysis_results)}")
    print(f"  • Markers analyzed: {analysis_results[0]['imc_data'].n_markers}")
    print(f"  • Average segments per ROI: {np.mean([r['segmentation']['n_segments'] for r in analysis_results]):.0f}")
    print(f"  • Average domains per ROI: {np.mean([r['clustering']['n_clusters'] for r in analysis_results]):.0f}")
    
    if 'error' not in validation_results:
        print(f"  • Validation score: {validation_results['summary']['overall_score']:.3f}")
    
    print(f"  • Figures saved: {len(saved_figures)}")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    main()
"""
Integration Tests for IMC Analysis Pipeline

Tests the complete analysis workflow from raw data to results.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from src.analysis.main_pipeline import (
    IMCAnalysisPipeline,
    run_complete_analysis
)
from types import SimpleNamespace


class TestIMCAnalysisPipeline:
    """Test the main analysis pipeline."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a minimal test configuration using SimpleNamespace."""
        # Create nested config structure that matches what pipeline expects
        return SimpleNamespace(
            multiscale=SimpleNamespace(
                scales_um=[10.0, 20.0, 40.0],
                enable_scale_analysis=True
            ),
            slic=SimpleNamespace(
                use_slic=True,
                compactness=10.0,
                sigma=2.0
            ),
            clustering=SimpleNamespace(
                optimization_method="comprehensive",
                k_range=[2, 8]
            ),
            storage=SimpleNamespace(
                format="hdf5",
                compression=True
            ),
            normalization=SimpleNamespace(
                method="arcsinh",
                cofactor=1.0
            ),
            # Add to_dict method for compatibility
            to_dict=lambda: {
                "multiscale": {"scales_um": [10.0, 20.0, 40.0], "enable_scale_analysis": True},
                "slic": {"use_slic": True, "compactness": 10.0, "sigma": 2.0},
                "clustering": {"optimization_method": "comprehensive", "k_range": [2, 8]},
                "storage": {"format": "hdf5", "compression": True},
                "normalization": {"method": "arcsinh", "cofactor": 1.0}
            }
        )
    
    @pytest.fixture
    def sample_roi_data(self):
        """Generate sample ROI data."""
        np.random.seed(42)
        n_points = 1000
        
        return {
            'coords': np.random.uniform(0, 100, (n_points, 2)),
            'ion_counts': {
                'CD45': np.random.negative_binomial(5, 0.3, n_points).astype(float),
                'CD31': np.random.negative_binomial(10, 0.5, n_points).astype(float),
                'CD11b': np.random.negative_binomial(3, 0.2, n_points).astype(float)
            },
            'dna1_intensities': np.random.poisson(800, n_points).astype(float),
            'dna2_intensities': np.random.poisson(600, n_points).astype(float),
            'protein_names': ['CD45', 'CD31', 'CD11b'],
            'n_measurements': n_points
        }
    
    def test_pipeline_initialization(self, mock_config):
        """Test pipeline initialization."""
        pipeline = IMCAnalysisPipeline(mock_config)
        
        assert pipeline.analysis_config == mock_config
        assert pipeline.results == {}
        assert pipeline.validation_results == {}
    
    def test_load_roi_data(self, mock_config):
        """Test ROI data loading."""
        pipeline = IMCAnalysisPipeline(mock_config)
        
        # Create temporary ROI file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write mock IMC data
            df = pd.DataFrame({
                'X': [10, 20, 30],
                'Y': [15, 25, 35],
                'CD45(Sm149Di)': [100, 200, 150],
                'CD31(Nd145Di)': [50, 300, 100],
                'DNA1(Ir191Di)': [800, 900, 850],
                'DNA2(Ir193Di)': [600, 650, 620]
            })
            df.to_csv(f.name, sep='\t', index=False)
            
            # Load data
            roi_data = pipeline.load_roi_data(f.name, ['CD45', 'CD31'])
            
        # Clean up
        Path(f.name).unlink()
        
        assert roi_data['coords'].shape == (3, 2)
        assert 'CD45' in roi_data['ion_counts']
        assert 'CD31' in roi_data['ion_counts']
        assert len(roi_data['dna1_intensities']) == 3
        assert roi_data['n_measurements'] == 3
    
    def test_analyze_single_roi(self, mock_config, sample_roi_data):
        """Test single ROI analysis."""
        pipeline = IMCAnalysisPipeline(mock_config)
        
        results = pipeline.analyze_single_roi(sample_roi_data)
        
        assert 'multiscale_results' in results
        assert 'consistency_results' in results
        assert 'configuration_used' in results
        assert 'metadata' in results
        
        # Check multiscale results
        for scale in mock_config.multiscale.scales_um:
            assert scale in results['multiscale_results']
            scale_result = results['multiscale_results'][scale]
            assert 'cluster_labels' in scale_result
            assert 'feature_matrix' in scale_result
    
    def test_analyze_single_roi_empty(self, mock_config):
        """Test handling of empty ROI data."""
        pipeline = IMCAnalysisPipeline(mock_config)
        
        empty_roi_data = {
            'coords': np.array([]),
            'ion_counts': {},
            'dna1_intensities': np.array([]),
            'dna2_intensities': np.array([]),
            'protein_names': [],
            'n_measurements': 0
        }
        
        # Should handle empty data gracefully
        results = pipeline.analyze_single_roi(empty_roi_data)
        assert results is not None
    
    @patch('src.analysis.main_pipeline.create_roi_batch_processor')
    @patch('src.analysis.main_pipeline.create_storage_backend')
    def test_run_batch_analysis(self, mock_storage, mock_processor, mock_config):
        """Test batch analysis of multiple ROIs."""
        pipeline = IMCAnalysisPipeline(mock_config)
        
        # Create mock batch processor
        mock_batch_fn = Mock()
        mock_batch_fn.return_value = ({'roi1': {}, 'roi2': {}}, [])
        mock_processor.return_value = mock_batch_fn
        
        # Create temporary ROI files
        roi_files = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                df = pd.DataFrame({
                    'X': [10, 20],
                    'Y': [15, 25],
                    'CD45(Sm149Di)': [100, 200],
                    'DNA1(Ir191Di)': [800, 900],
                    'DNA2(Ir193Di)': [600, 650]
                })
                df.to_csv(f.name, sep='\t', index=False)
                roi_files.append(f.name)
        
        # Run batch analysis
        with tempfile.TemporaryDirectory() as tmpdir:
            results, errors = pipeline.run_batch_analysis(
                roi_file_paths=roi_files,
                protein_names=['CD45'],
                output_dir=tmpdir
            )
        
        # Clean up
        for f in roi_files:
            Path(f).unlink()
        
        assert mock_batch_fn.called
        assert len(results) == 2
        assert len(errors) == 0
    
    def test_run_validation_study(self, mock_config, sample_roi_data):
        """Test validation study execution."""
        pipeline = IMCAnalysisPipeline(mock_config)
        
        # Create mock analysis results
        analysis_results = [{
            'multiscale_results': {
                10.0: {
                    'superpixel_labels': np.random.randint(0, 5, (50, 50)),
                    'composite_dna': np.random.rand(50, 50)
                },
                20.0: {
                    'superpixel_labels': np.random.randint(0, 3, (25, 25)),
                    'composite_dna': np.random.rand(25, 25)
                }
            }
        }]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            validation_summary = pipeline.run_validation_study(
                analysis_results, 
                output_dir=tmpdir
            )
        
        assert 'method' in validation_summary
        assert 'n_rois' in validation_summary
        assert 'scale_validations' in validation_summary
    
    def test_generate_summary_report(self, mock_config):
        """Test summary report generation."""
        pipeline = IMCAnalysisPipeline(mock_config)
        
        # Create mock results
        results = {
            'roi1': {
                'consistency_results': {
                    'overall': {
                        'mean_ari': 0.75,
                        'mean_nmi': 0.80,
                        'mean_centroid_distance': 5.2
                    }
                },
                'metadata': {
                    'n_measurements': 1000,
                    'scales_analyzed': [10.0, 20.0, 40.0]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            summary = pipeline.generate_summary_report(results, f.name)
            
        # Check summary structure
        assert 'experiment_metadata' in summary
        assert 'scale_consistency_summary' in summary
        assert 'roi_summaries' in summary
        
        # Clean up
        Path(f.name).unlink()


class TestMultiScaleIntegration:
    """Test multi-scale analysis integration."""
    
    def test_scale_consistency(self):
        """Test consistency metrics between scales."""
        from src.analysis.multiscale_analysis import compute_scale_consistency
        
        # Create mock multiscale results
        np.random.seed(42)
        multiscale_results = {
            10.0: {
                'cluster_labels': np.random.randint(0, 8, 100),
                'centroids': np.random.rand(100, 2)
            },
            20.0: {
                'cluster_labels': np.random.randint(0, 6, 100),
                'centroids': np.random.rand(100, 2)
            },
            40.0: {
                'cluster_labels': np.random.randint(0, 4, 100),
                'centroids': np.random.rand(100, 2)
            }
        }
        
        consistency = compute_scale_consistency(multiscale_results)
        
        assert 'pairwise' in consistency
        assert 'overall' in consistency
        
        # Check pairwise comparisons exist
        assert (10.0, 20.0) in consistency['pairwise']
        assert (20.0, 40.0) in consistency['pairwise']
        
        # Check overall metrics
        assert 'mean_ari' in consistency['overall']
        assert 0 <= consistency['overall']['mean_ari'] <= 1


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_protein_channel(self, mock_config):
        """Test handling of missing protein channels."""
        pipeline = IMCAnalysisPipeline(mock_config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write data missing requested protein
            df = pd.DataFrame({
                'X': [10, 20],
                'Y': [15, 25],
                'DNA1(Ir191Di)': [800, 900],
                'DNA2(Ir193Di)': [600, 650]
            })
            df.to_csv(f.name, sep='\t', index=False)
            
            # Should handle missing protein gracefully
            with pytest.warns(UserWarning):
                roi_data = pipeline.load_roi_data(f.name, ['CD45_MISSING'])
            
            assert 'CD45_MISSING' in roi_data['ion_counts']
            assert np.all(roi_data['ion_counts']['CD45_MISSING'] == 0)
        
        Path(f.name).unlink()
    
    def test_invalid_file_path(self, mock_config):
        """Test handling of invalid file paths."""
        pipeline = IMCAnalysisPipeline(mock_config)
        
        with pytest.raises(ValueError, match="Failed to load ROI data"):
            pipeline.load_roi_data("nonexistent_file.txt", ['CD45'])
    
    def test_batch_analysis_no_valid_rois(self, mock_config):
        """Test batch analysis with no valid ROIs."""
        pipeline = IMCAnalysisPipeline(mock_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No valid ROI data loaded"):
                pipeline.run_batch_analysis(
                    roi_file_paths=["invalid1.txt", "invalid2.txt"],
                    protein_names=['CD45'],
                    output_dir=tmpdir
                )


class TestConfigurationIntegration:
    """Test configuration management integration."""
    
    def test_config_override(self, mock_config, sample_roi_data):
        """Test configuration override in analysis."""
        pipeline = IMCAnalysisPipeline(mock_config)
        
        # Run with default config
        results_default = pipeline.analyze_single_roi(sample_roi_data)
        
        # Run with override (though not fully implemented)
        override = {"multiscale": {"scales_um": [5.0, 15.0]}}
        results_override = pipeline.analyze_single_roi(
            sample_roi_data, 
            override_config=override
        )
        
        # Both should complete successfully
        assert results_default is not None
        assert results_override is not None
    
    def test_config_serialization(self, mock_config):
        """Test configuration serialization in results."""
        pipeline = IMCAnalysisPipeline(mock_config)
        
        sample_roi_data = {
            'coords': np.random.rand(100, 2),
            'ion_counts': {'CD45': np.random.rand(100)},
            'dna1_intensities': np.random.rand(100),
            'dna2_intensities': np.random.rand(100),
            'protein_names': ['CD45'],
            'n_measurements': 100
        }
        
        results = pipeline.analyze_single_roi(sample_roi_data)
        
        assert 'configuration_used' in results
        assert isinstance(results['configuration_used'], dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
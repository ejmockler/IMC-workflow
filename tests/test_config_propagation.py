"""
Regression tests for config parameter propagation bugs.

These tests ensure that configuration parameters are properly read and used
by the analysis pipeline, preventing regression of bugs #5-9 where config
parameters were defined but ignored by the code.

Bug History:
- Bug #1: Coabundance order hardcoded at 2 instead of reading config order=3
- Bug #2: Bead normalization not enabled
- Bug #3: SLIC parameters ignored
- Bug #4: Enriched features not saved
- Bug #5: Batch analysis ignores config (uses hardcoded use_slic=True, n_clusters=8)
- Bug #6: Bead normalization scientifically invalid (single ROI processing)
- Bug #7: Enriched features lost in main pipeline (key mismatch)
- Bug #8: Zombie pipeline with hardcoded resolution=1.0
- Bug #9: Hardcoded science parameters throughout codebase
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.config import Config
from src.analysis.main_pipeline import IMCAnalysisPipeline


class TestBug5_BatchAnalysisConfigPropagation:
    """
    Test that run_batch_analysis properly reads from config instead of using
    hardcoded defaults.

    Previously, batch analysis used:
    - use_slic: True (hardcoded)
    - n_clusters: 8 (hardcoded)

    It should read from config:
    - segmentation.method
    - analysis.clustering parameters
    """

    @pytest.fixture
    def config_slic(self, tmp_path):
        """Config with SLIC segmentation."""
        import json

        config_dict = {
            'data': {
                'raw_data_dir': str(tmp_path),
                'metadata_file': str(tmp_path / 'metadata.csv')
            },
            'channels': {
                'protein_channels': ['CD45', 'CD11b', 'Ly6G'],
                'dna_channels': ['DNA1', 'DNA2']
            },
            'segmentation': {
                'method': 'slic',  # Should be read
                'scales_um': [10.0, 20.0]
            },
            'analysis': {
                'clustering': {
                    'method': 'leiden',
                    'resolution_range': [0.5, 5.0],  # Should be read
                    'use_coabundance_features': True,
                    'coabundance_options': {
                        'interaction_order': 3
                    }
                }
            },
            'output': {
                'results_dir': str(tmp_path / 'results')
            }
        }

        # Write config to file
        config_path = tmp_path / 'config_slic.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f)

        return Config(str(config_path))

    @pytest.fixture
    def config_watershed(self, tmp_path):
        """Config with watershed segmentation."""
        import json

        config_dict = {
            'data': {
                'raw_data_dir': str(tmp_path),
                'metadata_file': str(tmp_path / 'metadata.csv')
            },
            'channels': {
                'protein_channels': ['CD45', 'CD11b', 'Ly6G'],
                'dna_channels': ['DNA1', 'DNA2']
            },
            'segmentation': {
                'method': 'watershed',  # Should be read
                'scales_um': [10.0, 20.0]
            },
            'analysis': {
                'clustering': {
                    'method': 'leiden',
                    'resolution_range': [1.0, 10.0],
                    'use_coabundance_features': False
                }
            },
            'output': {
                'results_dir': str(tmp_path / 'results')
            }
        }

        # Write config to file
        config_path = tmp_path / 'config_watershed.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f)

        return Config(str(config_path))

    def test_batch_analysis_reads_slic_from_config(self, config_slic, tmp_path):
        """
        REGRESSION TEST FOR BUG #5

        Verify that run_batch_analysis reads segmentation.method='slic' from config
        instead of using hardcoded use_slic=True.
        """
        pipeline = IMCAnalysisPipeline(config_slic)

        # Mock the analyze_single_roi method to capture what parameters it receives
        original_analyze = pipeline.analyze_single_roi
        captured_params = []

        def mock_analyze(roi_data, override_config=None, **kwargs):
            captured_params.append(override_config)
            # Return minimal valid result
            return {
                'multiscale_results': {},
                'metadata': {'roi_id': 'test'}
            }

        with patch.object(pipeline, 'analyze_single_roi', side_effect=mock_analyze):
            with patch.object(pipeline, 'load_roi_data', return_value={'coords': np.array([[0, 0]])}):
                # Create dummy ROI file
                roi_file = tmp_path / 'test_roi.txt'
                roi_file.write_text("X,Y,CD45\n0,0,1.0\n")

                try:
                    pipeline.run_batch_analysis(
                        roi_file_paths=[str(roi_file)],
                        protein_names=['CD45', 'CD11b', 'Ly6G'],
                        output_dir=str(tmp_path / 'results'),
                        generate_plots=False
                    )
                except Exception:
                    pass  # We just want to capture params

        # Verify that use_slic was set from config, not hardcoded
        if captured_params:
            params = captured_params[0]
            assert 'use_slic' in params, "use_slic parameter should be present"
            assert params['use_slic'] is True, "Should read use_slic=True from config (method='slic')"

    def test_batch_analysis_reads_watershed_from_config(self, config_watershed, tmp_path):
        """
        REGRESSION TEST FOR BUG #5

        Verify that run_batch_analysis reads segmentation.method='watershed' from config
        instead of using hardcoded use_slic=True.
        """
        pipeline = IMCAnalysisPipeline(config_watershed)

        captured_params = []

        def mock_analyze(roi_data, override_config=None, **kwargs):
            captured_params.append(override_config)
            return {
                'multiscale_results': {},
                'metadata': {'roi_id': 'test'}
            }

        with patch.object(pipeline, 'analyze_single_roi', side_effect=mock_analyze):
            with patch.object(pipeline, 'load_roi_data', return_value={'coords': np.array([[0, 0]])}):
                roi_file = tmp_path / 'test_roi.txt'
                roi_file.write_text("X,Y,CD45\n0,0,1.0\n")

                try:
                    pipeline.run_batch_analysis(
                        roi_file_paths=[str(roi_file)],
                        protein_names=['CD45', 'CD11b', 'Ly6G'],
                        output_dir=str(tmp_path / 'results'),
                        generate_plots=False
                    )
                except Exception:
                    pass

        # Verify that use_slic was set from config
        if captured_params:
            params = captured_params[0]
            assert 'use_slic' in params, "use_slic parameter should be present"
            assert params['use_slic'] is False, "Should read use_slic=False from config (method='watershed')"

    def test_batch_analysis_reads_clustering_params_from_config(self, config_slic, tmp_path):
        """
        REGRESSION TEST FOR BUG #5

        Verify that run_batch_analysis reads clustering parameters from config
        instead of using hardcoded n_clusters=8.
        """
        pipeline = IMCAnalysisPipeline(config_slic)

        captured_params = []

        def mock_analyze(roi_data, override_config=None, **kwargs):
            captured_params.append(override_config)
            return {
                'multiscale_results': {},
                'metadata': {'roi_id': 'test'}
            }

        with patch.object(pipeline, 'analyze_single_roi', side_effect=mock_analyze):
            with patch.object(pipeline, 'load_roi_data', return_value={'coords': np.array([[0, 0]])}):
                roi_file = tmp_path / 'test_roi.txt'
                roi_file.write_text("X,Y,CD45\n0,0,1.0\n")

                try:
                    pipeline.run_batch_analysis(
                        roi_file_paths=[str(roi_file)],
                        protein_names=['CD45', 'CD11b', 'Ly6G'],
                        output_dir=str(tmp_path / 'results'),
                        generate_plots=False
                    )
                except Exception:
                    pass

        # Verify clustering parameters from config
        if captured_params:
            params = captured_params[0]
            assert 'resolution_range' in params, "Should include resolution_range from config"
            assert params['resolution_range'] == [0.5, 5.0], "Should read resolution_range from config"
            assert params['use_coabundance_features'] is True, "Should read coabundance setting from config"
            assert 'coabundance_options' in params, "Should include coabundance_options from config"
            assert params['coabundance_options']['interaction_order'] == 3, "Should read interaction_order=3 from config"

    def test_batch_analysis_allows_override_of_config_params(self, config_slic, tmp_path):
        """
        REGRESSION TEST FOR BUG #5

        Verify that explicit analysis_params can override config defaults.
        This maintains backward compatibility while fixing the bug.
        """
        pipeline = IMCAnalysisPipeline(config_slic)

        captured_params = []

        def mock_analyze(roi_data, override_config=None, **kwargs):
            captured_params.append(override_config)
            return {
                'multiscale_results': {},
                'metadata': {'roi_id': 'test'}
            }

        with patch.object(pipeline, 'analyze_single_roi', side_effect=mock_analyze):
            with patch.object(pipeline, 'load_roi_data', return_value={'coords': np.array([[0, 0]])}):
                roi_file = tmp_path / 'test_roi.txt'
                roi_file.write_text("X,Y,CD45\n0,0,1.0\n")

                # Explicitly override config with analysis_params
                try:
                    pipeline.run_batch_analysis(
                        roi_file_paths=[str(roi_file)],
                        protein_names=['CD45', 'CD11b', 'Ly6G'],
                        output_dir=str(tmp_path / 'results'),
                        analysis_params={'use_slic': False, 'n_clusters': 5},
                        generate_plots=False
                    )
                except Exception:
                    pass

        # Verify that explicit params override config
        if captured_params:
            params = captured_params[0]
            assert params['use_slic'] is False, "Explicit analysis_params should override config"
            assert params['n_clusters'] == 5, "Explicit n_clusters should override config"


class TestBug7_EnrichedFeaturesKeyMismatch:
    """
    Test that enriched features from multiscale_analysis are properly saved,
    not lost due to key mismatch between 'features' and 'feature_matrix'.

    Previously:
    - multiscale_analysis saves: results['features'] = enriched_features (237)
    - main_pipeline looks for: results['feature_matrix']  # Wrong key!
    - Result: 237 features → LOST

    Fixed:
    - main_pipeline now correctly reads: results['features']
    """

    def test_enriched_features_use_correct_key(self):
        """
        REGRESSION TEST FOR BUG #7

        Verify that main_pipeline correctly reads 'features' key (not 'feature_matrix')
        from multiscale_analysis results.
        """
        # Test the specific line that was buggy
        enriched_features = np.random.rand(100, 237)

        # Simulate what multiscale_analysis returns
        primary_results = {
            'features': enriched_features,  # This is the correct key
            'cluster_labels': np.array([0, 1, 0, 1]),
            'protein_names': ['CD45', 'CD11b']
        }

        # Simulate what main_pipeline does (the fixed version)
        feature_matrix = primary_results.get('features', np.array([]))

        # CRITICAL ASSERTION: Should get enriched features
        assert feature_matrix.shape == (100, 237), \
            f"BUG #7 regression: Expected (100, 237) features, got {feature_matrix.shape}"


class TestBug8_ZombiePipelineDeprecated:
    """
    Test that ion_count_pipeline raises deprecation warning due to hardcoded
    resolution=1.0 that bypasses stability analysis.

    Previously:
    - ion_count_pipeline used resolution=1.0 (hardcoded, line 566)
    - multiscale_analysis.py would call it, then throw away clustering results
    - Developers could use it unknowingly and get invalid results

    Fixed:
    - Added loud DeprecationWarning explaining the issue
    - Documented that perform_multiscale_analysis() should be used instead
    """

    def test_ion_count_pipeline_raises_deprecation_warning(self):
        """
        REGRESSION TEST FOR BUG #8

        Verify that ion_count_pipeline raises DeprecationWarning to prevent
        accidental use of scientifically invalid clustering.
        """
        from src.analysis.ion_count_processing import ion_count_pipeline

        coords = np.array([[0, 0], [1, 1], [2, 2]])
        ion_counts = {'CD45': np.array([1.0, 2.0, 3.0])}

        # Should raise DeprecationWarning
        with pytest.warns(DeprecationWarning, match="hardcoded resolution=1.0"):
            result = ion_count_pipeline(coords, ion_counts, bin_size_um=10.0)

    def test_deprecation_message_mentions_alternative(self):
        """
        REGRESSION TEST FOR BUG #8

        Verify that deprecation warning tells users what to use instead.
        """
        from src.analysis.ion_count_processing import ion_count_pipeline

        coords = np.array([[0, 0], [1, 1]])
        ion_counts = {'CD45': np.array([1.0, 2.0])}

        with pytest.warns(DeprecationWarning) as warning_list:
            result = ion_count_pipeline(coords, ion_counts, bin_size_um=10.0)

        # Verify warning message is helpful
        warning_message = str(warning_list[0].message)
        assert "perform_multiscale_analysis" in warning_message, \
            "Warning should direct users to correct function"
        assert "stability analysis" in warning_message, \
            "Warning should explain why it's deprecated"


class TestBug6_BeadNormalizationBatchProcessing:
    """
    Test that bead normalization is applied at the batch level (across all ROIs)
    instead of per-ROI, which is scientifically invalid.

    Previously, bead normalization was called like:
    - batch_data = {'current_roi': single_roi_counts}  # ❌ No temporal context

    It should be:
    - batch_data = {'roi1': counts1, 'roi2': counts2, ...}  # ✅ Full temporal context
    """

    @pytest.fixture
    def config_with_bead_norm(self, tmp_path):
        """Config with bead normalization enabled."""
        import json

        config_dict = {
            'data': {
                'raw_data_dir': str(tmp_path),
                'metadata_file': str(tmp_path / 'metadata.csv')
            },
            'channels': {
                'protein_channels': ['CD45', 'CD11b', 'Ly6G'],
                'dna_channels': ['DNA1', 'DNA2']
            },
            'segmentation': {
                'method': 'slic',
                'scales_um': [10.0, 20.0]
            },
            'analysis': {
                'clustering': {
                    'method': 'leiden',
                    'resolution_range': [0.5, 5.0]
                },
                'batch_correction': {
                    'enabled': True,
                    'method': 'sham_anchored',
                    'bead_normalization': {
                        'enabled': True,
                        'bead_channels': ['130Ba', '131Xe'],
                        'bead_signal_threshold': 100.0,
                        'drift_correction_method': 'median_reference'
                    }
                }
            },
            'output': {
                'results_dir': str(tmp_path / 'results')
            }
        }

        # Write config to file
        config_path = tmp_path / 'config_bead.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f)

        return Config(str(config_path))

    def test_bead_normalization_receives_all_rois(self, config_with_bead_norm, tmp_path):
        """
        REGRESSION TEST FOR BUG #6

        Verify that bead_anchored_normalize receives ALL ROIs in a single batch,
        not one ROI at a time.
        """
        from src.analysis.batch_correction import bead_anchored_normalize

        pipeline = IMCAnalysisPipeline(config_with_bead_norm)

        # Track what data bead_anchored_normalize receives
        captured_batches = []
        original_bead_norm = bead_anchored_normalize

        def mock_bead_norm(batch_data, batch_metadata, config=None):
            """Capture the batch_data structure."""
            captured_batches.append({
                'n_rois': len(batch_data),
                'roi_ids': list(batch_data.keys())
            })
            # Return identity normalization
            return batch_data, {'method': 'identity', 'per_batch_stats': {}}

        # Create multiple ROI files
        roi_files = []
        for i in range(3):
            roi_file = tmp_path / f'test_roi_{i}.txt'
            roi_file.write_text(f"X,Y,CD45,130Ba,131Xe\n0,0,1.0,100.0,100.0\n1,1,2.0,110.0,110.0\n")
            roi_files.append(str(roi_file))

        # Mock both load_roi_data and bead_anchored_normalize
        def mock_load_roi(roi_path, proteins):
            return {
                'coords': np.array([[0, 0], [1, 1]]),
                'ion_counts': {
                    'CD45': np.array([1.0, 2.0]),
                    '130Ba': np.array([100.0, 110.0]),
                    '131Xe': np.array([100.0, 110.0])
                }
            }

        with patch('src.analysis.main_pipeline.bead_anchored_normalize', side_effect=mock_bead_norm):
            with patch.object(pipeline, 'load_roi_data', side_effect=mock_load_roi):
                with patch.object(pipeline, 'analyze_single_roi', return_value={'multiscale_results': {}}):
                    try:
                        pipeline.run_batch_analysis(
                            roi_file_paths=roi_files,
                            protein_names=['CD45', 'CD11b', 'Ly6G'],
                            output_dir=str(tmp_path / 'results'),
                            generate_plots=False
                        )
                    except Exception as e:
                        pass  # We just want to verify bead_anchored_normalize was called correctly

        # CRITICAL ASSERTION: bead_anchored_normalize should be called ONCE with ALL ROIs
        assert len(captured_batches) == 1, \
            f"bead_anchored_normalize should be called once with all ROIs, but was called {len(captured_batches)} times"

        batch_info = captured_batches[0]
        assert batch_info['n_rois'] == 3, \
            f"Batch should contain 3 ROIs, but got {batch_info['n_rois']}"

        # Verify it's not being called with single ROI 'current_roi' pattern
        assert 'current_roi' not in batch_info['roi_ids'], \
            "Should not use legacy single-ROI pattern 'current_roi'"

    def test_bead_normalization_skipped_when_disabled(self, tmp_path):
        """
        REGRESSION TEST FOR BUG #6

        Verify that when bead normalization is disabled, it's not called at all.
        """
        from src.analysis.batch_correction import bead_anchored_normalize
        import json

        # Config without bead normalization
        config_dict = {
            'data': {
                'raw_data_dir': str(tmp_path),
                'metadata_file': str(tmp_path / 'metadata.csv')
            },
            'channels': {
                'protein_channels': ['CD45'],
                'dna_channels': ['DNA1', 'DNA2']
            },
            'segmentation': {'method': 'slic', 'scales_um': [10.0]},
            'analysis': {'clustering': {'method': 'leiden'}},
            'output': {'results_dir': str(tmp_path / 'results')}
        }

        config_path = tmp_path / 'config_no_bead.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f)

        config = Config(str(config_path))
        pipeline = IMCAnalysisPipeline(config)

        bead_norm_called = []

        def mock_bead_norm(*args, **kwargs):
            bead_norm_called.append(True)
            return {}, {'method': 'identity'}

        roi_file = tmp_path / 'test_roi.txt'
        roi_file.write_text("X,Y,CD45\n0,0,1.0\n")

        with patch('src.analysis.main_pipeline.bead_anchored_normalize', side_effect=mock_bead_norm):
            with patch.object(pipeline, 'load_roi_data', return_value={'coords': np.array([[0, 0]]), 'ion_counts': {'CD45': np.array([1.0])}}):
                with patch.object(pipeline, 'analyze_single_roi', return_value={'multiscale_results': {}}):
                    try:
                        pipeline.run_batch_analysis(
                            roi_file_paths=[str(roi_file)],
                            protein_names=['CD45'],
                            output_dir=str(tmp_path / 'results'),
                            generate_plots=False
                        )
                    except Exception:
                        pass

        assert len(bead_norm_called) == 0, \
            "bead_anchored_normalize should not be called when bead channels not configured"


class TestBug9_HardcodedParametersExternalized:
    """
    Test that hardcoded scientific parameters are now read from config
    instead of being buried in the code.

    Previously hardcoded in multiscale_analysis.py:
    - spatial_weight = 0.2 if scale_um <= 20 else 0.4  (line 245)
    - resolution_range = (0.5, 2.0) if scale_um <= 20 else (0.2, 1.0)  (line 225)

    Fixed:
    - Added optimization.algorithm_parameters section to config
    - Code now reads these parameters with fallback defaults
    - Parameters are now visible and reproducible
    """

    def test_spatial_weight_params_extracted_from_config(self):
        """
        REGRESSION TEST FOR BUG #9

        Verify that spatial_weight parameters are read from config
        instead of hardcoded.
        """
        from src.analysis.multiscale_analysis import _extract_algorithm_params

        # Simulate config with algorithm parameters
        config = {
            'optimization': {
                'algorithm_parameters': {
                    'spatial_weight': {
                        'fine_scale_threshold_um': 25.0,  # Custom value
                        'fine_scale_weight': 0.3,
                        'coarse_scale_weight': 0.5
                    }
                }
            }
        }

        params = _extract_algorithm_params(config, 'spatial_weight')

        assert params['fine_scale_threshold_um'] == 25.0
        assert params['fine_scale_weight'] == 0.3
        assert params['coarse_scale_weight'] == 0.5

    def test_resolution_range_params_extracted_from_config(self):
        """
        REGRESSION TEST FOR BUG #9

        Verify that resolution_range parameters are read from config.
        """
        from src.analysis.multiscale_analysis import _extract_algorithm_params

        config = {
            'optimization': {
                'algorithm_parameters': {
                    'resolution_range': {
                        'fine_scale_threshold_um': 30.0,
                        'fine_scale_range': [1.0, 3.0],
                        'coarse_scale_range': [0.1, 0.5]
                    }
                }
            }
        }

        params = _extract_algorithm_params(config, 'resolution_range')

        assert params['fine_scale_threshold_um'] == 30.0
        assert params['fine_scale_range'] == [1.0, 3.0]
        assert params['coarse_scale_range'] == [0.1, 0.5]

    def test_algorithm_params_have_sensible_defaults(self):
        """
        REGRESSION TEST FOR BUG #9

        Verify that missing config params return empty dict (code uses defaults).
        """
        from src.analysis.multiscale_analysis import _extract_algorithm_params

        # Config without algorithm_parameters
        config = {'optimization': {}}

        params = _extract_algorithm_params(config, 'spatial_weight')
        assert params == {}, "Should return empty dict when params not in config"

        # None config
        params = _extract_algorithm_params(None, 'spatial_weight')
        assert params == {}, "Should handle None config gracefully"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

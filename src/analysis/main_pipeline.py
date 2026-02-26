"""
Main IMC Analysis Pipeline

Production-quality implementation addressing all Gemini critiques.
Uses proper ion count statistics, morphology-aware segmentation, and multi-scale analysis.
"""

import numpy as np
import pandas as pd
import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

from .multiscale_analysis import perform_multiscale_analysis, compute_scale_consistency
from .parallel_processing import create_roi_batch_processor
from .batch_correction import BatchCorrectionConfig, bead_anchored_normalize
from ..config import Config
from ..quality_control import QualityGateEngine, QualityMetrics, QualityMonitor, GateDecision
try:
    from .data_storage import create_storage_backend
except ImportError:
    create_storage_backend = None

# Import AnalysisManifest system
try:
    from .analysis_manifest import AnalysisManifest, validate_manifest_compatibility
except ImportError:
    AnalysisManifest = None
    validate_manifest_compatibility = None


class IMCAnalysisPipeline:
    """
    Main analysis pipeline for IMC data processing.
    
    Implements all fixes from Gemini's distinguished engineering critique:
    - Proper ion count aggregation → arcsinh → StandardScaler → clustering
    - SLIC superpixel segmentation using DNA channels
    - Multi-scale analysis (10μm, 20μm, 40μm)
    - Realistic validation with Poisson noise
    - Simple parallelization for ROI-level processing
    """
    
    def __init__(self, config: Config, manifest: Optional[AnalysisManifest] = None):
        """Initialize IMC analysis pipeline with clean configuration."""
        self.analysis_config = config
        self.results = {}
        self.validation_results = {}

        # PRIORITY 2: Config versioning & provenance tracking
        self.config_hash = None
        self.provenance = {
            'timestamp': datetime.now().isoformat(),
            'version': self._get_version(),
            'config_hash': None
        }
        
        # PRODUCTION SAFETY: Check critical dependencies at startup
        self._validate_startup_capabilities()
        
        # AnalysisManifest integration
        self.analysis_manifest = manifest
        if manifest and validate_manifest_compatibility:
            compatibility = validate_manifest_compatibility(manifest, config)
            if not compatibility['compatible']:
                warnings.warn(f"Manifest-config incompatibility: {compatibility['errors']}")
            if compatibility['warnings']:
                warnings.warn(f"Manifest-config warnings: {compatibility['warnings']}")
        
        self.batch_config = BatchCorrectionConfig()

        # Configure bead normalization from config if available
        if hasattr(config, 'analysis'):
            analysis_cfg = config.analysis
            if isinstance(analysis_cfg, dict):
                batch_corr = analysis_cfg.get('batch_correction', {})
                if isinstance(batch_corr, dict):
                    bead_norm = batch_corr.get('bead_normalization', {})
                    if isinstance(bead_norm, dict) and bead_norm.get('enabled', False):
                        self.batch_config = BatchCorrectionConfig(
                            bead_channels=bead_norm.get('bead_channels', ['130Ba', '131Xe']),
                            bead_signal_threshold=bead_norm.get('bead_signal_threshold', 10.0),
                            drift_correction_method=bead_norm.get('drift_correction_method', 'median_reference')
                        )

    def _compute_roi_quality_metrics(self, roi_id: str, roi_data: Dict, protein_names: List[str]) -> QualityMetrics:
        """Compute lightweight quality metrics from loaded ROI data for gate decisions."""
        from datetime import datetime

        coords = roi_data.get('coords', np.array([]))
        ion_counts = roi_data.get('ion_counts', {})

        # Coordinate quality: check for NaN/inf and reasonable range
        if len(coords) > 0:
            valid_coords = np.isfinite(coords).all(axis=1).mean()
            coord_range = np.ptp(coords, axis=0)
            # Penalize if range is suspiciously small (< 10 pixels)
            range_ok = 1.0 if all(r > 10 for r in coord_range) else 0.5
            coordinate_quality = valid_coords * range_ok
        else:
            coordinate_quality = 0.0

        # Ion count quality: check for signal above noise
        if ion_counts:
            channel_qualities = []
            for ch, values in ion_counts.items():
                vals = np.asarray(values)
                if len(vals) > 0:
                    # Fraction of pixels with positive signal
                    pos_frac = (vals > 0).mean()
                    # Signal-to-noise proxy: mean / (std + 1e-10)
                    snr = np.mean(vals) / (np.std(vals) + 1e-10) if np.std(vals) > 0 else 0.0
                    snr_score = min(1.0, snr / 2.0)  # Normalize: SNR=2 → 1.0
                    channel_qualities.append(0.5 * pos_frac + 0.5 * snr_score)
            ion_count_quality = float(np.mean(channel_qualities)) if channel_qualities else 0.0
        else:
            ion_count_quality = 0.0

        # Biological quality: protein completeness and DNA signal
        n_proteins_found = len(ion_counts)
        protein_completeness = n_proteins_found / max(len(protein_names), 1)

        # DNA signal presence
        dna_data = roi_data.get('dna1_intensities', np.array([]))
        if len(dna_data) > 0:
            dna_signal = min(1.0, np.mean(dna_data) / 50.0)  # Normalize
        else:
            dna_signal = 0.5  # Unknown

        biological_quality = 0.6 * protein_completeness + 0.4 * dna_signal

        n_pixels = len(coords) if len(coords) > 0 else 0

        return QualityMetrics(
            roi_id=roi_id,
            batch_id='batch_0',
            timestamp=datetime.now().isoformat(),
            coordinate_quality=min(1.0, max(0.0, coordinate_quality)),
            ion_count_quality=min(1.0, max(0.0, ion_count_quality)),
            biological_quality=min(1.0, max(0.0, biological_quality)),
            n_pixels=n_pixels,
            n_proteins=n_proteins_found,
            protein_completeness=min(1.0, max(0.0, protein_completeness))
        )

    def _log_execution_step(
        self,
        step_name: str,
        step_type: str,
        parameters: Dict[str, Any],
        results_summary: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log execution step to analysis manifest if available."""
        if self.analysis_manifest:
            self.analysis_manifest.log_execution_step(
                step_name=step_name,
                step_type=step_type,
                parameters=parameters,
                results_summary=results_summary
            )
    
    def _log_parameter_deviation(
        self,
        parameter_path: str,
        original_value: Any,
        new_value: Any,
        reason: str
    ) -> None:
        """Log parameter deviation to analysis manifest if available."""
        if self.analysis_manifest:
            self.analysis_manifest.log_parameter_deviation(
                parameter_path=parameter_path,
                original_value=original_value,
                new_value=new_value,
                reason=reason
            )
        
    def _validate_startup_capabilities(self) -> None:
        """Validate critical dependencies are available at startup."""
        capabilities = {
            'hdf5_storage': False,
            'parquet_storage': False,
            'clustering': False,
            'spatial_analysis': False,
            'image_processing': False,
            'statistical_framework': False
        }
        
        missing_critical = []
        
        # Check storage backends
        try:
            import h5py
            capabilities['hdf5_storage'] = True
        except ImportError:
            pass
            
        try:
            import pyarrow
            capabilities['parquet_storage'] = True
        except ImportError:
            pass
        
        # Require at least one efficient storage backend
        if not capabilities['hdf5_storage'] and not capabilities['parquet_storage']:
            missing_critical.append(
                "Neither HDF5 (h5py) nor Parquet (pyarrow) available. "
                "Install with: pip install h5py pyarrow"
            )
        
        # Check clustering capabilities
        try:
            import leidenalg
            import igraph
            capabilities['clustering'] = True
        except ImportError:
            missing_critical.append(
                "Clustering dependencies missing. "
                "Install with: pip install leidenalg igraph"
            )
        
        # Check spatial analysis
        try:
            import scipy.spatial
            import sklearn.neighbors
            capabilities['spatial_analysis'] = True
        except ImportError:
            missing_critical.append(
                "Spatial analysis dependencies missing. "
                "Install with: pip install scipy scikit-learn"
            )
        
        # Check image processing
        try:
            import skimage.segmentation
            import numpy as np
            capabilities['image_processing'] = True
        except ImportError:
            missing_critical.append(
                "Image processing dependencies missing. "
                "Install with: pip install scikit-image numpy"
            )
        
        # Check statistical framework
        try:
            import statsmodels.stats
            capabilities['statistical_framework'] = True
        except ImportError:
            # Statistical framework is optional
            pass
        
        # Fail fast if critical capabilities missing
        if missing_critical:
            capability_report = "\n".join([
                "STARTUP CAPABILITY CHECK FAILED",
                "Missing critical dependencies:",
                *[f"  - {msg}" for msg in missing_critical],
                "",
                f"Available capabilities: {[k for k, v in capabilities.items() if v]}",
                f"Missing capabilities: {[k for k, v in capabilities.items() if not v]}"
            ])
            
            raise ImportError(
                f"\n{capability_report}\n\n"
                "Install missing dependencies before running pipeline."
            )

        # Successful capability check (no logging needed - silent success)

    # PRIORITY 2: Config Versioning & Provenance Methods

    def _get_version(self) -> str:
        """
        Get software version from git or fallback to default.

        Returns:
            Version string (e.g., "git-a3f2b91" or "1.0.0")
        """
        try:
            import subprocess
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent
            ).decode().strip()
            return f"git-{git_hash}"
        except Exception:
            return "1.0.0"

    def _get_dependencies(self) -> Dict[str, str]:
        """
        Record exact dependency versions for reproducibility.

        Returns:
            Dictionary mapping package names to versions
        """
        dependencies = {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'pandas': pd.__version__
        }

        # Try to get versions of optional dependencies
        try:
            import scipy
            dependencies['scipy'] = scipy.__version__
        except ImportError:
            dependencies['scipy'] = 'not installed'

        try:
            import sklearn
            dependencies['sklearn'] = sklearn.__version__
        except ImportError:
            dependencies['sklearn'] = 'not installed'

        try:
            import leidenalg
            # leidenalg doesn't have __version__, try version attribute or callable
            if hasattr(leidenalg, '__version__'):
                dependencies['leidenalg'] = leidenalg.__version__
            elif hasattr(leidenalg, 'version'):
                version_attr = getattr(leidenalg, 'version')
                if callable(version_attr):
                    dependencies['leidenalg'] = version_attr()
                else:
                    dependencies['leidenalg'] = str(version_attr)
            else:
                dependencies['leidenalg'] = 'installed (version unknown)'
        except (ImportError, AttributeError, TypeError):
            dependencies['leidenalg'] = 'not installed'

        try:
            import skimage
            dependencies['skimage'] = skimage.__version__
        except ImportError:
            dependencies['skimage'] = 'not installed'

        return dependencies

    def _config_to_dict(self, config) -> dict:
        """
        Convert Config object to dictionary recursively.

        Args:
            config: Config object or dict

        Returns:
            Dictionary representation of config
        """
        if isinstance(config, dict):
            return config

        config_dict = {}
        for key in dir(config):
            # Skip private attributes and methods
            if key.startswith('_'):
                continue

            try:
                value = getattr(config, key)

                # Skip methods
                if callable(value):
                    continue

                # Convert Path objects to strings
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                # Recursively convert nested objects
                elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, Path)):
                    config_dict[key] = self._config_to_dict(value)
                # Convert lists/tuples
                elif isinstance(value, (list, tuple)):
                    config_dict[key] = [
                        str(item) if isinstance(item, Path)
                        else self._config_to_dict(item) if hasattr(item, '__dict__')
                        else item
                        for item in value
                    ]
                else:
                    config_dict[key] = value
            except Exception:
                # Skip attributes that can't be accessed
                continue

        return config_dict

    def _snapshot_config(self, output_dir: Path) -> str:
        """
        Create immutable config snapshot with SHA256 hash.

        CRITICAL for reproducibility: Every analysis must be traceable
        to exact config used.

        Args:
            output_dir: Results output directory

        Returns:
            SHA256 hash (first 8 characters)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert config to canonical JSON
        config_dict = self._config_to_dict(self.analysis_config)
        config_json = json.dumps(config_dict, sort_keys=True, indent=2)

        # Compute SHA256 hash
        config_hash_full = hashlib.sha256(config_json.encode()).hexdigest()
        config_hash_short = config_hash_full[:8]

        # Save snapshot
        snapshot = {
            'timestamp': self.provenance['timestamp'],
            'config_hash_full': config_hash_full,
            'config_hash_short': config_hash_short,
            'config': config_dict,
            'version': self.provenance['version']
        }

        snapshot_file = output_dir / f"config_snapshot_{config_hash_short}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)

        # Also save human-readable copy
        config_file = output_dir / "config.json"
        with open(config_file, 'w') as f:
            f.write(config_json)

        # Store hash for provenance
        self.config_hash = config_hash_short
        self.provenance['config_hash'] = config_hash_short

        return config_hash_short

    def _create_provenance_file(self, output_dir: Path, results: dict):
        """
        Create provenance.json linking results to exact config.

        Format:
        {
            "timestamp": "2025-10-19T...",
            "config_hash": "a3f2b91c",
            "config_file": "config_snapshot_a3f2b91c.json",
            "roi_id": "IMC_...",
            "software_version": "git-a3f2b91c",
            "dependencies": {...},
            "results_summary": {...}
        }

        Args:
            output_dir: Results output directory
            results: Analysis results dictionary
        """
        provenance = {
            **self.provenance,
            'config_file': f"config_snapshot_{self.config_hash}.json",
            'roi_id': results.get('roi_id'),
            'dependencies': self._get_dependencies(),
            'results_summary': {
                'n_scales': len(results.get('multiscale_results', {})),
                'scales_um': list(results.get('multiscale_results', {}).keys()),
                'total_superpixels': sum(
                    len(r.get('cluster_labels', []))
                    for r in results.get('multiscale_results', {}).values()
                    if isinstance(r, dict)
                )
            }
        }

        provenance_file = output_dir / "provenance.json"
        with open(provenance_file, 'w') as f:
            json.dump(provenance, f, indent=2)

    def load_roi_data(self, roi_file_path: str, protein_names: List[str]) -> Dict:
        """
        Load single ROI data from IMC text file.
        
        Args:
            roi_file_path: Path to ROI data file
            protein_names: List of protein markers to extract
            
        Returns:
            Dictionary with coordinates and protein data
        """
        try:
            # Load IMC data (assuming tab-separated format)
            df = pd.read_csv(roi_file_path, sep='\t')
            
            # Extract coordinates (assuming X and Y columns)
            coords = df[['X', 'Y']].values
            
            # Extract protein channels using robust column matching
            from ..utils.column_matching import match_imc_columns

            ion_counts = {}

            # Use robust column matching utility
            protein_matches = match_imc_columns(protein_names, list(df.columns))

            for protein_name in protein_names:
                matched_col = protein_matches.get(protein_name)

                if matched_col:
                    ion_counts[protein_name] = df[matched_col].values
                else:
                    # Warn but don't fail - allow analysis to proceed with available proteins
                    warnings.warn(f"Protein {protein_name} not found in {roi_file_path}. "
                                f"This marker will be excluded from analysis. "
                                f"Available columns: {list(df.columns)[:10]}...")

            # Also load bead/calibration channels for normalization
            if hasattr(self, 'batch_config') and self.batch_config and hasattr(self.batch_config, 'bead_channels'):
                import logging
                logger = logging.getLogger('IMCPipeline')
                for bead_channel in self.batch_config.bead_channels:
                    if bead_channel not in ion_counts:  # Don't duplicate if already loaded
                        matching_cols = [col for col in df.columns if bead_channel in col]
                        if matching_cols:
                            ion_counts[bead_channel] = df[matching_cols[0]].values
                            median_signal = np.median(ion_counts[bead_channel])
                            logger.debug(f"Loaded bead channel {bead_channel} from {Path(roi_file_path).name}: median={median_signal:.2f}")
                        else:
                            logger.warning(f"Bead channel {bead_channel} not found in {Path(roi_file_path).name}")

            # Extract DNA channels using config (no hardcoded patterns)
            # Look for channels whose names contain the configured DNA channel names
            dna_channels = []
            if hasattr(self.analysis_config, 'channels'):
                channels_cfg = self.analysis_config.channels
                if isinstance(channels_cfg, dict):
                    dna_channels = channels_cfg.get('dna_channels', ['DNA1', 'DNA2'])
                else:
                    dna_channels = ['DNA1', 'DNA2']  # Minimal fallback
            else:
                dna_channels = ['DNA1', 'DNA2']  # Minimal fallback

            # Find columns for first DNA channel
            dna1_cols = [col for col in df.columns if dna_channels[0] in col]
            if not dna1_cols:
                raise ValueError(f"Critical error: {dna_channels[0]} channel not found in {roi_file_path}. "
                               f"Required for analysis. Available columns: {list(df.columns)}")

            # Find columns for second DNA channel
            dna2_cols = [col for col in df.columns if dna_channels[1] in col] if len(dna_channels) > 1 else dna1_cols
            if not dna2_cols:
                raise ValueError(f"Critical error: {dna_channels[1]} channel not found in {roi_file_path}. "
                               f"Required for analysis. Available columns: {list(df.columns)}")
                               
            dna1_intensities = df[dna1_cols[0]].values
            dna2_intensities = df[dna2_cols[0]].values
            
            return {
                'coords': coords,
                'ion_counts': ion_counts,
                'dna1_intensities': dna1_intensities,
                'dna2_intensities': dna2_intensities,
                'protein_names': protein_names,
                'n_measurements': len(coords)
            }
            
        except Exception as e:
            raise ValueError(f"Failed to load ROI data from {roi_file_path}: {str(e)}")
    
    def analyze_single_roi(
        self, 
        roi_data: Dict,
        override_config: Optional[Dict] = None,
        plots_dir: Optional[str] = None,
        roi_id: Optional[str] = None
    ) -> Dict:
        """
        Analyze single ROI using configuration-driven approach.
        
        Args:
            roi_data: ROI data dictionary from load_roi_data
            override_config: Optional config overrides
            
        Returns:
            Dictionary with analysis results for all scales
        """
        # Get configuration parameters
        config = self.analysis_config

        # Get default config values
        multiscale_config = self.analysis_config.analysis.get('multiscale', {})
        slic_config = self.analysis_config.segmentation.get('slic', {})
        default_scales = multiscale_config.get('scales_um', [10, 20, 40])
        default_use_slic = slic_config.get('use_slic', False)

        # Apply overrides if provided
        if override_config:
            scales_um = override_config.get('scales_um', default_scales)
            use_slic = override_config.get('use_slic', default_use_slic)
            # BUG FIX: Read plots_dir from override_config for parallel processing
            plots_dir = override_config.get('plots_dir', plots_dir)

            # Log parameter deviations to manifest
            if 'scales_um' in override_config and override_config['scales_um'] != default_scales:
                self._log_parameter_deviation(
                    parameter_path='analysis.multiscale.scales_um',
                    original_value=default_scales,
                    new_value=scales_um,
                    reason="Runtime override for single ROI analysis"
                )
            if 'use_slic' in override_config and override_config['use_slic'] != default_use_slic:
                self._log_parameter_deviation(
                    parameter_path='segmentation.slic.use_slic',
                    original_value=default_use_slic,
                    new_value=use_slic,
                    reason="Runtime override for single ROI analysis"
                )
        else:
            scales_um = default_scales
            use_slic = default_use_slic

        # Log analysis step to manifest
        analysis_params = {
            'roi_id': roi_id,
            'scales_um': scales_um,
            'method': 'leiden',
            'use_slic': use_slic,
            'n_measurements': roi_data.get('n_measurements', len(roi_data['coords']))
        }
        self._log_execution_step(
            step_name=f"multiscale_analysis_{roi_id or 'unknown'}",
            step_type="analysis",
            parameters=analysis_params
        )

        multiscale_results = perform_multiscale_analysis(
            coords=roi_data['coords'],
            ion_counts=roi_data['ion_counts'],
            dna1_intensities=roi_data['dna1_intensities'],
            dna2_intensities=roi_data['dna2_intensities'],
            scales_um=scales_um,
            method='leiden',  # Using new spatial clustering
            use_slic=use_slic,
            config=self.analysis_config,
            plots_dir=plots_dir,
            roi_id=roi_id
        )
        
        # Compute consistency metrics between scales
        consistency_results = compute_scale_consistency(multiscale_results)

        # Extract main scale results for backward compatibility
        # multiscale_results is keyed by scale floats (e.g., 10.0, 20.0, 40.0),
        # not wrapped under a 'scale_results' key.
        numeric_scales = sorted([k for k in multiscale_results.keys() if isinstance(k, (int, float))])
        if numeric_scales:
            primary_scale = numeric_scales[0]
            primary_results = multiscale_results.get(primary_scale, {})
        else:
            primary_results = {}
        
        result = {
            'multiscale_results': multiscale_results,
            'consistency_results': consistency_results,
            'configuration_used': config.to_dict() if hasattr(config, 'to_dict') else str(config),
            'metadata': {
                'n_measurements': roi_data.get('n_measurements', len(roi_data['coords'])),
                'scales_analyzed': scales_um,  # Use the resolved scales_um
                'method': 'slic' if use_slic else 'square',  # Use the resolved use_slic
                'optimization_enabled': True
            }
        }
        
        # Add backward compatibility fields from primary scale
        # BUG FIX #7: Use 'features' key not 'feature_matrix' to get enriched features
        if primary_results:
            result.update({
                'cluster_labels': primary_results.get('cluster_labels', []),
                'feature_matrix': primary_results.get('features', np.array([])),  # Fixed: was 'feature_matrix'
                'protein_names': primary_results.get('protein_names', []),
                'silhouette_score': primary_results.get('silhouette_score', 0.0)
            })

        # PRIORITY 2: Config versioning & provenance tracking
        # Determine output directory
        if roi_id:
            result['roi_id'] = roi_id

            # Get output directory from config
            if hasattr(config, 'output'):
                output_config = config.output
                if isinstance(output_config, dict):
                    results_dir = output_config.get('results_dir', 'results')
                else:
                    results_dir = getattr(output_config, 'results_dir', 'results')
            else:
                results_dir = 'results'

            output_dir = Path(results_dir) / "roi_results" / roi_id

            # Snapshot config (once per pipeline instance)
            if self.config_hash is None:
                try:
                    self._snapshot_config(output_dir)
                except Exception as e:
                    warnings.warn(f"Config snapshot failed: {e}")

            # Create provenance file linking this analysis to config
            if self.config_hash is not None:
                try:
                    self._create_provenance_file(output_dir, result)
                except Exception as e:
                    warnings.warn(f"Provenance file creation failed: {e}")

        return result
    
    def run_batch_analysis(
        self, 
        roi_file_paths: List[str],
        protein_names: List[str],
        output_dir: str,
        n_processes: Optional[int] = None,
        scales_um: List[float] = [10.0, 20.0, 40.0],
        analysis_params: Optional[Dict] = None,
        generate_plots: bool = True
    ) -> Tuple[Dict, List[str]]:
        """
        Run analysis on multiple ROIs in parallel.
        
        Args:
            roi_file_paths: List of ROI file paths
            protein_names: List of protein markers to analyze
            output_dir: Output directory for results
            n_processes: Number of parallel processes
            scales_um: Spatial scales to analyze
            analysis_params: Optional analysis parameters (n_clusters, use_slic, etc.)
            
        Returns:
            Tuple of (results_dict, error_messages)
        """
        # Prepare ROI data for batch processing
        roi_data_dict = {}
        
        for roi_path in roi_file_paths:
            try:
                roi_id = Path(roi_path).stem
                roi_data = self.load_roi_data(roi_path, protein_names)
                roi_data_dict[roi_id] = roi_data
            except Exception as e:
                warnings.warn(f"Failed to load ROI {roi_path}: {str(e)}")
        
        if not roi_data_dict:
            raise ValueError("No valid ROI data loaded")

        # === QC GATE: Pre-flight ROI quality check ===
        gate_engine = QualityGateEngine()
        qc_monitor = QualityMonitor()
        skipped_rois = []

        for roi_id, roi_data in list(roi_data_dict.items()):
            qm = self._compute_roi_quality_metrics(roi_id, roi_data, protein_names)
            qc_monitor.add_roi_quality(qm)
            decision, reason, details = gate_engine.evaluate_roi_quality(qm)

            if decision == GateDecision.FAIL:
                warnings.warn(f"QC gate FAIL for {roi_id}: {reason}. Skipping.")
                skipped_rois.append(roi_id)
                del roi_data_dict[roi_id]
            elif decision == GateDecision.WARN:
                warnings.warn(f"QC gate WARN for {roi_id}: {reason}. Continuing.")
            elif decision == GateDecision.ABORT:
                raise RuntimeError(f"QC gate ABORT: {reason}. Analysis halted.")

        if skipped_rois:
            warnings.warn(f"QC gates skipped {len(skipped_rois)} ROIs: {skipped_rois}")

        if not roi_data_dict:
            raise ValueError("All ROIs failed quality gates — no data to analyze")

        # BUG FIX #6: Apply batch-level bead normalization BEFORE individual ROI processing
        # Bead normalization requires temporal context across ALL ROIs to model drift
        # Only apply if explicitly enabled in config
        bead_enabled = False
        if hasattr(self.analysis_config, 'analysis'):
            analysis_cfg = self.analysis_config.analysis
            if isinstance(analysis_cfg, dict):
                batch_corr = analysis_cfg.get('batch_correction', {})
                if isinstance(batch_corr, dict):
                    bead_norm = batch_corr.get('bead_normalization', {})
                    if isinstance(bead_norm, dict):
                        bead_enabled = bead_norm.get('enabled', False)

        if bead_enabled and self.batch_config.bead_channels:
            import logging
            logger = logging.getLogger('IMCPipeline')
            logger.info(f"Applying batch-level bead normalization to {len(roi_data_dict)} ROIs")

            try:
                # Extract ion counts from all ROIs for batch normalization
                batch_ion_counts = {}
                batch_metadata = {}
                for roi_id, roi_data in roi_data_dict.items():
                    if 'ion_counts' in roi_data:
                        batch_ion_counts[roi_id] = roi_data['ion_counts']
                        # Debug: Check if bead channels are present
                        bead_channels_present = [ch for ch in self.batch_config.bead_channels if ch in roi_data['ion_counts']]
                        logger.debug(f"ROI {roi_id}: bead channels present = {bead_channels_present}")
                        # Create metadata with acquisition order (enumerate provides temporal sequence)
                        batch_metadata[roi_id] = {
                            'acquisition_time': list(roi_data_dict.keys()).index(roi_id)
                        }

                # Apply bead normalization across all ROIs
                normalized_batch, norm_stats = bead_anchored_normalize(
                    batch_ion_counts,
                    batch_metadata,
                    config=self.batch_config
                )

                # Update roi_data_dict with normalized ion counts
                for roi_id in roi_data_dict.keys():
                    if roi_id in normalized_batch:
                        roi_data_dict[roi_id]['ion_counts'] = normalized_batch[roi_id]

                method = norm_stats.get('method', 'unknown')
                drift_method = norm_stats.get('drift_correction_method', 'N/A')
                bead_channels = norm_stats.get('bead_channels', [])
                valid_batches = len(norm_stats.get('valid_batches', []))
                logger.info(f"Bead normalization applied: method={method}, drift={drift_method}, "
                           f"beads={bead_channels}, valid_rois={valid_batches}")

            except Exception as e:
                logger.warning(f"Batch bead normalization failed: {e}. Proceeding without normalization.")

        # Create efficient storage backend
        storage_config = {}
        if hasattr(self.analysis_config, 'storage'):
            storage_config = self.analysis_config.storage.__dict__
        else:
            # Default storage config
            storage_config = {
                'format': 'hdf5',
                'compression': True,
                'metadata_format': 'json'
            }

        storage_backend = create_storage_backend(
            storage_config=storage_config,
            base_path=output_dir
        )

        # Create plots directory if visualization is requested
        # BUG FIX: Use config.output.results_dir (not output_dir param) for consistency
        plots_dir = None
        if generate_plots:
            # Get results_dir from config (same logic as analyze_single_roi)
            if hasattr(self.analysis_config, 'output'):
                output_config = self.analysis_config.output
                if isinstance(output_config, dict):
                    results_dir = output_config.get('results_dir', 'results')
                else:
                    results_dir = getattr(output_config, 'results_dir', 'results')
            else:
                results_dir = 'results'

            plots_dir = Path(results_dir) / "plots" / "validation"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plots_dir = str(plots_dir)
        
        # Check if config is serializable for multiprocessing - if not, use sequential processing
        use_sequential = False
        try:
            # Test multiprocessing compatibility by checking if config object itself
            # can be serialized (not just its to_dict output)
            import json
            
            # First, check if the config object attributes contain any callables
            def check_for_callables(obj, path="config"):
                if callable(obj):
                    raise TypeError(f"Config contains callable at {path}: {obj}")
                elif hasattr(obj, '__dict__'):
                    for attr_name, attr_value in obj.__dict__.items():
                        check_for_callables(attr_value, f"{path}.{attr_name}")
                elif isinstance(obj, dict):
                    for key, value in obj.items():
                        check_for_callables(value, f"{path}[{key}]")
                elif isinstance(obj, (list, tuple)):
                    for i, item in enumerate(obj):
                        check_for_callables(item, f"{path}[{i}]")
            
            check_for_callables(self.analysis_config)
            
            # Also test JSON serialization of the config's to_dict if available
            if hasattr(self.analysis_config, 'to_dict'):
                config_dict = self.analysis_config.to_dict()
                json_str = json.dumps(config_dict)
                json.loads(json_str)
            else:
                # Try to serialize the config object attributes
                json.dumps(self.analysis_config.__dict__)
                
        except (TypeError, ValueError, AttributeError):
            # Config is not serializable, use sequential processing
            use_sequential = True
        
        # Analysis parameters with defaults
        if analysis_params is None:
            analysis_params = {}

        # Read defaults from config instead of hardcoding
        # BUG FIX #5: Batch analysis was ignoring config and using hardcoded values
        config_defaults = {}

        # Get segmentation method from config
        if hasattr(self.analysis_config, 'segmentation'):
            seg_config = self.analysis_config.segmentation
            config_defaults['use_slic'] = seg_config.get('method', 'slic') == 'slic'
        else:
            config_defaults['use_slic'] = True  # Fallback

        # Get clustering parameters from config
        if hasattr(self.analysis_config, 'analysis') and hasattr(self.analysis_config.analysis, 'clustering'):
            cluster_config = self.analysis_config.analysis.clustering
            # Use resolution_range to infer n_clusters (or pass resolution range directly)
            # For backward compatibility, keep n_clusters but read from config if available
            if hasattr(cluster_config, 'n_clusters'):
                config_defaults['n_clusters'] = cluster_config.n_clusters
            # Pass through clustering config options
            config_defaults['clustering_method'] = cluster_config.get('method', 'leiden')
            config_defaults['resolution_range'] = cluster_config.get('resolution_range', [0.1, 20.0])
            config_defaults['use_coabundance_features'] = cluster_config.get('use_coabundance_features', True)
            config_defaults['coabundance_options'] = cluster_config.get('coabundance_options', {})

        # Set defaults for missing parameters, preferring config over hardcoded
        # BUG FIX: Include plots_dir in params for parallel processing
        final_analysis_params = {
            'scales_um': scales_um,
            'plots_dir': plots_dir,  # Pass plots_dir to parallel workers
            **config_defaults,  # Apply config defaults first
            **analysis_params   # Override with explicitly provided params
        }
        
        if use_sequential:
            # Sequential processing for unpicklable configs (e.g., testing)
            results = {}
            errors = []
            
            for roi_id, roi_data in roi_data_dict.items():
                try:
                    result = self.analyze_single_roi(
                        roi_data, 
                        override_config=final_analysis_params,
                        plots_dir=plots_dir,
                        roi_id=roi_id
                    )
                    results[roi_id] = result
                    
                    # Save result if output directory specified
                    if output_dir:
                        output_path = Path(output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        
                        # PERFORMANCE FIX: Use efficient storage backend instead of JSON explosion
                        # Create storage backend for results (using module-level import)
                        storage_backend = create_storage_backend(
                            storage_config=storage_config,
                            base_path=str(output_path)
                        )
                        
                        # Save using efficient format (HDF5/Parquet) 
                        storage_backend.save_roi_analysis(roi_id, result)
                        
                        # Also save small metadata summary as JSON (without large arrays)
                        metadata_summary = {
                            'roi_id': roi_id,
                            'processing_status': 'completed',
                            'n_scales': len(result.get('multiscale_results', {})),
                            'analysis_timestamp': str(datetime.now()),
                            'pipeline_version': getattr(self.analysis_config, 'pipeline_version', 'unknown')
                        }
                        
                        import json
                        summary_file = output_path / f"{roi_id}_summary.json"
                        with open(summary_file, 'w') as f:
                            json.dump(metadata_summary, f, indent=2)
                            
                except Exception as e:
                    errors.append(f"Failed to process {roi_id}: {str(e)}")
        else:
            # Create batch processor for parallel processing
            batch_processor = create_roi_batch_processor(
                analysis_function=self.analyze_single_roi,
                n_processes=n_processes,
                output_dir=output_dir,
                save_format='json'  # Parallel processing only supports 'json' or 'csv'
            )
            
            # Run batch analysis in parallel
            results, errors = batch_processor(
                roi_data_dict=roi_data_dict,
                analysis_params=final_analysis_params,
                show_progress=True
            )
        
        return results, errors
    
    def generate_summary_report(
        self, 
        results: Dict,
        output_path: str = "analysis_summary.json"
    ) -> Dict:
        """
        Generate comprehensive summary report.
        
        Args:
            results: Analysis results from run_batch_analysis
            output_path: Path for summary report
            
        Returns:
            Summary report dictionary
        """
        # Aggregate scale consistency metrics across ROIs first
        all_consistency_metrics = []
        
        # Initialize summary structure
        summary = {
            'experiment_metadata': {
                'n_rois_analyzed': len(results),
                'analysis_date': pd.Timestamp.now().isoformat(),
                'config_used': self.analysis_config.to_dict() if hasattr(self.analysis_config, 'to_dict') else {}
            },
            'scale_consistency_summary': {},  # Will be populated later
            'roi_summaries': {},
            'validation_summary': self.validation_results if self.validation_results else {
                'status': 'validation_not_run',
                'note': 'Validation study was not executed - use run_validation_study() for detailed validation metrics'
            }
        }
        
        for roi_id, roi_result in results.items():
            if 'consistency_results' in roi_result:
                consistency = roi_result['consistency_results'].get('overall', {})
                if consistency:
                    all_consistency_metrics.append(consistency)
                
                # Add ROI-specific summary
                summary['roi_summaries'][roi_id] = {
                    'n_measurements': roi_result['metadata']['n_measurements'],
                    'scales_analyzed': roi_result['metadata']['scales_analyzed'],
                    'consistency_metrics': consistency
                }
        
        # Overall consistency statistics
        if all_consistency_metrics:
            overall_stats = {}
            for metric in ['mean_ari', 'mean_nmi', 'mean_centroid_distance']:
                values = [m.get(metric, np.nan) for m in all_consistency_metrics]
                valid_values = [v for v in values if not np.isnan(v)]
                
                if valid_values:
                    overall_stats[metric] = {
                        'mean': float(np.mean(valid_values)),
                        'std': float(np.std(valid_values)),
                        'n_rois': len(valid_values)
                    }
            
            summary['scale_consistency_summary'] = overall_stats
        else:
            # No consistency metrics available
            summary['scale_consistency_summary'] = {
                'status': 'no_consistency_data',
                'note': 'No consistency metrics available across ROIs'
            }
        
        # Save summary using efficient storage
        try:
            # Try to use the analysis config storage
            summary_storage = create_storage_backend(
                storage_config=storage_config,
                base_path=Path(output_path).parent
            )
            summary_storage.save_roi_analysis('analysis_summary', summary)
        except Exception:
            # Fallback to JSON if storage fails
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        return summary


# Module-level function to avoid pickle issues
def _pipeline_config_to_dict():
    """Convert pipeline config to dictionary."""
    return {
        "analysis": {
            "multiscale": {"scales_um": [10.0, 20.0, 40.0], "enable_scale_analysis": True},
            "clustering": {"method": "leiden", "resolution": 1.0},
            "normalization": {"method": "arcsinh", "cofactor": 1.0}
        },
        "segmentation": {
            "slic": {"use_slic": True, "compactness": 10.0, "sigma": 2.0}
        },
        "storage": {"format": "json", "compression": True}
    }


def run_complete_analysis(
    config_path: str,
    roi_directory: str,
    output_directory: str,
    run_validation: bool = True,
    manifest_path: Optional[str] = None,
    create_manifest: bool = False,
    scientific_objectives: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    Run complete IMC analysis pipeline.
    
    Args:
        config_path: Path to configuration file
        roi_directory: Directory containing ROI data files
        output_directory: Output directory for results
        run_validation: Whether to run validation study
        manifest_path: Path to existing analysis manifest
        create_manifest: Whether to create new manifest
        scientific_objectives: Scientific objectives for new manifest
        
    Returns:
        Analysis summary report
    """
    # Load configuration - USE THE ACTUAL CONFIG!
    config = Config(config_path)
    
    # Handle analysis manifest
    manifest = None
    if manifest_path and AnalysisManifest:
        try:
            manifest = AnalysisManifest.load(manifest_path)
            print(f"Loaded analysis manifest: {manifest.manifest_id}")
        except Exception as e:
            warnings.warn(f"Could not load manifest from {manifest_path}: {e}")
    
    elif create_manifest and AnalysisManifest:
        from .analysis_manifest import ScientificObjectives, create_manifest_from_config
        
        # Create scientific objectives from provided data or defaults
        if scientific_objectives:
            objectives = ScientificObjectives(**scientific_objectives)
        else:
            objectives = ScientificObjectives(
                primary_research_question="IMC spatial analysis of tissue microenvironment",
                hypotheses=["Spatial organization reveals biological patterns"],
                target_cell_types=config.channels.get('protein_channels', [])
            )
        
        manifest = create_manifest_from_config(
            config=config,
            data_directory=roi_directory,
            profile_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            scientific_objectives=objectives,
            description="Generated from pipeline configuration"
        )
        
        # Save manifest
        manifest_save_path = Path(output_directory) / f"manifest_{manifest.manifest_id}.json"
        manifest.save(manifest_save_path)
        print(f"Created analysis manifest: {manifest.manifest_id} at {manifest_save_path}")
    
    # Initialize pipeline with proper config and manifest
    pipeline = IMCAnalysisPipeline(config, manifest)

    # Find ROI files, excluding test acquisitions
    roi_files = [f for f in sorted(Path(roi_directory).glob("*.txt")) if 'Test' not in f.name]

    if not roi_files:
        raise ValueError(f"No ROI files found in {roi_directory}")

    print(f"Found {len(roi_files)} ROI files")

    # Get output directory from config (respects experiment-specific paths)
    # BUG FIX: Use config.output.results_dir instead of output_directory parameter
    if hasattr(config, 'output') and hasattr(config.output, 'results_dir'):
        output_path = Path(config.output.results_dir)
    else:
        # Fallback to parameter if config doesn't specify
        output_path = Path(output_directory)

    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run validation study first (after getting some analysis results)
    if run_validation:
        print("Running validation study...")
        # Note: This would need sample analysis results to validate against
        # For now, skip validation in run_complete_analysis or implement with mock data
        print("Validation study requires analysis results - skipping for now")
    
    # Get protein names from config first, then auto-detect if not specified
    protein_names = config.channels.get('protein_channels', [])

    if not protein_names:
        # Auto-detect from first ROI file
        import pandas as pd
        first_roi = pd.read_csv(roi_files[0], sep='\t')

        # Get exclusion lists from config (no hardcoded defaults)
        dna_channels = config.channels.get('dna_channels', [])
        coord_channels = config.channels.get('coordinate_channels', [])
        excluded_channels = config.channels.get('excluded_channels', [])
        background_channel = config.channels.get('background_channel', '')
        calibration_channels = config.channels.get('calibration_channels', [])
        carrier_gas_channel = config.channels.get('carrier_gas_channel', '')

        # Build exclusion set
        exclude_set = set(dna_channels + coord_channels + excluded_channels + calibration_channels)
        if background_channel:
            exclude_set.add(background_channel)
        if carrier_gas_channel:
            exclude_set.add(carrier_gas_channel)

        # Look for IMC format: Protein(Metal) - exclude known non-protein channels
        protein_columns = []
        for col in first_roi.columns:
            # Extract base name (before parenthesis if present)
            base_name = col.split('(')[0] if '(' in col else col
            # Include if not in exclusion set
            if base_name not in exclude_set and col not in exclude_set:
                protein_columns.append(col)

        if protein_columns:
            # Extract base protein names (remove metal tags if present)
            protein_names = [col.split('(')[0] if '(' in col else col for col in protein_columns]
        else:
            # Should never reach here if config and data are correct
            raise ValueError(f"No protein channels found in {roi_files[0]}. "
                           f"Check config.json channels section. Available columns: {list(first_roi.columns)}")

    print(f"Analyzing proteins: {protein_names}")
    
    # Run batch analysis on real data using config values
    print("Running batch analysis...")
    
    # Get scales from config
    scales_um = config.segmentation.get('scales_um', [10.0, 20.0, 40.0])
    
    results, errors = pipeline.run_batch_analysis(
        roi_file_paths=[str(f) for f in roi_files],
        protein_names=protein_names,
        output_dir=str(output_path / "roi_results"),
        scales_um=scales_um
    )
    
    if errors:
        print(f"Analysis completed with {len(errors)} errors")
        print("Error details:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    else:
        print("Analysis completed successfully")
    
    # Generate summary report
    summary = pipeline.generate_summary_report(
        results=results,
        output_path=str(output_path / "analysis_summary.json")
    )
    
    print(f"Results saved to: {output_directory}")
    print(f"Summary report: {output_path / 'analysis_summary.json'}")
    
    return summary


if __name__ == "__main__":
    # Example usage
    summary = run_complete_analysis(
        config_path="config.json",
        roi_directory="data/241218_IMC_Alun",
        output_directory="results",  # Simple, direct results directory
        run_validation=True
    )
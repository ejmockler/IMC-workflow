"""
Analysis Manifest System for IMC Pipeline

Provides cryptographically signed analysis specifications for transparent,
reproducible scientific analysis without rigid parameter lock-down.

This system creates signed manifests that capture analysis intent, dataset
provenance, and parameter profiles while maintaining flexibility for
scientific exploration.
"""

import json
import hashlib
import subprocess
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

from ..config import Config


class ManifestVersion(Enum):
    """Supported manifest versions."""
    V1_0 = "1.0"
    V1_1 = "1.1"  # Future extension


class SignatureMethod(Enum):
    """Supported cryptographic signature methods."""
    GPG = "gpg"
    NONE = "none"  # For development/testing


@dataclass
class DatasetFingerprint:
    """Cryptographic fingerprint of input dataset."""
    files: Dict[str, str] = field(default_factory=dict)  # filename -> sha256
    total_files: int = 0
    total_size_bytes: int = 0
    metadata_hash: Optional[str] = None
    directory_structure_hash: Optional[str] = None
    
    def add_file(self, filepath: Path, content_hash: str = None) -> None:
        """Add file to fingerprint."""
        if content_hash is None:
            content_hash = self._compute_file_hash(filepath)
        
        self.files[str(filepath.name)] = content_hash
        self.total_files += 1
        
        if filepath.exists():
            self.total_size_bytes += filepath.stat().st_size
    
    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of file content."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            warnings.warn(f"Could not hash file {filepath}: {e}")
            return "unavailable"
    
    def compute_overall_hash(self) -> str:
        """Compute overall dataset hash from file hashes."""
        combined = "|".join(sorted(f"{name}:{hash_val}" for name, hash_val in self.files.items()))
        return hashlib.sha256(combined.encode()).hexdigest()


@dataclass
class ParameterProfile:
    """Tissue-specific or experiment-specific parameter profile."""
    name: str
    description: str
    tissue_type: Optional[str] = None
    
    # Core analysis parameters
    segmentation_params: Dict[str, Any] = field(default_factory=dict)
    clustering_params: Dict[str, Any] = field(default_factory=dict)
    processing_params: Dict[str, Any] = field(default_factory=dict)
    
    # Quality control thresholds
    quality_thresholds: Dict[str, Any] = field(default_factory=dict)
    
    # Expected biological markers for validation
    expected_markers: List[str] = field(default_factory=list)
    marker_groups: Dict[str, List[str]] = field(default_factory=dict)
    
    # Scientific context
    scientific_rationale: Optional[str] = None
    literature_references: List[str] = field(default_factory=list)


@dataclass
class ProvenanceInfo:
    """Complete provenance information for analysis."""
    # Environment info
    git_commit_sha: Optional[str] = None
    git_branch: Optional[str] = None
    git_remote_url: Optional[str] = None
    
    # System info
    python_version: Optional[str] = None
    platform: Optional[str] = None
    container_info: Optional[Dict[str, str]] = None
    
    # Dependencies
    key_dependencies: Dict[str, str] = field(default_factory=dict)
    
    # Execution context
    working_directory: Optional[str] = None
    command_line_args: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScientificObjectives:
    """Scientific objectives and hypotheses for the analysis."""
    primary_research_question: str
    hypotheses: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    
    # Analysis scope
    target_cell_types: List[str] = field(default_factory=list)
    spatial_scales_of_interest: List[str] = field(default_factory=list)
    
    # Biological context
    tissue_context: Optional[str] = None
    experimental_conditions: List[str] = field(default_factory=list)
    
    # Success criteria
    success_metrics: List[str] = field(default_factory=list)
    quality_requirements: Dict[str, Any] = field(default_factory=dict)


class AnalysisManifest:
    """
    Cryptographically signed analysis manifest for reproducible IMC analysis.
    
    The manifest captures analysis intent, dataset provenance, parameter profiles,
    and scientific objectives while maintaining flexibility for exploration.
    """
    
    def __init__(
        self,
        manifest_id: Optional[str] = None,
        version: ManifestVersion = ManifestVersion.V1_0
    ):
        """Initialize analysis manifest."""
        self.manifest_id = manifest_id or self._generate_manifest_id()
        self.version = version
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        
        # Core components
        self.dataset_fingerprint = DatasetFingerprint()
        self.parameter_profile = None
        self.provenance_info = ProvenanceInfo()
        self.scientific_objectives = None
        
        # Signature information
        self.signature_method = SignatureMethod.NONE
        self.signature = None
        self.signer_info = None
        
        # Analysis tracking
        self.deviation_log: List[Dict[str, Any]] = []
        self.execution_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger('AnalysisManifest')
    
    def _generate_manifest_id(self) -> str:
        """Generate unique manifest ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
        return f"manifest_{timestamp}_{random_suffix}"
    
    def set_dataset_fingerprint(
        self,
        data_directory: Union[str, Path],
        file_pattern: str = "*.txt",
        metadata_file: Optional[Union[str, Path]] = None
    ) -> None:
        """Create dataset fingerprint from data directory."""
        data_dir = Path(data_directory)
        if not data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        self.logger.info(f"Creating dataset fingerprint for {data_dir}")
        
        # Hash all data files
        data_files = list(data_dir.glob(file_pattern))
        for file_path in data_files:
            self.dataset_fingerprint.add_file(file_path)
        
        # Hash metadata file if provided
        if metadata_file:
            metadata_path = Path(metadata_file)
            if metadata_path.exists():
                metadata_hash = self.dataset_fingerprint._compute_file_hash(metadata_path)
                self.dataset_fingerprint.metadata_hash = metadata_hash
        
        # Create directory structure hash
        dir_structure = sorted([str(p.relative_to(data_dir)) for p in data_dir.rglob("*") if p.is_file()])
        structure_str = "|".join(dir_structure)
        self.dataset_fingerprint.directory_structure_hash = hashlib.sha256(structure_str.encode()).hexdigest()
        
        self.logger.info(f"Dataset fingerprint created: {self.dataset_fingerprint.total_files} files, "
                        f"{self.dataset_fingerprint.total_size_bytes / (1024**2):.1f} MB")
    
    def set_parameter_profile(
        self,
        profile: ParameterProfile
    ) -> None:
        """Set parameter profile for analysis."""
        self.parameter_profile = profile
        self.updated_at = datetime.now(timezone.utc)
        self.logger.info(f"Parameter profile set: {profile.name}")
    
    def create_parameter_profile_from_config(
        self,
        config: Config,
        profile_name: str,
        description: str,
        tissue_type: Optional[str] = None
    ) -> ParameterProfile:
        """Create parameter profile from existing Config object."""
        profile = ParameterProfile(
            name=profile_name,
            description=description,
            tissue_type=tissue_type
        )
        
        # Extract key parameters from config
        if hasattr(config, 'segmentation'):
            profile.segmentation_params = {
                'method': config.segmentation.get('method', 'slic'),
                'scales_um': config.segmentation.get('scales_um', [10, 20, 40]),
                'slic_params': config.segmentation.get('slic_params', {})
            }
        
        if hasattr(config, 'analysis'):
            profile.clustering_params = {
                'method': config.analysis.get('clustering', {}).get('method', 'leiden'),
                'optimization_method': config.analysis.get('clustering', {}).get('optimization_method', 'stability'),
                'resolution_range': config.analysis.get('clustering', {}).get('resolution_range', [0.5, 2.0])
            }
        
        if hasattr(config, 'processing'):
            profile.processing_params = {
                'arcsinh_transform': config.processing.get('arcsinh_transform', {}),
                'normalization': config.processing.get('normalization', {}),
                'background_correction': config.processing.get('background_correction', {})
            }
        
        if hasattr(config, 'quality_control'):
            profile.quality_thresholds = config.quality_control.get('thresholds', {})
        
        # Extract expected markers
        if hasattr(config, 'channels'):
            profile.expected_markers = config.channels.get('protein_channels', [])
            profile.marker_groups = config.channels.get('channel_groups', {})
        
        return profile
    
    def set_scientific_objectives(self, objectives: ScientificObjectives) -> None:
        """Set scientific objectives for analysis."""
        self.scientific_objectives = objectives
        self.updated_at = datetime.now(timezone.utc)
        self.logger.info(f"Scientific objectives set: {objectives.primary_research_question}")
    
    def capture_provenance_info(self) -> None:
        """Capture current environment provenance information."""
        import sys
        import platform
        import os
        
        # Git information
        try:
            self.provenance_info.git_commit_sha = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
            ).decode().strip()
            
            self.provenance_info.git_branch = subprocess.check_output(
                ['git', 'branch', '--show-current'], stderr=subprocess.DEVNULL
            ).decode().strip()
            
            self.provenance_info.git_remote_url = subprocess.check_output(
                ['git', 'config', '--get', 'remote.origin.url'], stderr=subprocess.DEVNULL
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("Could not capture Git information")
        
        # System information
        self.provenance_info.python_version = sys.version
        self.provenance_info.platform = platform.platform()
        self.provenance_info.working_directory = os.getcwd()
        
        # Command line arguments
        self.provenance_info.command_line_args = sys.argv.copy()
        
        # Key environment variables (filtered for security)
        safe_env_vars = ['PATH', 'PYTHONPATH', 'USER', 'HOME', 'PWD']
        self.provenance_info.environment_variables = {
            var: os.environ.get(var, '') for var in safe_env_vars if var in os.environ
        }
        
        # Try to capture container information
        if os.path.exists('/.dockerenv'):
            self.provenance_info.container_info = {'type': 'docker'}
        elif os.environ.get('SINGULARITY_CONTAINER'):
            self.provenance_info.container_info = {
                'type': 'singularity',
                'image': os.environ.get('SINGULARITY_CONTAINER', '')
            }
        
        self.logger.info("Provenance information captured")
    
    def log_parameter_deviation(
        self,
        parameter_path: str,
        original_value: Any,
        new_value: Any,
        reason: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log deviation from original parameter profile."""
        deviation = {
            'timestamp': (timestamp or datetime.now(timezone.utc)).isoformat(),
            'parameter_path': parameter_path,
            'original_value': original_value,
            'new_value': new_value,
            'reason': reason
        }
        
        self.deviation_log.append(deviation)
        self.updated_at = datetime.now(timezone.utc)
        
        self.logger.info(f"Parameter deviation logged: {parameter_path} = {new_value} (was {original_value})")
    
    def log_execution_step(
        self,
        step_name: str,
        step_type: str,
        parameters: Dict[str, Any],
        results_summary: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log execution step for analysis tracking."""
        execution_step = {
            'timestamp': (timestamp or datetime.now(timezone.utc)).isoformat(),
            'step_name': step_name,
            'step_type': step_type,  # e.g., 'segmentation', 'clustering', 'validation'
            'parameters': parameters,
            'results_summary': results_summary or {}
        }
        
        self.execution_history.append(execution_step)
        self.updated_at = datetime.now(timezone.utc)
        
        self.logger.debug(f"Execution step logged: {step_name}")
    
    def validate_dataset_integrity(self, data_directory: Union[str, Path]) -> Dict[str, Any]:
        """Validate current dataset against manifest fingerprint."""
        if not self.dataset_fingerprint.files:
            return {
                'valid': False,
                'reason': 'No dataset fingerprint in manifest'
            }
        
        data_dir = Path(data_directory)
        validation_results = {
            'valid': True,
            'missing_files': [],
            'modified_files': [],
            'extra_files': [],
            'hash_mismatches': []
        }
        
        # Check for missing and modified files
        current_files = {}
        for filename, expected_hash in self.dataset_fingerprint.files.items():
            file_path = data_dir / filename
            if not file_path.exists():
                validation_results['missing_files'].append(filename)
                validation_results['valid'] = False
            else:
                current_hash = self.dataset_fingerprint._compute_file_hash(file_path)
                current_files[filename] = current_hash
                
                if current_hash != expected_hash:
                    validation_results['hash_mismatches'].append({
                        'filename': filename,
                        'expected_hash': expected_hash,
                        'current_hash': current_hash
                    })
                    validation_results['valid'] = False
        
        # Check for extra files
        expected_files = set(self.dataset_fingerprint.files.keys())
        actual_files = set(f.name for f in data_dir.glob("*.txt"))  # Assuming txt pattern
        extra_files = actual_files - expected_files
        if extra_files:
            validation_results['extra_files'] = list(extra_files)
        
        return validation_results
    
    def sign_manifest(
        self,
        signature_method: SignatureMethod = SignatureMethod.GPG,
        gpg_key_id: Optional[str] = None,
        passphrase: Optional[str] = None
    ) -> None:
        """Sign the manifest using specified cryptographic method."""
        if signature_method == SignatureMethod.NONE:
            self.signature_method = SignatureMethod.NONE
            self.signature = None
            self.signer_info = None
            return
        
        if signature_method == SignatureMethod.GPG:
            self._sign_with_gpg(gpg_key_id, passphrase)
        else:
            raise ValueError(f"Unsupported signature method: {signature_method}")
    
    def _sign_with_gpg(self, gpg_key_id: Optional[str] = None, passphrase: Optional[str] = None) -> None:
        """Sign manifest using GPG."""
        try:
            # Create signable content (excluding signature fields)
            signable_content = self._get_signable_content()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(signable_content, tmp_file, indent=2, default=str)
                tmp_path = tmp_file.name
            
            try:
                # Create GPG signature
                gpg_cmd = ['gpg', '--armor', '--detach-sign']
                if gpg_key_id:
                    gpg_cmd.extend(['--local-user', gpg_key_id])
                if passphrase:
                    gpg_cmd.extend(['--batch', '--yes', '--passphrase', passphrase])
                
                gpg_cmd.append(tmp_path)
                
                result = subprocess.run(gpg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"GPG signing failed: {result.stderr}")
                
                # Read signature
                sig_path = tmp_path + '.asc'
                with open(sig_path, 'r') as sig_file:
                    signature = sig_file.read()
                
                self.signature_method = SignatureMethod.GPG
                self.signature = signature
                
                # Get signer info
                self.signer_info = self._get_gpg_signer_info(gpg_key_id)
                
                # Clean up signature file
                Path(sig_path).unlink()
                
                self.logger.info(f"Manifest signed with GPG key: {gpg_key_id or 'default'}")
                
            finally:
                # Clean up temp file
                Path(tmp_path).unlink()
                
        except Exception as e:
            self.logger.error(f"GPG signing failed: {e}")
            raise RuntimeError(f"Could not sign manifest with GPG: {e}")
    
    def _get_gpg_signer_info(self, gpg_key_id: Optional[str] = None) -> Dict[str, str]:
        """Get information about GPG signer."""
        try:
            if gpg_key_id:
                cmd = ['gpg', '--list-keys', '--with-colons', gpg_key_id]
            else:
                cmd = ['gpg', '--list-secret-keys', '--with-colons']
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Parse GPG output (simplified)
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.startswith('uid:'):
                        parts = line.split(':')
                        if len(parts) > 9:
                            return {
                                'user_id': parts[9],
                                'key_id': gpg_key_id or 'default',
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            }
        except Exception:
            pass
        
        return {
            'key_id': gpg_key_id or 'default',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def verify_signature(self) -> Dict[str, Any]:
        """Verify manifest signature."""
        if self.signature_method == SignatureMethod.NONE:
            return {
                'valid': True,
                'method': 'none',
                'message': 'No signature to verify'
            }
        
        if self.signature_method == SignatureMethod.GPG:
            return self._verify_gpg_signature()
        
        return {
            'valid': False,
            'method': str(self.signature_method),
            'message': f'Unknown signature method: {self.signature_method}'
        }
    
    def _verify_gpg_signature(self) -> Dict[str, Any]:
        """Verify GPG signature."""
        if not self.signature:
            return {
                'valid': False,
                'method': 'gpg',
                'message': 'No signature present'
            }
        
        try:
            # Create signable content
            signable_content = self._get_signable_content()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as content_file:
                json.dump(signable_content, content_file, indent=2, default=str)
                content_path = content_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.asc', delete=False) as sig_file:
                sig_file.write(self.signature)
                sig_path = sig_file.name
            
            try:
                # Verify signature
                cmd = ['gpg', '--verify', sig_path, content_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    return {
                        'valid': True,
                        'method': 'gpg',
                        'message': 'Signature verified successfully',
                        'signer_info': self.signer_info
                    }
                else:
                    return {
                        'valid': False,
                        'method': 'gpg',
                        'message': f'Signature verification failed: {result.stderr}',
                        'signer_info': self.signer_info
                    }
            
            finally:
                Path(content_path).unlink()
                Path(sig_path).unlink()
        
        except Exception as e:
            return {
                'valid': False,
                'method': 'gpg',
                'message': f'Verification error: {e}'
            }
    
    def _get_signable_content(self) -> Dict[str, Any]:
        """Get content for signing (excluding signature fields)."""
        content = self.to_dict()
        # Remove signature-related fields
        content.pop('signature', None)
        content.pop('signature_method', None)
        content.pop('signer_info', None)
        return content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for serialization."""
        return {
            'manifest_id': self.manifest_id,
            'version': self.version.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'dataset_fingerprint': asdict(self.dataset_fingerprint),
            'parameter_profile': asdict(self.parameter_profile) if self.parameter_profile else None,
            'provenance_info': asdict(self.provenance_info),
            'scientific_objectives': asdict(self.scientific_objectives) if self.scientific_objectives else None,
            'signature_method': self.signature_method.value if self.signature_method else None,
            'signature': self.signature,
            'signer_info': self.signer_info,
            'deviation_log': self.deviation_log,
            'execution_history': self.execution_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisManifest':
        """Create manifest from dictionary."""
        manifest = cls(
            manifest_id=data['manifest_id'],
            version=ManifestVersion(data['version'])
        )
        
        manifest.created_at = datetime.fromisoformat(data['created_at'])
        manifest.updated_at = datetime.fromisoformat(data['updated_at'])
        
        # Reconstruct dataset fingerprint
        if data['dataset_fingerprint']:
            manifest.dataset_fingerprint = DatasetFingerprint(**data['dataset_fingerprint'])
        
        # Reconstruct parameter profile
        if data['parameter_profile']:
            manifest.parameter_profile = ParameterProfile(**data['parameter_profile'])
        
        # Reconstruct provenance info
        manifest.provenance_info = ProvenanceInfo(**data['provenance_info'])
        
        # Reconstruct scientific objectives
        if data['scientific_objectives']:
            manifest.scientific_objectives = ScientificObjectives(**data['scientific_objectives'])
        
        # Reconstruct signature info
        if data.get('signature_method'):
            manifest.signature_method = SignatureMethod(data['signature_method'])
            manifest.signature = data.get('signature')
            manifest.signer_info = data.get('signer_info')
        
        # Reconstruct logs
        manifest.deviation_log = data.get('deviation_log', [])
        manifest.execution_history = data.get('execution_history', [])
        
        return manifest
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save manifest to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Manifest saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'AnalysisManifest':
        """Load manifest from file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of manifest."""
        return {
            'manifest_id': self.manifest_id,
            'version': self.version.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'dataset_summary': {
                'total_files': self.dataset_fingerprint.total_files,
                'total_size_mb': self.dataset_fingerprint.total_size_bytes / (1024**2),
                'overall_hash': self.dataset_fingerprint.compute_overall_hash()
            },
            'parameter_profile': {
                'name': self.parameter_profile.name if self.parameter_profile else None,
                'tissue_type': self.parameter_profile.tissue_type if self.parameter_profile else None,
                'expected_markers': len(self.parameter_profile.expected_markers) if self.parameter_profile else 0
            },
            'scientific_objectives': {
                'research_question': self.scientific_objectives.primary_research_question if self.scientific_objectives else None,
                'hypothesis_count': len(self.scientific_objectives.hypotheses) if self.scientific_objectives else 0
            },
            'provenance': {
                'git_commit': self.provenance_info.git_commit_sha,
                'git_branch': self.provenance_info.git_branch,
                'platform': self.provenance_info.platform
            },
            'signature': {
                'method': self.signature_method.value if self.signature_method else 'none',
                'signed': self.signature is not None,
                'signer': self.signer_info.get('user_id') if self.signer_info else None
            },
            'execution': {
                'deviations_logged': len(self.deviation_log),
                'steps_executed': len(self.execution_history)
            }
        }


def create_manifest_from_config(
    config: Config,
    data_directory: Union[str, Path],
    profile_name: str,
    scientific_objectives: ScientificObjectives,
    description: str = "Generated from pipeline configuration",
    tissue_type: Optional[str] = None
) -> AnalysisManifest:
    """
    Factory function to create manifest from existing Config.
    
    Args:
        config: Existing Config object
        data_directory: Path to data directory
        profile_name: Name for parameter profile
        scientific_objectives: Scientific objectives for analysis
        description: Description of parameter profile
        tissue_type: Type of tissue being analyzed
        
    Returns:
        Configured AnalysisManifest
    """
    manifest = AnalysisManifest()
    
    # Set dataset fingerprint
    manifest.set_dataset_fingerprint(
        data_directory=data_directory,
        file_pattern="*.txt",  # Could be extracted from config
        metadata_file=config.data.get('metadata_file', None)
    )
    
    # Create parameter profile from config
    profile = manifest.create_parameter_profile_from_config(
        config=config,
        profile_name=profile_name,
        description=description,
        tissue_type=tissue_type
    )
    manifest.set_parameter_profile(profile)
    
    # Set scientific objectives
    manifest.set_scientific_objectives(scientific_objectives)
    
    # Capture provenance
    manifest.capture_provenance_info()
    
    return manifest


def validate_manifest_compatibility(
    manifest: AnalysisManifest,
    config: Config
) -> Dict[str, Any]:
    """
    Validate that a manifest is compatible with current configuration.
    
    Args:
        manifest: Analysis manifest to validate
        config: Current configuration
        
    Returns:
        Validation results dictionary
    """
    results = {
        'compatible': True,
        'warnings': [],
        'errors': [],
        'parameter_differences': []
    }
    
    if not manifest.parameter_profile:
        results['errors'].append("Manifest has no parameter profile")
        results['compatible'] = False
        return results
    
    # Check protein channels
    manifest_markers = set(manifest.parameter_profile.expected_markers)
    config_markers = set(config.channels.get('protein_channels', []))
    
    missing_markers = manifest_markers - config_markers
    extra_markers = config_markers - manifest_markers
    
    if missing_markers:
        results['warnings'].append(f"Config missing expected markers: {list(missing_markers)}")
    
    if extra_markers:
        results['warnings'].append(f"Config has extra markers not in manifest: {list(extra_markers)}")
    
    # Check key parameters
    if hasattr(config, 'segmentation'):
        manifest_scales = manifest.parameter_profile.segmentation_params.get('scales_um', [])
        config_scales = config.segmentation.get('scales_um', [])
        
        if manifest_scales != config_scales:
            results['parameter_differences'].append({
                'parameter': 'segmentation.scales_um',
                'manifest_value': manifest_scales,
                'config_value': config_scales
            })
    
    # Check clustering method
    if hasattr(config, 'analysis'):
        manifest_method = manifest.parameter_profile.clustering_params.get('method')
        config_method = config.analysis.get('clustering', {}).get('method')
        
        if manifest_method != config_method:
            results['parameter_differences'].append({
                'parameter': 'clustering.method',
                'manifest_value': manifest_method,
                'config_value': config_method
            })
    
    return results
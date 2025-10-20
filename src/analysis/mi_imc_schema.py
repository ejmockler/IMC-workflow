"""
Minimum Information for Imaging Mass Cytometry (MI-IMC) Metadata Schema

Provides standardized metadata schema for IMC experiments to ensure reproducibility,
publication compliance, and scientific transparency. Integrates seamlessly with 
existing IMC pipeline infrastructure while enabling standardized reporting.

This implementation follows FAIR data principles and established metadata standards
for biomedical imaging while being specifically designed for IMC workflows.
"""

import json
import hashlib
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import numpy as np
import pandas as pd

from ..config import Config
from ..utils.helpers import Metadata
from .data_storage import create_storage_backend


class MIIMCVersion(Enum):
    """Supported MI-IMC schema versions."""
    V1_0 = "1.0"
    V1_1 = "1.1"  # Future extension


class ExperimentType(Enum):
    """Types of IMC experiments."""
    TISSUE_MICROARRAY = "tissue_microarray"
    WHOLE_TISSUE_SECTION = "whole_tissue_section"
    SPATIAL_PROFILING = "spatial_profiling"
    TIME_COURSE = "time_course"
    DRUG_RESPONSE = "drug_response"
    CLINICAL_COHORT = "clinical_cohort"


class SampleType(Enum):
    """Types of biological samples."""
    FFPE_TISSUE = "ffpe_tissue"
    FROZEN_TISSUE = "frozen_tissue"
    CELL_PELLET = "cell_pellet"
    ORGANOID = "organoid"
    SINGLE_CELL = "single_cell"
    CYTOSPIN = "cytospin"


@dataclass
class InstrumentMetadata:
    """IMC instrument and acquisition metadata."""
    # Instrument identification
    instrument_model: str = "Hyperion"  # Standard CyTOF instrument
    instrument_serial: Optional[str] = None
    software_version: Optional[str] = None
    
    # Acquisition parameters
    acquisition_date: Optional[str] = None
    acquisition_frequency: float = 200.0  # Hz
    raster_size_um: float = 1.0  # μm pixel size
    acquisition_roi_width_um: Optional[float] = None
    acquisition_roi_height_um: Optional[float] = None
    
    # Laser and detection settings
    laser_power: Optional[float] = None
    laser_frequency: Optional[float] = None
    detection_efficiency: Optional[float] = None
    
    # Environmental conditions
    temperature_c: Optional[float] = None
    humidity_percent: Optional[float] = None
    atmospheric_pressure: Optional[float] = None


@dataclass 
class AntibodyMetadata:
    """Comprehensive antibody/marker metadata."""
    # Basic identification
    marker_name: str
    antibody_clone: Optional[str] = None
    antibody_supplier: Optional[str] = None
    catalog_number: Optional[str] = None
    
    # Metal tag information
    metal_tag: str = ""  # e.g., "163Dy", "165Ho"
    mass_channel: Optional[int] = None
    
    # Antibody properties
    antibody_concentration: Optional[float] = None  # μg/ml
    dilution_factor: Optional[str] = None  # e.g., "1:1000"
    incubation_time_hours: Optional[float] = None
    incubation_temperature_c: Optional[float] = None
    
    # Validation and performance
    specificity_validated: bool = False
    cross_reactivity_notes: Optional[str] = None
    lot_number: Optional[str] = None
    expiry_date: Optional[str] = None
    
    # Biological context
    target_cellular_location: Optional[str] = None  # "membrane", "cytoplasm", "nucleus"
    functional_role: Optional[str] = None
    expected_cell_types: List[str] = field(default_factory=list)
    
    # Quality control
    signal_to_noise_ratio: Optional[float] = None
    dynamic_range: Optional[float] = None
    coefficient_of_variation: Optional[float] = None


@dataclass
class SampleMetadata:
    """Comprehensive sample metadata."""
    # Sample identification
    sample_id: str
    patient_id: Optional[str] = None
    tissue_type: Optional[str] = None
    organ_system: Optional[str] = None
    anatomical_location: Optional[str] = None
    
    # Sample processing
    sample_type: Optional[SampleType] = None
    fixation_method: Optional[str] = None  # "formalin", "paraformaldehyde"
    fixation_duration_hours: Optional[float] = None
    processing_protocol: Optional[str] = None
    
    # Clinical/experimental context
    diagnosis: Optional[str] = None
    disease_stage: Optional[str] = None
    treatment_status: Optional[str] = None
    age_at_collection: Optional[int] = None
    sex: Optional[str] = None
    
    # Experimental variables
    experimental_group: Optional[str] = None
    timepoint: Optional[str] = None
    batch_id: Optional[str] = None
    
    # Quality metrics
    tissue_quality_score: Optional[float] = None
    dna_integrity_score: Optional[float] = None
    antigen_preservation_score: Optional[float] = None


@dataclass
class DataProcessingMetadata:
    """Metadata for data processing pipeline."""
    # Pipeline identification
    pipeline_name: str = "IMC_Analysis_Pipeline"
    pipeline_version: Optional[str] = None
    processing_date: Optional[str] = None
    
    # Core processing steps
    background_correction_method: Optional[str] = None
    spillover_correction_applied: bool = False
    batch_correction_method: Optional[str] = None
    
    # Segmentation parameters
    segmentation_method: str = "slic"  # "slic", "watershed", "manual"
    segmentation_parameters: Dict[str, Any] = field(default_factory=dict)
    nuclear_markers: List[str] = field(default_factory=lambda: ["DNA1", "DNA2"])
    
    # Analysis parameters
    transformation_method: str = "arcsinh"
    normalization_method: Optional[str] = None
    clustering_method: str = "leiden"
    clustering_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Spatial analysis
    spatial_scales_um: List[float] = field(default_factory=lambda: [10.0, 20.0, 40.0])
    spatial_analysis_methods: List[str] = field(default_factory=list)
    
    # Quality control
    quality_filters_applied: List[str] = field(default_factory=list)
    outlier_detection_method: Optional[str] = None
    minimum_cell_size: Optional[int] = None
    maximum_cell_size: Optional[int] = None


@dataclass
class StudyMetadata:
    """High-level study and publication metadata."""
    # Study identification (required fields first)
    study_title: str
    research_question: str
    
    # Optional study identification
    study_id: Optional[str] = None
    study_description: Optional[str] = None
    
    # Scientific context (optional fields with defaults)
    hypotheses: List[str] = field(default_factory=list)
    primary_outcomes: List[str] = field(default_factory=list)
    secondary_outcomes: List[str] = field(default_factory=list)
    
    # Authorship and attribution
    principal_investigator: Optional[str] = None
    data_analyst: Optional[str] = None
    contact_email: Optional[str] = None
    institution: Optional[str] = None
    
    # Publication and sharing
    publication_doi: Optional[str] = None
    preprint_doi: Optional[str] = None
    data_availability_statement: Optional[str] = None
    code_availability_statement: Optional[str] = None
    
    # Ethics and consent
    ethics_approval_number: Optional[str] = None
    consent_obtained: bool = False
    data_sharing_consent: bool = False
    
    # Funding
    funding_sources: List[str] = field(default_factory=list)
    grant_numbers: List[str] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Comprehensive quality control metrics."""
    # Sample-level quality
    sample_quality_score: float = 0.0
    tissue_coverage_percent: float = 0.0
    antibody_staining_quality: float = 0.0
    
    # Technical quality
    signal_to_noise_ratio: float = 0.0
    dynamic_range: float = 0.0
    acquisition_stability: float = 0.0
    
    # Analysis quality
    segmentation_quality_score: float = 0.0
    clustering_stability_score: float = 0.0
    spatial_coherence_score: float = 0.0
    
    # Data completeness
    marker_detection_rate: float = 0.0
    missing_data_percentage: float = 0.0
    outlier_percentage: float = 0.0
    
    # Cross-validation metrics
    reproducibility_score: Optional[float] = None
    technical_replicate_correlation: Optional[float] = None
    batch_effect_score: Optional[float] = None


class MIIMCSchema:
    """
    Minimum Information for Imaging Mass Cytometry (MI-IMC) metadata schema.
    
    Provides comprehensive, standardized metadata collection and validation
    for IMC experiments to ensure reproducibility and publication compliance.
    """
    
    def __init__(self, version: MIIMCVersion = MIIMCVersion.V1_0):
        """Initialize MI-IMC schema."""
        self.version = version
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        
        # Core metadata components
        self.study_metadata = None
        self.instrument_metadata = InstrumentMetadata()
        self.sample_metadata_list: List[SampleMetadata] = []
        self.antibody_panel: List[AntibodyMetadata] = []
        self.processing_metadata = DataProcessingMetadata()
        self.quality_metrics = QualityMetrics()
        
        # Compliance tracking
        self.compliance_checklist: Dict[str, bool] = {}
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
        self.logger = logging.getLogger('MIIMCSchema')
    
    def set_study_metadata(self, study: StudyMetadata) -> None:
        """Set study-level metadata."""
        self.study_metadata = study
        self.updated_at = datetime.now(timezone.utc)
        self.logger.info(f"Study metadata set: {study.study_title}")
    
    def add_sample_metadata(self, sample: SampleMetadata) -> None:
        """Add sample metadata to the schema."""
        self.sample_metadata_list.append(sample)
        self.updated_at = datetime.now(timezone.utc)
        self.logger.info(f"Sample metadata added: {sample.sample_id}")
    
    def set_antibody_panel(self, antibodies: List[AntibodyMetadata]) -> None:
        """Set the complete antibody panel."""
        self.antibody_panel = antibodies
        self.updated_at = datetime.now(timezone.utc)
        self.logger.info(f"Antibody panel set: {len(antibodies)} markers")
    
    def add_antibody(self, antibody: AntibodyMetadata) -> None:
        """Add single antibody to the panel."""
        self.antibody_panel.append(antibody)
        self.updated_at = datetime.now(timezone.utc)
        self.logger.info(f"Antibody added: {antibody.marker_name}")
    
    def set_instrument_metadata(self, instrument: InstrumentMetadata) -> None:
        """Set instrument and acquisition metadata."""
        self.instrument_metadata = instrument
        self.updated_at = datetime.now(timezone.utc)
        self.logger.info("Instrument metadata updated")
    
    def set_processing_metadata(self, processing: DataProcessingMetadata) -> None:
        """Set data processing metadata."""
        self.processing_metadata = processing
        self.updated_at = datetime.now(timezone.utc)
        self.logger.info("Processing metadata updated")
    
    def update_quality_metrics(self, metrics: QualityMetrics) -> None:
        """Update quality control metrics."""
        self.quality_metrics = metrics
        self.updated_at = datetime.now(timezone.utc)
        self.logger.info("Quality metrics updated")
    
    def import_from_config(self, config: Config) -> None:
        """Import metadata from existing Config object."""
        # Extract processing parameters
        processing = DataProcessingMetadata()
        
        # Segmentation information
        if hasattr(config, 'segmentation'):
            processing.segmentation_method = config.segmentation.get('method', 'slic')
            processing.segmentation_parameters = {
                'compactness': config.segmentation.get('compactness', 10.0),
                'sigma': config.segmentation.get('sigma', 2.0),
                'scales_um': config.segmentation.get('scales_um', [10.0, 20.0, 40.0])
            }
            processing.spatial_scales_um = config.segmentation.get('scales_um', [10.0, 20.0, 40.0])
        
        # Analysis parameters
        if hasattr(config, 'analysis'):
            analysis_config = config.analysis
            clustering_config = analysis_config.get('clustering', {})
            
            processing.clustering_method = clustering_config.get('method', 'leiden')
            processing.clustering_parameters = {
                'resolution_range': clustering_config.get('resolution_range', [0.5, 2.0]),
                'optimization_method': clustering_config.get('optimization_method', 'stability')
            }
        
        # Processing methods
        if hasattr(config, 'processing'):
            proc_config = config.processing
            processing.transformation_method = proc_config.get('transformation', {}).get('method', 'arcsinh')
            processing.normalization_method = proc_config.get('normalization', {}).get('method')
            
            # Extract nuclear markers
            dna_config = proc_config.get('dna_processing', {})
            processing.nuclear_markers = dna_config.get('channels', ['DNA1', 'DNA2'])
        
        # Channel information -> antibody panel
        if hasattr(config, 'channels'):
            protein_channels = config.channels.get('protein_channels', [])
            
            for protein in protein_channels:
                antibody = AntibodyMetadata(
                    marker_name=protein,
                    metal_tag="",  # Would need to be filled from data
                    specificity_validated=False  # Default
                )
                self.add_antibody(antibody)
        
        self.set_processing_metadata(processing)
        self.logger.info(f"Imported metadata from config: {len(self.antibody_panel)} markers")
    
    def import_from_helpers_metadata(self, metadata_list: List[Metadata]) -> None:
        """Import sample metadata from existing helpers.Metadata objects."""
        for meta in metadata_list:
            sample = SampleMetadata(
                sample_id=meta.replicate_id,
                experimental_group=meta.condition,
                timepoint=str(meta.timepoint) if meta.timepoint is not None else None,
                tissue_type=meta.region
            )
            self.add_sample_metadata(sample)
        
        self.logger.info(f"Imported {len(metadata_list)} sample metadata records")
    
    def extract_from_analysis_manifest(self, manifest_data: Dict[str, Any]) -> None:
        """Extract MI-IMC metadata from analysis manifest."""
        # Study information from scientific objectives
        if 'scientific_objectives' in manifest_data and manifest_data['scientific_objectives']:
            objectives = manifest_data['scientific_objectives']
            
            study = StudyMetadata(
                study_title=objectives.get('primary_research_question', 'IMC Study'),
                research_question=objectives.get('primary_research_question', ''),
                hypotheses=objectives.get('hypotheses', []),
                primary_outcomes=objectives.get('expected_outcomes', [])
            )
            self.set_study_metadata(study)
        
        # Processing information from parameter profile
        if 'parameter_profile' in manifest_data and manifest_data['parameter_profile']:
            profile = manifest_data['parameter_profile']
            
            processing = DataProcessingMetadata()
            
            # Segmentation parameters
            seg_params = profile.get('segmentation_params', {})
            processing.segmentation_method = seg_params.get('method', 'slic')
            processing.segmentation_parameters = seg_params.get('slic_params', {})
            processing.spatial_scales_um = seg_params.get('scales_um', [10.0, 20.0, 40.0])
            
            # Clustering parameters
            clust_params = profile.get('clustering_params', {})
            processing.clustering_method = clust_params.get('method', 'leiden')
            processing.clustering_parameters = clust_params
            
            # Processing parameters
            proc_params = profile.get('processing_params', {})
            processing.transformation_method = proc_params.get('arcsinh_transform', {}).get('method', 'arcsinh')
            processing.normalization_method = proc_params.get('normalization', {}).get('method')
            
            # Expected markers -> antibody panel
            expected_markers = profile.get('expected_markers', [])
            for marker in expected_markers:
                antibody = AntibodyMetadata(
                    marker_name=marker,
                    specificity_validated=True  # Assume validated in manifest
                )
                self.add_antibody(antibody)
            
            self.set_processing_metadata(processing)
        
        # Provenance information
        if 'provenance_info' in manifest_data:
            provenance = manifest_data['provenance_info']
            
            # Update processing metadata with version info
            if hasattr(self.processing_metadata, 'pipeline_version'):
                self.processing_metadata.pipeline_version = provenance.get('git_commit_sha', '')[:8]
            
        self.logger.info("Imported metadata from analysis manifest")
    
    def validate_compliance(self) -> Dict[str, Any]:
        """Validate MI-IMC compliance and generate report."""
        self.validation_errors.clear()
        self.validation_warnings.clear()
        self.compliance_checklist.clear()
        
        # Essential study metadata
        self._validate_study_metadata()
        
        # Sample metadata requirements
        self._validate_sample_metadata()
        
        # Antibody panel requirements
        self._validate_antibody_panel()
        
        # Instrument metadata
        self._validate_instrument_metadata()
        
        # Processing metadata
        self._validate_processing_metadata()
        
        # Quality metrics
        self._validate_quality_metrics()
        
        # Calculate overall compliance score
        total_checks = len(self.compliance_checklist)
        passed_checks = sum(self.compliance_checklist.values())
        compliance_score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        return {
            'compliance_score': compliance_score,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'compliance_checklist': self.compliance_checklist,
            'validation_errors': self.validation_errors,
            'validation_warnings': self.validation_warnings,
            'is_compliant': len(self.validation_errors) == 0 and compliance_score >= 0.8
        }
    
    def _validate_study_metadata(self) -> None:
        """Validate study-level metadata."""
        if self.study_metadata is None:
            self.validation_errors.append("Study metadata is required")
            self.compliance_checklist['study_metadata_present'] = False
            return
        
        self.compliance_checklist['study_metadata_present'] = True
        
        # Required fields
        if not self.study_metadata.study_title:
            self.validation_errors.append("Study title is required")
            self.compliance_checklist['study_title'] = False
        else:
            self.compliance_checklist['study_title'] = True
        
        if not self.study_metadata.research_question:
            self.validation_errors.append("Research question is required")
            self.compliance_checklist['research_question'] = False
        else:
            self.compliance_checklist['research_question'] = True
        
        # Recommended fields
        if not self.study_metadata.principal_investigator:
            self.validation_warnings.append("Principal investigator not specified")
            self.compliance_checklist['principal_investigator'] = False
        else:
            self.compliance_checklist['principal_investigator'] = True
        
        if not self.study_metadata.contact_email:
            self.validation_warnings.append("Contact email not provided")
            self.compliance_checklist['contact_email'] = False
        else:
            self.compliance_checklist['contact_email'] = True
    
    def _validate_sample_metadata(self) -> None:
        """Validate sample metadata."""
        if not self.sample_metadata_list:
            self.validation_errors.append("At least one sample metadata record is required")
            self.compliance_checklist['sample_metadata_present'] = False
            return
        
        self.compliance_checklist['sample_metadata_present'] = True
        
        # Check each sample
        sample_ids = set()
        for i, sample in enumerate(self.sample_metadata_list):
            # Unique sample IDs
            if sample.sample_id in sample_ids:
                self.validation_errors.append(f"Duplicate sample ID: {sample.sample_id}")
            sample_ids.add(sample.sample_id)
            
            # Required fields per sample
            if not sample.tissue_type:
                self.validation_warnings.append(f"Sample {sample.sample_id}: tissue type not specified")
            
            if sample.sample_type is None:
                self.validation_warnings.append(f"Sample {sample.sample_id}: sample type not specified")
        
        self.compliance_checklist['unique_sample_ids'] = len(sample_ids) == len(self.sample_metadata_list)
        self.compliance_checklist['adequate_sample_size'] = len(self.sample_metadata_list) >= 3
    
    def _validate_antibody_panel(self) -> None:
        """Validate antibody panel metadata."""
        if not self.antibody_panel:
            self.validation_errors.append("Antibody panel information is required")
            self.compliance_checklist['antibody_panel_present'] = False
            return
        
        self.compliance_checklist['antibody_panel_present'] = True
        
        # Check for duplicates and completeness
        marker_names = set()
        metal_tags = set()
        validated_count = 0
        
        for antibody in self.antibody_panel:
            # Unique marker names
            if antibody.marker_name in marker_names:
                self.validation_errors.append(f"Duplicate marker: {antibody.marker_name}")
            marker_names.add(antibody.marker_name)
            
            # Metal tag information
            if antibody.metal_tag:
                if antibody.metal_tag in metal_tags:
                    self.validation_errors.append(f"Duplicate metal tag: {antibody.metal_tag}")
                metal_tags.add(antibody.metal_tag)
            else:
                self.validation_warnings.append(f"Metal tag not specified for {antibody.marker_name}")
            
            # Validation status
            if antibody.specificity_validated:
                validated_count += 1
        
        self.compliance_checklist['unique_markers'] = len(marker_names) == len(self.antibody_panel)
        self.compliance_checklist['metal_tags_specified'] = len(metal_tags) >= len(self.antibody_panel) * 0.8
        self.compliance_checklist['antibodies_validated'] = validated_count >= len(self.antibody_panel) * 0.5
        self.compliance_checklist['adequate_panel_size'] = len(self.antibody_panel) >= 5
    
    def _validate_instrument_metadata(self) -> None:
        """Validate instrument and acquisition metadata."""
        # Required fields
        if not self.instrument_metadata.instrument_model:
            self.validation_warnings.append("Instrument model not specified")
            self.compliance_checklist['instrument_model'] = False
        else:
            self.compliance_checklist['instrument_model'] = True
        
        # Acquisition parameters
        if self.instrument_metadata.raster_size_um <= 0:
            self.validation_errors.append("Invalid raster size specified")
            self.compliance_checklist['valid_raster_size'] = False
        else:
            self.compliance_checklist['valid_raster_size'] = True
        
        if not self.instrument_metadata.acquisition_date:
            self.validation_warnings.append("Acquisition date not specified")
            self.compliance_checklist['acquisition_date'] = False
        else:
            self.compliance_checklist['acquisition_date'] = True
    
    def _validate_processing_metadata(self) -> None:
        """Validate data processing metadata."""
        # Pipeline identification
        if not self.processing_metadata.pipeline_name:
            self.validation_warnings.append("Processing pipeline not identified")
            self.compliance_checklist['pipeline_identified'] = False
        else:
            self.compliance_checklist['pipeline_identified'] = True
        
        # Core processing steps documented
        methods_documented = 0
        
        if self.processing_metadata.segmentation_method:
            methods_documented += 1
        if self.processing_metadata.transformation_method:
            methods_documented += 1
        if self.processing_metadata.clustering_method:
            methods_documented += 1
        
        self.compliance_checklist['processing_methods_documented'] = methods_documented >= 3
        
        # Spatial scales specified
        if self.processing_metadata.spatial_scales_um:
            self.compliance_checklist['spatial_scales_specified'] = True
        else:
            self.validation_warnings.append("Spatial analysis scales not specified")
            self.compliance_checklist['spatial_scales_specified'] = False
    
    def _validate_quality_metrics(self) -> None:
        """Validate quality control metrics."""
        metrics = self.quality_metrics
        
        # Basic quality scores should be present
        quality_scores = [
            metrics.sample_quality_score,
            metrics.signal_to_noise_ratio,
            metrics.segmentation_quality_score
        ]
        
        valid_scores = [score for score in quality_scores if score > 0]
        
        self.compliance_checklist['quality_metrics_present'] = len(valid_scores) >= 2
        
        if len(valid_scores) < 2:
            self.validation_warnings.append("Insufficient quality metrics provided")
    
    def generate_publication_report(self, output_format: str = "dict") -> Union[Dict[str, Any], str]:
        """Generate publication-ready metadata report."""
        # Ensure compliance validation is up to date
        compliance = self.validate_compliance()
        
        report = {
            'mi_imc_version': self.version.value,
            'report_generated': datetime.now(timezone.utc).isoformat(),
            'compliance_status': {
                'is_compliant': compliance['is_compliant'],
                'compliance_score': compliance['compliance_score'],
                'checks_passed': f"{compliance['passed_checks']}/{compliance['total_checks']}"
            }
        }
        
        # Study information
        if self.study_metadata:
            report['study_information'] = {
                'title': self.study_metadata.study_title,
                'research_question': self.study_metadata.research_question,
                'hypotheses': self.study_metadata.hypotheses,
                'principal_investigator': self.study_metadata.principal_investigator,
                'institution': self.study_metadata.institution,
                'funding_sources': self.study_metadata.funding_sources
            }
        
        # Sample summary
        report['sample_information'] = {
            'total_samples': len(self.sample_metadata_list),
            'tissue_types': list(set(s.tissue_type for s in self.sample_metadata_list if s.tissue_type)),
            'experimental_groups': list(set(s.experimental_group for s in self.sample_metadata_list if s.experimental_group)),
            'timepoints': list(set(s.timepoint for s in self.sample_metadata_list if s.timepoint))
        }
        
        # Antibody panel summary
        report['antibody_panel'] = {
            'total_markers': len(self.antibody_panel),
            'markers': [ab.marker_name for ab in self.antibody_panel],
            'metal_tags': [ab.metal_tag for ab in self.antibody_panel if ab.metal_tag],
            'validated_antibodies': len([ab for ab in self.antibody_panel if ab.specificity_validated])
        }
        
        # Technical specifications
        report['technical_specifications'] = {
            'instrument_model': self.instrument_metadata.instrument_model,
            'raster_size_um': self.instrument_metadata.raster_size_um,
            'acquisition_frequency_hz': self.instrument_metadata.acquisition_frequency,
            'segmentation_method': self.processing_metadata.segmentation_method,
            'clustering_method': self.processing_metadata.clustering_method,
            'transformation_method': self.processing_metadata.transformation_method,
            'spatial_scales_um': self.processing_metadata.spatial_scales_um
        }
        
        # Quality metrics summary
        report['quality_assessment'] = {
            'sample_quality_score': self.quality_metrics.sample_quality_score,
            'signal_to_noise_ratio': self.quality_metrics.signal_to_noise_ratio,
            'segmentation_quality': self.quality_metrics.segmentation_quality_score,
            'clustering_stability': self.quality_metrics.clustering_stability_score,
            'missing_data_percentage': self.quality_metrics.missing_data_percentage
        }
        
        # Compliance details
        if compliance['validation_errors']:
            report['compliance_issues'] = {
                'errors': compliance['validation_errors'],
                'warnings': compliance['validation_warnings']
            }
        
        if output_format == "json":
            return json.dumps(report, indent=2, default=str)
        elif output_format == "markdown":
            return self._format_report_as_markdown(report)
        else:
            return report
    
    def _format_report_as_markdown(self, report: Dict[str, Any]) -> str:
        """Format publication report as Markdown."""
        md_lines = []
        
        # Title
        md_lines.append("# MI-IMC Metadata Report")
        md_lines.append("")
        
        # Compliance status
        compliance = report['compliance_status']
        status_emoji = "✅" if compliance['is_compliant'] else "⚠️"
        md_lines.append(f"**Compliance Status:** {status_emoji} {compliance['checks_passed']} checks passed")
        md_lines.append(f"**Compliance Score:** {compliance['compliance_score']:.1%}")
        md_lines.append("")
        
        # Study information
        if 'study_information' in report:
            study = report['study_information']
            md_lines.append("## Study Information")
            md_lines.append(f"**Title:** {study.get('title', 'Not specified')}")
            md_lines.append(f"**Research Question:** {study.get('research_question', 'Not specified')}")
            md_lines.append(f"**Principal Investigator:** {study.get('principal_investigator', 'Not specified')}")
            md_lines.append(f"**Institution:** {study.get('institution', 'Not specified')}")
            md_lines.append("")
        
        # Sample information
        sample_info = report['sample_information']
        md_lines.append("## Sample Information")
        md_lines.append(f"**Total Samples:** {sample_info['total_samples']}")
        md_lines.append(f"**Tissue Types:** {', '.join(sample_info['tissue_types'])}")
        md_lines.append(f"**Experimental Groups:** {', '.join(sample_info['experimental_groups'])}")
        md_lines.append("")
        
        # Antibody panel
        panel = report['antibody_panel']
        md_lines.append("## Antibody Panel")
        md_lines.append(f"**Total Markers:** {panel['total_markers']}")
        md_lines.append(f"**Validated Antibodies:** {panel['validated_antibodies']}")
        md_lines.append("**Markers:** " + ", ".join(panel['markers']))
        md_lines.append("")
        
        # Technical specifications
        tech = report['technical_specifications']
        md_lines.append("## Technical Specifications")
        md_lines.append(f"**Instrument:** {tech['instrument_model']}")
        md_lines.append(f"**Raster Size:** {tech['raster_size_um']} μm")
        md_lines.append(f"**Segmentation Method:** {tech['segmentation_method']}")
        md_lines.append(f"**Clustering Method:** {tech['clustering_method']}")
        md_lines.append("")
        
        # Quality assessment
        quality = report['quality_assessment']
        md_lines.append("## Quality Assessment")
        md_lines.append(f"**Sample Quality Score:** {quality['sample_quality_score']:.3f}")
        md_lines.append(f"**Signal-to-Noise Ratio:** {quality['signal_to_noise_ratio']:.3f}")
        md_lines.append(f"**Segmentation Quality:** {quality['segmentation_quality']:.3f}")
        md_lines.append("")
        
        return "\n".join(md_lines)
    
    def export_to_storage(self, storage_backend, dataset_id: str = "mi_imc_metadata") -> None:
        """Export MI-IMC metadata to storage backend."""
        metadata_dict = self.to_dict()
        
        # Add export metadata
        metadata_dict['export_info'] = {
            'export_date': datetime.now(timezone.utc).isoformat(),
            'schema_version': self.version.value,
            'dataset_id': dataset_id
        }
        
        # Export using storage backend
        if hasattr(storage_backend, 'save_roi_analysis'):
            storage_backend.save_roi_analysis(dataset_id, metadata_dict)
        elif hasattr(storage_backend, 'save_analysis_results'):
            storage_backend.save_analysis_results(metadata_dict, dataset_id)
        else:
            raise ValueError("Storage backend does not support metadata export")
        
        self.logger.info(f"MI-IMC metadata exported to storage: {dataset_id}")
    
    def create_migration_report(self, existing_data_dir: Path) -> Dict[str, Any]:
        """Create migration report for existing datasets."""
        migration_report = {
            'migration_date': datetime.now(timezone.utc).isoformat(),
            'source_directory': str(existing_data_dir),
            'migration_status': 'planned',
            'identified_components': {},
            'missing_metadata': [],
            'recommended_actions': []
        }
        
        # Scan for existing metadata
        if existing_data_dir.exists():
            # Look for config files
            config_files = list(existing_data_dir.glob("**/config.json"))
            if config_files:
                migration_report['identified_components']['config_files'] = [str(f) for f in config_files]
            else:
                migration_report['missing_metadata'].append("No config.json files found")
            
            # Look for manifest files
            manifest_files = list(existing_data_dir.glob("**/manifest_*.json"))
            if manifest_files:
                migration_report['identified_components']['manifest_files'] = [str(f) for f in manifest_files]
            
            # Look for metadata files
            metadata_files = list(existing_data_dir.glob("**/*metadata*.json"))
            if metadata_files:
                migration_report['identified_components']['metadata_files'] = [str(f) for f in metadata_files]
            
            # Look for data files
            data_files = list(existing_data_dir.glob("**/*.txt")) + list(existing_data_dir.glob("**/*.csv"))
            migration_report['identified_components']['data_files'] = len(data_files)
        else:
            migration_report['missing_metadata'].append("Source directory does not exist")
        
        # Generate recommendations
        if not config_files:
            migration_report['recommended_actions'].append("Create MI-IMC schema from available data")
        
        if not migration_report['identified_components'].get('metadata_files'):
            migration_report['recommended_actions'].append("Manual metadata collection required")
        
        migration_report['recommended_actions'].append("Run compliance validation after migration")
        
        return migration_report
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for serialization."""
        return {
            'mi_imc_version': self.version.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'study_metadata': asdict(self.study_metadata) if self.study_metadata else None,
            'instrument_metadata': asdict(self.instrument_metadata),
            'sample_metadata_list': [asdict(sample) for sample in self.sample_metadata_list],
            'antibody_panel': [asdict(antibody) for antibody in self.antibody_panel],
            'processing_metadata': asdict(self.processing_metadata),
            'quality_metrics': asdict(self.quality_metrics),
            'compliance_checklist': self.compliance_checklist,
            'validation_errors': self.validation_errors,
            'validation_warnings': self.validation_warnings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MIIMCSchema':
        """Create schema from dictionary."""
        schema = cls(version=MIIMCVersion(data['mi_imc_version']))
        
        schema.created_at = datetime.fromisoformat(data['created_at'])
        schema.updated_at = datetime.fromisoformat(data['updated_at'])
        
        # Reconstruct components
        if data['study_metadata']:
            schema.study_metadata = StudyMetadata(**data['study_metadata'])
        
        schema.instrument_metadata = InstrumentMetadata(**data['instrument_metadata'])
        
        schema.sample_metadata_list = [
            SampleMetadata(**sample_data) for sample_data in data['sample_metadata_list']
        ]
        
        schema.antibody_panel = [
            AntibodyMetadata(**antibody_data) for antibody_data in data['antibody_panel']
        ]
        
        schema.processing_metadata = DataProcessingMetadata(**data['processing_metadata'])
        schema.quality_metrics = QualityMetrics(**data['quality_metrics'])
        
        schema.compliance_checklist = data.get('compliance_checklist', {})
        schema.validation_errors = data.get('validation_errors', [])
        schema.validation_warnings = data.get('validation_warnings', [])
        
        return schema
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save schema to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"MI-IMC schema saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'MIIMCSchema':
        """Load schema from file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


def create_schema_from_config(
    config: Config,
    study_title: str,
    research_question: str,
    principal_investigator: str = "",
    contact_email: str = ""
) -> MIIMCSchema:
    """
    Factory function to create MI-IMC schema from existing Config.
    
    Args:
        config: Existing Config object
        study_title: Title of the study
        research_question: Primary research question
        principal_investigator: PI name
        contact_email: Contact email
        
    Returns:
        Configured MIIMCSchema
    """
    schema = MIIMCSchema()
    
    # Set study metadata
    study = StudyMetadata(
        study_title=study_title,
        research_question=research_question,
        principal_investigator=principal_investigator,
        contact_email=contact_email
    )
    schema.set_study_metadata(study)
    
    # Import from config
    schema.import_from_config(config)
    
    return schema


def create_schema_from_manifest(manifest_dict: Dict[str, Any]) -> MIIMCSchema:
    """
    Create MI-IMC schema from analysis manifest.
    
    Args:
        manifest_dict: Analysis manifest dictionary
        
    Returns:
        MIIMCSchema with imported metadata
    """
    schema = MIIMCSchema()
    schema.extract_from_analysis_manifest(manifest_dict)
    return schema


def migrate_existing_dataset(
    data_directory: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[MIIMCSchema, Dict[str, Any]]:
    """
    Migrate existing dataset to MI-IMC schema.
    
    Args:
        data_directory: Path to existing dataset
        config_path: Optional path to config file
        output_path: Optional output path for migrated schema
        
    Returns:
        Tuple of (migrated_schema, migration_report)
    """
    data_dir = Path(data_directory)
    schema = MIIMCSchema()
    
    # Create migration report
    migration_report = schema.create_migration_report(data_dir)
    
    # Import from config if available
    if config_path and Path(config_path).exists():
        config = Config(str(config_path))
        schema.import_from_config(config)
        migration_report['config_imported'] = True
    
    # Look for existing metadata files
    metadata_files = list(data_dir.glob("**/*metadata*.json"))
    if metadata_files:
        # Try to import from first metadata file
        try:
            with open(metadata_files[0], 'r') as f:
                metadata = json.load(f)
                # Extract sample information if available
                if 'roi_metadata' in metadata:
                    roi_meta = metadata['roi_metadata']
                    sample = SampleMetadata(
                        sample_id=roi_meta.get('filename', 'unknown'),
                        experimental_group=roi_meta.get('condition'),
                        timepoint=str(roi_meta.get('timepoint', '')),
                        tissue_type=roi_meta.get('region')
                    )
                    schema.add_sample_metadata(sample)
                    migration_report['samples_imported'] = 1
        except Exception as e:
            migration_report['import_errors'] = [str(e)]
    
    # Set default study metadata
    if not schema.study_metadata:
        study = StudyMetadata(
            study_title="Migrated IMC Dataset",
            research_question="Spatial analysis of tissue microenvironment",
            study_description=f"Dataset migrated from {data_dir}"
        )
        schema.set_study_metadata(study)
    
    # Save migrated schema if output path provided
    if output_path:
        schema.save(output_path)
        migration_report['output_saved'] = str(output_path)
    
    migration_report['migration_status'] = 'completed'
    migration_report['schema_compliance'] = schema.validate_compliance()
    
    return schema, migration_report


def validate_dataset_compliance(dataset_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate an existing dataset for MI-IMC compliance.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Validation report dictionary
    """
    dataset_dir = Path(dataset_path)
    
    # Try to find and load existing schema
    schema_files = list(dataset_dir.glob("**/mi_imc_schema.json"))
    
    if schema_files:
        # Load existing schema and validate
        schema = MIIMCSchema.load(schema_files[0])
        compliance = schema.validate_compliance()
        compliance['existing_schema_found'] = True
        compliance['schema_file'] = str(schema_files[0])
    else:
        # Create schema from available data and validate
        schema, migration_report = migrate_existing_dataset(dataset_dir)
        compliance = schema.validate_compliance()
        compliance['existing_schema_found'] = False
        compliance['migration_report'] = migration_report
    
    return compliance
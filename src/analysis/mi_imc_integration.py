"""
MI-IMC Schema Integration with Existing IMC Pipeline

Demonstrates seamless integration of MI-IMC metadata schema with existing
Config, storage systems, and analysis workflows. Provides utility functions
for real-world usage scenarios.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

from .mi_imc_schema import (
    MIIMCSchema, StudyMetadata, SampleMetadata, AntibodyMetadata,
    InstrumentMetadata, DataProcessingMetadata, QualityMetrics,
    create_schema_from_config, migrate_existing_dataset
)
from .data_storage import create_storage_backend
from ..config import Config
from ..utils.helpers import Metadata as HelperMetadata
from ..utils.streamlined_loader import StreamlinedIMCLoader

logger = logging.getLogger(__name__)


class MIIMCPipelineIntegration:
    """
    Integration layer between MI-IMC schema and existing IMC pipeline.
    
    Provides seamless workflow integration while maintaining backward
    compatibility with existing code and data structures.
    """
    
    def __init__(self, config: Config, schema: Optional[MIIMCSchema] = None):
        """Initialize integration with config and optional existing schema."""
        self.config = config
        self.schema = schema or MIIMCSchema()
        self.logger = logging.getLogger('MIIMCIntegration')
        
        # Create storage backend for schema persistence
        self.storage_backend = self._create_storage_backend()
    
    def _create_storage_backend(self):
        """Create storage backend using existing config."""
        try:
            storage_config = self.config.output.__dict__ if hasattr(self.config, 'output') else {}
            base_path = getattr(self.config, 'output_dir', Path('results'))
            
            return create_storage_backend(
                storage_config=storage_config,
                base_path=base_path / "mi_imc_metadata"
            )
        except Exception as e:
            self.logger.warning(f"Could not create storage backend: {e}")
            return None
    
    def initialize_from_config(
        self,
        study_title: str,
        research_question: str,
        principal_investigator: str = "",
        contact_email: str = "",
        tissue_type: str = ""
    ) -> Dict[str, Any]:
        """
        Initialize MI-IMC schema from existing Config object.
        
        Args:
            study_title: Title of the study
            research_question: Primary research question
            principal_investigator: Principal investigator name
            contact_email: Contact email
            tissue_type: Type of tissue being analyzed
            
        Returns:
            Initialization summary report
        """
        # Create study metadata
        study = StudyMetadata(
            study_title=study_title,
            research_question=research_question,
            principal_investigator=principal_investigator,
            contact_email=contact_email
        )
        self.schema.set_study_metadata(study)
        
        # Import processing metadata from config
        self.schema.import_from_config(self.config)
        
        # Set instrument metadata with defaults
        instrument = InstrumentMetadata(
            instrument_model="Hyperion",
            acquisition_frequency=200.0,
            raster_size_um=1.0
        )
        self.schema.set_instrument_metadata(instrument)
        
        # Update antibody panel with tissue context
        if tissue_type:
            for antibody in self.schema.antibody_panel:
                antibody.target_cellular_location = self._infer_cellular_location(antibody.marker_name)
                antibody.expected_cell_types = self._infer_cell_types(antibody.marker_name, tissue_type)
        
        # Generate initialization report
        compliance = self.schema.validate_compliance()
        
        report = {
            'initialization_date': self.schema.updated_at.isoformat(),
            'study_title': study_title,
            'markers_imported': len(self.schema.antibody_panel),
            'processing_methods_captured': bool(self.schema.processing_metadata.segmentation_method),
            'compliance_score': compliance['compliance_score'],
            'immediate_issues': compliance['validation_errors'],
            'recommendations': compliance['validation_warnings']
        }
        
        self.logger.info(f"MI-IMC schema initialized: {report['markers_imported']} markers, "
                        f"compliance score: {compliance['compliance_score']:.1%}")
        
        return report
    
    def import_sample_metadata_from_loader(self, loader: StreamlinedIMCLoader) -> Dict[str, Any]:
        """
        Import sample metadata from StreamlinedIMCLoader.
        
        Args:
            loader: Configured StreamlinedIMCLoader instance
            
        Returns:
            Import summary report
        """
        metadata_df = loader.metadata_df
        
        if metadata_df.empty:
            return {
                'samples_imported': 0,
                'error': 'No metadata available in loader'
            }
        
        samples_imported = 0
        
        for _, row in metadata_df.iterrows():
            sample = SampleMetadata(
                sample_id=str(row.get('roi_id', 'unknown')),
                patient_id=str(row.get('mouse', '')),
                experimental_group=str(row.get('condition', '')),
                timepoint=str(row.get('timepoint', '')),
                tissue_type=str(row.get('region', '')),
                batch_id=str(row.get('batch', ''))
            )
            
            self.schema.add_sample_metadata(sample)
            samples_imported += 1
        
        # Update quality metrics from loader data
        quality_df = loader.quality_df
        if not quality_df.empty:
            avg_quality = quality_df.select_dtypes(include=[float, int]).mean()
            
            quality_metrics = QualityMetrics(
                sample_quality_score=avg_quality.get('overall_quality', 0.0),
                signal_to_noise_ratio=avg_quality.get('signal_to_noise', 0.0),
                segmentation_quality_score=avg_quality.get('coordinate_quality', 0.0),
                tissue_coverage_percent=avg_quality.get('tissue_coverage', 0.0)
            )
            
            self.schema.update_quality_metrics(quality_metrics)
        
        report = {
            'samples_imported': samples_imported,
            'quality_metrics_updated': not quality_df.empty,
            'unique_conditions': len(metadata_df['condition'].unique()) if 'condition' in metadata_df.columns else 0,
            'unique_timepoints': len(metadata_df['timepoint'].unique()) if 'timepoint' in metadata_df.columns else 0
        }
        
        self.logger.info(f"Imported {samples_imported} sample metadata records from loader")
        
        return report
    
    def import_from_analysis_manifest(self, manifest_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Import metadata from existing analysis manifest.
        
        Args:
            manifest_path: Path to analysis manifest file
            
        Returns:
            Import summary report
        """
        manifest_file = Path(manifest_path)
        
        if not manifest_file.exists():
            return {
                'success': False,
                'error': f'Manifest file not found: {manifest_path}'
            }
        
        try:
            with open(manifest_file, 'r') as f:
                manifest_data = json.load(f)
            
            # Extract metadata from manifest
            self.schema.extract_from_analysis_manifest(manifest_data)
            
            # Update study metadata if available
            if 'scientific_objectives' in manifest_data and manifest_data['scientific_objectives']:
                objectives = manifest_data['scientific_objectives']
                
                if not self.schema.study_metadata:
                    study = StudyMetadata(
                        study_title=objectives.get('primary_research_question', 'IMC Study'),
                        research_question=objectives.get('primary_research_question', ''),
                        hypotheses=objectives.get('hypotheses', [])
                    )
                    self.schema.set_study_metadata(study)
            
            report = {
                'success': True,
                'manifest_file': str(manifest_file),
                'markers_imported': len(self.schema.antibody_panel),
                'processing_methods_captured': bool(self.schema.processing_metadata.clustering_method),
                'provenance_captured': bool(manifest_data.get('provenance_info'))
            }
            
            self.logger.info(f"Imported metadata from manifest: {manifest_file}")
            
            return report
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to import manifest: {str(e)}'
            }
    
    def update_quality_metrics_from_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update quality metrics from analysis results.
        
        Args:
            analysis_results: Results from IMC analysis pipeline
            
        Returns:
            Update summary report
        """
        metrics_updated = {}
        
        # Extract quality metrics from multiscale results
        if 'multiscale_results' in analysis_results:
            multiscale = analysis_results['multiscale_results']
            
            quality_scores = []
            
            # Aggregate quality across scales
            for scale_key, scale_data in multiscale.items():
                if isinstance(scale_data, dict):
                    # Extract various quality indicators
                    if 'silhouette_score' in scale_data:
                        quality_scores.append(scale_data['silhouette_score'])
                    
                    if 'optimization_results' in scale_data:
                        opt_results = scale_data['optimization_results']
                        if 'stability_score' in opt_results:
                            quality_scores.append(opt_results['stability_score'])
            
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                self.schema.quality_metrics.clustering_stability_score = avg_quality
                metrics_updated['clustering_stability'] = avg_quality
        
        # Extract consistency metrics
        if 'consistency_results' in analysis_results:
            consistency = analysis_results['consistency_results']
            
            if 'overall' in consistency:
                overall_consistency = consistency['overall']
                
                # Use ARI as spatial coherence indicator
                if 'mean_ari' in overall_consistency:
                    self.schema.quality_metrics.spatial_coherence_score = overall_consistency['mean_ari']
                    metrics_updated['spatial_coherence'] = overall_consistency['mean_ari']
        
        # Calculate missing data percentage
        if 'metadata' in analysis_results:
            metadata = analysis_results['metadata']
            n_measurements = metadata.get('n_measurements', 1)
            
            # Estimate missing data from feature matrix
            if 'feature_matrix' in analysis_results:
                feature_matrix = analysis_results['feature_matrix']
                if hasattr(feature_matrix, 'shape') and len(feature_matrix.shape) > 0:
                    missing_percentage = (1.0 - len(feature_matrix) / n_measurements) * 100
                    self.schema.quality_metrics.missing_data_percentage = missing_percentage
                    metrics_updated['missing_data_percentage'] = missing_percentage
        
        report = {
            'metrics_updated': list(metrics_updated.keys()),
            'quality_scores': metrics_updated,
            'analysis_date': analysis_results.get('metadata', {}).get('analysis_date', 'unknown')
        }
        
        self.logger.info(f"Updated {len(metrics_updated)} quality metrics from analysis results")
        
        return report
    
    def generate_publication_metadata(self, output_format: str = "dict") -> Union[Dict[str, Any], str]:
        """
        Generate publication-ready metadata report.
        
        Args:
            output_format: Output format ("dict", "json", "markdown")
            
        Returns:
            Publication-ready metadata in requested format
        """
        # Ensure schema is up to date
        compliance = self.schema.validate_compliance()
        
        # Generate publication report
        pub_report = self.schema.generate_publication_report(output_format)
        
        # Add pipeline-specific information
        if output_format == "dict":
            pub_report['pipeline_information'] = {
                'config_file_used': str(self.config.config_path) if hasattr(self.config, 'config_path') else 'unknown',
                'output_directory': str(self.config.output_dir) if hasattr(self.config, 'output_dir') else 'unknown',
                'protein_channels_configured': len(getattr(self.config, 'proteins', [])),
                'scales_analyzed': getattr(self.config.segmentation, 'scales_um', []) if hasattr(self.config, 'segmentation') else []
            }
        
        return pub_report
    
    def save_schema_to_storage(self, dataset_id: str = "mi_imc_metadata") -> Dict[str, Any]:
        """
        Save MI-IMC schema to configured storage backend.
        
        Args:
            dataset_id: Identifier for the dataset
            
        Returns:
            Save operation report
        """
        if not self.storage_backend:
            return {
                'success': False,
                'error': 'No storage backend available'
            }
        
        try:
            self.schema.export_to_storage(self.storage_backend, dataset_id)
            
            # Also save publication report
            pub_report = self.generate_publication_metadata()
            publication_id = f"{dataset_id}_publication_metadata"
            
            if hasattr(self.storage_backend, 'save_roi_analysis'):
                self.storage_backend.save_roi_analysis(publication_id, pub_report)
            
            report = {
                'success': True,
                'dataset_id': dataset_id,
                'publication_metadata_id': publication_id,
                'storage_backend': str(type(self.storage_backend).__name__),
                'compliance_score': self.schema.validate_compliance()['compliance_score']
            }
            
            self.logger.info(f"MI-IMC schema saved to storage: {dataset_id}")
            
            return report
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to save to storage: {str(e)}'
            }
    
    def create_compliance_report(self) -> Dict[str, Any]:
        """
        Create comprehensive compliance report for the current schema.
        
        Returns:
            Detailed compliance report
        """
        # Validate compliance
        compliance = self.schema.validate_compliance()
        
        # Add context from pipeline
        enhanced_report = compliance.copy()
        enhanced_report.update({
            'pipeline_context': {
                'config_source': str(self.config.config_path) if hasattr(self.config, 'config_path') else 'unknown',
                'proteins_configured': len(getattr(self.config, 'proteins', [])),
                'has_storage_backend': self.storage_backend is not None,
                'schema_version': self.schema.version.value
            },
            'recommendations': self._generate_compliance_recommendations(compliance),
            'priority_actions': self._prioritize_compliance_actions(compliance)
        })
        
        return enhanced_report
    
    def _generate_compliance_recommendations(self, compliance: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on compliance results."""
        recommendations = []
        
        if not compliance['is_compliant']:
            if 'Study metadata is required' in compliance['validation_errors']:
                recommendations.append("Use initialize_from_config() to set basic study metadata")
            
            if any('antibody' in error.lower() for error in compliance['validation_errors']):
                recommendations.append("Use update_antibody_panel() to add metal tags and validation status")
            
            if compliance['compliance_score'] < 0.5:
                recommendations.append("Consider using migrate_existing_dataset() for comprehensive metadata collection")
        
        if len(self.schema.sample_metadata_list) < 3:
            recommendations.append("Add more sample metadata using import_sample_metadata_from_loader()")
        
        if self.schema.quality_metrics.sample_quality_score == 0:
            recommendations.append("Update quality metrics using update_quality_metrics_from_analysis()")
        
        return recommendations
    
    def _prioritize_compliance_actions(self, compliance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize compliance actions by importance."""
        actions = []
        
        # Critical actions (affect compliance status)
        for error in compliance['validation_errors']:
            actions.append({
                'priority': 'critical',
                'action': f"Fix error: {error}",
                'category': 'compliance'
            })
        
        # High priority actions (improve compliance score)
        missing_checks = [
            check for check, passed in compliance['compliance_checklist'].items()
            if not passed
        ]
        
        for check in missing_checks[:3]:  # Top 3 missing checks
            actions.append({
                'priority': 'high',
                'action': f"Address missing requirement: {check}",
                'category': 'improvement'
            })
        
        # Medium priority actions (warnings)
        for warning in compliance['validation_warnings'][:2]:  # Top 2 warnings
            actions.append({
                'priority': 'medium',
                'action': f"Address warning: {warning}",
                'category': 'enhancement'
            })
        
        return actions
    
    def _infer_cellular_location(self, marker_name: str) -> str:
        """Infer cellular location from marker name."""
        marker_lower = marker_name.lower()
        
        membrane_markers = ['cd', 'pecam', 'vcam', 'icam', 'sma']
        nuclear_markers = ['dna', 'ki67', 'pcna', 'p53']
        cytoplasm_markers = ['cytokeratin', 'vimentin', 'actin', 'tubulin']
        
        if any(mem in marker_lower for mem in membrane_markers):
            return "membrane"
        elif any(nuc in marker_lower for nuc in nuclear_markers):
            return "nucleus"
        elif any(cyto in marker_lower for cyto in cytoplasm_markers):
            return "cytoplasm"
        else:
            return "unknown"
    
    def _infer_cell_types(self, marker_name: str, tissue_type: str) -> List[str]:
        """Infer expected cell types from marker and tissue context."""
        marker_lower = marker_name.lower()
        tissue_lower = tissue_type.lower()
        
        cell_types = []
        
        # Common immune markers
        if 'cd45' in marker_lower:
            cell_types.extend(['leukocytes', 'immune_cells'])
        elif 'cd11b' in marker_lower:
            cell_types.extend(['macrophages', 'monocytes'])
        elif 'cd206' in marker_lower:
            cell_types.extend(['m2_macrophages'])
        elif 'ly6g' in marker_lower:
            cell_types.extend(['neutrophils'])
        
        # Endothelial markers
        elif 'cd31' in marker_lower or 'pecam' in marker_lower:
            cell_types.extend(['endothelial_cells'])
        elif 'cd34' in marker_lower:
            cell_types.extend(['endothelial_cells', 'stem_cells'])
        
        # Fibroblast markers
        elif 'cd140' in marker_lower or 'pdgfr' in marker_lower:
            cell_types.extend(['fibroblasts'])
        elif 'sma' in marker_lower or 'acta2' in marker_lower:
            cell_types.extend(['smooth_muscle_cells', 'myofibroblasts'])
        
        # Tissue-specific additions
        if 'kidney' in tissue_lower:
            if 'cd44' in marker_lower:
                cell_types.extend(['tubular_epithelial_cells'])
        
        return cell_types or ['unknown']


def create_integration_from_config(
    config_path: Union[str, Path],
    study_title: str,
    research_question: str,
    principal_investigator: str = "",
    contact_email: str = ""
) -> MIIMCPipelineIntegration:
    """
    Factory function to create integration from config file.
    
    Args:
        config_path: Path to configuration file
        study_title: Title of the study
        research_question: Primary research question
        principal_investigator: Principal investigator name
        contact_email: Contact email
        
    Returns:
        Configured MIIMCPipelineIntegration
    """
    config = Config(str(config_path))
    integration = MIIMCPipelineIntegration(config)
    
    initialization_report = integration.initialize_from_config(
        study_title=study_title,
        research_question=research_question,
        principal_investigator=principal_investigator,
        contact_email=contact_email
    )
    
    logger.info(f"Integration created: {initialization_report['markers_imported']} markers imported")
    
    return integration


def create_integration_from_existing_data(
    data_directory: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    study_title: str = "Migrated IMC Study",
    research_question: str = "Spatial analysis of tissue microenvironment"
) -> Tuple[MIIMCPipelineIntegration, Dict[str, Any]]:
    """
    Create integration from existing dataset with migration.
    
    Args:
        data_directory: Path to existing dataset
        config_path: Optional path to config file
        study_title: Title for the study
        research_question: Research question
        
    Returns:
        Tuple of (integration, migration_report)
    """
    # Load config
    if config_path and Path(config_path).exists():
        config = Config(str(config_path))
    else:
        # Create minimal config
        config = Config()  # Will use defaults
    
    # Migrate dataset to MI-IMC schema
    schema, migration_report = migrate_existing_dataset(
        data_directory=data_directory,
        config_path=config_path
    )
    
    # Create integration
    integration = MIIMCPipelineIntegration(config, schema)
    
    # Update study metadata if not already set
    if not schema.study_metadata:
        integration.initialize_from_config(
            study_title=study_title,
            research_question=research_question
        )
    
    return integration, migration_report


def demonstrate_complete_workflow(
    config_path: Union[str, Path],
    data_directory: Union[str, Path],
    output_directory: Union[str, Path]
) -> Dict[str, Any]:
    """
    Demonstrate complete MI-IMC workflow integration.
    
    Args:
        config_path: Path to configuration file
        data_directory: Path to data directory
        output_directory: Path for outputs
        
    Returns:
        Workflow demonstration report
    """
    workflow_report = {
        'workflow_start': str(pd.Timestamp.now()),
        'steps_completed': [],
        'outputs_generated': [],
        'compliance_evolution': []
    }
    
    try:
        # Step 1: Create integration from config
        integration = create_integration_from_config(
            config_path=config_path,
            study_title="Demonstration IMC Study",
            research_question="How does spatial organization change with treatment?",
            principal_investigator="Dr. IMC Researcher",
            contact_email="researcher@university.edu"
        )
        workflow_report['steps_completed'].append("Integration created from config")
        
        # Check initial compliance
        compliance1 = integration.create_compliance_report()
        workflow_report['compliance_evolution'].append({
            'step': 'initial',
            'score': compliance1['compliance_score']
        })
        
        # Step 2: Import sample metadata if loader available
        try:
            loader = StreamlinedIMCLoader(integration.config.to_dict())
            import_report = integration.import_sample_metadata_from_loader(loader)
            workflow_report['steps_completed'].append(f"Imported {import_report['samples_imported']} samples")
            
            # Check compliance after sample import
            compliance2 = integration.create_compliance_report()
            workflow_report['compliance_evolution'].append({
                'step': 'after_sample_import',
                'score': compliance2['compliance_score']
            })
            
        except Exception as e:
            workflow_report['steps_completed'].append(f"Sample import failed: {str(e)}")
        
        # Step 3: Update antibody panel with realistic metadata
        for antibody in integration.schema.antibody_panel[:3]:  # Update first 3 antibodies
            antibody.specificity_validated = True
            antibody.antibody_clone = f"Clone_{antibody.marker_name}_01"
            antibody.metal_tag = f"{150 + len(antibody.marker_name)}Dy"  # Dummy metal tag
        
        workflow_report['steps_completed'].append("Updated antibody panel metadata")
        
        # Step 4: Generate publication metadata
        pub_metadata = integration.generate_publication_metadata("markdown")
        
        # Save outputs
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save schema
        schema_path = output_dir / "mi_imc_schema.json"
        integration.schema.save(schema_path)
        workflow_report['outputs_generated'].append(str(schema_path))
        
        # Save publication metadata
        pub_path = output_dir / "publication_metadata.md"
        with open(pub_path, 'w') as f:
            f.write(pub_metadata)
        workflow_report['outputs_generated'].append(str(pub_path))
        
        # Save compliance report
        final_compliance = integration.create_compliance_report()
        compliance_path = output_dir / "compliance_report.json"
        with open(compliance_path, 'w') as f:
            json.dump(final_compliance, f, indent=2, default=str)
        workflow_report['outputs_generated'].append(str(compliance_path))
        
        workflow_report['compliance_evolution'].append({
            'step': 'final',
            'score': final_compliance['compliance_score']
        })
        
        # Save to storage backend if available
        save_report = integration.save_schema_to_storage("demonstration_dataset")
        if save_report['success']:
            workflow_report['steps_completed'].append("Saved to storage backend")
        
        workflow_report['workflow_status'] = 'completed_successfully'
        workflow_report['final_compliance_score'] = final_compliance['compliance_score']
        
    except Exception as e:
        workflow_report['workflow_status'] = 'failed'
        workflow_report['error'] = str(e)
    
    workflow_report['workflow_end'] = str(pd.Timestamp.now())
    
    return workflow_report


if __name__ == "__main__":
    # Example usage and testing
    import pandas as pd
    
    print("MI-IMC Schema Integration Demonstration")
    print("=" * 50)
    
    # This would typically be run with real config and data paths
    # For demonstration, we'll show the structure
    
    demo_config_path = Path("config.json")
    demo_data_dir = Path("data/example_imc")
    demo_output_dir = Path("results/mi_imc_demo")
    
    if demo_config_path.exists():
        report = demonstrate_complete_workflow(
            config_path=demo_config_path,
            data_directory=demo_data_dir,
            output_directory=demo_output_dir
        )
        
        print(f"Workflow Status: {report['workflow_status']}")
        print(f"Steps Completed: {len(report['steps_completed'])}")
        print(f"Outputs Generated: {len(report['outputs_generated'])}")
        print(f"Final Compliance Score: {report.get('final_compliance_score', 'N/A')}")
    else:
        print("Demo config file not found. Integration is ready for real data.")
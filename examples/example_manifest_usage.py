#!/usr/bin/env python3
"""
Example usage of the AnalysisManifest system.

This script demonstrates how to create, use, and validate analysis manifests
for reproducible IMC analysis.
"""

import tempfile
import json
from pathlib import Path

# Mock the Config class for demonstration
class MockConfig:
    def __init__(self, config_path):
        self.channels = {
            'protein_channels': ['CD45', 'CD31', 'CD11b', 'CD206', 'Ly6G'],
            'dna_channels': ['DNA1', 'DNA2']
        }
        self.segmentation = {
            'method': 'slic',
            'scales_um': [10, 20, 40],
            'slic_params': {'compactness': 10.0, 'sigma': 1.5}
        }
        self.analysis = {
            'clustering': {
                'method': 'leiden',
                'resolution_range': [0.5, 2.0],
                'optimization_method': 'stability'
            }
        }
        self.processing = {
            'arcsinh_transform': {'cofactor': 1.0},
            'normalization': {'method': 'standardize'}
        }
        self.quality_control = {
            'thresholds': {'min_signal': 10.0}
        }
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def to_dict(self):
        return {
            'channels': self.channels,
            'segmentation': self.segmentation,
            'analysis': self.analysis,
            'processing': self.processing,
            'quality_control': self.quality_control
        }


def example_basic_manifest():
    """Example 1: Create a basic manifest."""
    print("=== Example 1: Basic Manifest Creation ===")
    
    # For this example, we'll work with the manifest components directly
    # since we don't have numpy/pandas installed
    
    print("Creating a basic analysis manifest...")
    
    # In real usage, you would import:
    # from src.analysis.analysis_manifest import AnalysisManifest, ParameterProfile, ScientificObjectives
    
    # For demonstration, we'll show the structure
    manifest_structure = {
        'manifest_id': 'manifest_20241003_example_12345678',
        'version': '1.0',
        'created_at': '2024-10-03T14:25:30+00:00',
        'updated_at': '2024-10-03T14:25:30+00:00',
        
        'dataset_fingerprint': {
            'files': {
                'roi_1.txt': 'sha256_hash_1',
                'roi_2.txt': 'sha256_hash_2',
                'roi_3.txt': 'sha256_hash_3'
            },
            'total_files': 3,
            'total_size_bytes': 1048576,  # 1MB
            'metadata_hash': 'metadata_sha256_hash'
        },
        
        'parameter_profile': {
            'name': 'kidney_injury_analysis',
            'description': 'Multi-scale analysis optimized for kidney injury studies',
            'tissue_type': 'kidney',
            'segmentation_params': {
                'method': 'slic',
                'scales_um': [10, 20, 40],
                'slic_params': {'compactness': 10.0, 'sigma': 1.5}
            },
            'clustering_params': {
                'method': 'leiden',
                'resolution_range': [0.5, 2.0]
            },
            'expected_markers': ['CD45', 'CD31', 'CD11b', 'CD206', 'Ly6G'],
            'marker_groups': {
                'immune': ['CD45', 'CD11b', 'CD206', 'Ly6G'],
                'vascular': ['CD31']
            }
        },
        
        'scientific_objectives': {
            'primary_research_question': 'How does spatial cellular organization change during kidney injury repair?',
            'hypotheses': [
                'Neutrophil recruitment occurs primarily along vascular networks within 24h',
                'Macrophage polarization shows spatial clustering by day 3'
            ],
            'target_cell_types': ['CD45+ leukocytes', 'CD31+ endothelium', 'CD206+ macrophages'],
            'tissue_context': 'Murine kidney cortex following ischemia-reperfusion injury'
        },
        
        'provenance_info': {
            'git_commit_sha': 'a1b2c3d4e5f67890abcdef1234567890abcdef12',
            'git_branch': 'main',
            'python_version': '3.12.0',
            'platform': 'macOS-14.0-arm64'
        },
        
        'signature_method': 'gpg',
        'signature': None,  # Would contain PGP signature
        'signer_info': None,
        
        'execution_history': [],
        'deviation_log': []
    }
    
    print(f"✓ Created manifest structure: {manifest_structure['manifest_id']}")
    print(f"  Profile: {manifest_structure['parameter_profile']['name']}")
    print(f"  Research Question: {manifest_structure['scientific_objectives']['primary_research_question']}")
    print(f"  Dataset: {manifest_structure['dataset_fingerprint']['total_files']} files")
    print()


def example_parameter_profile():
    """Example 2: Creating parameter profiles."""
    print("=== Example 2: Parameter Profile Creation ===")
    
    # Kidney-specific profile
    kidney_profile = {
        'name': 'kidney_injury_multiscale',
        'description': 'Optimized for kidney tissue injury and repair studies',
        'tissue_type': 'kidney',
        
        'segmentation_params': {
            'method': 'slic',
            'scales_um': [10, 20, 40],  # Capillary, tubular, architectural scales
            'slic_params': {
                'compactness': 10.0,  # Good for kidney morphology
                'sigma': 1.5,  # Preserve fine structures
                'n_segments_per_mm2': 2500
            }
        },
        
        'clustering_params': {
            'method': 'leiden',
            'resolution_range': [0.5, 2.0],
            'optimization_method': 'stability',
            'use_coabundance_features': True
        },
        
        'processing_params': {
            'arcsinh_transform': {
                'optimization_method': 'percentile',
                'percentile_threshold': 5.0
            },
            'normalization': {
                'method': 'standardize',
                'per_channel': True
            }
        },
        
        'quality_thresholds': {
            'min_signal_to_background': 3.0,
            'min_tissue_coverage_percent': 10,
            'max_calibration_cv': 0.2
        },
        
        'expected_markers': [
            'CD45',   # Pan-leukocyte
            'CD11b',  # Myeloid
            'Ly6G',   # Neutrophils
            'CD206',  # M2 macrophages
            'CD31',   # Endothelium
            'CD140a', # PDGFRα (fibroblasts)
            'CD44'    # Adhesion molecule
        ],
        
        'marker_groups': {
            'immune_markers': {
                'pan_leukocyte': ['CD45'],
                'myeloid': ['CD11b', 'Ly6G', 'CD206']
            },
            'stromal_markers': {
                'fibroblasts': ['CD140a']
            },
            'vascular_markers': {
                'endothelial': ['CD31']
            },
            'adhesion_markers': ['CD44']
        },
        
        'scientific_rationale': (
            'Multi-scale analysis captures both cellular (10μm) and tissue-level (40μm) '
            'organization. SLIC segmentation preserves kidney tubular structure while '
            'Leiden clustering with coabundance features captures functional cell states.'
        ),
        
        'literature_references': [
            'doi:10.1038/s41586-019-1631-3',  # Kidney injury atlas
            'doi:10.1016/j.cell.2019.06.025'  # Spatial omics methods
        ]
    }
    
    print("✓ Created kidney-specific parameter profile:")
    print(f"  Name: {kidney_profile['name']}")
    print(f"  Tissue: {kidney_profile['tissue_type']}")
    print(f"  Scales: {kidney_profile['segmentation_params']['scales_um']} μm")
    print(f"  Markers: {len(kidney_profile['expected_markers'])} proteins")
    print(f"  Rationale: {kidney_profile['scientific_rationale'][:80]}...")
    print()
    
    # Brain-specific profile (different parameters)
    brain_profile = {
        'name': 'brain_cortex_analysis',
        'description': 'Optimized for brain cortical tissue analysis',
        'tissue_type': 'brain',
        
        'segmentation_params': {
            'method': 'slic',
            'scales_um': [5, 15, 30],  # Smaller scales for brain
            'slic_params': {
                'compactness': 15.0,  # Higher compactness for brain morphology
                'sigma': 1.0
            }
        },
        
        'expected_markers': [
            'NeuN',    # Neurons
            'GFAP',    # Astrocytes  
            'Iba1',    # Microglia
            'MBP',     # Oligodendrocytes
            'CD31'     # Vasculature
        ],
        
        'marker_groups': {
            'neural_cells': ['NeuN', 'GFAP', 'Iba1', 'MBP'],
            'vascular': ['CD31']
        }
    }
    
    print("✓ Created brain-specific parameter profile:")
    print(f"  Name: {brain_profile['name']}")
    print(f"  Tissue: {brain_profile['tissue_type']}")
    print(f"  Scales: {brain_profile['segmentation_params']['scales_um']} μm")
    print()


def example_scientific_objectives():
    """Example 3: Defining scientific objectives."""
    print("=== Example 3: Scientific Objectives ===")
    
    # Comprehensive research objectives
    objectives = {
        'primary_research_question': (
            'How does spatial cellular organization change during the progression '
            'from acute kidney injury to chronic kidney disease?'
        ),
        
        'hypotheses': [
            'Neutrophil recruitment occurs preferentially along peritubular capillaries within 24h of injury',
            'Macrophage polarization shows distinct spatial clustering patterns by day 3 post-injury',
            'Fibroblast activation creates organized repair zones that predict long-term outcomes',
            'Vascular remodeling correlates with local immune cell density and activation state'
        ],
        
        'expected_outcomes': [
            'Quantified spatial relationships between immune and vascular compartments',
            'Temporal progression maps of cellular organization changes',
            'Validated spatial biomarkers of tissue repair vs. fibrosis progression',
            'Mechanistic insights into cell-cell communication patterns'
        ],
        
        'target_cell_types': [
            'CD45+ leukocytes (all immune cells)',
            'CD11b+ myeloid cells',
            'Ly6G+ neutrophils', 
            'CD206+ M2 macrophages',
            'CD31+ endothelial cells',
            'CD140a+ fibroblasts'
        ],
        
        'spatial_scales_of_interest': [
            'Cellular interactions (10μm) - direct cell contact and paracrine signaling',
            'Microenvironment organization (20μm) - local tissue architecture',
            'Tissue-level patterns (40μm) - cortical vs medullary organization'
        ],
        
        'tissue_context': (
            'Murine kidney cortex following bilateral ischemia-reperfusion injury. '
            'Analysis focuses on the transition zone between healthy and injured tissue.'
        ),
        
        'experimental_conditions': [
            'Sham control (no injury)',
            '1 day post-injury (acute inflammatory response)',
            '3 days post-injury (peak inflammation and early repair)',
            '7 days post-injury (resolution vs. chronic inflammation)',
            '14 days post-injury (repair completion vs. fibrosis onset)'
        ],
        
        'success_metrics': [
            'Spatial clustering metrics with statistical significance (p < 0.05)',
            'Reproducible patterns across biological replicates (n≥5 per group)',
            'Correlation with functional outcomes (serum creatinine, histology)',
            'Cross-validation with independent dataset',
            'Clinical relevance assessment via human tissue correlation'
        ],
        
        'quality_requirements': {
            'spatial_resolution': 'Subcellular protein localization preserved',
            'statistical_power': 'Minimum 80% power to detect 20% effect size',
            'reproducibility': 'ICC > 0.75 for key spatial metrics across technical replicates',
            'biological_validity': 'Consistent with known kidney injury pathophysiology'
        }
    }
    
    print("✓ Defined comprehensive scientific objectives:")
    print(f"  Research Question: {objectives['primary_research_question'][:80]}...")
    print(f"  Hypotheses: {len(objectives['hypotheses'])} specific hypotheses")
    print(f"  Target Cells: {len(objectives['target_cell_types'])} cell types")
    print(f"  Conditions: {len(objectives['experimental_conditions'])} experimental groups")
    print(f"  Success Metrics: {len(objectives['success_metrics'])} quantitative criteria")
    print()


def example_execution_tracking():
    """Example 4: Execution tracking and parameter deviations."""
    print("=== Example 4: Execution Tracking ===")
    
    # Example execution history
    execution_history = [
        {
            'timestamp': '2024-10-03T14:30:00+00:00',
            'step_name': 'dataset_fingerprinting',
            'step_type': 'preprocessing',
            'parameters': {
                'data_directory': 'data/241218_IMC_Alun',
                'file_pattern': '*.txt',
                'total_files': 25
            },
            'results_summary': {
                'total_size_mb': 125.5,
                'overall_hash': 'a1b2c3d4...'
            }
        },
        {
            'timestamp': '2024-10-03T14:32:15+00:00',
            'step_name': 'multiscale_analysis_cortex_001',
            'step_type': 'analysis',
            'parameters': {
                'roi_id': 'cortex_001',
                'scales_um': [10, 20, 40],
                'method': 'leiden',
                'use_slic': True,
                'n_measurements': 12847
            },
            'results_summary': {
                'n_clusters': 8,
                'silhouette_score': 0.75,
                'scale_consistency': 0.82
            }
        },
        {
            'timestamp': '2024-10-03T14:35:42+00:00',
            'step_name': 'quality_control_validation',
            'step_type': 'validation',
            'parameters': {
                'check_signal_to_background': True,
                'check_spatial_artifacts': True,
                'check_calibration_drift': True
            },
            'results_summary': {
                'qc_passed': True,
                'avg_snr': 4.2,
                'calibration_cv': 0.15
            }
        }
    ]
    
    print("✓ Example execution history:")
    for i, step in enumerate(execution_history, 1):
        print(f"  {i}. {step['step_name']} ({step['step_type']})")
        print(f"     Time: {step['timestamp']}")
        if 'n_clusters' in step['results_summary']:
            print(f"     Result: {step['results_summary']['n_clusters']} clusters, "
                  f"silhouette = {step['results_summary']['silhouette_score']}")
        elif 'qc_passed' in step['results_summary']:
            print(f"     Result: QC {'passed' if step['results_summary']['qc_passed'] else 'failed'}")
    print()
    
    # Example parameter deviations
    deviation_log = [
        {
            'timestamp': '2024-10-03T14:33:10+00:00',
            'parameter_path': 'clustering.resolution',
            'original_value': 1.0,
            'new_value': 1.5,
            'reason': 'Better cluster separation observed for high-density cortical regions'
        },
        {
            'timestamp': '2024-10-03T14:38:22+00:00',
            'parameter_path': 'segmentation.slic_params.compactness',
            'original_value': 10.0,
            'new_value': 12.0,
            'reason': 'Improved boundary adherence for tubular structures in ROI cortex_003'
        },
        {
            'timestamp': '2024-10-03T14:42:55+00:00',
            'parameter_path': 'quality_control.min_signal_threshold',
            'original_value': 10.0,
            'new_value': 8.0,
            'reason': 'Accommodate lower signal in medullary regions while maintaining specificity'
        }
    ]
    
    print("✓ Example parameter deviations:")
    for i, dev in enumerate(deviation_log, 1):
        print(f"  {i}. {dev['parameter_path']}: {dev['original_value']} → {dev['new_value']}")
        print(f"     Reason: {dev['reason']}")
        print(f"     Time: {dev['timestamp']}")
    print()


def example_cli_usage():
    """Example 5: CLI usage patterns."""
    print("=== Example 5: CLI Usage Patterns ===")
    
    print("1. Create a new manifest:")
    print("   python manifest_cli.py create \\")
    print("     --config-path config.json \\")
    print("     --data-directory data/241218_IMC_Alun \\")
    print("     --research-question 'How does injury affect spatial organization?' \\")
    print("     --profile-name kidney_injury_analysis \\")
    print("     --output manifest.json \\")
    print("     --tissue-type kidney \\")
    print("     --hypotheses 'Neutrophils recruit along vessels' 'Macrophages cluster by day 3' \\")
    print("     --sign --gpg-key researcher@institution.edu")
    print()
    
    print("2. Validate an existing manifest:")
    print("   python manifest_cli.py validate manifest.json \\")
    print("     --config config.json \\")
    print("     --data-directory data/241218_IMC_Alun \\")
    print("     --check-dataset")
    print()
    
    print("3. Inspect a manifest:")
    print("   python manifest_cli.py inspect manifest.json --show-history --show-deviations")
    print()
    
    print("4. Sign an existing manifest:")
    print("   python manifest_cli.py sign manifest.json --gpg-key researcher@institution.edu")
    print()


def example_integration_workflow():
    """Example 6: Complete integration workflow."""
    print("=== Example 6: Complete Integration Workflow ===")
    
    workflow_steps = [
        "1. Project Setup",
        "   - Create config.json with analysis parameters",
        "   - Organize data directory with ROI files",
        "   - Define scientific objectives and hypotheses",
        
        "2. Manifest Creation", 
        "   - python manifest_cli.py create --config config.json ...",
        "   - Review and sign manifest",
        "   - Commit manifest to version control",
        
        "3. Analysis Execution",
        "   - python run_analysis.py --manifest manifest.json",
        "   - Pipeline automatically logs execution steps",
        "   - Parameter deviations tracked in real-time",
        
        "4. Quality Control",
        "   - Review execution history and deviations",
        "   - Validate results against scientific objectives",
        "   - Check dataset integrity and signatures",
        
        "5. Results Sharing",
        "   - Save final manifest with complete execution log",
        "   - Share signed manifest for reproducibility",
        "   - Archive data and manifest together",
        
        "6. Replication/Extension",
        "   - Load shared manifest: AnalysisManifest.load('shared_manifest.json')",
        "   - Verify signatures and dataset integrity", 
        "   - Adapt parameters for new datasets while tracking deviations"
    ]
    
    for step in workflow_steps:
        print(step)
    print()


def main():
    """Run all examples."""
    print("Analysis Manifest System - Usage Examples")
    print("=" * 50)
    print()
    
    example_basic_manifest()
    example_parameter_profile()
    example_scientific_objectives()
    example_execution_tracking()
    example_cli_usage()
    example_integration_workflow()
    
    print("🎉 All examples completed!")
    print()
    print("Next steps:")
    print("1. Install dependencies: pip install numpy pandas h5py")
    print("2. Create your first manifest: python manifest_cli.py create --help")
    print("3. Run analysis with manifest: python run_analysis.py")
    print("4. Review the ANALYSIS_MANIFEST_README.md for complete documentation")


if __name__ == '__main__':
    main()
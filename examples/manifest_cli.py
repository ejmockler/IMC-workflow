#!/usr/bin/env python3
"""
CLI tool for IMC Analysis Manifest creation, validation, and management.

This tool provides command-line interfaces for working with analysis manifests
to ensure reproducible, transparent scientific analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import Config
from src.analysis.analysis_manifest import (
    AnalysisManifest, ParameterProfile, ScientificObjectives, 
    SignatureMethod, create_manifest_from_config, validate_manifest_compatibility
)


def cmd_create_manifest(args) -> None:
    """Create a new analysis manifest."""
    try:
        # Load configuration
        config = Config(args.config_path)
        print(f"Loaded configuration from {args.config_path}")
        
        # Create scientific objectives
        objectives = ScientificObjectives(
            primary_research_question=args.research_question,
            hypotheses=args.hypotheses if args.hypotheses else [],
            expected_outcomes=args.expected_outcomes if args.expected_outcomes else [],
            target_cell_types=args.target_cell_types if args.target_cell_types else [],
            tissue_context=args.tissue_context,
            experimental_conditions=args.experimental_conditions if args.experimental_conditions else []
        )
        
        # Create manifest
        manifest = create_manifest_from_config(
            config=config,
            data_directory=args.data_directory,
            profile_name=args.profile_name,
            scientific_objectives=objectives,
            description=args.description,
            tissue_type=args.tissue_type
        )
        
        # Sign manifest if requested
        if args.sign:
            if args.gpg_key:
                manifest.sign_manifest(SignatureMethod.GPG, args.gpg_key)
                print(f"Manifest signed with GPG key: {args.gpg_key}")
            else:
                print("Warning: --sign specified but no --gpg-key provided. Skipping signature.")
        
        # Save manifest
        output_path = Path(args.output)
        manifest.save(output_path)
        
        print(f"Created analysis manifest: {manifest.manifest_id}")
        print(f"Saved to: {output_path}")
        
        # Print summary
        summary = manifest.get_summary_report()
        print("\nManifest Summary:")
        print(f"  Dataset: {summary['dataset_summary']['total_files']} files, "
              f"{summary['dataset_summary']['total_size_mb']:.1f} MB")
        print(f"  Profile: {summary['parameter_profile']['name']}")
        print(f"  Research Question: {summary['scientific_objectives']['research_question']}")
        print(f"  Git Commit: {summary['provenance']['git_commit']}")
        
    except Exception as e:
        print(f"Error creating manifest: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_validate_manifest(args) -> None:
    """Validate a manifest against current configuration."""
    try:
        # Load manifest
        manifest = AnalysisManifest.load(args.manifest_path)
        print(f"Loaded manifest: {manifest.manifest_id}")
        
        # Verify signature if present
        if manifest.signature:
            sig_result = manifest.verify_signature()
            if sig_result['valid']:
                print(f"✓ Signature verification: {sig_result['message']}")
            else:
                print(f"✗ Signature verification failed: {sig_result['message']}")
                if not args.ignore_signature:
                    sys.exit(1)
        else:
            print("ℹ No signature present")
        
        # Validate dataset integrity if requested
        if args.check_dataset:
            data_validation = manifest.validate_dataset_integrity(args.data_directory)
            if data_validation['valid']:
                print("✓ Dataset integrity verified")
            else:
                print("✗ Dataset integrity check failed:")
                for issue_type, issues in data_validation.items():
                    if issue_type != 'valid' and issues:
                        print(f"  {issue_type}: {issues}")
                if not args.ignore_dataset:
                    sys.exit(1)
        
        # Validate compatibility with config if provided
        if args.config_path:
            config = Config(args.config_path)
            compatibility = validate_manifest_compatibility(manifest, config)
            
            if compatibility['compatible']:
                print("✓ Manifest compatible with configuration")
            else:
                print("✗ Manifest-configuration compatibility issues:")
                for error in compatibility['errors']:
                    print(f"  Error: {error}")
                
            if compatibility['warnings']:
                print("  Warnings:")
                for warning in compatibility['warnings']:
                    print(f"    {warning}")
            
            if compatibility['parameter_differences']:
                print("  Parameter differences:")
                for diff in compatibility['parameter_differences']:
                    print(f"    {diff['parameter']}: manifest={diff['manifest_value']}, config={diff['config_value']}")
            
            if not compatibility['compatible'] and not args.ignore_compatibility:
                sys.exit(1)
        
        print("✓ Manifest validation passed")
        
    except Exception as e:
        print(f"Error validating manifest: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_inspect_manifest(args) -> None:
    """Inspect a manifest and show detailed information."""
    try:
        manifest = AnalysisManifest.load(args.manifest_path)
        
        if args.format == 'json':
            print(json.dumps(manifest.to_dict(), indent=2, default=str))
        else:
            # Human-readable format
            print(f"Analysis Manifest: {manifest.manifest_id}")
            print(f"Version: {manifest.version.value}")
            print(f"Created: {manifest.created_at}")
            print(f"Updated: {manifest.updated_at}")
            print()
            
            # Dataset information
            print("Dataset Fingerprint:")
            print(f"  Files: {manifest.dataset_fingerprint.total_files}")
            print(f"  Total Size: {manifest.dataset_fingerprint.total_size_bytes / (1024**2):.1f} MB")
            print(f"  Overall Hash: {manifest.dataset_fingerprint.compute_overall_hash()}")
            print()
            
            # Parameter profile
            if manifest.parameter_profile:
                profile = manifest.parameter_profile
                print("Parameter Profile:")
                print(f"  Name: {profile.name}")
                print(f"  Description: {profile.description}")
                print(f"  Tissue Type: {profile.tissue_type}")
                print(f"  Expected Markers: {len(profile.expected_markers)}")
                print(f"  Marker Groups: {len(profile.marker_groups)}")
                print()
            
            # Scientific objectives
            if manifest.scientific_objectives:
                obj = manifest.scientific_objectives
                print("Scientific Objectives:")
                print(f"  Research Question: {obj.primary_research_question}")
                print(f"  Hypotheses: {len(obj.hypotheses)}")
                print(f"  Target Cell Types: {len(obj.target_cell_types)}")
                print()
            
            # Provenance
            prov = manifest.provenance_info
            print("Provenance Information:")
            print(f"  Git Commit: {prov.git_commit_sha}")
            print(f"  Git Branch: {prov.git_branch}")
            print(f"  Platform: {prov.platform}")
            print(f"  Python Version: {prov.python_version}")
            print()
            
            # Signature
            print("Signature:")
            print(f"  Method: {manifest.signature_method.value if manifest.signature_method else 'none'}")
            print(f"  Signed: {'Yes' if manifest.signature else 'No'}")
            if manifest.signer_info:
                print(f"  Signer: {manifest.signer_info.get('user_id', 'Unknown')}")
            print()
            
            # Execution history
            print(f"Execution History: {len(manifest.execution_history)} steps")
            print(f"Parameter Deviations: {len(manifest.deviation_log)} logged")
            
            if args.show_history and manifest.execution_history:
                print("\nExecution Steps:")
                for i, step in enumerate(manifest.execution_history[-5:], 1):  # Show last 5
                    print(f"  {i}. {step['step_name']} ({step['step_type']}) - {step['timestamp']}")
            
            if args.show_deviations and manifest.deviation_log:
                print("\nParameter Deviations:")
                for i, dev in enumerate(manifest.deviation_log[-5:], 1):  # Show last 5
                    print(f"  {i}. {dev['parameter_path']}: {dev['original_value']} → {dev['new_value']}")
                    print(f"     Reason: {dev['reason']}")
    
    except Exception as e:
        print(f"Error inspecting manifest: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_sign_manifest(args) -> None:
    """Sign an existing manifest."""
    try:
        manifest = AnalysisManifest.load(args.manifest_path)
        print(f"Loaded manifest: {manifest.manifest_id}")
        
        if args.gpg_key:
            manifest.sign_manifest(SignatureMethod.GPG, args.gpg_key)
            print(f"Manifest signed with GPG key: {args.gpg_key}")
        else:
            print("Error: GPG key required for signing", file=sys.stderr)
            sys.exit(1)
        
        # Save signed manifest
        output_path = Path(args.output) if args.output else Path(args.manifest_path)
        manifest.save(output_path)
        print(f"Signed manifest saved to: {output_path}")
        
    except Exception as e:
        print(f"Error signing manifest: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="IMC Analysis Manifest CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new manifest
  python manifest_cli.py create --config config.json --data-dir data/ \\
    --research-question "How does spatial organization change after injury?" \\
    --profile-name kidney_injury_analysis \\
    --output manifest.json

  # Validate a manifest
  python manifest_cli.py validate manifest.json --config config.json \\
    --data-dir data/ --check-dataset

  # Inspect a manifest
  python manifest_cli.py inspect manifest.json --show-history

  # Sign a manifest
  python manifest_cli.py sign manifest.json --gpg-key user@example.com
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new analysis manifest')
    create_parser.add_argument('--config-path', required=True, help='Path to configuration file')
    create_parser.add_argument('--data-directory', required=True, help='Path to data directory')
    create_parser.add_argument('--research-question', required=True, help='Primary research question')
    create_parser.add_argument('--profile-name', required=True, help='Name for parameter profile')
    create_parser.add_argument('--output', required=True, help='Output path for manifest')
    create_parser.add_argument('--description', default="Generated via CLI", help='Profile description')
    create_parser.add_argument('--tissue-type', help='Type of tissue being analyzed')
    create_parser.add_argument('--hypotheses', nargs='+', help='Scientific hypotheses')
    create_parser.add_argument('--expected-outcomes', nargs='+', help='Expected outcomes')
    create_parser.add_argument('--target-cell-types', nargs='+', help='Target cell types')
    create_parser.add_argument('--tissue-context', help='Tissue context description')
    create_parser.add_argument('--experimental-conditions', nargs='+', help='Experimental conditions')
    create_parser.add_argument('--sign', action='store_true', help='Sign the manifest')
    create_parser.add_argument('--gpg-key', help='GPG key ID for signing')
    create_parser.set_defaults(func=cmd_create_manifest)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate analysis manifest')
    validate_parser.add_argument('manifest_path', help='Path to manifest file')
    validate_parser.add_argument('--config-path', help='Path to configuration file')
    validate_parser.add_argument('--data-directory', help='Path to data directory for integrity check')
    validate_parser.add_argument('--check-dataset', action='store_true', help='Check dataset integrity')
    validate_parser.add_argument('--ignore-signature', action='store_true', help='Ignore signature validation failures')
    validate_parser.add_argument('--ignore-dataset', action='store_true', help='Ignore dataset validation failures')
    validate_parser.add_argument('--ignore-compatibility', action='store_true', help='Ignore compatibility validation failures')
    validate_parser.set_defaults(func=cmd_validate_manifest)
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect analysis manifest')
    inspect_parser.add_argument('manifest_path', help='Path to manifest file')
    inspect_parser.add_argument('--format', choices=['human', 'json'], default='human', help='Output format')
    inspect_parser.add_argument('--show-history', action='store_true', help='Show execution history')
    inspect_parser.add_argument('--show-deviations', action='store_true', help='Show parameter deviations')
    inspect_parser.set_defaults(func=cmd_inspect_manifest)
    
    # Sign command
    sign_parser = subparsers.add_parser('sign', help='Sign analysis manifest')
    sign_parser.add_argument('manifest_path', help='Path to manifest file')
    sign_parser.add_argument('--gpg-key', required=True, help='GPG key ID for signing')
    sign_parser.add_argument('--output', help='Output path (default: overwrite input)')
    sign_parser.set_defaults(func=cmd_sign_manifest)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
"""ROI analysis module for IMC data."""

from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np

from src.config import Config
from src.utils.helpers import (
    Metadata, find_roi_files, canonicalize_pair
)
from src.utils.data_loader import load_roi_data
from src.analysis.spatial import (
    identify_expression_blobs,
    analyze_blob_spatial_relationships,
    identify_tissue_neighborhoods,
    calculate_neighborhood_entropy_map,
    analyze_multiscale_neighborhoods
)
from src.analysis.pipeline import parse_roi_metadata


class ROIAnalyzer:
    """Analyzes single ROI."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def _normalize_region(self, region: Optional[str]) -> str:
        """Normalize region names to configured canonical set; fallback to Unknown."""
        exp = self.config.raw.get('experimental', {})
        configured = {str(r).strip().lower(): str(r) for r in exp.get('regions', [])}
        if region is None:
            return 'Unknown'
        key = str(region).strip().lower()
        # Map common variants
        aliases = {
            'kidney': 'Unknown',
            'renal cortex': 'Cortex',
            'renal medulla': 'Medulla'
        }
        if key in configured:
            return configured[key]
        if key in aliases and aliases[key] in configured.values():
            return aliases[key]
        # Title-case unknown but do not force "Kidney"
        return region.title()
    
    def analyze(self, roi_file: Path) -> Dict:
        """Complete analysis of single ROI."""
        # Load data
        coords, values, protein_names = load_roi_data(roi_file, 'config.json')
        
        # Identify blobs with validation
        result = identify_expression_blobs(
            coords, values, protein_names, validate=True
        )
        
        # Handle both old and new return formats (with or without validation)
        if len(result) == 4:
            blob_labels, blob_signatures, blob_type_mapping, validation_results = result
        else:
            blob_labels, blob_signatures, blob_type_mapping = result
            validation_results = None
        
        # Get contacts
        blob_contacts = analyze_blob_spatial_relationships(
            coords, blob_labels, blob_signatures, blob_type_mapping
        )
        
        # Identify neighborhoods - multi-scale if configured
        multiscale_neighborhoods = analyze_multiscale_neighborhoods(
            coords, blob_labels, blob_signatures, blob_type_mapping, 'config.json'
        )
        
        # For backward compatibility, also store default scale
        default_scale = self.config.raw.get('neighborhood_analysis', {}).get('default_scale', 'microenvironment')
        if default_scale in multiscale_neighborhoods:
            neighborhoods = multiscale_neighborhoods[default_scale]
        else:
            # Use first available scale
            neighborhoods = list(multiscale_neighborhoods.values())[0] if multiscale_neighborhoods else None
        
        entropy_map = calculate_neighborhood_entropy_map(
            coords, blob_labels, blob_type_mapping, config_path='config.json'
        )
        
        # Get metadata using pipeline parser
        metadata_dict = parse_roi_metadata(roi_file.name, 'config.json')
        metadata = Metadata(
            condition=metadata_dict.get('condition', 'Unknown'),
            injury_day=metadata_dict.get('timepoint'),
            tissue_region=self._normalize_region(
                metadata_dict.get('region') or metadata_dict.get('tissue_region') or metadata_dict.get('Region')
            ),
            mouse_replicate=metadata_dict.get('mouse_id', 'Unknown')
        )
        
        # Canonicalize contacts
        canonical_contacts = self._canonicalize_contacts(blob_contacts, blob_signatures)
        
        return {
            'filename': roi_file.name,
            'metadata': metadata,
            'coords': coords,
            'values': values,
            'protein_names': protein_names,
            'blob_labels': blob_labels,
            'blob_signatures': blob_signatures,
            'blob_type_mapping': blob_type_mapping,
            'blob_contacts': blob_contacts,
            'canonical_contacts': canonical_contacts,
            'neighborhoods': neighborhoods,
            'multiscale_neighborhoods': multiscale_neighborhoods,
            'entropy_map': entropy_map,
            'validation_results': validation_results,
            'total_pixels': len(coords)
        }
    
    def _canonicalize_contacts(self, blob_contacts, blob_signatures):
        """Canonicalize all contact pairs."""
        canonical = {}
        for blob_id, contacts in blob_contacts.items():
            blob1_sig = '+'.join(blob_signatures[blob_id]['dominant_proteins'][:2])
            for neighbor_id, freq in contacts.items():
                if freq > 0:
                    blob2_sig = '+'.join(blob_signatures[neighbor_id]['dominant_proteins'][:2])
                    pair = canonicalize_pair(blob1_sig, blob2_sig)
                    canonical[pair] = max(canonical.get(pair, 0), freq)
        return canonical


class BatchAnalyzer:
    """Analyzes multiple ROIs."""
    
    def __init__(self, config: Config):
        self.config = config
        self.roi_analyzer = ROIAnalyzer(config)
    
    def analyze_all(self) -> List[Dict]:
        """Analyze all ROIs in data directory."""
        roi_files = find_roi_files(self.config.data_dir)
        results = []
        
        print(f"Analyzing {len(roi_files)} ROIs...")
        for i, roi_file in enumerate(roi_files, 1):
            print(f"  [{i}/{len(roi_files)}] {roi_file.name}")
            try:
                result = self.roi_analyzer.analyze(roi_file)
                results.append(result)
            except Exception as e:
                print(f"    ERROR: {e}")
        
        # Save results
        output_file = self.config.output_dir / 'analysis_results.json'
        
        def make_serializable(obj):
            """Convert numpy types to Python types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        with open(output_file, 'w') as f:
            # Convert metadata objects to dicts for serialization
            serializable = []
            for r in results:
                r_copy = r.copy()
                if hasattr(r_copy['metadata'], 'to_dict'):
                    r_copy['metadata'] = r_copy['metadata'].to_dict()
                elif hasattr(r_copy['metadata'], '_asdict'):
                    r_copy['metadata'] = r_copy['metadata']._asdict()
                # Convert numpy types (keeps all spatial data for visualization)
                r_copy = make_serializable(r_copy)
                serializable.append(r_copy)
            json.dump(serializable, f, indent=2)
        
        print(f"Saved results to {output_file}")
        return results
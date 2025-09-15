"""ROI analysis module for IMC data."""

from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np

from src.config import Config
from src.utils.helpers import (
    Metadata, find_roi_files
)
from src.utils.data_loader import load_roi_data
from src.analysis.pipeline import parse_roi_metadata

# Import the new robust analysis engine
from src.analysis.spatial_engine_final import analyze_spatial_organization_robust, SPATIAL_LIBS_AVAILABLE

class ROIAnalyzer:
    """Analyzes single ROI using the robust superpixel-based engine."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def _normalize_region(self, region: Optional[str]) -> str:
        """Normalize region names to configured canonical set; fallback to Unknown."""
        exp = self.config.raw.get('experimental', {})
        configured = {str(r).strip().lower(): str(r) for r in exp.get('regions', [])}
        if region is None:
            return 'Unknown'
        key = str(region).strip().lower()
        aliases = {
            'kidney': 'Unknown',
            'renal cortex': 'Cortex',
            'renal medulla': 'Medulla'
        }
        if key in configured:
            return configured[key]
        if key in aliases and aliases[key] in configured.values():
            return aliases[key]
        return region.title()
    
    def analyze(self, roi_file: Path) -> Dict:
        """Complete analysis of single ROI using the robust engine."""
        if not SPATIAL_LIBS_AVAILABLE:
            raise ImportError("Spatial libraries (squidpy, scanpy) not found. Cannot run analysis.")

        # Load data
        coords, values, protein_names = load_roi_data(roi_file, 'config.json')
        
        print("   Running robust spatial analysis engine...")
        # Run the new, robust analysis
        analysis_result = analyze_spatial_organization_robust(
            coords, values, protein_names, 'config.json'
        )
        
        # Get metadata
        metadata_dict = parse_roi_metadata(roi_file.name, 'config.json')
        metadata = Metadata(
            condition=metadata_dict.get('condition', 'Unknown'),
            injury_day=metadata_dict.get('timepoint'),
            tissue_region=self._normalize_region(
                metadata_dict.get('region') or metadata_dict.get('tissue_region') or metadata_dict.get('Region')
            ),
            mouse_replicate=metadata_dict.get('mouse_id', 'Unknown')
        )
        
        # Combine results into a single dictionary for output
        final_result = {
            'filename': roi_file.name,
            'metadata': metadata,
            **analysis_result
        }
        
        return final_result


class BatchAnalyzer:
    """Analyzes multiple ROIs."""
    
    def __init__(self, config: Config):
        self.config = config
        self.roi_analyzer = ROIAnalyzer(config)
    
    def analyze_all(self) -> List[Dict]:
        """Analyze all ROIs in data directory."""
        roi_files = find_roi_files(self.config.data_dir)
        results = []
        
        print(f"Analyzing {len(roi_files)} ROIs using the ROBUST SUPERPIXEL ENGINE...")
        for i, roi_file in enumerate(roi_files, 1):
            print(f"  [{i}/{len(roi_files)}] {roi_file.name}")
            try:
                result = self.roi_analyzer.analyze(roi_file)
                results.append(result)
            except Exception as e:
                print(f"    ERROR: {e}")
                # Optionally, decide if you want to skip failed ROIs or stop
                # continue
        
        # Save results
        output_file = self.config.output_dir / 'analysis_results_robust.json'
        
        def make_serializable(obj):
            """Convert numpy/pathlib types to Python types for JSON serialization."""
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        with open(output_file, 'w') as f:
            serializable_results = []
            for r in results:
                r_copy = r.copy()
                if hasattr(r_copy.get('metadata'), 'to_dict'):
                    r_copy['metadata'] = r_copy['metadata'].to_dict()
                elif hasattr(r_copy.get('metadata'), '_asdict'):
                    r_copy['metadata'] = r_copy['metadata']._asdict()
                
                serializable_results.append(make_serializable(r_copy))
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved robust results to {output_file}")
        return results
"""ROI analysis module for IMC data using MSPT framework."""

from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np

from src.config import Config
from src.utils.helpers import Metadata, find_roi_files
from src.utils.data_loader import parse_roi_metadata
from .mspt_pipeline import MSPTPipeline, analyze_roi_with_mspt


class ROIAnalyzer:
    """Analyzes single ROI using MSPT framework."""
    
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = MSPTPipeline(config)
    
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
        """Complete analysis of single ROI with MSPT framework.
        
        Args:
            roi_file: Path to ROI data file
            
        Returns:
            Unified analysis results based on protein territories
        """
        print("   Running MSPT analysis pipeline...")
        
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
        
        # Run MSPT analysis
        analysis_result = analyze_roi_with_mspt(
            roi_file, self.config, 
            metadata=metadata.to_dict() if hasattr(metadata, 'to_dict') else metadata._asdict()
        )
        
        # Add filename for compatibility
        analysis_result['filename'] = roi_file.name
        analysis_result['metadata'] = metadata
        
        return analysis_result


class BatchAnalyzer:
    """Analyzes multiple ROIs using MSPT framework."""
    
    def __init__(self, config: Config):
        self.config = config
        self.roi_analyzer = ROIAnalyzer(config)
    
    def analyze_all(self) -> List[Dict]:
        """Analyze all ROIs in data directory.
        
        Returns:
            List of analysis results for all ROIs
        """
        roi_files = find_roi_files(self.config.data_dir)
        results = []
        
        print(f"Analyzing {len(roi_files)} ROIs using MSPT framework...")
        print("Framework: Multi-Scale Protein Territory analysis")
        print("    • Unified spatial pixel representation")
        print("    • Territory-based protein expression patterns")
        print("    • Multi-scale spatial interaction analysis")
        print("    • Statistical validation with permutation testing")
        
        successful = 0
        failed = 0
        
        for i, roi_file in enumerate(roi_files, 1):
            print(f"  [{i}/{len(roi_files)}] {roi_file.name}")
            try:
                result = self.roi_analyzer.analyze(roi_file)
                results.append(result)
                successful += 1
                
                # Print territory summary
                if 'territory_discovery' in result:
                    td = result['territory_discovery']
                    coverage = result.get('coverage_metrics', {})
                    print(f"    Territories: {td['n_territories']}, "
                          f"Pixels: {result.get('total_spatial_pixels', 0)}, "
                          f"Coverage: {coverage.get('analyzed_fraction', 0):.1%}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                failed += 1
                # Store error result for debugging
                results.append({
                    'filename': roi_file.name,
                    'error': str(e),
                    'status': 'failed'
                })
        
        print(f"\nAnalysis complete: {successful} successful, {failed} failed")
        
        # Save results
        output_file = self.config.output_dir / 'mspt_analysis_results.json'
        
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
        
        print(f"Saved results to {output_file}")
        
        # Print analysis summary
        self._print_analysis_summary(results)
        
        return results
    
    def _print_analysis_summary(self, results: List[Dict]):
        """Print summary of MSPT analysis results."""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY - Multi-Scale Protein Territory Framework")
        print("="*60)
        
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            print("No successful analyses to summarize")
            return
        
        # Territory and coverage statistics
        territory_counts = []
        pixel_counts = []
        coverage_fractions = []
        nuclear_fractions = []
        
        for r in successful_results:
            if 'territory_discovery' in r:
                territory_counts.append(r['territory_discovery']['n_territories'])
            
            if 'total_spatial_pixels' in r:
                pixel_counts.append(r['total_spatial_pixels'])
            
            if 'coverage_metrics' in r:
                cm = r['coverage_metrics']
                coverage_fractions.append(cm.get('analyzed_fraction', 0))
                nuclear_fractions.append(cm.get('nuclear_fraction', 0))
        
        print("\nSpatial Pixel Analysis:")
        if pixel_counts:
            print(f"  Average spatial pixels per ROI: {np.mean(pixel_counts):.0f} ± {np.std(pixel_counts):.0f}")
            print(f"  Pixel size: {self.config.raw.get('mspt_analysis', {}).get('pixel_size_um', 2.0)}μm × {self.config.raw.get('mspt_analysis', {}).get('pixel_size_um', 2.0)}μm")
        
        print("\nTerritory Discovery:")
        if territory_counts:
            print(f"  Average territories per ROI: {np.mean(territory_counts):.1f} ± {np.std(territory_counts):.1f}")
            print(f"  Territory range: {min(territory_counts)} - {max(territory_counts)}")
        
        print("\nTissue Coverage:")
        if coverage_fractions:
            print(f"  Average coverage: {np.mean(coverage_fractions):.1%} ± {np.std(coverage_fractions):.1%}")
        if nuclear_fractions:
            print(f"  Nuclear context: {np.mean(nuclear_fractions):.1%} ± {np.std(nuclear_fractions):.1%}")
        
        print("\nSpatial Analysis:")
        spatial_radii = self.config.raw.get('mspt_analysis', {}).get('spatial_radii', [10, 25, 50])
        print(f"  Multi-scale radii: {spatial_radii} μm")
        print(f"  Statistical validation: {self.config.raw.get('mspt_analysis', {}).get('spatial_analysis', {}).get('n_permutations', 1000)} permutations")
        
        print("\nMETHODOLOGICAL STRENGTHS:")
        print("  ✓ Unified spatial pixel representation eliminates data type incompatibility")
        print("  ✓ Territory-based analysis avoids cell-type overreach")
        print("  ✓ Multi-scale spatial analysis reveals scale-dependent organization")
        print("  ✓ Statistical validation with permutation testing")
        print("  ✓ Nuclear information as enrichment features (not separate analysis)")
        
        print("\nINTERPRETATION GUIDELINES:")
        print("  • Territories represent spatial protein expression patterns")
        print("  • Statistical significance indicates non-random spatial associations")
        print("  • Multi-scale analysis reveals scale-dependent spatial organization")
        print("  • Results describe protein distribution patterns in tissue space")
        print("  • Nuclear context provides spatial enrichment, not cell counts")
        
        print("="*60)
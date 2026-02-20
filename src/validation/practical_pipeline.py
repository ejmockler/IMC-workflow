"""
Practical Validation Pipeline for IMC Data

Simple, transparent validation that logs all metrics clearly.
No academic fluff - just practical quality checks with clear output.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from scipy import stats


class PracticalValidationPipeline:
    """Simple, practical validation pipeline with transparent logging."""

    def __init__(self, expected_proteins: Optional[List[str]] = None):
        self.logger = logging.getLogger(__name__)
        # Default to kidney markers if not specified, but allow config override
        self.expected_proteins = expected_proteins or ['CD45', 'CD11b', 'CD31', 'CD140a', 'CD140b', 'CD206']
        
    def validate_single_roi(self, roi_file: Path) -> Dict[str, Any]:
        """Validate single ROI with clear metric logging."""
        start_time = time.time()
        
        try:
            # Load data
            roi_data = pd.read_csv(roi_file, sep='\t')
            coords = roi_data[['X', 'Y']].values
            n_cells = len(coords)
            
            self.logger.info(f"=== Validating {roi_file.name} ===")
            self.logger.info(f"  Cells: {n_cells}")
            
            # Basic checks
            issues = []
            
            # 1. Cell count check
            if n_cells < 100:
                issues.append(f"Low cell count: {n_cells}")
                self.logger.warning(f"  ⚠️  Low cell count: {n_cells} < 100")
            else:
                self.logger.info(f"  ✓ Cell count OK: {n_cells}")
            
            # 2. Coordinate range check
            x_range = coords[:, 0].max() - coords[:, 0].min()
            y_range = coords[:, 1].max() - coords[:, 1].min()
            
            if x_range == 0 or y_range == 0:
                issues.append("Degenerate coordinates")
                self.logger.error(f"  ❌ Degenerate coordinates: X range={x_range}, Y range={y_range}")
                return {'roi_file': roi_file.name, 'status': 'FAIL', 'issues': issues}
            else:
                area = x_range * y_range
                density = n_cells / area
                self.logger.info(f"  ✓ Spatial extent: {x_range:.1f} x {y_range:.1f} μm")
                self.logger.info(f"  ✓ Cell density: {density:.3f} cells/μm²")
            
            # 3. Protein channel checks
            found_proteins = []
            missing_proteins = []

            for protein in self.expected_proteins:
                matching_cols = [col for col in roi_data.columns if protein in col]
                if matching_cols:
                    found_proteins.append((protein, matching_cols[0]))
                    
                    # Check signal quality
                    intensities = roi_data[matching_cols[0]].values
                    signal_95 = np.percentile(intensities, 95)
                    noise_5 = np.percentile(intensities, 5)
                    snr = signal_95 / noise_5 if noise_5 > 0 else float('inf')
                    zero_fraction = np.sum(intensities == 0) / len(intensities)
                    
                    self.logger.info(f"  ✓ {protein}: SNR={snr:.2f}, Zero%={zero_fraction:.2f}")
                    
                    if snr < 1.5:
                        issues.append(f"Low SNR {protein}: {snr:.2f}")
                        self.logger.warning(f"    ⚠️  Low SNR: {snr:.2f} < 1.5")
                    
                    if zero_fraction > 0.9:
                        issues.append(f"High zero fraction {protein}: {zero_fraction:.2f}")
                        self.logger.warning(f"    ⚠️  High zero fraction: {zero_fraction:.2f}")
                else:
                    missing_proteins.append(protein)
                    self.logger.error(f"  ❌ Missing {protein}")
            
            if missing_proteins:
                issues.append(f"Missing proteins: {missing_proteins}")
            
            # 4. DNA channel checks
            dna_channels = ['DNA1', 'DNA2']
            for dna in dna_channels:
                dna_cols = [col for col in roi_data.columns if dna in col]
                if dna_cols:
                    dna_signal = roi_data[dna_cols[0]].values
                    mean_signal = np.mean(dna_signal)
                    median_signal = np.median(dna_signal)

                    self.logger.info(f"  ✓ {dna}: Mean={mean_signal:.1f}, Median={median_signal:.1f}")

                    # Check mean instead of min (min=0 is normal for background pixels)
                    if mean_signal < 5.0:
                        issues.append(f"Low DNA signal {dna}: mean={mean_signal:.1f}")
                        self.logger.warning(f"    ⚠️  Low DNA signal: mean={mean_signal:.1f} < 5.0")
                else:
                    issues.append(f"Missing {dna}")
                    self.logger.error(f"  ❌ Missing {dna}")
            
            # 5. DNA correlation check
            dna_cols = [col for col in roi_data.columns if 'DNA' in col]
            if len(dna_cols) >= 2:
                dna_corr = np.corrcoef(roi_data[dna_cols[0]], roi_data[dna_cols[1]])[0, 1]
                self.logger.info(f"  ✓ DNA correlation: {dna_corr:.3f}")
                
                if dna_corr < 0.6:
                    issues.append(f"Low DNA correlation: {dna_corr:.3f}")
                    self.logger.warning(f"    ⚠️  Low DNA correlation: {dna_corr:.3f} < 0.6")
            
            execution_time = (time.time() - start_time) * 1000
            
            # Determine status
            critical_issues = [issue for issue in issues if any(word in issue for word in ['Missing', 'Degenerate'])]
            if critical_issues:
                status = 'FAIL'
                self.logger.error(f"  ❌ FAIL: {len(critical_issues)} critical issues")
            elif len(issues) > 5:
                status = 'POOR'
                self.logger.warning(f"  ⚠️  POOR: {len(issues)} issues")
            elif len(issues) > 2:
                status = 'FAIR'
                self.logger.warning(f"  ⚠️  FAIR: {len(issues)} issues")
            elif len(issues) > 0:
                status = 'GOOD'
                self.logger.info(f"  ✓ GOOD: {len(issues)} minor issues")
            else:
                status = 'EXCELLENT'
                self.logger.info(f"  ✓ EXCELLENT: No issues")
            
            self.logger.info(f"  Status: {status} ({execution_time:.1f}ms)")
            
            return {
                'roi_file': roi_file.name,
                'status': status,
                'n_cells': n_cells,
                'issues': issues,
                'critical_issues': len(critical_issues),
                'total_issues': len(issues),
                'execution_time_ms': execution_time
            }
            
        except Exception as e:
            self.logger.error(f"  ❌ ERROR validating {roi_file.name}: {e}")
            return {
                'roi_file': roi_file.name,
                'status': 'ERROR',
                'issues': [f"Validation error: {str(e)}"],
                'execution_time_ms': (time.time() - start_time) * 1000
            }
    
    def validate_dataset(self, roi_files: List[Path]) -> Dict[str, Any]:
        """Validate entire dataset with clear summary."""
        start_time = time.time()
        self.logger.info(f"Starting practical validation of {len(roi_files)} ROI files")
        
        # Validate all ROIs
        results = []
        for roi_file in roi_files:
            result = self.validate_single_roi(roi_file)
            results.append(result)
        
        # Compile summary
        status_counts = {}
        for result in results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_issues = sum(result['total_issues'] for result in results)
        critical_failures = sum(1 for result in results if result['status'] in ['FAIL', 'ERROR'])
        
        total_time = (time.time() - start_time) * 1000
        
        self.logger.info(f"\n=== VALIDATION SUMMARY ===")
        self.logger.info(f"Total ROIs: {len(roi_files)}")
        for status, count in status_counts.items():
            self.logger.info(f"  {status}: {count} ROIs")
        self.logger.info(f"Total issues: {total_issues}")
        self.logger.info(f"Critical failures: {critical_failures}")
        self.logger.info(f"Validation time: {total_time:.1f}ms")
        
        # Decision
        can_proceed = critical_failures == 0
        if can_proceed:
            usable_rois = len(roi_files) - critical_failures
            self.logger.info(f"✓ PROCEED: {usable_rois}/{len(roi_files)} ROIs can be analyzed")
        else:
            self.logger.error(f"❌ BLOCKED: {critical_failures} critical failures")
        
        return {
            'can_proceed': can_proceed,
            'total_rois': len(roi_files),
            'status_distribution': status_counts,
            'critical_failures': critical_failures,
            'total_issues': total_issues,
            'usable_rois': len(roi_files) - critical_failures,
            'execution_time_ms': total_time,
            'roi_results': results
        }


def create_practical_pipeline(expected_proteins: Optional[List[str]] = None) -> PracticalValidationPipeline:
    """Factory function for practical validation pipeline.

    Args:
        expected_proteins: Optional list of protein markers to validate.
                          If None, defaults to kidney panel.
    """
    return PracticalValidationPipeline(expected_proteins=expected_proteins)
"""
Test structure-guided segmentation (CD31+CD140b) vs DNA-based segmentation.

Tests on 3 representative ROIs:
- D1_M1_01_9: Low stability (0.08)
- D3_M1_01_15: Medium stability (0.60)
- D7_M1_01_21: High stability (0.81)
"""

import sys
from pathlib import Path
from src.config import Config
from src.analysis.main_pipeline import IMCAnalysisPipeline

def main():
    print("="*80)
    print("Structure-Guided Segmentation Test")
    print("Testing CD31 (vascular) + CD140b (stromal) vs DNA1+DNA2")
    print("="*80)

    # Load config with new segmentation channels
    config = Config('config.json')
    print(f"\n✓ Config loaded")
    print(f"  Segmentation channels: {config.raw['segmentation']['slic_input_channels']}")

    # Initialize pipeline
    pipeline = IMCAnalysisPipeline(config)

    # Get protein names from config
    protein_names = config.raw['channels']['protein_channels']
    print(f"  Protein markers: {protein_names}")

    # Test ROIs
    test_files = [
        ('D1_M1_01_9 (stability=0.08, LOW)', 'IMC_241218_Alun_ROI_D1_M1_01_9.txt'),
        ('D3_M1_01_15 (stability=0.60, MED)', 'IMC_241218_Alun_ROI_D3_M1_01_15.txt'),
        ('D7_M1_01_21 (stability=0.81, HIGH)', 'IMC_241218_Alun_ROI_D7_M1_01_21.txt'),
    ]

    data_dir = Path('data/241218_IMC_Alun')
    output_dir = Path('results/structure_segmentation_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for label, filename in test_files:
        print(f"\n{'─'*80}")
        print(f"Processing: {label}")
        print(f"File: {filename}")
        print(f"{'─'*80}")

        roi_path = data_dir / filename
        roi_id = filename.replace('.txt', '')

        try:
            # Load ROI data
            print(f"  Loading ROI data...")
            roi_data = pipeline.load_roi_data(str(roi_path), protein_names)
            print(f"  ✓ Loaded {roi_data['n_measurements']} cells")

            # Run full pipeline
            print(f"  Running analysis...")
            result = pipeline.analyze_single_roi(
                roi_data=roi_data,
                roi_id=roi_id
            )

            # Extract key metrics
            consistency = result.get('consistency_results', {}).get('overall', {})

            print(f"\n✓ Analysis complete for {roi_id}")
            print(f"\n  Clustering Stability Metrics:")
            if consistency:
                cluster_stability = consistency.get('cluster_count_stability', consistency.get('mean_ari', 0.0))
                assignment_stability = consistency.get('assignment_stability', consistency.get('mean_nmi', 0.0))

                print(f"    Cluster count stability: {cluster_stability:.3f}")
                print(f"    Assignment stability: {assignment_stability:.3f}")

                # Store for comparison
                results[label] = {
                    'roi_id': roi_id,
                    'cluster_stability': cluster_stability,
                    'assignment_stability': assignment_stability,
                    'n_scales': len(result.get('multiscale_results', {}))
                }
            else:
                print(f"    No consistency metrics available")
                results[label] = {
                    'roi_id': roi_id,
                    'cluster_stability': None,
                    'assignment_stability': None,
                    'n_scales': len(result.get('multiscale_results', {}))
                }

        except Exception as e:
            print(f"\n❌ Exception during processing: {e}")
            import traceback
            traceback.print_exc()
            results[label] = {'status': 'failed', 'error': str(e)}

    # Summary comparison
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY - Structure-Guided Segmentation")
    print(f"{'='*80}")

    print(f"\n{'ROI':<40} {'Old Stability':<15} {'New Stability':<15} {'Change':<10}")
    print(f"{'─'*80}")

    # Original stabilities for comparison
    original = {
        'D1_M1_01_9 (stability=0.08, LOW)': 0.08,
        'D3_M1_01_15 (stability=0.60, MED)': 0.60,
        'D7_M1_01_21 (stability=0.81, HIGH)': 0.81,
    }

    for label, orig_stability in original.items():
        if label in results and 'cluster_stability' in results[label]:
            new_stability = results[label]['cluster_stability']
            change = new_stability - orig_stability
            change_str = f"{change:+.3f}"

            print(f"{label:<40} {orig_stability:<15.3f} {new_stability:<15.3f} {change_str:<10}")
        else:
            print(f"{label:<40} {orig_stability:<15.3f} {'FAILED':<15} {'N/A':<10}")

    print(f"\n{'='*80}")
    print("Validation Criteria:")
    print("  ✓ Low stability ROI improves (D1_M1_01_9: 0.08 → >0.3)")
    print("  ✓ Medium stability ROI improves (D3_M1_01_15: 0.60 → >0.7)")
    print("  ✓ High stability ROI maintains (D7_M1_01_21: 0.81 → ~0.8)")
    print("  ✓ Biological validation: clusters align with injury timeline")
    print(f"{'='*80}\n")

    return results

if __name__ == '__main__':
    main()

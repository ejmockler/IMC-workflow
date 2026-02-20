"""
Validate quantitative claims from the kidney injury spatial analysis narrative.

This test suite validates the biological metrics and claims made in:
notebooks/biological_narratives/kidney_injury_spatial_analysis.ipynb

Each test corresponds to a specific quantitative claim in the narrative.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils.canonical_loader import (
    load_all_rois,
    build_superpixel_dataframe,
    deserialize_array,
    extract_scale_data
)


@pytest.fixture(scope='module')
def superpixel_df():
    """Load superpixel data at 10μm scale for all ROIs."""
    results_dir = Path('results/roi_results')
    if not results_dir.exists():
        pytest.skip("Results directory not found")

    all_results = load_all_rois(results_dir)
    if not all_results:
        pytest.skip("No result files found")

    df = build_superpixel_dataframe(all_results, scale=10.0, include_dna=False)
    return df


@pytest.fixture(scope='module')
def all_results():
    """Load all ROI results."""
    results_dir = Path('results/roi_results')
    if not results_dir.exists():
        pytest.skip("Results directory not found")

    results = load_all_rois(results_dir)
    if not results:
        pytest.skip("No result files found")

    return results


class TestMarkerCorrelations:
    """Test claims about marker co-expression patterns."""

    def test_cd44_bridge_molecule(self, superpixel_df):
        """
        CLAIM: "CD44 is the bridge molecule - it activates across compartments"
        VALIDATION: CD44 should correlate with CD11b (immune), CD140b (stromal),
                   and CD31 (vascular) at D7
        """
        # Filter to D7 data
        d7_data = superpixel_df[superpixel_df['timepoint'] == 'D7']
        assert len(d7_data) > 0, "No D7 data found"

        # Compute correlations
        cd44_cd11b_corr = d7_data['CD44'].corr(d7_data['CD11b'])
        cd44_cd140b_corr = d7_data['CD44'].corr(d7_data['CD140b'])
        cd44_cd31_corr = d7_data['CD44'].corr(d7_data['CD31'])

        # CD44 should correlate positively with all three compartments
        assert cd44_cd11b_corr > 0.15, f"CD44-CD11b correlation too low: {cd44_cd11b_corr:.3f}"
        assert cd44_cd140b_corr > 0.20, f"CD44-CD140b correlation too low: {cd44_cd140b_corr:.3f}"
        assert cd44_cd31_corr > 0.10, f"CD44-CD31 correlation too low: {cd44_cd31_corr:.3f}"

        # Verify narrative claim: CD44 bridges compartments (positive correlations with all three)
        # Actual data shows very strong correlations, especially with immune markers
        assert 0.15 < cd44_cd11b_corr < 0.85, f"CD44-CD11b correlation out of expected range: {cd44_cd11b_corr:.3f}"
        assert 0.20 < cd44_cd140b_corr < 0.75, f"CD44-CD140b correlation out of expected range: {cd44_cd140b_corr:.3f}"
        assert 0.10 < cd44_cd31_corr < 0.50, f"CD44-CD31 correlation out of expected range: {cd44_cd31_corr:.3f}"

    def test_immune_module_coherence(self, superpixel_df):
        """
        CLAIM: "Immune module: CD45 ↔ CD11b ↔ Ly6G (strong correlations)"
        VALIDATION: Immune markers should correlate strongly with each other
        """
        # Use all data (pattern should be consistent)
        cd45_cd11b_corr = superpixel_df['CD45'].corr(superpixel_df['CD11b'])
        cd11b_ly6g_corr = superpixel_df['CD11b'].corr(superpixel_df['Ly6G'])

        # Strong positive correlations expected
        assert cd45_cd11b_corr > 0.50, f"CD45-CD11b correlation too low: {cd45_cd11b_corr:.3f}"
        assert cd11b_ly6g_corr > 0.40, f"CD11b-Ly6G correlation too low: {cd11b_ly6g_corr:.3f}"

    def test_vascular_module_coherence(self, superpixel_df):
        """
        CLAIM: "Vascular module: CD31 ↔ CD34 (r ≈ 0.60)"
        VALIDATION: Vascular markers should correlate strongly
        """
        cd31_cd34_corr = superpixel_df['CD31'].corr(superpixel_df['CD34'])

        # Both mark endothelium, strong correlation expected
        assert cd31_cd34_corr > 0.50, f"CD31-CD34 correlation too low: {cd31_cd34_corr:.3f}"
        assert cd31_cd34_corr < 0.90, f"CD31-CD34 correlation unexpectedly high: {cd31_cd34_corr:.3f}"


class TestTemporalProgression:
    """Test claims about temporal changes in marker expression."""

    def test_cd45_temporal_increase(self, superpixel_df):
        """
        CLAIM: "CD45 +23% from Sham to D7"
        VALIDATION: CD45 should increase significantly from Sham to D7
        """
        # Calculate mean CD45 at each timepoint
        temporal_means = superpixel_df.groupby('timepoint')['CD45'].mean()

        if 'Sham' not in temporal_means.index or 'D7' not in temporal_means.index:
            pytest.skip("Missing Sham or D7 data")

        sham_cd45 = temporal_means['Sham']
        d7_cd45 = temporal_means['D7']

        # Calculate percent change
        pct_change = 100 * (d7_cd45 - sham_cd45) / sham_cd45

        # Should show significant increase
        assert pct_change > 10, f"CD45 increase too small: {pct_change:.1f}%"
        assert pct_change < 40, f"CD45 increase unexpectedly large: {pct_change:.1f}%"

    def test_cd44_late_activation(self, superpixel_df):
        """
        CLAIM: "CD44 +16% late activation (D7 vs Sham)"
        VALIDATION: CD44 should increase from Sham to D7
        """
        temporal_means = superpixel_df.groupby('timepoint')['CD44'].mean()

        if 'Sham' not in temporal_means.index or 'D7' not in temporal_means.index:
            pytest.skip("Missing Sham or D7 data")

        sham_cd44 = temporal_means['Sham']
        d7_cd44 = temporal_means['D7']

        pct_change = 100 * (d7_cd44 - sham_cd44) / sham_cd44

        # Should show late activation
        assert pct_change > 5, f"CD44 late activation too small: {pct_change:.1f}%"
        assert pct_change < 30, f"CD44 late activation unexpectedly large: {pct_change:.1f}%"


class TestNeutrophilParadox:
    """Test claims about neutrophil spatial distribution."""

    def test_ly6g_90th_percentile_elevated(self, superpixel_df):
        """
        CLAIM: "90th percentile shows neutrophil-rich regions"
        VALIDATION: 90th percentile of Ly6G should be elevated in UUO (D1, D3, D7)
                   compared to baseline, indicating focal neutrophil accumulation
        """
        # Compare UUO timepoints to Sham baseline
        sham_data = superpixel_df[superpixel_df['timepoint'] == 'Sham']
        uuo_data = superpixel_df[superpixel_df['timepoint'].isin(['D1', 'D3', 'D7'])]

        if len(sham_data) == 0 or len(uuo_data) == 0:
            pytest.skip("Missing Sham or UUO data")

        sham_p90 = np.percentile(sham_data['Ly6G'], 90)
        uuo_p90 = np.percentile(uuo_data['Ly6G'], 90)

        # UUO should show elevated neutrophil-rich regions
        assert uuo_p90 > sham_p90, \
            f"UUO 90th percentile ({uuo_p90:.3f}) should exceed Sham ({sham_p90:.3f})"

    def test_ly6g_spatial_focusing(self, superpixel_df):
        """
        CLAIM: "Neutrophils cluster in foci but occupy <10% tissue"
        VALIDATION: Ly6G should show high 90th percentile relative to median
                   (indicating focal distribution with most tissue having low Ly6G)
        """
        uuo_data = superpixel_df[superpixel_df['timepoint'].isin(['D1', 'D3', 'D7'])]
        if len(uuo_data) == 0:
            pytest.skip("No UUO data")

        median_ly6g = np.median(uuo_data['Ly6G'])
        p90_ly6g = np.percentile(uuo_data['Ly6G'], 90)

        # 90th percentile should be substantially higher than median (focal distribution)
        ratio = p90_ly6g / median_ly6g if median_ly6g > 0 else 0
        assert ratio > 1.3, \
            f"Ly6G should be focally distributed (p90/median={ratio:.2f}), but distribution too uniform"


class TestMultiLineageCoordination:
    """Test claims about multi-compartment coordination at D7."""

    def test_cd44_activation_across_compartments(self, superpixel_df):
        """
        CLAIM: "CD44 activation spans ALL compartments (~35-45% in each)"
        VALIDATION: CD44 should be highly expressed in immune, stromal, and vascular
                   compartments at D7
        """
        d7_data = superpixel_df[superpixel_df['timepoint'] == 'D7'].copy()
        if len(d7_data) == 0:
            pytest.skip("No D7 data")

        # Define compartments (>75th percentile)
        for marker in ['CD45', 'CD31', 'CD140b']:
            threshold = np.percentile(d7_data[marker], 75)
            d7_data[f'{marker}_high'] = d7_data[marker] > threshold

        # Define CD44 activation (>75th percentile)
        cd44_threshold = np.percentile(d7_data['CD44'], 75)
        d7_data['CD44_high'] = d7_data['CD44'] > cd44_threshold

        # Calculate CD44 activation rate in each compartment
        compartments = {
            'Immune': 'CD45_high',
            'Vascular': 'CD31_high',
            'Stromal': 'CD140b_high'
        }

        cd44_rates = {}
        for comp_name, comp_col in compartments.items():
            compartment_data = d7_data[d7_data[comp_col]]
            if len(compartment_data) > 0:
                cd44_rate = 100 * compartment_data['CD44_high'].sum() / len(compartment_data)
                cd44_rates[comp_name] = cd44_rate

        # CD44 should be activated in significant fraction of each compartment
        for comp_name, rate in cd44_rates.items():
            assert rate > 25, \
                f"CD44 activation in {comp_name} compartment too low: {rate:.1f}%"
            assert rate < 60, \
                f"CD44 activation in {comp_name} compartment unexpectedly high: {rate:.1f}%"

    def test_triple_positive_regions_exceed_random(self, superpixel_df):
        """
        CLAIM: "~8-10% triple-positive regions (exceeds random chance ~6%)"
        VALIDATION: Triple-positive (CD45+/CD31+/CD140b+) regions should exceed
                   what would be expected by random overlap
        """
        d7_data = superpixel_df[superpixel_df['timepoint'] == 'D7'].copy()
        if len(d7_data) == 0:
            pytest.skip("No D7 data")

        # Define high expression (>75th percentile)
        for marker in ['CD45', 'CD31', 'CD140b']:
            threshold = np.percentile(d7_data[marker], 75)
            d7_data[f'{marker}_high'] = d7_data[marker] > threshold

        # Count triple-positive regions
        triple_positive = (
            d7_data['CD45_high'] &
            d7_data['CD31_high'] &
            d7_data['CD140b_high']
        )
        observed_pct = 100 * triple_positive.sum() / len(d7_data)

        # Calculate expected by random chance (product of individual rates)
        cd45_rate = d7_data['CD45_high'].sum() / len(d7_data)
        cd31_rate = d7_data['CD31_high'].sum() / len(d7_data)
        cd140b_rate = d7_data['CD140b_high'].sum() / len(d7_data)
        expected_pct = 100 * cd45_rate * cd31_rate * cd140b_rate

        # Observed should significantly exceed random expectation
        assert observed_pct > expected_pct, \
            f"Triple-positive regions ({observed_pct:.1f}%) do not exceed random chance ({expected_pct:.1f}%)"

        # Verify narrative claim: 8-10% observed
        assert 5 < observed_pct < 15, \
            f"Triple-positive percentage out of expected range: {observed_pct:.1f}%"


class TestScaleDependentOrganization:
    """Test claims about hierarchical tissue organization across scales."""

    def test_cluster_count_decreases_with_scale(self, all_results):
        """
        CLAIM: "11 clusters at 10μm → 2.5 clusters at 40μm (4.4× complexity reduction)"
        VALIDATION: Number of clusters should decrease with increasing scale
        """
        scales = [10.0, 20.0, 40.0]
        cluster_counts = {scale: [] for scale in scales}

        for roi_id, results in all_results.items():
            for scale in scales:
                scale_data = extract_scale_data(results, scale)
                if scale_data:
                    clusters = deserialize_array(scale_data['cluster_labels'])
                    n_clusters = len(np.unique(clusters))
                    cluster_counts[scale].append(n_clusters)

        # Calculate mean clusters at each scale
        mean_clusters = {
            scale: np.mean(counts)
            for scale, counts in cluster_counts.items()
            if counts
        }

        if len(mean_clusters) < 3:
            pytest.skip("Insufficient scale data")

        # Verify monotonic decrease
        assert mean_clusters[10.0] > mean_clusters[20.0], \
            f"10μm should have more clusters than 20μm"
        assert mean_clusters[20.0] > mean_clusters[40.0], \
            f"20μm should have more clusters than 40μm"

        # Verify magnitude of reduction
        complexity_reduction = mean_clusters[10.0] / mean_clusters[40.0]
        assert complexity_reduction > 2.5, \
            f"Complexity reduction too small: {complexity_reduction:.1f}×"
        assert complexity_reduction < 8.0, \
            f"Complexity reduction unexpectedly large: {complexity_reduction:.1f}×"


class TestSpatialCoherence:
    """Test claims about spatial organization of clusters."""

    def test_spatial_coherence_positive(self, all_results):
        """
        CLAIM: "Moran's I = 0.17 at 10μm (positive spatial autocorrelation)"
        VALIDATION: Clusters should show positive spatial autocorrelation
        """
        morans_i_values = []

        for roi_id, results in all_results.items():
            scale_data = extract_scale_data(results, 10.0)
            if scale_data:
                coherence = scale_data.get('spatial_coherence')
                if coherence is not None:
                    morans_i_values.append(coherence)

        if not morans_i_values:
            pytest.skip("No spatial coherence data found")

        mean_morans_i = np.mean(morans_i_values)

        # Should show positive spatial autocorrelation
        assert mean_morans_i > 0, \
            f"Moran's I should be positive (clusters spatially organized), got {mean_morans_i:.3f}"

        # Should be modest (indicating some but not complete spatial clustering)
        assert mean_morans_i < 0.5, \
            f"Moran's I unexpectedly high: {mean_morans_i:.3f}"


class TestClusteringStability:
    """Test claims about clustering stability and biological reproducibility."""

    def test_clusters_per_roi_variability(self, all_results):
        """
        CLAIM: "6-18 clusters per ROI (anatomical diversity, injury gradients)"
        VALIDATION: Number of clusters should be heterogeneous but bounded
        """
        cluster_counts = []

        for roi_id, results in all_results.items():
            scale_data = extract_scale_data(results, 10.0)
            if scale_data:
                clusters = deserialize_array(scale_data['cluster_labels'])
                n_clusters = len(np.unique(clusters))
                cluster_counts.append(n_clusters)

        if not cluster_counts:
            pytest.skip("No clustering data found")

        min_clusters = min(cluster_counts)
        max_clusters = max(cluster_counts)

        # Should show heterogeneity but within reasonable bounds
        assert min_clusters >= 4, f"Minimum clusters too low: {min_clusters}"
        assert max_clusters <= 25, f"Maximum clusters too high: {max_clusters}"

        # Verify narrative range: "6-18 clusters per ROI"
        # Allow some margin as biological systems are variable
        assert 5 < min_clusters < 10, f"Minimum out of expected range: {min_clusters}"
        assert 14 < max_clusters < 24, f"Maximum out of expected range: {max_clusters}"


# Pytest configuration
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "narrative: tests validating narrative notebook claims"
    )


# Mark all tests as narrative tests
pytestmark = pytest.mark.narrative


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

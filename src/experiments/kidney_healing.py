"""Kidney healing experiment framework with injury recovery analysis."""

from typing import Dict, List, Any
import numpy as np
from collections import defaultdict, Counter

from .base import ExperimentFramework
from src.config import Config


class KidneyHealingExperiment(ExperimentFramework):
    """Concrete experiment framework for kidney injury-healing research."""
    
    def __init__(self, config: Config):
        """Initialize kidney healing experiment framework."""
        super().__init__(config)
        self.injury_timepoints = config.raw.get('experimental', {}).get('timepoints', [0, 1, 3, 7])
        self.tissue_regions = config.raw.get('experimental', {}).get('regions', ['Cortex', 'Medulla'])
        self.biological_replicates = config.raw.get('experimental', {}).get('replicates', ['MS1', 'MS2'])
    
    def get_figure_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Define kidney healing-specific figure specifications."""
        return {
            'temporal_dynamics': {
                'method': 'create_temporal_figure',
                'params': {},
                'description': 'Temporal progression of tissue domains and contacts',
                'priority': 1
            },
            'regional_comparison': {
                'method': 'create_condition_figure', 
                'params': {},
                'description': 'Cortex vs Medulla tissue architecture comparison',
                'priority': 2
            },
            'functional_group_dynamics': {
                'method': 'create_functional_group_dynamics_figure',
                'params': {},
                'description': 'Inflammation, repair, and vascular dynamics over time',
                'priority': 3
            },
            'neighborhood_evolution': {
                'method': 'create_neighborhood_temporal_evolution',
                'params': {},
                'description': 'Multi-scale neighborhood evolution during healing',
                'priority': 4
            },
            'regional_neighborhoods': {
                'method': 'create_neighborhood_regional_comparison',
                'params': {},
                'description': 'Region-specific neighborhood organization patterns',
                'priority': 5
            },
            'replicate_variance': {
                'method': 'create_replicate_variance_figure',
                'params': {},
                'description': 'Biological replicate concordance analysis',
                'priority': 6
            },
            'network_architecture': {
                'method': 'create_network_figure',
                'params': {},
                'description': 'Protein colocalization networks across healing phases',
                'priority': 7
            },
            'regional_trajectories': {
                'method': 'create_region_temporal_trajectories',
                'params': {},
                'description': 'Region-specific temporal trajectory analysis',
                'priority': 8
            },
            'tmd_overlay_spatial': {
                'method': 'create_tmd_spatial_overlay',
                'params': {},
                'description': 'Tissue Microdomains overlaid on spatial coordinates',
                'priority': 9
            }
        }
    
    def get_experimental_groupings(self) -> Dict[str, List[str]]:
        """Define kidney healing experimental groupings."""
        # Convert timepoints to labels
        timepoint_labels = []
        for tp in self.injury_timepoints:
            if tp == 0:
                timepoint_labels.append('Sham')
            else:
                timepoint_labels.append(f'Day{tp}')
        
        return {
            'timepoints': timepoint_labels,
            'regions': self.tissue_regions,
            'replicates': self.biological_replicates
        }
    
    def get_functional_groups(self) -> Dict[str, List[str]]:
        """Define kidney healing functional protein groups."""
        return self.config.raw.get('proteins', {}).get('functional_groups', {
            'kidney_inflammation': ['CD45', 'CD11b', 'Ly6G'],
            'kidney_repair': ['CD206', 'CD44'],
            'kidney_vasculature': ['CD31', 'CD34', 'CD140a', 'CD140b'],
            'structural_controls': ['DNA1', 'DNA2']
        })
    
    def create_experimental_summary_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Create kidney healing-specific summary metrics."""
        # Group ROIs by experimental conditions
        by_timepoint = self._group_by_timepoint(results)
        by_region = self._group_by_region(results)
        by_replicate = self._group_by_replicate(results)
        by_region_time = self._group_by_region_timepoint(results)
        
        functional_groups = self.get_functional_groups()
        
        # Calculate summary statistics for each grouping
        metrics = {
            'experiment_overview': {
                'total_rois': len(results),
                'timepoints_analyzed': list(by_timepoint.keys()),
                'regions_analyzed': list(by_region.keys()),
                'replicates_analyzed': list(by_replicate.keys())
            },
            'temporal_progression': self._analyze_temporal_progression(by_timepoint, functional_groups),
            'regional_differences': self._analyze_regional_differences(by_region, functional_groups),
            'replicate_concordance': self._analyze_replicate_concordance(by_replicate),
            'region_time_interactions': self._analyze_region_time_interactions(by_region_time, functional_groups),
            'key_biological_insights': self._extract_biological_insights(results, functional_groups)
        }
        
        return metrics
    
    def _group_by_timepoint(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group ROIs by injury timepoint."""
        by_timepoint = defaultdict(list)
        
        for roi in results:
            metadata = self._extract_metadata(roi)
            day = metadata.get('injury_day') or metadata.get('timepoint')
            
            if day is not None:
                if day == 0:
                    key = 'Sham'
                else:
                    key = f'Day{day}'
                by_timepoint[key].append(roi)
        
        return dict(by_timepoint)
    
    def _group_by_region(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group ROIs by tissue region."""
        by_region = defaultdict(list)
        
        for roi in results:
            metadata = self._extract_metadata(roi)
            region = metadata.get('tissue_region', 'Unknown')
            by_region[region].append(roi)
        
        return dict(by_region)
    
    def _group_by_replicate(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group ROIs by biological replicate."""
        by_replicate = defaultdict(list)
        
        for roi in results:
            metadata = self._extract_metadata(roi)
            replicate = metadata.get('mouse_replicate', 'Unknown')
            by_replicate[replicate].append(roi)
        
        return dict(by_replicate)
    
    def _group_by_region_timepoint(self, results: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
        """Group ROIs by region and timepoint."""
        by_region_time = defaultdict(lambda: defaultdict(list))
        
        for roi in results:
            metadata = self._extract_metadata(roi)
            region = metadata.get('tissue_region', 'Unknown')
            day = metadata.get('injury_day') or metadata.get('timepoint')
            
            if day is not None:
                timepoint = 'Sham' if day == 0 else f'Day{day}'
                by_region_time[region][timepoint].append(roi)
        
        return {region: dict(timepoints) for region, timepoints in by_region_time.items()}
    
    def _extract_metadata(self, roi: Dict) -> Dict:
        """Extract metadata from ROI, handling different formats."""
        metadata = roi.get('metadata', {})
        if hasattr(metadata, 'to_dict'):
            return metadata.to_dict()
        elif hasattr(metadata, '_asdict'):
            return metadata._asdict()
        return metadata
    
    def _analyze_temporal_progression(self, by_timepoint: Dict, functional_groups: Dict) -> Dict:
        """Analyze temporal progression of functional groups."""
        progression = {}
        
        for timepoint, rois in by_timepoint.items():
            if not rois:
                continue
                
            # Calculate functional group percentages
            group_percentages = {}
            for group_name, proteins in functional_groups.items():
                if group_name == 'structural_controls':
                    continue
                    
                total_coverage = 0
                group_coverage = 0
                
                for roi in rois:
                    total_pixels = roi.get('total_pixels', 0)
                    if total_pixels > 0:
                        total_coverage += total_pixels
                        
                        # Calculate group coverage from blob signatures
                        blob_sigs = roi.get('blob_signatures', {})
                        for sig in blob_sigs.values():
                            dominant_proteins = sig.get('dominant_proteins', [])[:2]
                            if any(p in proteins for p in dominant_proteins):
                                group_coverage += sig.get('size', 0)
                
                if total_coverage > 0:
                    group_percentages[group_name] = 100.0 * group_coverage / total_coverage
                else:
                    group_percentages[group_name] = 0.0
            
            # Calculate tissue domain and contact statistics
            domain_counts = [len(roi.get('blob_signatures', {})) for roi in rois]
            contact_counts = [len(roi.get('canonical_contacts', {})) for roi in rois]
            
            progression[timepoint] = {
                'n_rois': len(rois),
                'functional_groups_pct': group_percentages,
                'domains_mean': float(np.mean(domain_counts)) if domain_counts else 0.0,
                'domains_std': float(np.std(domain_counts)) if domain_counts else 0.0,
                'contacts_mean': float(np.mean(contact_counts)) if contact_counts else 0.0,
                'contacts_std': float(np.std(contact_counts)) if contact_counts else 0.0
            }
        
        return progression
    
    def _analyze_regional_differences(self, by_region: Dict, functional_groups: Dict) -> Dict:
        """Analyze differences between tissue regions."""
        regional_analysis = {}
        
        for region, rois in by_region.items():
            if not rois:
                continue
            
            # Similar analysis as temporal but for regions
            group_percentages = {}
            for group_name, proteins in functional_groups.items():
                if group_name == 'structural_controls':
                    continue
                    
                total_coverage = 0
                group_coverage = 0
                
                for roi in rois:
                    total_pixels = roi.get('total_pixels', 0)
                    if total_pixels > 0:
                        total_coverage += total_pixels
                        
                        blob_sigs = roi.get('blob_signatures', {})
                        for sig in blob_sigs.values():
                            dominant_proteins = sig.get('dominant_proteins', [])[:2]
                            if any(p in proteins for p in dominant_proteins):
                                group_coverage += sig.get('size', 0)
                
                if total_coverage > 0:
                    group_percentages[group_name] = 100.0 * group_coverage / total_coverage
                else:
                    group_percentages[group_name] = 0.0
            
            domain_counts = [len(roi.get('blob_signatures', {})) for roi in rois]
            contact_counts = [len(roi.get('canonical_contacts', {})) for roi in rois]
            
            regional_analysis[region] = {
                'n_rois': len(rois),
                'functional_groups_pct': group_percentages,
                'domains_mean': float(np.mean(domain_counts)) if domain_counts else 0.0,
                'domains_std': float(np.std(domain_counts)) if domain_counts else 0.0,
                'contacts_mean': float(np.mean(contact_counts)) if contact_counts else 0.0,
                'contacts_std': float(np.std(contact_counts)) if contact_counts else 0.0
            }
        
        return regional_analysis
    
    def _analyze_replicate_concordance(self, by_replicate: Dict) -> Dict:
        """Analyze concordance between biological replicates."""
        concordance = {}
        
        # Calculate key metrics for each replicate
        replicate_metrics = {}
        for replicate, rois in by_replicate.items():
            if not rois:
                continue
                
            domain_counts = [len(roi.get('blob_signatures', {})) for roi in rois]
            contact_counts = [len(roi.get('canonical_contacts', {})) for roi in rois]
            
            replicate_metrics[replicate] = {
                'n_rois': len(rois),
                'domains_mean': float(np.mean(domain_counts)) if domain_counts else 0.0,
                'contacts_mean': float(np.mean(contact_counts)) if contact_counts else 0.0
            }
        
        # Calculate concordance statistics
        if len(replicate_metrics) >= 2:
            replicates = list(replicate_metrics.keys())
            
            # Correlation between domain counts
            rep1_domains = replicate_metrics[replicates[0]]['domains_mean']
            rep2_domains = replicate_metrics[replicates[1]]['domains_mean']
            
            rep1_contacts = replicate_metrics[replicates[0]]['contacts_mean']
            rep2_contacts = replicate_metrics[replicates[1]]['contacts_mean']
            
            concordance = {
                'replicate_metrics': replicate_metrics,
                'domain_fold_difference': abs(rep1_domains - rep2_domains) / max(rep1_domains, rep2_domains, 1e-6),
                'contact_fold_difference': abs(rep1_contacts - rep2_contacts) / max(rep1_contacts, rep2_contacts, 1e-6),
                'overall_concordance': 'High' if max(
                    abs(rep1_domains - rep2_domains) / max(rep1_domains, rep2_domains, 1e-6),
                    abs(rep1_contacts - rep2_contacts) / max(rep1_contacts, rep2_contacts, 1e-6)
                ) < 0.5 else 'Moderate'
            }
        else:
            concordance = {'replicate_metrics': replicate_metrics, 'note': 'Insufficient replicates for concordance analysis'}
        
        return concordance
    
    def _analyze_region_time_interactions(self, by_region_time: Dict, functional_groups: Dict) -> Dict:
        """Analyze interactions between region and time."""
        interactions = {}
        
        for region, timepoint_data in by_region_time.items():
            region_progression = {}
            
            for timepoint, rois in timepoint_data.items():
                if not rois:
                    continue
                
                # Calculate functional group coverage
                group_percentages = {}
                for group_name, proteins in functional_groups.items():
                    if group_name == 'structural_controls':
                        continue
                        
                    total_coverage = 0
                    group_coverage = 0
                    
                    for roi in rois:
                        total_pixels = roi.get('total_pixels', 0)
                        if total_pixels > 0:
                            total_coverage += total_pixels
                            
                            blob_sigs = roi.get('blob_signatures', {})
                            for sig in blob_sigs.values():
                                dominant_proteins = sig.get('dominant_proteins', [])[:2]
                                if any(p in proteins for p in dominant_proteins):
                                    group_coverage += sig.get('size', 0)
                    
                    if total_coverage > 0:
                        group_percentages[group_name] = 100.0 * group_coverage / total_coverage
                    else:
                        group_percentages[group_name] = 0.0
                
                region_progression[timepoint] = group_percentages
            
            interactions[region] = region_progression
        
        return interactions
    
    def _extract_biological_insights(self, results: List[Dict], functional_groups: Dict) -> Dict:
        """Extract key biological insights from the analysis."""
        insights = {}
        
        # Group by timepoint for trajectory analysis
        by_timepoint = self._group_by_timepoint(results)
        
        # Track inflammation trajectory
        inflammation_trajectory = []
        repair_trajectory = []
        vascular_trajectory = []
        
        for timepoint in ['Sham', 'Day1', 'Day3', 'Day7']:
            if timepoint not in by_timepoint:
                continue
                
            rois = by_timepoint[timepoint]
            
            # Calculate group coverages
            for group_name, proteins in functional_groups.items():
                if group_name == 'structural_controls':
                    continue
                    
                total_coverage = 0
                group_coverage = 0
                
                for roi in rois:
                    total_pixels = roi.get('total_pixels', 0)
                    if total_pixels > 0:
                        total_coverage += total_pixels
                        
                        blob_sigs = roi.get('blob_signatures', {})
                        for sig in blob_sigs.values():
                            dominant_proteins = sig.get('dominant_proteins', [])[:2]
                            if any(p in proteins for p in dominant_proteins):
                                group_coverage += sig.get('size', 0)
                
                coverage_pct = 100.0 * group_coverage / total_coverage if total_coverage > 0 else 0.0
                
                if group_name == 'kidney_inflammation':
                    inflammation_trajectory.append(coverage_pct)
                elif group_name == 'kidney_repair':
                    repair_trajectory.append(coverage_pct)
                elif group_name == 'kidney_vasculature':
                    vascular_trajectory.append(coverage_pct)
        
        # Determine key insights
        insights = {
            'inflammation_pattern': self._characterize_trajectory(inflammation_trajectory),
            'repair_pattern': self._characterize_trajectory(repair_trajectory),
            'vascular_pattern': self._characterize_trajectory(vascular_trajectory),
            'peak_inflammation_phase': self._find_peak_phase(inflammation_trajectory),
            'peak_repair_phase': self._find_peak_phase(repair_trajectory),
            'healing_dynamics': self._assess_healing_dynamics(inflammation_trajectory, repair_trajectory)
        }
        
        return insights
    
    def _characterize_trajectory(self, trajectory: List[float]) -> str:
        """Characterize the pattern of a trajectory."""
        if len(trajectory) < 2:
            return 'Insufficient data'
        
        # Calculate trends
        early_trend = trajectory[1] - trajectory[0] if len(trajectory) > 1 else 0
        late_trend = trajectory[-1] - trajectory[-2] if len(trajectory) > 2 else 0
        
        if early_trend > 5 and late_trend < -5:
            return 'Early peak with resolution'
        elif early_trend > 5:
            return 'Progressive increase'
        elif late_trend < -5:
            return 'Progressive decrease'
        elif max(trajectory) - min(trajectory) < 5:
            return 'Stable/unchanged'
        else:
            return 'Variable pattern'
    
    def _find_peak_phase(self, trajectory: List[float]) -> str:
        """Find the phase with peak activity."""
        if not trajectory:
            return 'Unknown'
        
        phases = ['Sham', 'Day1', 'Day3', 'Day7'][:len(trajectory)]
        peak_idx = trajectory.index(max(trajectory))
        return phases[peak_idx]
    
    def _assess_healing_dynamics(self, inflammation: List[float], repair: List[float]) -> str:
        """Assess overall healing dynamics pattern."""
        if len(inflammation) < 3 or len(repair) < 3:
            return 'Insufficient data for assessment'
        
        # Check if inflammation peaks early and repair peaks later
        inflammation_peak = inflammation.index(max(inflammation))
        repair_peak = repair.index(max(repair))
        
        if inflammation_peak < repair_peak:
            return 'Sequential: Inflammation followed by repair'
        elif inflammation_peak == repair_peak:
            return 'Concurrent: Inflammation and repair overlap'
        else:
            return 'Delayed: Repair precedes peak inflammation'
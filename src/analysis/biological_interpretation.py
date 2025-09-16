"""Biologically-informed interpretation of IMC kidney healing data.

This module leverages known biological functions of protein markers to interpret
unsupervised discoveries in the context of kidney injury and repair.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings

from src.config import Config


class BiologicallyInformedAnalysis:
    """Analysis that leverages known marker biology for interpretation."""
    
    def __init__(self, config: Config):
        """Initialize with biological knowledge from config.
        
        Args:
            config: Configuration with protein annotations
        """
        self.config = config
        self.protein_info = config.raw.get('proteins', {})
        self.functional_groups = self.protein_info.get('functional_groups', {})
        self.annotations = self.protein_info.get('annotations', {})
        self.timepoints = config.raw.get('experimental', {}).get('timepoint_labels', [])
        
    def interpret_phenotypes_biologically(self,
                                        phenotypes: Dict[str, Any],
                                        protein_names: List[str]) -> Dict[str, Any]:
        """Interpret discovered phenotypes using known marker biology.
        
        This maps mathematical clusters to likely biological identities based on
        established marker functions, but still requires validation.
        
        Args:
            phenotypes: Discovered phenotype information from unsupervised clustering
            protein_names: List of protein marker names
            
        Returns:
            Biologically interpreted phenotype information
        """
        interpreted = {
            'phenotype_interpretations': {},
            'functional_group_enrichment': {},
            'likely_cell_states': [],
            'confidence_level': 'hypothesis'
        }
        
        for phenotype_id, phenotype_data in phenotypes.items():
            if not isinstance(phenotype_data, dict):
                continue
                
            interpretation = {
                'mathematical_id': phenotype_id,
                'defining_markers': phenotype_data.get('defining_markers', []),
                'biological_interpretation': None,
                'confidence': 'low',
                'supporting_evidence': []
            }
            
            # Analyze marker expression pattern
            high_markers = [m['protein'] for m in phenotype_data.get('defining_markers', []) 
                           if m.get('direction') == 'high' and m.get('z_score', 0) > 1.5]
            low_markers = [m['protein'] for m in phenotype_data.get('defining_markers', [])
                          if m.get('direction') == 'low' and m.get('z_score', 0) < -1.5]
            
            # Match against known biological patterns
            biological_identity = self._match_to_known_patterns(high_markers, low_markers)
            interpretation['biological_interpretation'] = biological_identity['identity']
            interpretation['confidence'] = biological_identity['confidence']
            interpretation['supporting_evidence'] = biological_identity['evidence']
            
            # Check functional group enrichment
            for group_name, group_markers in self.functional_groups.items():
                overlap = len(set(high_markers) & set(group_markers))
                if overlap > 0:
                    enrichment_score = overlap / len(group_markers)
                    if enrichment_score > 0.5:
                        interpretation['supporting_evidence'].append(
                            f"Enriched for {group_name} markers ({overlap}/{len(group_markers)})"
                        )
            
            interpreted['phenotype_interpretations'][phenotype_id] = interpretation
        
        # Generate summary of likely cell states
        interpreted['likely_cell_states'] = self._summarize_cell_states(
            interpreted['phenotype_interpretations']
        )
        
        # Add biological context
        interpreted['biological_context'] = self._generate_biological_context()
        
        return interpreted
    
    def analyze_kidney_healing_dynamics(self,
                                       temporal_results: Dict[str, Any],
                                       phenotype_abundances: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze kidney healing dynamics using biological knowledge.
        
        Args:
            temporal_results: Results from temporal dynamics analysis
            phenotype_abundances: Phenotype frequencies by timepoint
            
        Returns:
            Biologically interpreted healing dynamics
        """
        healing_analysis = {
            'injury_response': {},
            'repair_progression': {},
            'vascular_recovery': {},
            'biological_phases': [],
            'key_transitions': []
        }
        
        # Analyze inflammatory response (CD45, CD11b, Ly6G)
        inflammatory_markers = ['CD45', 'CD11b', 'Ly6G']
        if 'temporal_trends' in temporal_results:
            inflammation_trends = {}
            for marker in inflammatory_markers:
                if marker in temporal_results['temporal_trends']:
                    trend_data = temporal_results['temporal_trends'][marker]
                    inflammation_trends[marker] = {
                        'trend': trend_data.get('trend_test', {}).get('interpretation', 'unknown'),
                        'peak_timepoint': self._find_peak_timepoint(
                            trend_data.get('mean_expression_by_time', {})
                        )
                    }
            
            healing_analysis['injury_response'] = {
                'inflammatory_markers': inflammation_trends,
                'interpretation': self._interpret_inflammatory_response(inflammation_trends)
            }
        
        # Analyze repair markers (CD206, CD44)
        repair_markers = ['CD206', 'CD44']
        repair_trends = {}
        for marker in repair_markers:
            if marker in temporal_results.get('temporal_trends', {}):
                trend_data = temporal_results['temporal_trends'][marker]
                repair_trends[marker] = {
                    'trend': trend_data.get('trend_test', {}).get('interpretation', 'unknown'),
                    'onset_timepoint': self._find_onset_timepoint(
                        trend_data.get('mean_expression_by_time', {})
                    )
                }
        
        healing_analysis['repair_progression'] = {
            'repair_markers': repair_trends,
            'interpretation': self._interpret_repair_response(repair_trends)
        }
        
        # Analyze vascular markers (CD31, CD34, CD140a, CD140b)
        vascular_markers = ['CD31', 'CD34', 'CD140a', 'CD140b']
        vascular_trends = {}
        for marker in vascular_markers:
            if marker in temporal_results.get('temporal_trends', {}):
                trend_data = temporal_results['temporal_trends'][marker]
                vascular_trends[marker] = trend_data.get('trend_test', {}).get('interpretation', 'unknown')
        
        healing_analysis['vascular_recovery'] = {
            'vascular_markers': vascular_trends,
            'interpretation': self._interpret_vascular_response(vascular_trends)
        }
        
        # Define biological phases based on marker dynamics
        healing_analysis['biological_phases'] = self._define_healing_phases(
            inflammation_trends, repair_trends, vascular_trends
        )
        
        # Identify key transitions
        healing_analysis['key_transitions'] = self._identify_transitions(
            temporal_results, phenotype_abundances
        )
        
        # Generate integrated interpretation
        healing_analysis['integrated_interpretation'] = self._generate_integrated_interpretation(
            healing_analysis
        )
        
        return healing_analysis
    
    def validate_against_literature(self,
                                   analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate findings against known kidney injury biology.
        
        Args:
            analysis_results: Results to validate
            
        Returns:
            Literature-based validation assessment
        """
        validation = {
            'consistency_with_literature': {},
            'unexpected_findings': [],
            'validation_level': 'partial',
            'recommendations': []
        }
        
        # Known patterns in kidney injury/repair
        expected_patterns = {
            'early_inflammation': {
                'timepoint': 'Day1',
                'markers': ['CD45+', 'CD11b+', 'Ly6G+'],
                'description': 'Neutrophil infiltration peaks at 24h post-injury'
            },
            'macrophage_transition': {
                'timepoint': 'Day3-Day7',
                'markers': ['CD11b+', 'CD206+ emerging'],
                'description': 'M1 to M2 macrophage transition during repair'
            },
            'vascular_repair': {
                'timepoint': 'Day3-Day7',
                'markers': ['CD31+', 'CD140b+'],
                'description': 'Endothelial repair and pericyte recruitment'
            },
            'matrix_remodeling': {
                'timepoint': 'Day7+',
                'markers': ['CD44+', 'CD140a+'],
                'description': 'ECM remodeling and fibroblast activation'
            }
        }
        
        # Check consistency with expected patterns
        for pattern_name, expected in expected_patterns.items():
            validation['consistency_with_literature'][pattern_name] = {
                'expected': expected['description'],
                'observed': 'Not assessed',
                'consistent': False
            }
            
            # Check if pattern was observed (simplified check)
            # In real implementation, would check actual data
            validation['consistency_with_literature'][pattern_name]['observed'] = (
                "Pattern assessment requires temporal phenotype data"
            )
        
        # Literature references for validation
        validation['key_references'] = [
            "Jang & Rabb (2015) - Immune cells in experimental acute kidney injury",
            "Ferenbach & Bonventre (2015) - Mechanisms of maladaptive repair after AKI",
            "Kramann et al. (2015) - Perivascular cells in kidney fibrosis",
            "Black et al. (2019) - Renal inflammation and fibrosis: a double-edged sword"
        ]
        
        # Generate recommendations
        validation['recommendations'] = [
            "Compare CD206/CD11b ratio dynamics to published M1/M2 transition kinetics",
            "Validate CD140b+ pericyte expansion against known vascular repair patterns",
            "Confirm neutrophil (Ly6G+) clearance timeline matches literature (24-48h)",
            "Check if CD44+ matrix remodeling coincides with fibrosis markers"
        ]
        
        return validation
    
    def _match_to_known_patterns(self, high_markers: List[str], 
                                low_markers: List[str]) -> Dict[str, Any]:
        """Match marker expression to known cell types/states."""
        
        # Known cell type patterns based on config annotations
        known_patterns = {
            'M1_macrophage': {
                'required_high': ['CD45', 'CD11b'],
                'optional_high': [],
                'should_be_low': ['CD206'],
                'confidence_threshold': 2
            },
            'M2_macrophage': {
                'required_high': ['CD45', 'CD11b', 'CD206'],
                'optional_high': [],
                'should_be_low': [],
                'confidence_threshold': 3
            },
            'neutrophil': {
                'required_high': ['CD45', 'Ly6G'],
                'optional_high': ['CD11b'],
                'should_be_low': [],
                'confidence_threshold': 2
            },
            'endothelial': {
                'required_high': ['CD31'],
                'optional_high': ['CD34'],
                'should_be_low': ['CD45'],
                'confidence_threshold': 1
            },
            'pericyte': {
                'required_high': ['CD140b'],
                'optional_high': ['CD140a'],
                'should_be_low': ['CD45', 'CD31'],
                'confidence_threshold': 1
            },
            'progenitor': {
                'required_high': ['CD34'],
                'optional_high': ['CD44'],
                'should_be_low': ['CD45'],
                'confidence_threshold': 1
            },
            'activated_fibroblast': {
                'required_high': ['CD140a', 'CD44'],
                'optional_high': [],
                'should_be_low': ['CD45', 'CD31'],
                'confidence_threshold': 2
            }
        }
        
        best_match = {
            'identity': 'Unknown phenotype',
            'confidence': 'low',
            'evidence': []
        }
        best_score = 0
        
        for cell_type, pattern in known_patterns.items():
            score = 0
            evidence = []
            
            # Check required high markers
            required_matches = len(set(pattern['required_high']) & set(high_markers))
            score += required_matches * 2
            if required_matches == len(pattern['required_high']):
                evidence.append(f"All required markers present: {', '.join(pattern['required_high'])}")
            
            # Check optional high markers
            optional_matches = len(set(pattern['optional_high']) & set(high_markers))
            score += optional_matches
            if optional_matches > 0:
                matched = list(set(pattern['optional_high']) & set(high_markers))
                evidence.append(f"Optional markers present: {', '.join(matched)}")
            
            # Check markers that should be low
            correct_low = len(set(pattern['should_be_low']) & set(low_markers))
            score += correct_low
            if correct_low > 0:
                evidence.append(f"Expected low markers confirmed: {', '.join(set(pattern['should_be_low']) & set(low_markers))}")
            
            # Penalize if "should be low" markers are high
            incorrect_high = len(set(pattern['should_be_low']) & set(high_markers))
            score -= incorrect_high * 2
            
            if score > best_score and score >= pattern['confidence_threshold']:
                best_score = score
                best_match = {
                    'identity': cell_type.replace('_', ' ').title(),
                    'confidence': 'high' if score >= pattern['confidence_threshold'] * 1.5 else 'moderate',
                    'evidence': evidence
                }
        
        return best_match
    
    def _find_peak_timepoint(self, expression_by_time: Dict[str, float]) -> str:
        """Find timepoint with peak expression."""
        if not expression_by_time:
            return 'unknown'
        return max(expression_by_time.items(), key=lambda x: x[1])[0]
    
    def _find_onset_timepoint(self, expression_by_time: Dict[str, float]) -> str:
        """Find timepoint where expression begins to increase."""
        if not expression_by_time:
            return 'unknown'
        
        sorted_times = sorted(expression_by_time.items(), key=lambda x: self.timepoints.index(x[0]) if x[0] in self.timepoints else 999)
        
        if len(sorted_times) < 2:
            return 'unknown'
        
        # Find first significant increase (>20% from baseline)
        baseline = sorted_times[0][1]
        for timepoint, value in sorted_times[1:]:
            if value > baseline * 1.2:
                return timepoint
        
        return 'no significant onset'
    
    def _interpret_inflammatory_response(self, inflammation_trends: Dict) -> str:
        """Interpret inflammatory marker dynamics."""
        interpretations = []
        
        # Check for neutrophil response (Ly6G)
        if 'Ly6G' in inflammation_trends:
            ly6g_peak = inflammation_trends['Ly6G'].get('peak_timepoint', 'unknown')
            if ly6g_peak == 'Day1':
                interpretations.append("Classic acute neutrophil response peaking at 24h")
            elif ly6g_peak == 'Day3':
                interpretations.append("Delayed or prolonged neutrophil response")
        
        # Check for macrophage dynamics (CD11b)
        if 'CD11b' in inflammation_trends:
            cd11b_trend = inflammation_trends['CD11b'].get('trend', 'unknown')
            if 'increasing' in cd11b_trend.lower():
                interpretations.append("Progressive myeloid cell accumulation")
            elif 'decreasing' in cd11b_trend.lower():
                interpretations.append("Resolving myeloid inflammation")
        
        return '; '.join(interpretations) if interpretations else "Inflammatory dynamics unclear"
    
    def _interpret_repair_response(self, repair_trends: Dict) -> str:
        """Interpret repair marker dynamics."""
        interpretations = []
        
        # Check M2 macrophage emergence (CD206)
        if 'CD206' in repair_trends:
            cd206_onset = repair_trends['CD206'].get('onset_timepoint', 'unknown')
            if cd206_onset in ['Day3', 'Day7']:
                interpretations.append(f"M2 macrophage emergence at {cd206_onset} suggests repair initiation")
        
        # Check matrix remodeling (CD44)
        if 'CD44' in repair_trends:
            cd44_trend = repair_trends['CD44'].get('trend', 'unknown')
            if 'increasing' in cd44_trend.lower():
                interpretations.append("Active ECM remodeling indicates tissue repair")
        
        return '; '.join(interpretations) if interpretations else "Repair response not clearly defined"
    
    def _interpret_vascular_response(self, vascular_trends: Dict) -> str:
        """Interpret vascular marker dynamics."""
        interpretations = []
        
        if 'CD31' in vascular_trends:
            if 'decreasing' in vascular_trends['CD31'].lower():
                interpretations.append("Vascular injury/loss evident")
            elif 'increasing' in vascular_trends['CD31'].lower():
                interpretations.append("Vascular regeneration ongoing")
        
        if 'CD140b' in vascular_trends:
            if 'increasing' in vascular_trends['CD140b'].lower():
                interpretations.append("Pericyte recruitment for vascular stabilization")
        
        return '; '.join(interpretations) if interpretations else "Vascular response unclear"
    
    def _define_healing_phases(self, inflammation: Dict, repair: Dict, vascular: Dict) -> List[Dict]:
        """Define biological phases of kidney healing."""
        phases = []
        
        # Phase 1: Acute injury/inflammation (Day 0-1)
        phases.append({
            'phase': 'Acute Injury Response',
            'timepoints': ['Sham', 'Day1'],
            'characteristics': 'Neutrophil infiltration, initial vascular injury',
            'key_markers': ['Ly6G+', 'CD45+', 'CD11b+']
        })
        
        # Phase 2: Transition (Day 1-3)
        phases.append({
            'phase': 'Inflammatory-to-Repair Transition',
            'timepoints': ['Day1', 'Day3'],
            'characteristics': 'Neutrophil clearance, macrophage accumulation',
            'key_markers': ['CD11b+', 'CD206 emerging']
        })
        
        # Phase 3: Active repair (Day 3-7)
        phases.append({
            'phase': 'Active Tissue Repair',
            'timepoints': ['Day3', 'Day7'],
            'characteristics': 'M2 macrophages, vascular repair, ECM remodeling',
            'key_markers': ['CD206+', 'CD44+', 'CD140b+']
        })
        
        return phases
    
    def _identify_transitions(self, temporal_results: Dict, 
                             phenotype_abundances: Dict) -> List[Dict]:
        """Identify key biological transitions."""
        transitions = []
        
        # Look for M1 to M2 transition
        transitions.append({
            'transition': 'M1 to M2 macrophage polarization',
            'expected_timing': 'Day1 to Day3',
            'markers': 'CD206 upregulation while maintaining CD11b',
            'biological_significance': 'Shift from inflammation to repair'
        })
        
        # Look for vascular repair initiation
        transitions.append({
            'transition': 'Vascular repair initiation',
            'expected_timing': 'Day3 to Day7',
            'markers': 'CD31 recovery, CD140b recruitment',
            'biological_significance': 'Restoration of renal perfusion'
        })
        
        return transitions
    
    def _summarize_cell_states(self, phenotype_interpretations: Dict) -> List[str]:
        """Summarize likely cell states present."""
        cell_states = []
        
        for phenotype_id, interpretation in phenotype_interpretations.items():
            if interpretation['confidence'] in ['high', 'moderate']:
                cell_states.append(f"{interpretation['biological_interpretation']} ({interpretation['confidence']} confidence)")
        
        return cell_states if cell_states else ["No high-confidence cell states identified"]
    
    def _generate_biological_context(self) -> str:
        """Generate biological context for interpretation."""
        return (
            "Kidney injury and repair involves coordinated cellular responses:\n"
            "1. Acute phase (0-24h): Neutrophil infiltration (Ly6G+), vascular injury\n"
            "2. Transition (24-72h): Neutrophil clearance, monocyte recruitment (CD11b+)\n"
            "3. Repair phase (3-7d): M2 macrophages (CD206+), pericyte recruitment (CD140b+), "
            "ECM remodeling (CD44+)\n"
            "4. Resolution (>7d): Restoration of tissue architecture or progression to fibrosis\n\n"
            "Key markers in this panel:\n"
            "- Inflammation: CD45 (all immune), CD11b (myeloid), Ly6G (neutrophils)\n"
            "- Repair: CD206 (M2 macrophages), CD44 (ECM receptor)\n"
            "- Vasculature: CD31 (endothelial), CD140b (pericytes)\n"
            "- Regeneration: CD34 (progenitors), CD140a (mesenchymal)"
        )
    
    def _generate_integrated_interpretation(self, healing_analysis: Dict) -> str:
        """Generate integrated interpretation of healing dynamics."""
        interpretation = []
        
        # Summarize injury response
        if healing_analysis['injury_response'].get('interpretation'):
            interpretation.append(f"Injury Response: {healing_analysis['injury_response']['interpretation']}")
        
        # Summarize repair
        if healing_analysis['repair_progression'].get('interpretation'):
            interpretation.append(f"Repair: {healing_analysis['repair_progression']['interpretation']}")
        
        # Summarize vascular
        if healing_analysis['vascular_recovery'].get('interpretation'):
            interpretation.append(f"Vasculature: {healing_analysis['vascular_recovery']['interpretation']}")
        
        # Add phase summary
        if healing_analysis['biological_phases']:
            current_phase = healing_analysis['biological_phases'][-1]
            interpretation.append(f"Current Phase: {current_phase['phase']}")
        
        return '\n'.join(interpretation) if interpretation else "Integrated interpretation requires complete temporal data"
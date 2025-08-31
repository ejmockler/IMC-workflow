"""Temporal visualization module for time-based analysis."""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from src.config import Config


class TemporalVisualizer:
    """Creates temporal progression visualizations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_temporal_figure(self, results: List[Dict]) -> plt.Figure:
        """Create temporal progression analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Group by timepoint
        by_timepoint = {}
        for roi in results:
            day = roi['metadata'].get('injury_day') if isinstance(roi['metadata'], dict) else getattr(roi['metadata'], 'injury_day', None)
            if day is not None:
                key = f"D{day}"
                if key not in by_timepoint:
                    by_timepoint[key] = []
                by_timepoint[key].append(roi)
        
        if not by_timepoint:
            fig.suptitle('Temporal Analysis - No temporal data available')
            return fig
        
        timepoints = sorted(by_timepoint.keys(), key=lambda x: int(x[1:]))
        
        # Build display labels and x positions using config
        exp_cfg = self.config.experimental
        tp_vals = exp_cfg.get('timepoints', [])
        tp_labels = exp_cfg.get('timepoint_labels', [])
        label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}
        display_labels = [label_map.get(tp, tp) for tp in timepoints]
        x_pos = np.arange(len(timepoints))

        # 1. Domain count progression
        ax = axes[0, 0]
        self._plot_domain_progression(ax, by_timepoint, timepoints, display_labels)
        
        # 2. Contact diversity progression  
        ax = axes[0, 1]
        self._plot_contact_diversity(ax, by_timepoint, timepoints, display_labels)
        
        # 3. Biological process dynamics
        ax = axes[1, 0]
        self._plot_biological_processes(ax, by_timepoint, timepoints, display_labels, x_pos)
        
        # 4. Top contact evolution
        ax = axes[1, 1]
        self._plot_contact_evolution(ax, results, by_timepoint, timepoints, display_labels, x_pos)
        
        fig.suptitle('Kidney Injury Recovery Timeline', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_domain_progression(self, ax, by_timepoint, timepoints, display_labels):
        """Plot domain count progression over time."""
        domain_counts = []
        for tp in timepoints:
            counts = [len(roi['blob_signatures']) for roi in by_timepoint[tp]]
            domain_counts.append(counts)
        
        bp = ax.boxplot(domain_counts, tick_labels=display_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_ylabel('Number of Domains')
        ax.set_title('Cellular Domain Formation During Recovery')
        ax.grid(True, alpha=0.3)
    
    def _plot_contact_diversity(self, ax, by_timepoint, timepoints, display_labels):
        """Plot contact diversity progression over time."""
        contact_counts = []
        for tp in timepoints:
            counts = [len(roi['canonical_contacts']) for roi in by_timepoint[tp]]
            contact_counts.append(counts)
        
        bp = ax.boxplot(contact_counts, tick_labels=display_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        ax.set_ylabel('Number of Contact Pairs')
        ax.set_title('Cellular Interaction Complexity')
        ax.grid(True, alpha=0.3)
    
    def _plot_biological_processes(self, ax, by_timepoint, timepoints, display_labels, x_pos):
        """Plot biologically-relevant domain evolution."""
        # Key kidney repair processes
        key_domains = {
            'CD31+CD34': 'Vascular Regeneration',    # Endothelial repair in kidney
            'CD44+CD45': 'Inflammation Resolution',  # ECM remodeling + immune clearance
            'CD11b+CD44': 'Myeloid-Matrix Response', # Inflammatory cells + matrix repair
            'CD206+CD44': 'Kidney Tissue Repair'    # M2 macrophages + ECM healing
        }
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (domain, process_name) in enumerate(key_domains.items()):
            means = []
            for tp in timepoints:
                tp_percentages = []
                for roi in by_timepoint[tp]:
                    total_size = sum(sig['size'] for sig in roi['blob_signatures'].values())
                    domain_size = sum(sig['size'] for name, sig in roi['blob_signatures'].items() 
                                    if '+'.join(sorted(sig['dominant_proteins'][:2])) == domain)
                    if total_size > 0:
                        tp_percentages.append(100 * domain_size / total_size)
                means.append(np.mean(tp_percentages) if tp_percentages else 0)
            
            ax.plot(x_pos, means, marker='o', label=process_name, 
                   linewidth=2, color=colors[i % len(colors)])
        
        ax.set_ylabel('Biological Process Representation (%)')
        ax.set_title('Kidney Repair Process Dynamics')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_labels)
    
    def _plot_contact_evolution(self, ax, results, by_timepoint, timepoints, display_labels, x_pos):
        """Plot evolution of top contact pairs over time."""
        # Collect all contact data and score them - DEDUPLICATE PAIRS
        contact_data = {}
        
        for roi in results:
            if 'canonical_contacts' in roi:
                for contact, strength in roi['canonical_contacts'].items():
                    # Normalize contact pairs to handle undirected interactions
                    normalized_contact = self._normalize_contact_pair(contact)
                    
                    if normalized_contact not in contact_data:
                        contact_data[normalized_contact] = {'strengths': [], 'timepoint_presence': {}}
                    contact_data[normalized_contact]['strengths'].append(strength)
                    
                    # Track which timepoints this contact appears in
                    day = roi['metadata'].get('injury_day') if isinstance(roi['metadata'], dict) else getattr(roi['metadata'], 'injury_day', None)
                    if day is not None:
                        tp_key = f"D{day}"
                        if tp_key not in contact_data[normalized_contact]['timepoint_presence']:
                            contact_data[normalized_contact]['timepoint_presence'][tp_key] = []
                        contact_data[normalized_contact]['timepoint_presence'][tp_key].append(strength)
        
        # Score contacts for temporal relevance
        contact_scores = {}
        for contact, data in contact_data.items():
            strengths = data['strengths']
            tp_presence = data['timepoint_presence']
            
            # Scoring criteria
            avg_strength = np.mean(strengths)
            frequency = len(strengths)
            temporal_spread = len(tp_presence)
            
            # Calculate temporal dynamics
            tp_means = [np.mean(vals) for vals in tp_presence.values()]
            temporal_variation = np.std(tp_means) / np.mean(tp_means) if len(tp_means) > 1 and np.mean(tp_means) > 0 else 0
            
            # Combined score
            score = (
                avg_strength * 2.0 +
                min(frequency / 10, 1.0) +
                min(temporal_spread / 4, 1.0) +
                temporal_variation * 1.5
            )
            
            # Only consider contacts with minimum frequency and strength
            if frequency >= 3 and avg_strength >= 0.1:
                contact_scores[contact] = score
        
        # Select contacts using biological diversity scoring to reduce redundancy
        top_n = self.config.get('visualization.top_interactions_count', 8)
        key_contacts = self._select_diverse_interactions(contact_scores, top_n)

        colors = plt.cm.tab10(np.linspace(0, 1, len(key_contacts)))

        # Plot contact evolution and collect final positions for smart labeling
        final_positions = []
        plot_data = []
        
        for i, contact in enumerate(key_contacts):
            strengths = []
            for tp in timepoints:
                tp_strengths = []
                for roi in by_timepoint[tp]:
                    if contact in roi['canonical_contacts']:
                        tp_strengths.append(roi['canonical_contacts'][contact])
                strengths.append(np.mean(tp_strengths) if tp_strengths else 0)
            
            ax.plot(x_pos, strengths, marker='s', linewidth=2, color=colors[i])
            final_positions.append(strengths[-1])
            plot_data.append((contact, strengths[-1], colors[i]))
        
        # Smart label positioning to avoid overlaps
        label_positions = []
        min_distance = 0.05  # Minimum vertical distance between labels (as fraction of y-range)
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        min_dist_abs = min_distance * y_range
        
        # Sort by final y position to process from bottom to top
        sorted_data = sorted(enumerate(plot_data), key=lambda x: x[1][1])
        
        for original_idx, (contact, base_y, color) in sorted_data:
            # Find a position that doesn't overlap with existing labels
            best_y = base_y
            
            # Check against all previously placed labels
            while any(abs(best_y - pos) < min_dist_abs for pos in label_positions):
                # Move up to avoid overlap
                best_y += min_dist_abs * 1.2
            
            label_positions.append(best_y)
            
            # Draw a subtle connector line if label was moved significantly
            if abs(best_y - base_y) > min_dist_abs * 0.5:
                ax.plot([x_pos[-1], x_pos[-1] + 0.03], [base_y, best_y], 
                       color='gray', alpha=0.4, linewidth=0.8)
            
            # Place the label
            ax.text(x_pos[-1] + 0.05, best_y, contact, fontsize=7,
                    va='center', color=color, weight='bold')
        
        ax.set_ylabel('Mean Contact Strength')
        ax.set_title(f'Top {len(key_contacts)} Cellular Interactions Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_labels)
    
    def _select_diverse_interactions(self, contact_scores: Dict[str, float], top_n: int) -> List[str]:
        """Select interactions with biological diversity scoring to reduce redundancy."""
        
        # Define biological process groups for diversity scoring
        biological_processes = {
            'immune': ['CD45', 'CD11b', 'Ly6G'],
            'vascular': ['CD31', 'CD34'], 
            'repair': ['CD44', 'CD206'],
            'mesenchymal': ['CD140a', 'CD140b']
        }
        
        # Sort all candidates by base score
        candidates = sorted(contact_scores.items(), key=lambda x: x[1], reverse=True)
        selected_contacts = []
        used_proteins = set()
        process_counts = {proc: 0 for proc in biological_processes.keys()}
        
        for contact, base_score in candidates:
            if len(selected_contacts) >= top_n:
                break
                
            # Extract all proteins from this interaction
            if ' ↔ ' in contact:
                domain1, domain2 = contact.split(' ↔ ')
                proteins1 = set(domain1.strip().split('+'))
                proteins2 = set(domain2.strip().split('+'))
                interaction_proteins = proteins1.union(proteins2)
            else:
                continue
            
            # Calculate diversity penalties and bonuses
            overlap_penalty = len(interaction_proteins.intersection(used_proteins)) / len(interaction_proteins)
            
            # Biological process diversity bonus
            interaction_processes = set()
            for protein in interaction_proteins:
                for process, process_proteins in biological_processes.items():
                    if protein in process_proteins:
                        interaction_processes.add(process)
            
            # Bonus for spanning multiple processes
            process_diversity_bonus = len(interaction_processes) - 1 if len(interaction_processes) > 1 else 0
            
            # Penalty for over-representing a biological process
            process_saturation_penalty = 0
            for process in interaction_processes:
                if process_counts[process] >= 1:  # More aggressive - only 1 per process
                    process_saturation_penalty += 0.5
            
            # Heavy penalty for hub proteins (CD31, CD34, CD44 appear too frequently)
            hub_proteins = {'CD31', 'CD34', 'CD44'}
            hub_penalty = len(interaction_proteins.intersection(hub_proteins)) * 0.3
            
            # Calculate final diversity score with more aggressive penalties
            diversity_score = (
                base_score * (1.0 - overlap_penalty * 0.8) +  # More aggressive overlap penalty
                process_diversity_bonus * 0.6 +               # Higher bonus for cross-process
                - process_saturation_penalty                   # Stricter process limits
                - hub_penalty                                  # Penalty for hub proteins
            )
            
            # Only select if it adds meaningful biological diversity
            min_new_proteins = 2 if len(selected_contacts) < 2 else 1  # Require 2 new proteins for first selections
            new_protein_count = len(interaction_proteins - used_proteins)
            
            if new_protein_count >= min_new_proteins:
                selected_contacts.append(contact)
                used_proteins.update(interaction_proteins)
                
                # Update process counts
                for process in interaction_processes:
                    process_counts[process] += 1
        
        return selected_contacts
    
    def _normalize_contact_pair(self, contact: str) -> str:
        """Normalize contact pairs to handle undirected interactions."""
        if ' ↔ ' in contact:
            domain1, domain2 = contact.split(' ↔ ')
            # Sort domains to ensure consistent ordering
            sorted_domains = sorted([domain1.strip(), domain2.strip()])
            return f"{sorted_domains[0]} ↔ {sorted_domains[1]}"
        return contact
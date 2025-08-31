"""Replicate variance visualization module."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from collections import defaultdict

from src.config import Config
from src.utils.helpers import top_n_items


class ReplicateVisualizer:
    """Creates replicate variance visualizations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_replicate_variance_figure(self, results: List[Dict]) -> plt.Figure:
        """Show replicate variance for top-N contacts as separate boxplots across timepoints."""
        # Get configured interaction count
        top_n = self.config.get('visualization.replicate_interactions_count', 12)
        
        # Calculate grid dimensions dynamically
        n_cols = 4
        n_rows = (top_n + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten()

        # Group by timepoint
        by_timepoint = {}
        for roi in results:
            day = roi['metadata'].get('injury_day') if isinstance(roi['metadata'], dict) else getattr(roi['metadata'], 'injury_day', None)
            if day is not None:
                key = f"D{day}"
                by_timepoint.setdefault(key, []).append(roi)
        
        if not by_timepoint:
            fig.suptitle('Replicate Variance - No temporal data available')
            return fig

        timepoints = sorted(by_timepoint.keys(), key=lambda x: int(x[1:]))
        
        # Build label map
        exp_cfg = self.config.experimental
        tp_vals = exp_cfg.get('timepoints', [])
        tp_labels = exp_cfg.get('timepoint_labels', [])
        label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}
        display_labels = [label_map.get(tp, tp) for tp in timepoints]

        # Find top contacts across all data with biological diversity scoring
        all_contact_scores = defaultdict(list)
        contact_data = {}
        
        for roi in results:
            for contact, score in roi.get('canonical_contacts', {}).items():
                # Normalize contact pairs to handle undirected interactions
                normalized_contact = self._normalize_contact_pair(contact)
                all_contact_scores[normalized_contact].append(score)
                
                if normalized_contact not in contact_data:
                    contact_data[normalized_contact] = {'strengths': [], 'timepoint_presence': {}}
                contact_data[normalized_contact]['strengths'].append(score)
                
                # Track timepoint presence
                day = roi['metadata'].get('injury_day') if isinstance(roi['metadata'], dict) else getattr(roi['metadata'], 'injury_day', None)
                if day is not None:
                    tp_key = f"D{day}"
                    if tp_key not in contact_data[normalized_contact]['timepoint_presence']:
                        contact_data[normalized_contact]['timepoint_presence'][tp_key] = []
                    contact_data[normalized_contact]['timepoint_presence'][tp_key].append(score)

        # Score contacts for temporal relevance and replicate variance
        contact_scores = {}
        for contact, data in contact_data.items():
            strengths = data['strengths']
            tp_presence = data['timepoint_presence']
            
            if len(strengths) >= 3:  # Must appear in at least 3 ROIs
                avg_strength = np.mean(strengths)
                frequency = len(strengths)
                temporal_spread = len(tp_presence)
                
                # Calculate temporal dynamics
                tp_means = [np.mean(vals) for vals in tp_presence.values()]
                temporal_variation = np.std(tp_means) / np.mean(tp_means) if len(tp_means) > 1 and np.mean(tp_means) > 0 else 0
                
                # Add replicate variance component (higher variance = more interesting for this plot)
                replicate_variance = np.var(strengths) / np.mean(strengths) if np.mean(strengths) > 0 else 0
                
                # Combined score
                score = (
                    avg_strength * 2.0 +
                    min(frequency / 10, 1.0) +
                    min(temporal_spread / 4, 1.0) +
                    temporal_variation * 1.5 +
                    replicate_variance * 0.5  # Bonus for variance (interesting for replicate analysis)
                )
                
                contact_scores[contact] = score

        # Select contacts using biological diversity scoring to reduce redundancy
        # top_n is already defined above from config
        key_contacts = self._select_diverse_interactions(contact_scores, top_n)
        
        # Convert back to (contact, stats) format for compatibility
        top_contacts = []
        for contact in key_contacts:
            strengths = contact_data[contact]['strengths']
            stats = {
                'mean': np.mean(strengths),
                'frequency': len(strengths),
                'variance': np.var(strengths)
            }
            top_contacts.append((contact, stats))

        # Create subplot for each top contact
        for i, (contact, stats) in enumerate(top_contacts):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Collect replicate data for this contact across timepoints
            replicate_data = []
            valid_timepoints = []
            
            for tp in timepoints:
                tp_scores = []
                for roi in by_timepoint[tp]:
                    if contact in roi.get('canonical_contacts', {}):
                        tp_scores.append(roi['canonical_contacts'][contact])
                
                if tp_scores:  # Only include timepoints where this contact appears
                    replicate_data.append(tp_scores)
                    valid_timepoints.append(label_map.get(tp, tp))
            
            if replicate_data:
                # Create boxplot showing replicate variance at each timepoint
                bp = ax.boxplot(replicate_data, labels=valid_timepoints, patch_artist=True)
                
                # Color boxes by timepoint
                colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add individual points to show actual replicate values
                for j, tp_scores in enumerate(replicate_data):
                    y = tp_scores
                    x = np.random.normal(j+1, 0.04, size=len(y))  # Add jitter
                    ax.scatter(x, y, alpha=0.6, s=20, color='black')
                
                ax.set_title(f'{contact}\n(mean={stats["mean"]:.3f}, n={stats["frequency"]})', 
                           fontsize=10)
                ax.set_ylabel('Contact Strength')
                ax.grid(True, alpha=0.3)
                
                # Add coefficient of variation annotation
                overall_scores = [score for tp_scores in replicate_data for score in tp_scores]
                cv = np.std(overall_scores) / np.mean(overall_scores) if np.mean(overall_scores) > 0 else 0
                ax.text(0.02, 0.98, f'CV={cv:.2f}', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            else:
                ax.text(0.5, 0.5, f'No data for\n{contact}', ha='center', va='center')
                ax.set_title(contact, fontsize=10)

        # Hide unused subplots
        for i in range(len(top_contacts), len(axes)):
            axes[i].axis('off')

        fig.suptitle('Biological Replicate Variance - Top Cellular Contacts', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_timepoint_region_contact_grid(self, results: List[Dict], config: Config) -> plt.Figure:
        """Create region × time contact grid analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Group by region and timepoint
        by_region_time = defaultdict(lambda: defaultdict(list))
        for roi in results:
            region = roi['metadata'].get('tissue_region') if isinstance(roi['metadata'], dict) else getattr(roi['metadata'], 'tissue_region', None) or 'Unknown'
            day = roi['metadata'].get('injury_day') if isinstance(roi['metadata'], dict) else getattr(roi['metadata'], 'injury_day', None)
            if day is not None:
                tp = f"D{day}"
                by_region_time[region][tp].append(roi)
        
        regions = list(by_region_time.keys())
        timepoints = set()
        for region_data in by_region_time.values():
            timepoints.update(region_data.keys())
        timepoints = sorted(list(timepoints), key=lambda x: int(x[1:]))
        
        # Build display labels
        exp_cfg = config.experimental
        tp_vals = exp_cfg.get('timepoints', [])
        tp_labels = exp_cfg.get('timepoint_labels', [])
        label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}
        display_labels = [label_map.get(tp, tp) for tp in timepoints]
        
        # 1. Domain count grid
        ax = axes[0, 0]
        self._plot_region_time_grid(ax, by_region_time, regions, timepoints, display_labels,
                                   'domains', 'Number of Domains', 
                                   lambda roi: len(roi['blob_signatures']))
        
        # 2. Contact diversity grid
        ax = axes[0, 1]
        self._plot_region_time_grid(ax, by_region_time, regions, timepoints, display_labels,
                                   'contacts', 'Contact Diversity',
                                   lambda roi: len(roi['canonical_contacts']))
        
        # 3. Functional group ratios grid
        ax = axes[1, 0]
        self._plot_functional_ratio_grid(ax, by_region_time, regions, timepoints, display_labels)
        
        # 4. Top contact strength grid
        ax = axes[1, 1]
        self._plot_top_contact_grid(ax, by_region_time, regions, timepoints, display_labels)
        
        fig.suptitle('Regional × Temporal Analysis Grid', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_region_time_grid(self, ax, by_region_time, regions, timepoints, display_labels,
                              metric_name, title, metric_func):
        """Plot a region × time grid for a given metric."""
        # Build matrix
        matrix = np.zeros((len(regions), len(timepoints)))
        for i, region in enumerate(regions):
            for j, tp in enumerate(timepoints):
                if tp in by_region_time[region]:
                    values = [metric_func(roi) for roi in by_region_time[region][tp]]
                    matrix[i, j] = np.mean(values) if values else 0
        
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(timepoints)))
        ax.set_xticklabels(display_labels)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions)
        ax.set_title(title)
        
        # Add text annotations
        for i in range(len(regions)):
            for j in range(len(timepoints)):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                           color='white' if val > matrix.max()/2 else 'black')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_functional_ratio_grid(self, ax, by_region_time, regions, timepoints, display_labels):
        """Plot functional group ratio grid."""
        functional_groups = self.config.functional_groups
        
        matrix = np.zeros((len(regions), len(timepoints)))
        for i, region in enumerate(regions):
            for j, tp in enumerate(timepoints):
                if tp in by_region_time[region]:
                    ratios = []
                    for roi in by_region_time[region][tp]:
                        inflammation_size = 0
                        repair_size = 0
                        
                        for sig in roi['blob_signatures'].values():
                            domain_proteins = sig['dominant_proteins'][:2]
                            
                            is_inflammation = any(p in functional_groups.get('kidney_inflammation', []) 
                                                for p in domain_proteins)
                            is_repair = any(p in functional_groups.get('kidney_repair', []) 
                                          for p in domain_proteins)
                            
                            if is_inflammation:
                                inflammation_size += sig['size']
                            elif is_repair:
                                repair_size += sig['size']
                        
                        if repair_size > 0:
                            ratios.append(inflammation_size / repair_size)
                    
                    matrix[i, j] = np.mean(ratios) if ratios else 0
        
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=0, vmax=2)
        ax.set_xticks(range(len(timepoints)))
        ax.set_xticklabels(display_labels)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions)
        ax.set_title('Inflammation/Repair Ratio')
        
        # Add text annotations
        for i in range(len(regions)):
            for j in range(len(timepoints)):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_top_contact_grid(self, ax, by_region_time, regions, timepoints, display_labels):
        """Plot top contact strength grid."""
        # Find the most common contact across all data
        all_contacts = defaultdict(list)
        for region_data in by_region_time.values():
            for tp_data in region_data.values():
                for roi in tp_data:
                    for contact, score in roi.get('canonical_contacts', {}).items():
                        all_contacts[contact].append(score)
        
        # Get the top contact by frequency and strength
        if not all_contacts:
            ax.text(0.5, 0.5, 'No contact data available', ha='center', va='center')
            ax.axis('off')
            return
            
        top_contact = max(all_contacts.keys(), 
                         key=lambda x: len(all_contacts[x]) * np.mean(all_contacts[x]))
        
        matrix = np.zeros((len(regions), len(timepoints)))
        for i, region in enumerate(regions):
            for j, tp in enumerate(timepoints):
                if tp in by_region_time[region]:
                    scores = []
                    for roi in by_region_time[region][tp]:
                        if top_contact in roi.get('canonical_contacts', {}):
                            scores.append(roi['canonical_contacts'][top_contact])
                    matrix[i, j] = np.mean(scores) if scores else 0
        
        im = ax.imshow(matrix, cmap='plasma', aspect='auto')
        ax.set_xticks(range(len(timepoints)))
        ax.set_xticklabels(display_labels)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions)
        ax.set_title(f'Top Contact: {top_contact}')
        
        # Add text annotations
        for i in range(len(regions)):
            for j in range(len(timepoints)):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                           color='white' if val > matrix.max()/2 else 'black')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
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
            
            # Penalty for over-representing a biological process (less aggressive for 12 contacts)
            process_saturation_penalty = 0
            for process in interaction_processes:
                if process_counts[process] >= 3:  # Allow 3 per process for 12 total
                    process_saturation_penalty += 0.3
            
            # Hub protein penalty (less aggressive for replicate analysis)
            hub_proteins = {'CD31', 'CD34', 'CD44'}
            hub_penalty = len(interaction_proteins.intersection(hub_proteins)) * 0.2
            
            # Calculate final diversity score
            diversity_score = (
                base_score * (1.0 - overlap_penalty * 0.6) +  # Moderate overlap penalty
                process_diversity_bonus * 0.6 +               # Bonus for cross-process
                - process_saturation_penalty                   # Process limits
                - hub_penalty                                  # Moderate hub penalty
            )
            
            # Only select if it adds meaningful biological diversity (relaxed for 12 contacts)
            min_new_proteins = 1 if len(selected_contacts) < 4 else 0  # Relaxed requirement
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
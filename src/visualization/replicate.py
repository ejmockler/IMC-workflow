"""Replicate variance visualization module."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from collections import defaultdict

from src.config import Config
from src.utils.helpers import top_n_items
from scipy.stats import ttest_ind
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


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
    
    def create_region_temporal_trajectories(self, results: List[Dict], config: Config) -> plt.Figure:
        """Create region-specific temporal trajectory analysis."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        
        # Group by region and timepoint
        by_region_time = defaultdict(lambda: defaultdict(list))
        for roi in results:
            metadata = roi['metadata']
            if isinstance(metadata, dict):
                day = metadata.get('injury_day')
                region = metadata.get('tissue_region', 'Unknown')
            else:
                day = getattr(metadata, 'injury_day', None)
                region = getattr(metadata, 'tissue_region', 'Unknown')
            
            if day is not None and region in ['Cortex', 'Medulla']:
                tp_key = f"D{day}"
                by_region_time[region][tp_key].append(roi)
        
        # Get timepoints and labels
        exp_cfg = config.raw.get('experimental', {})
        tp_vals = exp_cfg.get('timepoints', [])
        tp_labels = exp_cfg.get('timepoint_labels', [])
        timepoints = sorted([f"D{tp}" for tp in tp_vals], key=lambda x: int(x[1:]))
        label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}
        
        regions = ['Cortex', 'Medulla']
        functional_groups = config.raw.get('proteins', {}).get('functional_groups', {})
        
        # Plot functional group trajectories for each region
        metrics = ['domains', 'contacts', 'inflammation', 'repair']
        colors = {'Cortex': '#e74c3c', 'Medulla': '#3498db'}
        
        for col, metric in enumerate(metrics):
            ax = axes[0, col]
            
            for region in regions:
                values = []
                errors = []
                
                for tp in timepoints:
                    if tp in by_region_time[region]:
                        rois = by_region_time[region][tp]
                        
                        if metric == 'domains':
                            vals = [len(r['blob_signatures']) for r in rois]
                        elif metric == 'contacts':
                            vals = [len(r.get('canonical_contacts', {})) for r in rois]
                        elif metric == 'inflammation':
                            vals = [self._calculate_functional_percentage(r, functional_groups.get('kidney_inflammation', []), functional_groups) for r in rois]
                        elif metric == 'repair':
                            vals = [self._calculate_functional_percentage(r, functional_groups.get('kidney_repair', []), functional_groups) for r in rois]
                        
                        values.append(np.mean(vals) if vals else 0)
                        errors.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                    else:
                        values.append(0)
                        errors.append(0)
                
                x_pos = range(len(timepoints))
                ax.errorbar(x_pos, values, yerr=errors, marker='o', linewidth=2, 
                           label=region, color=colors[region], markersize=6)
            
            ax.set_xticks(range(len(timepoints)))
            ax.set_xticklabels([label_map.get(tp, tp) for tp in timepoints])
            ax.set_title(f'{metric.capitalize()} Progression by Region')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Bottom row: Statistical comparisons and differential analysis
        self._plot_regional_differences(axes[1, :], by_region_time, timepoints, label_map, functional_groups, config)
        
        fig.suptitle('Region-Specific Kidney Healing Trajectories', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _calculate_functional_percentage(self, roi, target_proteins, all_functional_groups):
        """Calculate percentage of tissue covered by functional group."""
        total_pixels = roi.get('total_pixels', 0) or sum(sig['size'] for sig in roi['blob_signatures'].values())
        if total_pixels <= 0:
            return 0.0
        
        group_size = 0
        for sig in roi['blob_signatures'].values():
            domain_proteins = sig['dominant_proteins'][:2]
            if any(p in target_proteins for p in domain_proteins):
                group_size += sig['size']
        
        return 100.0 * group_size / total_pixels
    
    def _plot_regional_differences(self, axes, by_region_time, timepoints, label_map, functional_groups, config):
        """Plot statistical comparisons between regions."""
        regions = ['Cortex', 'Medulla']
        
        # Panel 1: Differential inflammation/repair ratio
        ax = axes[0]
        inflammation_markers = functional_groups.get('kidney_inflammation', [])
        repair_markers = functional_groups.get('kidney_repair', [])
        
        cortex_ratios = []
        medulla_ratios = []
        timepoint_labels = []
        
        for tp in timepoints:
            if tp in by_region_time['Cortex'] and tp in by_region_time['Medulla']:
                cortex_inflam = np.mean([self._calculate_functional_percentage(r, inflammation_markers, functional_groups) for r in by_region_time['Cortex'][tp]])
                cortex_repair = np.mean([self._calculate_functional_percentage(r, repair_markers, functional_groups) for r in by_region_time['Cortex'][tp]])
                medulla_inflam = np.mean([self._calculate_functional_percentage(r, inflammation_markers, functional_groups) for r in by_region_time['Medulla'][tp]])
                medulla_repair = np.mean([self._calculate_functional_percentage(r, repair_markers, functional_groups) for r in by_region_time['Medulla'][tp]])
                
                cortex_ratio = cortex_inflam / max(cortex_repair, 1.0)
                medulla_ratio = medulla_inflam / max(medulla_repair, 1.0)
                
                cortex_ratios.append(cortex_ratio)
                medulla_ratios.append(medulla_ratio)
                timepoint_labels.append(label_map.get(tp, tp))
        
        x_pos = range(len(timepoint_labels))
        ax.plot(x_pos, cortex_ratios, 'o-', label='Cortex', color='#e74c3c', linewidth=2)
        ax.plot(x_pos, medulla_ratios, 's-', label='Medulla', color='#3498db', linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(timepoint_labels)
        ax.set_ylabel('Inflammation/Repair Ratio')
        ax.set_title('Regional Inflammation Balance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Vascular integrity comparison
        ax = axes[1]
        vascular_markers = functional_groups.get('kidney_vasculature', [])
        
        cortex_vasc = []
        medulla_vasc = []
        
        for tp in timepoints:
            if tp in by_region_time['Cortex'] and tp in by_region_time['Medulla']:
                cortex_v = np.mean([self._calculate_functional_percentage(r, vascular_markers, functional_groups) for r in by_region_time['Cortex'][tp]])
                medulla_v = np.mean([self._calculate_functional_percentage(r, vascular_markers, functional_groups) for r in by_region_time['Medulla'][tp]])
                
                cortex_vasc.append(cortex_v)
                medulla_vasc.append(medulla_v)
        
        ax.plot(x_pos, cortex_vasc, 'o-', label='Cortex', color='#e74c3c', linewidth=2)
        ax.plot(x_pos, medulla_vasc, 's-', label='Medulla', color='#3498db', linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(timepoint_labels)
        ax.set_ylabel('Vascular Coverage (%)')
        ax.set_title('Regional Vascular Integrity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Domain complexity comparison
        ax = axes[2]
        
        cortex_domains = []
        medulla_domains = []
        
        for tp in timepoints:
            if tp in by_region_time['Cortex'] and tp in by_region_time['Medulla']:
                cortex_d = np.mean([len(r['blob_signatures']) for r in by_region_time['Cortex'][tp]])
                medulla_d = np.mean([len(r['blob_signatures']) for r in by_region_time['Medulla'][tp]])
                
                cortex_domains.append(cortex_d)
                medulla_domains.append(medulla_d)
        
        ax.plot(x_pos, cortex_domains, 'o-', label='Cortex', color='#e74c3c', linewidth=2)
        ax.plot(x_pos, medulla_domains, 's-', label='Medulla', color='#3498db', linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(timepoint_labels)
        ax.set_ylabel('Number of Domains')
        ax.set_title('Regional Complexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Statistical significance
        ax = axes[3]
        self._plot_significance_matrix(ax, by_region_time, timepoints, label_map, functional_groups)
    
    def _plot_significance_matrix(self, ax, by_region_time, timepoints, label_map, functional_groups):
        """Plot statistical significance matrix for regional differences."""
        metrics = ['Inflammation', 'Repair', 'Vascular']
        p_values = np.ones((len(metrics), len(timepoints)))
        
        inflammation_markers = functional_groups.get('kidney_inflammation', [])
        repair_markers = functional_groups.get('kidney_repair', [])
        vascular_markers = functional_groups.get('kidney_vasculature', [])
        
        for j, tp in enumerate(timepoints):
            if tp in by_region_time['Cortex'] and tp in by_region_time['Medulla']:
                cortex_rois = by_region_time['Cortex'][tp]
                medulla_rois = by_region_time['Medulla'][tp]
                
                if len(cortex_rois) >= 2 and len(medulla_rois) >= 2:
                    # Inflammation
                    cortex_inflam = [self._calculate_functional_percentage(r, inflammation_markers, functional_groups) for r in cortex_rois]
                    medulla_inflam = [self._calculate_functional_percentage(r, inflammation_markers, functional_groups) for r in medulla_rois]
                    _, p_values[0, j] = ttest_ind(cortex_inflam, medulla_inflam)
                    
                    # Repair
                    cortex_repair = [self._calculate_functional_percentage(r, repair_markers, functional_groups) for r in cortex_rois]
                    medulla_repair = [self._calculate_functional_percentage(r, repair_markers, functional_groups) for r in medulla_rois]
                    _, p_values[1, j] = ttest_ind(cortex_repair, medulla_repair)
                    
                    # Vascular
                    cortex_vasc = [self._calculate_functional_percentage(r, vascular_markers, functional_groups) for r in cortex_rois]
                    medulla_vasc = [self._calculate_functional_percentage(r, vascular_markers, functional_groups) for r in medulla_rois]
                    _, p_values[2, j] = ttest_ind(cortex_vasc, medulla_vasc)
        
        # Create significance heatmap
        sig_matrix = -np.log10(p_values + 1e-10)  # -log10(p-value)
        im = ax.imshow(sig_matrix, cmap='Reds', aspect='auto', vmax=2)  # p=0.01 → 2
        
        ax.set_xticks(range(len(timepoints)))
        ax.set_xticklabels([label_map.get(tp, tp) for tp in timepoints])
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics)
        ax.set_title('Regional Significance\n(-log10 p-value)')
        
        # Add significance annotations
        for i in range(len(metrics)):
            for j in range(len(timepoints)):
                p_val = p_values[i, j]
                if p_val < 0.001:
                    marker = '***'
                elif p_val < 0.01:
                    marker = '**'
                elif p_val < 0.05:
                    marker = '*'
                else:
                    marker = ''
                
                if marker:
                    ax.text(j, i, marker, ha='center', va='center', 
                           color='white' if sig_matrix[i, j] > 1 else 'black', fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def create_region_time_interaction_heatmap(self, results: List[Dict], config: Config) -> plt.Figure:
        """Create comprehensive region × time interaction heatmap for top protein domains."""
        # Group by region and timepoint
        by_region_time = defaultdict(lambda: defaultdict(list))
        for roi in results:
            metadata = roi['metadata']
            if isinstance(metadata, dict):
                day = metadata.get('injury_day')
                region = metadata.get('tissue_region', 'Unknown')
            else:
                day = getattr(metadata, 'injury_day', None)
                region = getattr(metadata, 'tissue_region', 'Unknown')
            
            if day is not None and region in ['Cortex', 'Medulla']:
                tp_key = f"D{day}"
                by_region_time[region][tp_key].append(roi)
        
        # Get timepoints and create region-time combinations
        exp_cfg = config.raw.get('experimental', {})
        tp_vals = exp_cfg.get('timepoints', [])
        tp_labels = exp_cfg.get('timepoint_labels', [])
        timepoints = sorted([f"D{tp}" for tp in tp_vals], key=lambda x: int(x[1:]))
        label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}
        
        # Create region-time combinations
        regions = ['Cortex', 'Medulla']
        region_time_combos = []
        combo_labels = []
        
        for tp in timepoints:
            for region in regions:
                region_time_combos.append((region, tp))
                combo_labels.append(f"{region}\n{label_map.get(tp, tp)}")
        
        # Collect all protein domains and their frequencies
        domain_frequencies = defaultdict(lambda: defaultdict(float))
        
        for region in regions:
            for tp in timepoints:
                if tp in by_region_time[region]:
                    for roi in by_region_time[region][tp]:
                        total_pixels = sum(sig['size'] for sig in roi['blob_signatures'].values())
                        for domain_name, sig in roi['blob_signatures'].items():
                            # Create canonical domain name from dominant proteins
                            proteins = sorted(sig['dominant_proteins'][:2])
                            canonical_domain = '+'.join(proteins)
                            
                            percentage = 100.0 * sig['size'] / total_pixels if total_pixels > 0 else 0
                            domain_frequencies[(region, tp)][canonical_domain] += percentage
                    
                    # Average across ROIs in this region-time combination
                    n_rois = len(by_region_time[region][tp])
                    for domain in domain_frequencies[(region, tp)]:
                        domain_frequencies[(region, tp)][domain] /= n_rois
        
        # Get top domains by overall frequency
        all_domain_scores = defaultdict(float)
        for combo_freqs in domain_frequencies.values():
            for domain, freq in combo_freqs.items():
                all_domain_scores[domain] += freq
        
        # Select top 20 domains for the heatmap
        top_domains = sorted(all_domain_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        domain_names = [domain for domain, _ in top_domains]
        
        # Create the data matrix
        n_domains = len(domain_names)
        n_combos = len(region_time_combos)
        matrix = np.zeros((n_domains, n_combos))
        
        for j, (region, tp) in enumerate(region_time_combos):
            combo_data = domain_frequencies.get((region, tp), {})
            for i, domain in enumerate(domain_names):
                matrix[i, j] = combo_data.get(domain, 0)
        
        # Create the figure with adjusted layout
        fig = plt.figure(figsize=(18, 12))
        
        # Create GridSpec for layout with dendrograms - increased wspace for colorbar clearance
        gs = fig.add_gridspec(2, 3, width_ratios=[0.15, 1, 1], hspace=0.3, wspace=0.35, 
                             left=0.08, right=0.92, bottom=0.1, top=0.93)
        
        # Prepare data for clustering - TOP ROW (main heatmap and regional differences)
        # We'll cluster based on the full matrix across all region-time combinations
        if n_domains > 1:
            # Calculate distance matrix and perform hierarchical clustering for top row
            distances_top = pdist(matrix, metric='euclidean')
            linkage_matrix_top = linkage(distances_top, method='ward')
            
            # Get dendrogram ordering for top row
            dendro_top = dendrogram(linkage_matrix_top, no_plot=True)
            row_order_top = dendro_top['leaves']
            
            # Reorder matrix and domain names for top row
            matrix_clustered = matrix[row_order_top, :]
            domain_names_clustered_top = [domain_names[i] for i in row_order_top]
            
            # Prepare data for BOTTOM ROW clustering (Cortex and Medulla temporal profiles)
            # Extract combined temporal data for clustering
            temporal_matrix = np.zeros((n_domains, len(timepoints) * 2))  # Cortex + Medulla
            for j, tp in enumerate(timepoints):
                cortex_idx = region_time_combos.index(('Cortex', tp)) if ('Cortex', tp) in region_time_combos else None
                medulla_idx = region_time_combos.index(('Medulla', tp)) if ('Medulla', tp) in region_time_combos else None
                if cortex_idx is not None:
                    temporal_matrix[:, j] = matrix[:, cortex_idx]
                if medulla_idx is not None:
                    temporal_matrix[:, j + len(timepoints)] = matrix[:, medulla_idx]
            
            # Calculate distance matrix and perform hierarchical clustering for bottom row
            distances_bottom = pdist(temporal_matrix, metric='euclidean')
            linkage_matrix_bottom = linkage(distances_bottom, method='ward')
            
            # Get dendrogram ordering for bottom row
            dendro_bottom = dendrogram(linkage_matrix_bottom, no_plot=True)
            row_order_bottom = dendro_bottom['leaves']
            
            # Domain names for bottom row
            domain_names_clustered_bottom = [domain_names[i] for i in row_order_bottom]
        else:
            matrix_clustered = matrix
            domain_names_clustered_top = domain_names
            domain_names_clustered_bottom = domain_names
            linkage_matrix_top = None
            linkage_matrix_bottom = None
            row_order_top = [0]
            row_order_bottom = [0]
        
        # Dendrogram for top row
        if linkage_matrix_top is not None:
            ax_dendro_top = fig.add_subplot(gs[0, 0])
            dendro_plot = dendrogram(linkage_matrix_top, orientation='left', ax=ax_dendro_top, 
                                    color_threshold=0, above_threshold_color='black')
            ax_dendro_top.set_xticks([])
            ax_dendro_top.set_yticks([])
            ax_dendro_top.spines['top'].set_visible(False)
            ax_dendro_top.spines['right'].set_visible(False)
            ax_dendro_top.spines['bottom'].set_visible(False)
            ax_dendro_top.spines['left'].set_visible(False)
        
        # Main clustered heatmap (top-middle)
        ax_main = fig.add_subplot(gs[0, 1])
        im = ax_main.imshow(matrix_clustered, cmap='YlOrRd', aspect='auto', vmin=0, vmax=np.percentile(matrix, 95))
        ax_main.set_xticks(range(n_combos))
        ax_main.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=8)
        ax_main.set_yticks(range(len(domain_names_clustered_top)))
        ax_main.set_yticklabels(domain_names_clustered_top, fontsize=7)
        ax_main.set_title('Clustered Protein Domains Across Regions and Time', fontsize=12, fontweight='bold')
        
        # Add colorbar for main heatmap
        cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
        cbar.set_label('Tissue Coverage (%)', fontsize=9)
        
        # Regional comparison (difference between Cortex and Medulla) - also clustered
        ax_diff = fig.add_subplot(gs[0, 2])
        
        # Create difference matrix (Cortex - Medulla)
        diff_matrix = np.zeros((n_domains, len(timepoints)))
        for j, tp in enumerate(timepoints):
            cortex_idx = region_time_combos.index(('Cortex', tp)) if ('Cortex', tp) in region_time_combos else None
            medulla_idx = region_time_combos.index(('Medulla', tp)) if ('Medulla', tp) in region_time_combos else None
            
            if cortex_idx is not None and medulla_idx is not None:
                diff_matrix[:, j] = matrix[:, cortex_idx] - matrix[:, medulla_idx]
        
        # Apply same clustering order (top row) to difference matrix
        if linkage_matrix_top is not None:
            diff_matrix_clustered = diff_matrix[row_order_top, :]
        else:
            diff_matrix_clustered = diff_matrix
        
        im_diff = ax_diff.imshow(diff_matrix_clustered, cmap='RdBu_r', aspect='auto', 
                                vmin=-np.max(np.abs(diff_matrix)), vmax=np.max(np.abs(diff_matrix)))
        ax_diff.set_xticks(range(len(timepoints)))
        ax_diff.set_xticklabels([label_map.get(tp, tp) for tp in timepoints], fontsize=9)
        ax_diff.set_yticks(range(len(domain_names_clustered_top)))
        ax_diff.set_yticklabels(domain_names_clustered_top, fontsize=7)  # Keep y-axis labels on rightmost plot
        ax_diff.set_title('Regional Differences (Cortex - Medulla)', fontsize=12, fontweight='bold')
        
        # Add colorbar for difference heatmap with more padding
        cbar_diff = plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.08)
        cbar_diff.set_label('Difference (%)', fontsize=9)
        
        # Dendrogram for bottom row (different clustering based on temporal profiles)
        if linkage_matrix_bottom is not None:
            ax_dendro_bottom = fig.add_subplot(gs[1, 0])
            dendro_plot2 = dendrogram(linkage_matrix_bottom, orientation='left', ax=ax_dendro_bottom, 
                                     color_threshold=0, above_threshold_color='black')
            ax_dendro_bottom.set_xticks([])
            ax_dendro_bottom.set_yticks([])
            ax_dendro_bottom.spines['top'].set_visible(False)
            ax_dendro_bottom.spines['right'].set_visible(False)
            ax_dendro_bottom.spines['bottom'].set_visible(False)
            ax_dendro_bottom.spines['left'].set_visible(False)
        
        # Temporal profile heatmap for Cortex
        ax_cortex = fig.add_subplot(gs[1, 1])
        
        # Extract Cortex data only - use bottom row clustering order
        cortex_matrix = np.zeros((len(domain_names_clustered_bottom), len(timepoints)))
        for j, tp in enumerate(timepoints):
            cortex_idx = region_time_combos.index(('Cortex', tp)) if ('Cortex', tp) in region_time_combos else None
            if cortex_idx is not None:
                cortex_matrix[:, j] = matrix[row_order_bottom, cortex_idx]
        
        im_cortex = ax_cortex.imshow(cortex_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=np.percentile(matrix, 95))
        ax_cortex.set_xticks(range(len(timepoints)))
        ax_cortex.set_xticklabels([label_map.get(tp, tp) for tp in timepoints], fontsize=9)
        ax_cortex.set_yticks(range(len(domain_names_clustered_bottom)))
        ax_cortex.set_yticklabels(domain_names_clustered_bottom, fontsize=7)
        ax_cortex.set_title('Cortex Temporal Progression', fontsize=12, fontweight='bold')
        ax_cortex.set_xlabel('Timepoint', fontsize=10)
        
        # Add colorbar
        cbar_cortex = plt.colorbar(im_cortex, ax=ax_cortex, fraction=0.046, pad=0.04)
        cbar_cortex.set_label('Coverage (%)', fontsize=9)
        
        # Temporal profile heatmap for Medulla
        ax_medulla = fig.add_subplot(gs[1, 2])
        
        # Extract Medulla data only - use bottom row clustering order
        medulla_matrix = np.zeros((len(domain_names_clustered_bottom), len(timepoints)))
        for j, tp in enumerate(timepoints):
            medulla_idx = region_time_combos.index(('Medulla', tp)) if ('Medulla', tp) in region_time_combos else None
            if medulla_idx is not None:
                medulla_matrix[:, j] = matrix[row_order_bottom, medulla_idx]
        
        im_medulla = ax_medulla.imshow(medulla_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=np.percentile(matrix, 95))
        ax_medulla.set_xticks(range(len(timepoints)))
        ax_medulla.set_xticklabels([label_map.get(tp, tp) for tp in timepoints], fontsize=9)
        ax_medulla.set_yticks(range(len(domain_names_clustered_bottom)))
        ax_medulla.set_yticklabels(domain_names_clustered_bottom, fontsize=7)  # Keep y-axis labels on rightmost plot
        ax_medulla.set_title('Medulla Temporal Progression', fontsize=12, fontweight='bold')
        ax_medulla.set_xlabel('Timepoint', fontsize=10)
        
        # Add colorbar with more padding to avoid overlap
        cbar_medulla = plt.colorbar(im_medulla, ax=ax_medulla, fraction=0.046, pad=0.08)
        cbar_medulla.set_label('Coverage (%)', fontsize=9)
        
        fig.suptitle('Hierarchical Clustering of Protein Domain Landscapes Across Kidney Healing', 
                    fontsize=14, fontweight='bold')
        
        return fig
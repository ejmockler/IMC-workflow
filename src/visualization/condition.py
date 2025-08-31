"""Condition comparison visualization module."""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from src.config import Config


class ConditionVisualizer:
    """Creates condition comparison visualizations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_condition_figure(self, results: List[Dict]) -> plt.Figure:
        """Create condition comparison analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Group by condition  
        by_condition = {}
        for roi in results:
            condition = roi['metadata'].condition
            if condition not in by_condition:
                by_condition[condition] = []
            by_condition[condition].append(roi)
        
        conditions = list(by_condition.keys())
        
        # 1. Functional group composition by condition (config-driven)
        ax = axes[0, 0]
        functional_groups = self.config.functional_groups
        group_names = [name for name in functional_groups.keys() if name != 'structural_controls']
        group_data = {group: [] for group in group_names}
        
        for condition in conditions:
            for group_name in group_names:
                group_proteins = functional_groups[group_name]
                percentages = []
                
                for roi in by_condition[condition]:
                    total_size = sum(sig['size'] for sig in roi['blob_signatures'].values())
                    group_size = 0
                    
                    for sig in roi['blob_signatures'].values():
                        domain_proteins = sig['dominant_proteins'][:2]
                        if any(p in group_proteins for p in domain_proteins):
                            group_size += sig['size']
                    
                    if total_size > 0:
                        percentages.append(100 * group_size / total_size)
                
                group_data[group_name].append(np.mean(percentages) if percentages else 0)
        
        x = np.arange(len(conditions))
        width = 0.25  # Reduced bar width
        colors = ['#ff9999', '#66b3ff', '#99ff99']  # Red for inflammation, Blue for repair, Green for vasculature
        
        # Calculate positions - bars touching within group
        for i, group_name in enumerate(group_names):
            label = group_name.replace('_', ' ').title()
            # Position bars side by side within each condition
            positions = x + (i - len(group_names)/2 + 0.5) * width
            ax.bar(positions, group_data[group_name], width, 
                  label=label, color=colors[i % len(colors)])
        
        ax.set_xlabel('Experimental Condition')
        ax.set_ylabel('Functional Group Representation (%)')
        ax.set_title('Functional Group Distribution by Condition')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Contact patterns by condition
        ax = axes[0, 1]
        contact_counts = []
        for condition in conditions:
            counts = [len(roi['canonical_contacts']) for roi in by_condition[condition]]
            contact_counts.append(counts)
        
        bp = ax.boxplot(contact_counts, labels=conditions, patch_artist=True)
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        ax.set_ylabel('Number of Contact Pairs')
        ax.set_title('Contact Diversity by Condition')
        ax.grid(True, alpha=0.3)
        
        # 3. Tissue complexity
        ax = axes[1, 0]
        complexity_data = []
        for condition in conditions:
            complexities = []
            for roi in by_condition[condition]:
                # Complexity = number of domains * contact diversity
                n_domains = len(roi['blob_signatures'])
                n_contacts = len(roi['canonical_contacts'])
                complexity = n_domains * (n_contacts / 100)  # Normalized
                complexities.append(complexity)
            complexity_data.append(complexities)
        
        bp = ax.boxplot(complexity_data, labels=conditions, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        ax.set_ylabel('Tissue Complexity Index')
        ax.set_title('Tissue Organization Complexity')
        ax.grid(True, alpha=0.3)
        
        # 4. Functional group ratios by condition
        ax = axes[1, 1]
        self._plot_functional_group_ratios(ax, results, by_condition)
        
        fig.suptitle('Sham vs Injury Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_functional_group_ratios(self, ax, results, by_condition):
        """Plot functional group ratios by experimental condition."""
        functional_groups = self.config.functional_groups
        conditions = list(by_condition.keys())
        
        # Calculate functional group ratios by condition
        ratios = []
        for condition in conditions:
            condition_ratios = []
            for roi in by_condition[condition]:
                inflammation_size = 0
                repair_size = 0
                
                for sig in roi['blob_signatures'].values():
                    domain_proteins = sig['dominant_proteins'][:2]
                    
                    # Check functional group membership
                    is_inflammation = any(p in functional_groups.get('kidney_inflammation', []) for p in domain_proteins)
                    is_repair = any(p in functional_groups.get('kidney_repair', []) for p in domain_proteins)
                    
                    if is_inflammation:
                        inflammation_size += sig['size']
                    elif is_repair:
                        repair_size += sig['size']
                
                # Calculate ratio (inflammation/repair)
                if repair_size > 0:
                    ratio = inflammation_size / repair_size
                    condition_ratios.append(ratio)
            
            ratios.append(condition_ratios)
        
        if any(ratios):
            bp = ax.boxplot(ratios, labels=conditions, patch_artist=True)
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Inflammation/Repair Ratio')
            ax.set_title('Inflammation vs Repair Balance by Condition')
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line at ratio = 1
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Balance')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient data for ratio analysis', ha='center', va='center')
            ax.axis('off')
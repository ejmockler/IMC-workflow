#!/usr/bin/env python3
"""
IMC Analysis Final - The clean, unified pipeline
All analysis and visualization in one coherent system
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from imc_utils import (
    Metadata, PlotGrid, load_config, find_roi_files,
    canonicalize_pair, add_percentage_labels, top_n_items
)
from imc_pipeline import parse_roi_metadata

from spatial_blob_analysis import (
    load_roi_data, identify_expression_blobs,
    analyze_blob_spatial_relationships
)

from blob_visualization_components import (
    plot_spatial_domains, plot_domain_signatures_heatmap,
    plot_spatial_contact_matrix, plot_domain_size_distribution,
    plot_top_domain_contacts, plot_aggregated_contact_matrix,
    plot_aggregated_domain_signatures
)

from network_analysis import NetworkDiscovery
from network_visualizer import NetworkVisualizer

# === CONFIGURATION ===

class Config:
    """Single source of truth for all configuration"""
    def __init__(self, config_path='config.json'):
        self.raw = load_config(config_path)
        self.data_dir = Path(self.raw.get('data_dir', 'data/241218_IMC_Alun'))
        self.output_dir = Path(self.raw.get('output_dir', 'results'))
        self.proteins = self.raw.get('proteins', [])
        self.metadata_lookup = self.raw.get('metadata_lookup', {})
        self.contact_radius = self.raw.get('contact_radius', 15.0)
        self.min_blob_size = self.raw.get('min_blob_size', 50)
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)

# === CORE ANALYSIS ===

class ROIAnalyzer:
    """Analyzes single ROI"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def _normalize_region(self, region: Optional[str]) -> str:
        """Normalize region names to configured canonical set; fallback to Unknown."""
        exp = self.config.raw.get('experimental', {})
        configured = {str(r).strip().lower(): str(r) for r in exp.get('regions', [])}
        if region is None:
            return 'Unknown'
        key = str(region).strip().lower()
        # Map common variants
        aliases = {
            'kidney': 'Unknown',
            'renal cortex': 'Cortex',
            'renal medulla': 'Medulla'
        }
        if key in configured:
            return configured[key]
        if key in aliases and aliases[key] in configured.values():
            return aliases[key]
        # Title-case unknown but do not force "Kidney"
        return region.title()
    
    def analyze(self, roi_file: Path) -> Dict:
        """Complete analysis of single ROI"""
        # Load data
        coords, values, protein_names = load_roi_data(roi_file, 'config.json')
        
        # Identify blobs
        blob_labels, blob_signatures, blob_type_mapping = identify_expression_blobs(
            coords, values, protein_names
        )
        
        # Get contacts
        blob_contacts = analyze_blob_spatial_relationships(
            coords, blob_labels, blob_signatures, blob_type_mapping
        )
        
        # Get metadata using pipeline parser
        metadata_dict = parse_roi_metadata(roi_file.name, 'config.json')
        metadata = Metadata(
            condition=metadata_dict.get('condition', 'Unknown'),
            injury_day=metadata_dict.get('timepoint'),
            tissue_region=self._normalize_region(
                metadata_dict.get('region') or metadata_dict.get('tissue_region') or metadata_dict.get('Region')
            ),
            mouse_replicate=metadata_dict.get('mouse_id', 'Unknown')
        )
        
        # Canonicalize contacts
        canonical_contacts = self._canonicalize_contacts(blob_contacts, blob_signatures)
        
        return {
            'filename': roi_file.name,
            'metadata': metadata,
            'coords': coords,
            'values': values,
            'protein_names': protein_names,
            'blob_labels': blob_labels,
            'blob_signatures': blob_signatures,
            'blob_type_mapping': blob_type_mapping,
            'blob_contacts': blob_contacts,
            'canonical_contacts': canonical_contacts,
            'total_pixels': len(coords)
        }
    
    def _canonicalize_contacts(self, blob_contacts, blob_signatures):
        """Canonicalize all contact pairs"""
        canonical = {}
        for blob_id, contacts in blob_contacts.items():
            blob1_sig = '+'.join(blob_signatures[blob_id]['dominant_proteins'][:2])
            for neighbor_id, freq in contacts.items():
                if freq > 0:
                    blob2_sig = '+'.join(blob_signatures[neighbor_id]['dominant_proteins'][:2])
                    pair = canonicalize_pair(blob1_sig, blob2_sig)
                    canonical[pair] = max(canonical.get(pair, 0), freq)
        return canonical

class BatchAnalyzer:
    """Analyzes multiple ROIs"""
    
    def __init__(self, config: Config):
        self.config = config
        self.roi_analyzer = ROIAnalyzer(config)
    
    def analyze_all(self) -> List[Dict]:
        """Analyze all ROIs in data directory"""
        roi_files = find_roi_files(self.config.data_dir)
        results = []
        
        print(f"Analyzing {len(roi_files)} ROIs...")
        for i, roi_file in enumerate(roi_files, 1):
            print(f"  [{i}/{len(roi_files)}] {roi_file.name}")
            try:
                result = self.roi_analyzer.analyze(roi_file)
                results.append(result)
            except Exception as e:
                print(f"    Error: {e}")
        
        return results

# === VISUALIZATION ===

class ROIVisualizer:
    """Creates visualizations for single ROI"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_figure(self, roi_data: Dict) -> plt.Figure:
        """Create comprehensive ROI visualization"""
        grid = PlotGrid(3, 3, figsize=(24, 18))  # Expand to 3x3 grid
        
        # Row 0: All heatmaps
        ax = grid.get(0, 0)
        # Row-only dendrogram clustering for domain signatures (rows=proteins, cols=domains)
        from imc_utils import add_dendrogram_to_heatmap
        # Build matrix domains x proteins from blob_signatures
        blob_compositions = []
        blob_names = []
        valid_blobs = sorted(roi_data['blob_signatures'].keys(),
                             key=lambda x: roi_data['blob_signatures'][x]['size'], reverse=True)[:15]
        for key in valid_blobs:
            sig = roi_data['blob_signatures'][key]
            blob_names.append('+'.join(sig['dominant_proteins'][:2]))
            blob_compositions.append(sig['mean_expression'])
        if blob_compositions:
            import numpy as np
            M = np.array(blob_compositions)
            if M.max() > 0:
                M = M / M.max()
            # Add dendrogram on rows (proteins)
            heat_ax, Z, order = add_dendrogram_to_heatmap(ax, M.T)
            proteins = roi_data['protein_names'][:M.shape[1]]
            proteins_ordered = [proteins[i] for i in order]
            M_ordered = (M.T)[order, :]
            im = heat_ax.imshow(M_ordered, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            heat_ax.set_yticks(range(len(proteins_ordered)))
            heat_ax.set_yticklabels(proteins_ordered, fontsize=8)
            heat_ax.set_xticks(range(len(blob_names)))
            heat_ax.set_xticklabels(blob_names, rotation=45, ha='right', fontsize=8)
            heat_ax.set_title('Domain Protein Signatures')
            plt.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04)
        else:
            plot_domain_signatures_heatmap(ax, roi_data['blob_signatures'], roi_data['protein_names'])
        ax = grid.get(0, 1)
        # Row-only dendrogram for protein-domain contact matrix (rows=proteins)
        from imc_utils import add_dendrogram_to_heatmap
        M, row_labels, col_labels = plot_spatial_contact_matrix(
            ax,
            roi_data['blob_labels'],
            roi_data['blob_signatures'],
            roi_data['coords'],
            radius=self.config.contact_radius,
            sample_step=1,
            max_neighbors=100,
            blob_type_mapping=roi_data['blob_type_mapping'],
            protein_names=roi_data['protein_names'],
            return_data=True
        ) or (None, None, None)
        if M is not None:
            heat_ax, Z, order = add_dendrogram_to_heatmap(ax, M)
            M_ordered = M[order, :]
            rows_ordered = [row_labels[i] for i in order]
            im = heat_ax.imshow(M_ordered, cmap='Blues', aspect='auto', vmin=0)
            heat_ax.set_yticks(range(len(rows_ordered)))
            heat_ax.set_yticklabels(rows_ordered, fontsize=8)
            heat_ax.set_xticks(range(len(col_labels)))
            heat_ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
            heat_ax.set_title('Protein-Domain Spatial Contacts (Adjacency frequency)')
            plt.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04)
        ax = grid.get(0, 2)
        self._plot_colocalization_matrix(ax, roi_data)
        
        # Row 1: All bar plots
        ax = grid.get(1, 0)
        plot_domain_size_distribution(ax, roi_data['blob_signatures'],
                                     roi_data['protein_names'])
        ax = grid.get(1, 1)
        plot_top_domain_contacts(ax, roi_data['blob_contacts'],
                                roi_data['blob_signatures'])
        ax = grid.get(1, 2)
        self._plot_organization_scales(ax, roi_data)
        
        # Row 2: Spatial map + temporal/group dynamics
        ax = grid.get(2, 0)
        plot_spatial_domains(ax, roi_data['coords'], 
                           roi_data['blob_labels'],
                           roi_data['blob_signatures'])
        ax = grid.get(2, 1)
        self._plot_spatial_autocorrelation(ax, roi_data)
        ax = grid.get(2, 2)
        self._plot_functional_groups(ax, roi_data)
        
        # Title
        meta = roi_data['metadata']
        title = f"Spatial Tissue Organization: {meta.display_name} — {roi_data['filename']}"
        grid.fig.suptitle(title, fontsize=14, fontweight='bold')
        
        return grid.fig
    
    def _plot_functional_groups(self, ax, roi_data):
        """Plot functional group composition analysis"""
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        functional_groups = config['proteins']['functional_groups']
        
        # Calculate functional group percentages
        group_sizes = {}
        total_pixels = roi_data['total_pixels']
        
        for group_name, proteins in functional_groups.items():
            if group_name == 'structural_controls':
                continue
                
            group_size = 0
            for sig in roi_data['blob_signatures'].values():
                domain_proteins = sig['dominant_proteins'][:2]
                # Check if domain belongs to this functional group
                if any(p in proteins for p in domain_proteins):
                    group_size += sig['size']
            
            group_sizes[group_name] = 100 * group_size / total_pixels
        
        # Create pie chart
        if group_sizes:
            labels = [name.replace('_', ' ').title() for name in group_sizes.keys()]
            sizes = list(group_sizes.values())
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(sizes)]
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
            ax.set_title('Functional Group Composition')
            
            # Enhance text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax.text(0.5, 0.5, 'No functional group data', ha='center', va='center')
            ax.axis('off')
    
    def _plot_spatial_autocorrelation(self, ax, roi_data):
        """Plot spatial autocorrelation by distance (top proteins by organization)"""
        pipeline_file = roi_data['filename'].replace('.txt', '_pipeline_analysis.json')
        pipeline_path = self.config.data_dir / pipeline_file
        
        if pipeline_path.exists():
            with open(pipeline_path, 'r') as f:
                pipeline_data = json.load(f)
            
            autocorr_data = pipeline_data.get('spatial_autocorrelation', {})
            if autocorr_data:
                # Determine available distances from data
                sample = next(iter(autocorr_data.values())) if autocorr_data else {}
                try:
                    distances = sorted([int(d) for d in sample.keys()])
                except Exception:
                    distances = [5, 10, 25, 50]
                
                # Rank proteins by mean autocorrelation
                means = []
                for protein, values in autocorr_data.items():
                    corrs = [values.get(str(d), 0) for d in distances]
                    mean_corr = float(np.mean(corrs)) if corrs else 0.0
                    means.append((protein.split('(')[0], mean_corr))
                means.sort(key=lambda x: x[1], reverse=True)
                
                top = means[:8]
                for protein, _ in top:
                    values = autocorr_data.get(protein, {}) or next((v for k, v in autocorr_data.items() if k.startswith(protein)), {})
                    corrs = [values.get(str(d), 0) for d in distances]
                    ax.plot(distances, corrs, marker='o', label=protein, linewidth=2)
                
                ax.set_xlabel('Distance (μm)')
                ax.set_ylabel('Spatial Autocorrelation')
                ax.set_title('Top Protein Spatial Organization')
                if top:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No autocorrelation data', ha='center', va='center')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Pipeline data not found', ha='center', va='center')
            ax.axis('off')
    
    def _plot_colocalization_matrix(self, ax, roi_data):
        """Plot proteins (rows) vs top colocalized pairs (columns), with row clustering."""
        pipeline_file = roi_data['filename'].replace('.txt', '_pipeline_analysis.json')
        pipeline_path = self.config.data_dir / pipeline_file
        
        if pipeline_path.exists():
            with open(pipeline_path, 'r') as f:
                pipeline_data = json.load(f)
            
            coloc_data = pipeline_data.get('colocalization', {})
            if coloc_data:
                # Build list of pairs and scores
                pairs = []  # (label, p1, p2, score)
                for _, data in coloc_data.items():
                    score = data.get('colocalization_score', 0.0)
                    p1 = data.get('protein_1', '').split('(')[0]
                    p2 = data.get('protein_2', '').split('(')[0]
                    if p1 and p2:
                        label = f"{p1}↔{p2}"
                        pairs.append((label, p1, p2, score))
                # Sort and filter
                pairs.sort(key=lambda x: x[3], reverse=True)
                pairs = [p for p in pairs if p[3] > 0.05][:20]
                
                if pairs:
                    # Columns: pair labels
                    pair_labels = [label for label, _, _, _ in pairs]
                    # Rows: unique proteins involved
                    from collections import Counter
                    prot_counts = Counter([p1 for _, p1, _, _ in pairs] + [p2 for _, _, p2, _ in pairs])
                    proteins = [p for p, _ in prot_counts.most_common()]
                    n_rows = len(proteins)
                    n_cols = len(pair_labels)
                    matrix = np.zeros((n_rows, n_cols))
                    row_index = {p: i for i, p in enumerate(proteins)}
                    for j, (_, p1, p2, s) in enumerate(pairs):
                        if p1 in row_index:
                            matrix[row_index[p1], j] = s
                        if p2 in row_index:
                            matrix[row_index[p2], j] = s
                    # Cluster rows (proteins)
                    if n_rows > 1:
                        from scipy.spatial.distance import pdist
                        from scipy.cluster.hierarchy import linkage, leaves_list
                        dist = pdist(matrix, metric='euclidean')
                        Z = linkage(dist, method='average')
                        order = leaves_list(Z)
                        matrix = matrix[order, :]
                        proteins = [proteins[i] for i in order]
                    
                    from imc_utils import add_dendrogram_to_heatmap
                    heat_ax, Z, order = add_dendrogram_to_heatmap(ax, matrix)
                    matrix = matrix[order, :]
                    proteins = [proteins[i] for i in order]
                    im = heat_ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
                    heat_ax.set_yticks(range(len(proteins)))
                    heat_ax.set_yticklabels(proteins, fontsize=8)
                    heat_ax.set_xticks(range(len(pair_labels)))
                    heat_ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
                    heat_ax.set_xlabel('Top colocalized pairs')
                    heat_ax.set_title('Protein vs top colocalized pairs (Pearson r)')
                    plt.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04, label='Pearson r')
                else:
                    ax.text(0.5, 0.5, 'No protein pairs found', ha='center', va='center')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No colocalization data', ha='center', va='center')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Pipeline data not found', ha='center', va='center')
            ax.axis('off')
    
    def _plot_individual_proteins(self, ax, roi_data):
        """Plot individual protein expression levels"""
        pipeline_file = roi_data['filename'].replace('.txt', '_pipeline_analysis.json')
        pipeline_path = Path('data/241218_IMC_Alun') / pipeline_file
        
        if pipeline_path.exists():
            with open(pipeline_path, 'r') as f:
                pipeline_data = json.load(f)
            
            autocorr_data = pipeline_data.get('spatial_autocorrelation', {})
            if autocorr_data:
                # Get protein names and their max autocorrelation (organization strength)
                proteins = []
                org_strengths = []
                
                for protein, distances in autocorr_data.items():
                    proteins.append(protein)
                    # Use max autocorrelation as organization strength
                    max_autocorr = max(distances.values()) if distances.values() else 0
                    org_strengths.append(max_autocorr)
                
                # Sort by organization strength
                sorted_data = sorted(zip(proteins, org_strengths), key=lambda x: x[1], reverse=True)
                proteins, org_strengths = zip(*sorted_data) if sorted_data else ([], [])
                
                if proteins:
                    bars = ax.bar(range(len(proteins)), org_strengths, 
                                 color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'][:len(proteins)])
                    ax.set_xticks(range(len(proteins)))
                    ax.set_xticklabels(proteins, rotation=45, ha='right', fontsize=8)
                    ax.set_ylabel('Organization Strength')
                    ax.set_title('Individual Protein Organization')
                    
                    # Add value labels
                    for bar, strength in zip(bars, org_strengths):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{strength:.2f}', ha='center', va='bottom', fontsize=7)
                else:
                    ax.text(0.5, 0.5, 'No protein data', ha='center', va='center')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No autocorrelation data', ha='center', va='center')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Pipeline data not found', ha='center', va='center')
            ax.axis('off')
    
    def _plot_organization_scales(self, ax, roi_data):
        """Plot organization scales by protein (top-N)"""
        pipeline_file = roi_data['filename'].replace('.txt', '_pipeline_analysis.json')
        pipeline_path = self.config.data_dir / pipeline_file
        
        if pipeline_path.exists():
            with open(pipeline_path, 'r') as f:
                pipeline_data = json.load(f)
            
            org_data = pipeline_data.get('spatial_organization', {})
            if org_data:
                items = []
                for protein, data in org_data.items():
                    protein_clean = protein.split('(')[0]
                    items.append((protein_clean, data.get('organization_scale', 0)))
                items.sort(key=lambda x: x[1], reverse=True)
                
                proteins = [p for p, _ in items[:10]]
                scales = [s for _, s in items[:10]]
                
                if proteins:
                    bars = ax.bar(proteins, scales, color='lightcoral')
                    ax.set_ylabel('Organization Scale (μm)')
                    ax.set_title('Top Organization Scales')
                    ax.tick_params(axis='x', rotation=45)
                    
                    for bar, scale in zip(bars, scales):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{scale}μm', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No organization data', ha='center', va='center')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No organization data', ha='center', va='center')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Pipeline data not found', ha='center', va='center')
            ax.axis('off')

class SummaryVisualizer:
    """Creates summary visualizations across all ROIs"""
    
    def create_trends_figure(self, results: List[Dict]) -> plt.Figure:
        """Create trends summary figure"""
        grid = PlotGrid(2, 3, figsize=(20, 14))
        
        # Group results by metadata
        by_condition = self._group_by(results, 'condition')
        by_region = self._group_by(results, 'tissue_region')
        by_day = self._group_by_day(results)
        
        # 1. Domains by condition
        ax = grid.get(0, 0)
        self._plot_domains_by_group(ax, by_condition, 'Condition')
        
        # 2. Domains by region
        ax = grid.get(0, 1)
        self._plot_domains_by_group(ax, by_region, 'Region')
        
        # 3. Temporal trends
        ax = grid.get(0, 2)
        self._plot_temporal_trends(ax, by_day)
        
        # 4. Contact frequency distribution
        ax = grid.get(1, 0)
        self._plot_contact_distribution(ax, results)
        
        # 5. Top universal contacts
        ax = grid.get(1, 1)
        self._plot_universal_contacts(ax, results)
        
        # 6. Experiment coverage: count of ROIs per condition x day
        ax = grid.get(1, 2)
        self._plot_experiment_coverage(ax, results)
        
        grid.fig.suptitle('Acute Kidney Injury Recovery Analysis', fontsize=16, fontweight='bold')
        return grid.fig
    
    def _group_by(self, results, field):
        """Group results by metadata field"""
        groups = {}
        for roi in results:
            value = getattr(roi['metadata'], field, 'Unknown')
            if value not in groups:
                groups[value] = []
            groups[value].append(roi)
        return groups
    
    def _group_by_day(self, results):
        """Group results by injury day"""
        groups = {}
        for roi in results:
            day = roi['metadata'].injury_day
            if day is not None:
                key = f"D{day}"
                if key not in groups:
                    groups[key] = []
                groups[key].append(roi)
        return groups
    
    def _plot_domains_by_group(self, ax, groups, title):
        """Plot domain counts by group"""
        group_names = list(groups.keys())
        domain_counts = []
        
        for group in group_names:
            counts = [len(roi['blob_signatures']) for roi in groups[group]]
            domain_counts.append(counts)
        
        if domain_counts:
            bp = ax.boxplot(domain_counts, tick_labels=group_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax.set_ylabel('Number of Domains')
            kidney_titles = {
                'Condition': 'Sham vs Injury Domain Formation',
                'Region': 'Cortex vs Medulla Domain Patterns'
            }
            ax.set_title(kidney_titles.get(title, f'Domains by {title}'))
            ax.grid(True, alpha=0.3)
    
    def _plot_temporal_trends(self, ax, by_day):
        """Plot temporal trends"""
        if not by_day:
            ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center')
            ax.axis('off')
            return
        
        days = sorted(by_day.keys(), key=lambda x: int(x[1:]))
        mean_domains = []
        std_domains = []
        
        for day in days:
            counts = [len(roi['blob_signatures']) for roi in by_day[day]]
            mean_domains.append(np.mean(counts))
            std_domains.append(np.std(counts))
        
        # Map D-codes to configured timepoint labels if available
        try:
            with open('config.json', 'r') as f:
                cfg = json.load(f)
            exp_cfg = cfg.get('experimental', {})
            tp_vals = exp_cfg.get('timepoints', [])
            tp_labels = exp_cfg.get('timepoint_labels', [])
            label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}
            display_labels = [label_map.get(d, d) for d in days]
        except Exception:
            display_labels = days
        x = range(len(days))
        ax.errorbar(x, mean_domains, yerr=std_domains, marker='o', capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels)
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Mean Domain Count')
        ax.set_title('Temporal Domain Dynamics')
        ax.grid(True, alpha=0.3)
    
    def _plot_contact_distribution(self, ax, results):
        """Plot distribution of contact frequencies"""
        all_freqs = []
        for roi in results:
            all_freqs.extend(roi['canonical_contacts'].values())
        
        if all_freqs:
            ax.hist(all_freqs, bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Contact Frequency')
            ax.set_ylabel('Count')
            ax.set_title('Contact Frequency Distribution')
            ax.axvline(np.mean(all_freqs), color='red', linestyle='--',
                      label=f'Mean: {np.mean(all_freqs):.2f}')
            ax.legend()
    
    def _plot_universal_contacts(self, ax, results):
        """Plot most common contact pairs across all ROIs"""
        all_contacts = {}
        for roi in results:
            for pair, freq in roi['canonical_contacts'].items():
                if pair not in all_contacts:
                    all_contacts[pair] = []
                all_contacts[pair].append(freq)
        
        # Calculate mean frequency for each pair
        mean_contacts = {pair: np.mean(freqs) 
                        for pair, freqs in all_contacts.items()
                        if len(freqs) >= 3}  # Present in at least 3 ROIs
        
        if mean_contacts:
            top_pairs = top_n_items(mean_contacts, 15)
            pairs, freqs = zip(*top_pairs)
            
            bars = ax.barh(range(len(pairs)), freqs)
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels(pairs, fontsize=8)
            ax.set_xlabel('Mean Contact Frequency')
            ax.set_title('Universal Domain Contacts')
            
            for bar, freq in zip(bars, freqs):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{freq:.2f}', va='center', fontsize=8)
    
    def _add_summary_stats(self, ax, results):
        """Add summary statistics"""
        n_rois = len(results)
        total_domains = sum(len(roi['blob_signatures']) for roi in results)
        all_contacts = set()
        for roi in results:
            all_contacts.update(roi['canonical_contacts'].keys())
        
        conditions = set(roi['metadata'].condition for roi in results)
        regions = set(roi['metadata'].tissue_region for roi in results)
        
        stats = f"""
Analysis Summary
================
ROIs Analyzed: {n_rois}
Total Domains: {total_domains}
Avg Domains/ROI: {total_domains/n_rois:.1f}
Unique Contact Pairs: {len(all_contacts)}

Conditions: {', '.join(conditions)}
Regions: {', '.join(regions)}

Data Quality:
  Min domains/ROI: {min(len(r['blob_signatures']) for r in results)}
  Max domains/ROI: {max(len(r['blob_signatures']) for r in results)}
  Avg pixels/ROI: {np.mean([r['total_pixels'] for r in results]):.0f}
"""
        ax.text(0.1, 0.9, stats, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               family='monospace')
        ax.axis('off')
    
    def _plot_functional_progression(self, ax, results):
        """Plot functional group changes over time"""
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        functional_groups = config['proteins']['functional_groups']
        
        # Group by timepoint
        by_timepoint = {}
        for roi in results:
            day = roi['metadata'].injury_day
            if day is not None:
                key = f"D{day}"
                if key not in by_timepoint:
                    by_timepoint[key] = []
                by_timepoint[key].append(roi)
        
        if not by_timepoint:
            ax.text(0.5, 0.5, 'No temporal data available', ha='center', va='center')
            ax.axis('off')
            return
        
        timepoints = sorted(by_timepoint.keys(), key=lambda x: int(x[1:]))
        # Build display labels and x positions using config timepoint_labels when available
        try:
            with open('config.json', 'r') as f:
                _cfg = json.load(f)
            _exp = _cfg.get('experimental', {})
            _tp_vals = _exp.get('timepoints', [])
            _tp_labels = _exp.get('timepoint_labels', [])
            _map = {f"D{tp}": lbl for tp, lbl in zip(_tp_vals, _tp_labels)}
        except Exception:
            _map = {}
        display_labels = [_map.get(tp, tp) for tp in timepoints]
        x_pos = np.arange(len(timepoints))
        # Build label map for timepoints and x positions
        try:
            with open('config.json', 'r') as f:
                cfg = json.load(f)
            exp_cfg = cfg.get('experimental', {})
            tp_vals = exp_cfg.get('timepoints', [])
            tp_labels = exp_cfg.get('timepoint_labels', [])
            _label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}
        except Exception:
            _label_map = {}
        display_labels = [_label_map.get(tp, tp) for tp in timepoints]
        x_pos = np.arange(len(timepoints))
        # Build label map
        try:
            with open('config.json', 'r') as f:
                cfg = json.load(f)
            exp_cfg = cfg.get('experimental', {})
            tp_vals = exp_cfg.get('timepoints', [])
            tp_labels = exp_cfg.get('timepoint_labels', [])
            label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}
        except Exception:
            label_map = {}
        display_labels = [label_map.get(tp, tp) for tp in timepoints]
        x_pos = np.arange(len(timepoints))
        
        # Calculate functional group percentages over time
        for group_name, proteins in functional_groups.items():
            if group_name == 'structural_controls':
                continue
                
            percentages = []
            for tp in timepoints:
                tp_percentages = []
                for roi in by_timepoint[tp]:
                    total_pixels = roi['total_pixels']
                    group_size = 0
                    
                    for sig in roi['blob_signatures'].values():
                        domain_proteins = sig['dominant_proteins'][:2]
                        if any(p in proteins for p in domain_proteins):
                            group_size += sig['size']
                    
                    tp_percentages.append(100 * group_size / total_pixels)
                
                percentages.append(np.mean(tp_percentages) if tp_percentages else 0)
            
            label = group_name.replace('_', ' ').title()
            ax.plot(timepoints, percentages, marker='o', label=label, linewidth=2)
        
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Functional Group Composition (%)')
        ax.set_title('Functional Group Dynamics')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_experiment_coverage(self, ax, results):
        """Heatmap of ROI counts by condition x timepoint (communicates dataset leverage)"""
        # Collect dimensions
        conditions = sorted(list({r['metadata'].condition for r in results}))
        days = sorted(list({f"D{r['metadata'].injury_day}" for r in results if r['metadata'].injury_day is not None}), key=lambda x: int(x[1:]))
        if not conditions or not days:
            ax.text(0.5, 0.5, 'Insufficient metadata for coverage map', ha='center', va='center')
            ax.axis('off')
            return
        
        # Build count matrix
        mat = np.zeros((len(conditions), len(days)), dtype=int)
        cond_idx = {c: i for i, c in enumerate(conditions)}
        day_idx = {d: i for i, d in enumerate(days)}
        for r in results:
            c = r['metadata'].condition
            d = r['metadata'].injury_day
            if d is None or c not in cond_idx:
                continue
            key = f"D{d}"
            if key in day_idx:
                mat[cond_idx[c], day_idx[key]] += 1
        
        im = ax.imshow(mat, cmap='Greens', aspect='auto')
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions)
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels(days)
        ax.set_title('Experiment Coverage (ROIs per Condition x Day)')
        
        # Annotate counts
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if val > 0:
                    ax.text(j, i, str(val), ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

class ExperimentalVisualizer:
    """Creates experimental analysis visualizations"""
    
    def create_temporal_figure(self, results: List[Dict]) -> plt.Figure:
        """Create temporal progression analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Group by timepoint
        by_timepoint = {}
        for roi in results:
            day = roi['metadata'].injury_day
            if day is not None:
                key = f"D{day}"
                if key not in by_timepoint:
                    by_timepoint[key] = []
                by_timepoint[key].append(roi)
        
        if not by_timepoint:
            fig.suptitle('Temporal Analysis - No temporal data available')
            return fig
        
        timepoints = sorted(by_timepoint.keys(), key=lambda x: int(x[1:]))
        
        # Build display labels and x positions
        try:
            with open('config.json', 'r') as f:
                cfg = json.load(f)
            exp_cfg = cfg.get('experimental', {})
            tp_vals = exp_cfg.get('timepoints', [])
            tp_labels = exp_cfg.get('timepoint_labels', [])
            label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}
        except Exception:
            label_map = {}
        display_labels = [label_map.get(tp, tp) for tp in timepoints]
        x_pos = np.arange(len(timepoints))

        # 1. Domain count progression
        ax = axes[0, 0]
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
        
        # 2. Contact diversity progression
        ax = axes[0, 1]
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
        
        # 3. Biologically-relevant domain evolution using config annotations
        ax = axes[1, 0]
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Use config to identify key kidney repair processes
        key_domains = {
            'CD31+CD34': 'Vascular Regeneration',    # Endothelial repair in kidney
            'CD44+CD45': 'Inflammation Resolution',  # ECM remodeling + immune clearance
            'CD11b+CD44': 'Myeloid-Matrix Response', # Inflammatory cells + matrix repair
            'CD206+CD44': 'Kidney Tissue Repair'    # M2 macrophages + ECM healing
        }
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (domain, process_name) in enumerate(key_domains.items()):
            means = []
            stds = []
            for tp in timepoints:
                tp_percentages = []
                for roi in by_timepoint[tp]:
                    total_size = sum(sig['size'] for sig in roi['blob_signatures'].values())
                    domain_size = sum(sig['size'] for name, sig in roi['blob_signatures'].items() 
                                    if '+'.join(sorted(sig['dominant_proteins'][:2])) == domain)
                    if total_size > 0:
                        tp_percentages.append(100 * domain_size / total_size)
                means.append(np.mean(tp_percentages) if tp_percentages else 0)
                stds.append(np.std(tp_percentages) if tp_percentages else 0)
            means_arr = np.array(means)
            ax.plot(x_pos, means_arr, marker='o', label=process_name, 
                   linewidth=2, color=colors[i % len(colors)])
        
        ax.set_ylabel('Biological Process Representation (%)')
        ax.set_title('Kidney Repair Process Dynamics')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_labels)
        
        # 4. Contact strength evolution - data-driven selection
        ax = axes[1, 1]
        
        # Collect all contact data
        contact_data = {}
        
        for roi in results:
            if 'canonical_contacts' in roi:
                for contact, strength in roi['canonical_contacts'].items():
                    if contact not in contact_data:
                        contact_data[contact] = {'strengths': [], 'timepoint_presence': {}}
                    contact_data[contact]['strengths'].append(strength)
                    
                    # Track which timepoints this contact appears in
                    day = roi['metadata'].injury_day
                    if day is not None:
                        tp_key = f"D{day}"
                        if tp_key not in contact_data[contact]['timepoint_presence']:
                            contact_data[contact]['timepoint_presence'][tp_key] = []
                        contact_data[contact]['timepoint_presence'][tp_key].append(strength)
        
        # Score contacts based on multiple criteria
        contact_scores = {}
        for contact, data in contact_data.items():
            strengths = data['strengths']
            tp_presence = data['timepoint_presence']
            
            # Scoring criteria:
            avg_strength = np.mean(strengths)           # Higher = better
            frequency = len(strengths)                  # Higher = better  
            consistency = 1 - np.std(strengths)        # Lower variance = better
            temporal_spread = len(tp_presence)         # More timepoints = better
            
            # Calculate temporal dynamics (coefficient of variation across timepoints)
            tp_means = [np.mean(vals) for vals in tp_presence.values()]
            temporal_variation = np.std(tp_means) / np.mean(tp_means) if len(tp_means) > 1 and np.mean(tp_means) > 0 else 0
            
            # Combined score - weight factors based on what makes a contact "notable"
            score = (
                avg_strength * 2.0 +           # Strength is important
                min(frequency / 10, 1.0) +     # Frequency matters but cap influence
                consistency * 0.5 +            # Consistency is good
                min(temporal_spread / 4, 1.0) + # Temporal spread matters
                temporal_variation * 1.5        # Temporal dynamics are interesting
            )
            
            # Only consider contacts with minimum frequency and strength
            if frequency >= 3 and avg_strength >= 0.1:
                contact_scores[contact] = score
        
        # Select top contacts by score (limit for readability)
        top_n = 8
        top_contacts = sorted(contact_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        key_contacts = [contact for contact, score in top_contacts]

        colors = plt.cm.tab10(np.linspace(0, 1, len(key_contacts)))

        # Plot clean mean traces with end-of-line labels (replicate variance shown in separate figure)
        for i, contact in enumerate(key_contacts):
            strengths = []
            for tp in timepoints:
                tp_strengths = []
                for roi in by_timepoint[tp]:
                    if contact in roi['canonical_contacts']:
                        tp_strengths.append(roi['canonical_contacts'][contact])
                strengths.append(np.mean(tp_strengths) if tp_strengths else 0)
            strengths_arr = np.array(strengths)
            ax.plot(x_pos, strengths_arr, marker='s', linewidth=2, color=colors[i])
            # End label at last point
            ax.text(x_pos[-1] + 0.05, strengths_arr[-1], contact, fontsize=7,
                    va='center', color=colors[i])
        
        ax.set_ylabel('Mean Contact Strength')
        ax.set_title(f'Top {top_n} Cellular Interactions Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_labels)
        
        fig.suptitle('Kidney Injury Recovery Timeline', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_condition_figure(self, results: List[Dict]) -> plt.Figure:
        """Create condition comparison analysis"""
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
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        functional_groups = config['proteins']['functional_groups']
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
        
        x = np.arange(len(conditions)) * 1.5  # Add spacing between condition groups
        width = 0.3  # Bar width
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
        
        bp = ax.boxplot(contact_counts, tick_labels=conditions, patch_artist=True)
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
        
        bp = ax.boxplot(complexity_data, tick_labels=conditions, patch_artist=True)
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

    def create_replicate_variance_figure(self, results: List[Dict]) -> plt.Figure:
        """Show replicate variance for top-N contacts as separate boxplots across timepoints."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Group by timepoint
        by_timepoint = {}
        for roi in results:
            day = roi['metadata'].injury_day
            if day is not None:
                key = f"D{day}"
                by_timepoint.setdefault(key, []).append(roi)
        if not by_timepoint:
            fig.suptitle('Replicate Variance - No temporal data available')
            return fig

        timepoints = sorted(by_timepoint.keys(), key=lambda x: int(x[1:]))
        # Build label map
        try:
            with open('config.json', 'r') as f:
                cfg = json.load(f)
            exp_cfg = cfg.get('experimental', {})
            tp_vals = exp_cfg.get('timepoints', [])
            tp_labels = exp_cfg.get('timepoint_labels', [])
            label_map = {f"D{tp}": lbl for tp, lbl in zip(tp_vals, tp_labels)}
        except Exception:
            label_map = {}
        display_labels = [label_map.get(tp, tp) for tp in timepoints]

        # Score contacts similar to temporal figure
        contact_data = {}
        for roi in results:
            if 'canonical_contacts' in roi:
                for contact, strength in roi['canonical_contacts'].items():
                    if contact not in contact_data:
                        contact_data[contact] = {'strengths': [], 'timepoint_presence': {}}
                    contact_data[contact]['strengths'].append(strength)
                    day = roi['metadata'].injury_day
                    if day is not None:
                        tp_key = f"D{day}"
                        contact_data[contact]['timepoint_presence'].setdefault(tp_key, []).append(strength)

        contact_scores = {}
        for contact, data in contact_data.items():
            strengths = data['strengths']
            tp_presence = data['timepoint_presence']
            avg_strength = np.mean(strengths)
            frequency = len(strengths)
            consistency = 1 - np.std(strengths)
            temporal_spread = len(tp_presence)
            tp_means = [np.mean(vals) for vals in tp_presence.values()]
            temporal_variation = np.std(tp_means) / np.mean(tp_means) if len(tp_means) > 1 and np.mean(tp_means) > 0 else 0
            score = avg_strength * 2.0 + min(frequency / 10, 1.0) + consistency * 0.5 + min(temporal_spread / 4, 1.0) + temporal_variation * 1.5
            if frequency >= 3 and avg_strength >= 0.1:
                contact_scores[contact] = score

        top_n = 6
        top_contacts = sorted(contact_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        key_contacts = [contact for contact, _ in top_contacts]

        # For each contact, build per-timepoint distributions across replicates
        for idx, contact in enumerate(key_contacts):
            ax = axes[idx]
            series = []
            for tp in timepoints:
                vals = []
                for roi in by_timepoint[tp]:
                    if contact in roi.get('canonical_contacts', {}):
                        vals.append(roi['canonical_contacts'][contact])
                series.append(vals)
            if any(series):
                # Violin plots per timepoint
                parts = ax.violinplot(series, showmeans=True, showmedians=False, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor('#cfe2f3')
                    pc.set_edgecolor('#3d85c6')
                    pc.set_alpha(0.6)
                if 'cmeans' in parts:
                    parts['cmeans'].set_color('#3d85c6')
                ax.set_xticks(range(1, len(timepoints) + 1))
                ax.set_xticklabels(display_labels, rotation=0)
                ax.set_title(contact, fontsize=9)
                ax.set_ylabel('Contact Strength')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.axis('off')

        # Hide any unused axes
        for j in range(len(key_contacts), len(axes)):
            axes[j].axis('off')

        fig.suptitle('Replicate Variance for Top Colocalization Pairs', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_spatial_fields_figure(self, results: List[Dict]) -> plt.Figure:
        """Create spatial field analysis from pipeline data"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Collect all spatial autocorrelation data
        all_autocorr = {}
        all_coloc = {}
        
        for roi in results:
            pipeline_file = roi['filename'].replace('.txt', '_pipeline_analysis.json')
            pipeline_path = Path('data/241218_IMC_Alun') / pipeline_file
            
            if pipeline_path.exists():
                with open(pipeline_path, 'r') as f:
                    pipeline_data = json.load(f)
                
                # Autocorrelation data
                autocorr = pipeline_data.get('spatial_autocorrelation', {})
                for protein, values in autocorr.items():
                    if protein not in all_autocorr:
                        all_autocorr[protein] = []
                    all_autocorr[protein].append(values)
                
                # Colocalization data
                coloc = pipeline_data.get('colocalization', {})
                for pair, data in coloc.items():
                    if pair not in all_coloc:
                        all_coloc[pair] = []
                    all_coloc[pair].append(data['colocalization_score'])
        
        # 1. Enhanced spatial autocorrelation heatmap
        ax = axes[0, 0]
        distances = [5, 10, 25, 50]
        
        # Get ALL proteins present in the data, not just a subset
        available_proteins = list(all_autocorr.keys())
        
        autocorr_matrix = []
        protein_labels = []
        for protein in available_proteins:
            mean_autocorr = []
            for d in distances:
                values = [data.get(str(d), 0) for data in all_autocorr[protein]]
                mean_autocorr.append(np.mean(values))
            autocorr_matrix.append(mean_autocorr)
            protein_labels.append(protein)
        
        if autocorr_matrix:
            autocorr_matrix = np.array(autocorr_matrix)
            im = ax.imshow(autocorr_matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(distances)))
            ax.set_xticklabels([f'{d}μm' for d in distances])
            ax.set_yticks(range(len(protein_labels)))
            ax.set_yticklabels(protein_labels, fontsize=8)
            ax.set_title('Spatial Organization Across All Proteins')
            
            # Add text annotations for high values
            for i in range(len(protein_labels)):
                for j in range(len(distances)):
                    if autocorr_matrix[i, j] > 0.1:  # Show significant values
                        ax.text(j, i, f'{autocorr_matrix[i, j]:.2f}', 
                               ha='center', va='center', fontsize=6,
                               color='white' if autocorr_matrix[i, j] > 0.2 else 'black')
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 2. Organization scales distribution
        ax = axes[0, 1]
        scales_by_protein = {}
        for roi in results:
            pipeline_file = roi['filename'].replace('.txt', '_pipeline_analysis.json')
            pipeline_path = Path('data/241218_IMC_Alun') / pipeline_file
            
            if pipeline_path.exists():
                with open(pipeline_path, 'r') as f:
                    pipeline_data = json.load(f)
                
                org_data = pipeline_data.get('spatial_organization', {})
                for protein, data in org_data.items():
                    protein_clean = protein.split('(')[0]
                    if protein_clean in available_proteins:  # Use available_proteins from above
                        if protein_clean not in scales_by_protein:
                            scales_by_protein[protein_clean] = []
                        scales_by_protein[protein_clean].append(data['organization_scale'])
        
        if scales_by_protein:
            proteins = list(scales_by_protein.keys())
            scale_data = [scales_by_protein[p] for p in proteins]
            bp = ax.boxplot(scale_data, tick_labels=proteins, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightgreen')
            ax.set_ylabel('Organization Scale (μm)')
            ax.set_title('Protein Organization Scales')
            ax.tick_params(axis='x', rotation=45)
        
        # 3. Colocalization network
        ax = axes[0, 2]
        if all_coloc:
            # Top colocalization pairs
            top_pairs = sorted(all_coloc.items(), key=lambda x: np.mean(x[1]), reverse=True)[:15]
            pair_names, scores = zip(*[(pair, np.mean(scores)) for pair, scores in top_pairs])
            
            bars = ax.barh(range(len(pair_names)), scores)
            ax.set_yticks(range(len(pair_names)))
            ax.set_yticklabels([p.replace('(', '\n(') for p in pair_names], fontsize=7)
            ax.set_xlabel('Mean Colocalization Score')
            ax.set_title('Top Protein Colocalization')
        
        # 4. Spatial heterogeneity
        ax = axes[1, 0]
        heterogeneity_scores = []
        roi_names = []
        for roi in results:
            # Heterogeneity = coefficient of variation in domain sizes
            domain_sizes = [sig['size'] for sig in roi['blob_signatures'].values()]
            if len(domain_sizes) > 1:
                cv = np.std(domain_sizes) / np.mean(domain_sizes)
                heterogeneity_scores.append(cv)
                roi_names.append(roi['filename'][:15])  # Short name
        
        if heterogeneity_scores:
            ax.hist(heterogeneity_scores, bins=15, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Spatial Heterogeneity (CV)')
            ax.set_ylabel('Count')
            ax.set_title('Tissue Spatial Heterogeneity')
            ax.axvline(np.mean(heterogeneity_scores), color='red', linestyle='--',
                      label=f'Mean: {np.mean(heterogeneity_scores):.2f}')
            ax.legend()
        
        # 5. Field correlation analysis
        ax = axes[1, 1]
        # Correlation between spatial scales and domain diversity
        scales = []
        diversities = []
        for roi in results:
            pipeline_file = roi['filename'].replace('.txt', '_pipeline_analysis.json')
            pipeline_path = Path('data/241218_IMC_Alun') / pipeline_file
            
            if pipeline_path.exists():
                with open(pipeline_path, 'r') as f:
                    pipeline_data = json.load(f)
                
                org_data = pipeline_data.get('spatial_organization', {})
                if org_data:
                    roi_scales = [data['organization_scale'] for data in org_data.values()]
                    mean_scale = np.mean(roi_scales)
                    diversity = len(roi['blob_signatures'])
                    
                    scales.append(mean_scale)
                    diversities.append(diversity)
        
        if scales and diversities:
            ax.scatter(scales, diversities, alpha=0.6)
            ax.set_xlabel('Mean Organization Scale (μm)')
            ax.set_ylabel('Domain Diversity')
            ax.set_title('Scale vs Diversity Relationship')
            
            # Add trend line
            z = np.polyfit(scales, diversities, 1)
            p = np.poly1d(z)
            ax.plot(scales, p(scales), "r--", alpha=0.8)
            
            # Calculate correlation
            corr = np.corrcoef(scales, diversities)[0,1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # 6. Functional group network analysis
        ax = axes[1, 2]
        self._plot_functional_network(ax, results, all_coloc)
        
        fig.suptitle('Kidney Regional Architecture Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_timepoint_region_contact_grid(self, results: List[Dict], config: Config) -> plt.Figure:
        """Grid of aggregated contact matrices: rows=regions, cols=timepoints."""
        # Group by region and time
        by_region = {}
        for roi in results:
            region = roi['metadata'].tissue_region
            day = roi['metadata'].injury_day
            if day is None:
                continue
            key_day = f"D{day}"
            by_region.setdefault(region, {}).setdefault(key_day, []).append(roi)
        if not by_region:
            fig = plt.figure(figsize=(8, 4))
            fig.suptitle('No region/timepoint metadata available')
            return fig
        regions = sorted(by_region.keys())
        # Determine ordered timepoints from data
        timepoints = sorted({tp for d in by_region.values() for tp in d.keys()}, key=lambda x: int(x[1:]))
        fig, axes = plt.subplots(len(regions), len(timepoints), figsize=(4*len(timepoints), 3.5*len(regions)))
        # Normalize axes to a 2D list shape [rows][cols]
        if len(regions) == 1 and len(timepoints) == 1:
            axes = [[axes]]
        elif len(regions) == 1:
            axes = [axes]
        elif len(timepoints) == 1:
            axes = [[ax] for ax in axes]
        for i, region in enumerate(regions):
            for j, tp in enumerate(timepoints):
                ax = axes[i][j]
                rois = by_region[region].get(tp, [])
                if rois:
                    ax_drawn = plot_aggregated_contact_matrix(ax, rois)
                else:
                    ax.text(0.5, 0.5, '—', ha='center', va='center')
                    ax.axis('off')
                if i == 0:
                    (ax_drawn if rois else ax).set_title(tp)
            # Row label
            axes[i][0].set_ylabel(region)
        fig.suptitle('Kidney Architecture: Cortex vs Medulla Recovery Patterns', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_functional_group_ratios(self, ax, results, by_condition):
        """Plot functional group ratios by experimental condition"""
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        functional_groups = config['proteins']['functional_groups']
        conditions = list(by_condition.keys())
        
        # Calculate immune/vascular ratios by condition
        ratios = []
        for condition in conditions:
            condition_ratios = []
            for roi in by_condition[condition]:
                immune_size = 0
                vascular_size = 0
                total_pixels = roi['total_pixels']
                
                for sig in roi['blob_signatures'].values():
                    domain_proteins = sig['dominant_proteins'][:2]
                    
                    # Check functional group membership
                    is_inflammation = any(p in functional_groups['kidney_inflammation'] for p in domain_proteins)
                    is_repair = any(p in functional_groups['kidney_repair'] for p in domain_proteins)
                    is_vascular = any(p in functional_groups['kidney_vasculature'] for p in domain_proteins)
                    
                    if is_inflammation:
                        immune_size += sig['size']
                    elif is_vascular:
                        vascular_size += sig['size']
                
                # Calculate ratio (inflammation/vascular)
                if vascular_size > 0:
                    ratio = immune_size / vascular_size
                    condition_ratios.append(ratio)
            
            ratios.append(condition_ratios)
        
        if any(ratios):
            bp = ax.boxplot(ratios, tick_labels=conditions, patch_artist=True)
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Inflammation/Vascular Ratio')
            ax.set_title('Kidney Inflammation vs Vascular Response')
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line at ratio = 1
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Balance')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient data for ratio analysis', ha='center', va='center')
            ax.axis('off')
    
    def _plot_functional_network(self, ax, results, all_coloc):
        """Plot functional group interaction network"""
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        functional_groups = config['proteins']['functional_groups']
        
        # Calculate inter-group and intra-group colocalization scores
        group_interactions = {
            'immune_immune': [],
            'vascular_vascular': [],
            'immune_vascular': []
        }
        
        for pair, scores in all_coloc.items():
            proteins = pair.split(' ↔ ')
            if len(proteins) == 2:
                p1_groups = []
                p2_groups = []
                
                for group_name, group_proteins in functional_groups.items():
                    if group_name == 'structural_controls':
                        continue
                    
                    if any(p in proteins[0] for p in group_proteins):
                        p1_groups.append(group_name)
                    if any(p in proteins[1] for p in group_proteins):
                        p2_groups.append(group_name)
                
                # Categorize interaction type
                if 'immune_activation' in p1_groups and 'immune_activation' in p2_groups:
                    group_interactions['immune_immune'].extend(scores)
                elif 'vascular_remodeling' in p1_groups and 'vascular_remodeling' in p2_groups:
                    group_interactions['vascular_vascular'].extend(scores)
                elif ('immune_activation' in p1_groups and 'vascular_remodeling' in p2_groups) or \
                     ('vascular_remodeling' in p1_groups and 'immune_activation' in p2_groups):
                    group_interactions['immune_vascular'].extend(scores)
        
        # Create violin plot
        data = []
        labels = []
        for interaction_type, scores in group_interactions.items():
            if scores:
                data.append(scores)
                label = interaction_type.replace('_', '-').title()
                labels.append(label)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Colocalization Score')
            ax.set_title('Functional Group Interactions')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No functional group interactions detected', ha='center', va='center')
            ax.axis('off')

# === MAIN PIPELINE ===

def main():
    """Execute complete analysis pipeline"""
    print("=" * 60)
    print("IMC ANALYSIS - FINAL UNIFIED PIPELINE")
    print("=" * 60)
    
    # Load configuration
    config = Config('config.json')
    
    # Run batch analysis
    batch_analyzer = BatchAnalyzer(config)
    results = batch_analyzer.analyze_all()
    
    if not results:
        print("No ROIs successfully analyzed")
        return
    
    print(f"\n✅ Successfully analyzed {len(results)} ROIs")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Summary visualization
    summary_viz = SummaryVisualizer()
    summary_fig = summary_viz.create_trends_figure(results)
    summary_path = config.output_dir / 'imc_analysis_summary.png'
    summary_fig.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close(summary_fig)
    print(f"  Saved: {summary_path}")
    
    # Create experimental breakdown visualizations
    print("  Creating experimental breakdowns...")
    exp_viz = ExperimentalVisualizer()
    
    # 1. Temporal progression visualization
    temp_fig = exp_viz.create_temporal_figure(results)
    temp_path = config.output_dir / 'temporal_progression.png'
    temp_fig.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close(temp_fig)
    print(f"  Saved: {temp_path}")
    
    # 2. Condition comparison
    cond_fig = exp_viz.create_condition_figure(results)
    cond_path = config.output_dir / 'condition_comparison.png'
    cond_fig.savefig(cond_path, dpi=300, bbox_inches='tight')
    plt.close(cond_fig)
    print(f"  Saved: {cond_path}")
    
    # Skipped region×time grid to simplify outputs

    # Replicate variance for top interactions (separate figure)
    try:
        rep_fig = exp_viz.create_replicate_variance_figure(results)
        rep_path = config.output_dir / 'temporal_replicate_variance.png'
        rep_fig.savefig(rep_path, dpi=300, bbox_inches='tight')
        plt.close(rep_fig)
        print(f"  Saved: {rep_path}")
    except Exception as e:
        print(f"  Skipped replicate variance figure: {e}")
    
    # 4. Network Analysis - Load pipeline data if available
    pipeline_data_path = Path('batch_experimental_results.json')
    if pipeline_data_path.exists():
        print("  Creating network analysis...")
        try:
            with open(pipeline_data_path, 'r') as f:
                pipeline_data = json.load(f)
            
            # Extract batch results for network analysis (contains colocalization data)
            batch_results = pipeline_data.get('batch_results', [])
            
            if batch_results:
                # Initialize network analysis
                network_discovery = NetworkDiscovery('config.json')
                network_visualizer = NetworkVisualizer('config.json')
                
                # Discover networks using batch results (has colocalization data)
                network_results = network_discovery.discover_spatial_networks(batch_results)
                
                # Create comprehensive faceted network grid using batch results
                network_report_path = config.output_dir / 'network_analysis_comprehensive.png'
                network_visualizer.create_faceted_network_grid(batch_results, str(network_report_path))
                print(f"  Saved: {network_report_path}")
                
                # Create individual network visualizations
                networks_by_condition = network_results.get('networks_by_condition', {})
                if networks_by_condition:
                    pass  # Skip detached/ambiguous metrics and interaction evolution plots
                
                # Skip temporal network evolution output (redundant)
                
                # Discovery analysis for most interesting network
                if networks_by_condition:
                    # Use injury network for discovery if available, otherwise first network
                    discovery_network = networks_by_condition.get('Injury', list(networks_by_condition.values())[0])
                    discovery_fig = network_visualizer.plot_colocalization_discovery(discovery_network)
                    discovery_path = config.output_dir / 'colocalization_discovery.png'
                    discovery_fig.savefig(discovery_path, dpi=300, bbox_inches='tight')
                    plt.close(discovery_fig)
                    print(f"  Saved: {discovery_path}")
                
            else:
                print("    No analysis results found in pipeline data")
                
        except Exception as e:
            print(f"    Network analysis failed: {e}")
    else:
        print("    Pipeline data not found - skipping network analysis")

    # Individual ROI visualizations (ALL ROIs)
    roi_viz = ROIVisualizer(config)
    print(f"  Creating {len(results)} individual ROI visualizations...")
    for i, roi_data in enumerate(results, 1):
        print(f"    [{i}/{len(results)}] {roi_data['filename']}")
        fig = roi_viz.create_figure(roi_data)
        filename = roi_data['filename'].replace('.txt', '_analysis.png')
        output_path = config.output_dir / filename
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        # print(f"  Saved: {output_path}")  # Remove verbose output
    
    # Save results as JSON
    json_path = config.output_dir / 'analysis_results.json'
    with open(json_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_data = []
        for roi in results:
            # Convert canonical_contacts values to native Python floats
            contacts = {}
            for contact, strength in roi['canonical_contacts'].items():
                contacts[contact] = float(strength) if hasattr(strength, 'dtype') else strength
            
            json_roi = {
                'filename': roi['filename'],
                'metadata': {
                    'condition': roi['metadata'].condition,
                    'injury_day': int(roi['metadata'].injury_day) if roi['metadata'].injury_day is not None else None,
                    'tissue_region': roi['metadata'].tissue_region,
                    'mouse_replicate': roi['metadata'].mouse_replicate
                },
                'n_domains': len(roi['blob_signatures']),
                'n_contacts': len(roi['canonical_contacts']),
                'total_pixels': int(roi['total_pixels']) if hasattr(roi['total_pixels'], 'dtype') else roi['total_pixels'],
                'canonical_contacts': contacts
            }
            json_data.append(json_roi)
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")
    
    # Generate individual ROI detailed visualizations
    print(f"\nGenerating individual ROI detailed visualizations...")
    roi_visualizer = ROIVisualizer(config)
    
    for i, roi_data in enumerate(results):
        roi_name = Path(roi_data['filename']).stem
        print(f"  [{i+1}/{len(results)}] {roi_name}")
        
        # Create ROI figure
        roi_fig = roi_visualizer.create_figure(roi_data)
        
        # Save individual ROI analysis
        roi_path = config.output_dir / f'{roi_name}_analysis.png'
        roi_fig.savefig(roi_path, dpi=300, bbox_inches='tight')
        plt.close(roi_fig)
        
        # Removed redundant two-panel export (domain signatures + contacts)
    
    print(f"✅ Generated {len(results)} individual ROI visualizations")
    
    print("\n🎯 Analysis complete!")

if __name__ == '__main__':
    main()
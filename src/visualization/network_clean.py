#!/usr/bin/env python3
"""
Clean protein network visualization.
Shows proteins as nodes, colocalization strength as edges.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class CleanNetworkVisualizer:
    """Clean protein network visualization - no bullshit."""
    
    def __init__(self):
        # Biological function-based color scheme
        self.protein_colors = {
            # Immune/Inflammatory (Reds)
            'CD45': '#E74C3C',    # Pan-leukocyte
            'CD11b': '#C0392B',   # Myeloid cells
            'Ly6G': '#922B21',    # Neutrophils
            
            # Vascular/Endothelial (Blues) 
            'CD31': '#3498DB',    # Endothelial
            'CD34': '#2980B9',    # Progenitor/stem
            
            # ECM/Repair (Greens)
            'CD44': '#27AE60',    # ECM receptor
            'CD206': '#229954',   # M2 macrophage (repair)
            
            # Mesenchymal/Structural (Teals)
            'CD140a': '#17A2B8',  # PDGFRA
            'CD140b': '#138496',  # Pericytes
            
            'default': '#95A5A6'  # Gray
        }
        
        # Biological process colors for edges
        self.process_colors = {
            'immune': '#E74C3C',      # Red - immune/inflammatory
            'vascular': '#3498DB',    # Blue - vascular/endothelial  
            'repair': '#27AE60',      # Green - ECM/tissue repair
            'mesenchymal': '#17A2B8', # Teal - stromal/structural
            'default': '#95A5A6'      # Gray - other
        }
        
        # Define biological interactions
        self.biological_interactions = {
            # Immune/Inflammatory processes
            ('CD45', 'CD11b'): {'process': 'immune', 'description': 'Immune cell infiltration'},
            ('CD45', 'Ly6G'): {'process': 'immune', 'description': 'Neutrophil recruitment'},
            ('CD11b', 'Ly6G'): {'process': 'immune', 'description': 'Myeloid-neutrophil response'},
            ('CD11b', 'CD206'): {'process': 'immune', 'description': 'M1→M2 macrophage transition'},
            
            # Vascular processes  
            ('CD31', 'CD34'): {'process': 'vascular', 'description': 'Endothelial-progenitor repair'},
            ('CD31', 'CD44'): {'process': 'vascular', 'description': 'Vessel-ECM interaction'},
            ('CD34', 'CD44'): {'process': 'vascular', 'description': 'Progenitor-matrix interaction'},
            
            # Repair/ECM processes
            ('CD44', 'CD206'): {'process': 'repair', 'description': 'M2 macrophage tissue repair'},
            ('CD44', 'CD140b'): {'process': 'repair', 'description': 'Pericyte-ECM repair'},
            ('CD44', 'CD140a'): {'process': 'repair', 'description': 'Mesenchymal-ECM repair'},
            
            # Mesenchymal/structural
            ('CD140a', 'CD140b'): {'process': 'mesenchymal', 'description': 'Mesenchymal signaling'},
            ('CD140b', 'CD31'): {'process': 'mesenchymal', 'description': 'Pericyte-endothelial support'},
        }
    
    def create_network_grid(self, results: List[Dict], output_path: str):
        """Create clean protein network grid."""
        
        print("\n=== CLEAN PROTEIN NETWORKS ===")
        
        # Define 4 key temporal panels for biological narrative
        panels = [
            {
                'title': 'Sham Baseline',
                'filter': {'condition': 'Sham'}, 
                'description': 'Healthy kidney homeostasis'
            },
            {
                'title': 'Day 1: Acute Injury',
                'filter': {'injury_day': 1},
                'description': 'Immune infiltration & inflammation'
            },
            {
                'title': 'Day 3: Early Recovery', 
                'filter': {'injury_day': 3},
                'description': 'Repair initiation & M2 activation'
            },
            {
                'title': 'Day 7: Tissue Healing',
                'filter': {'injury_day': 7},
                'description': 'ECM remodeling & regeneration'
            }
        ]
        
        # Create 2x2 figure for biological narrative
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Kidney Injury & Healing: Protein Interaction Networks', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        # Generate biological narrative networks
        for i, panel in enumerate(panels):
            ax = axes[i]
            protein_network = self._build_protein_network(results, panel['filter'])
            self._draw_biological_network(ax, protein_network, panel)
            print(f"  {panel['title']}: {len(protein_network['nodes'])} proteins, {len(protein_network['edges'])} interactions")
        
        # Add biological legend
        self._add_biological_legend(fig)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make space for legend
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Biological protein networks saved to: {output_path}")
        plt.close()
    
    def _add_biological_legend(self, fig):
        """Add biological process legend."""
        # Create legend for biological processes
        legend_elements = []
        for process, color in self.process_colors.items():
            if process != 'default':
                label = process.capitalize()
                legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, label=label))
        
        # Add to figure
        fig.legend(handles=legend_elements, 
                  title='Biological Processes',
                  loc='center right',
                  bbox_to_anchor=(0.98, 0.5),
                  fontsize=10)
    
    def _build_protein_network(self, results: List[Dict], filters: Dict) -> Dict:
        """Build protein-protein interaction network from domain colocalization."""
        
        # Filter results
        filtered_results = []
        for result in results:
            meta = result['metadata']
            include = True
            
            for filter_key, filter_val in filters.items():
                if hasattr(meta, filter_key):
                    if getattr(meta, filter_key) != filter_val:
                        include = False
                        break
                elif isinstance(meta, dict):
                    if meta.get(filter_key) != filter_val:
                        include = False
                        break
            
            if include:
                filtered_results.append(result)
        
        if not filtered_results:
            return {'nodes': [], 'edges': []}
        
        # Aggregate protein-protein interactions
        protein_interactions = {}
        
        for result in filtered_results:
            canonical_contacts = result.get('canonical_contacts', {})
            
            # Deduplicate contact pairs first
            deduplicated_contacts = {}
            for domain_pair, strength in canonical_contacts.items():
                if strength < 0.05:  # Skip very weak interactions
                    continue
                    
                # Normalize pairs to avoid duplicates like A↔B and B↔A
                normalized_pair = self._normalize_contact_pair(domain_pair)
                if normalized_pair in deduplicated_contacts:
                    # Average if we see the same pair twice (shouldn't happen but safety)
                    deduplicated_contacts[normalized_pair] = (deduplicated_contacts[normalized_pair] + strength) / 2
                else:
                    deduplicated_contacts[normalized_pair] = strength
            
            # Process deduplicated contacts
            for domain_pair, strength in deduplicated_contacts.items():
                if ' ↔ ' in domain_pair:
                    domain1, domain2 = domain_pair.split(' ↔ ')
                    
                    # Extract proteins from each domain
                    proteins1 = domain1.split('+')
                    proteins2 = domain2.split('+')
                    
                    # Create protein-protein interactions
                    for p1 in proteins1:
                        for p2 in proteins2:
                            if p1 != p2:  # No self-loops
                                edge_key = tuple(sorted([p1, p2]))
                                if edge_key not in protein_interactions:
                                    protein_interactions[edge_key] = []
                                protein_interactions[edge_key].append(strength)
        
        # Build final network
        if not protein_interactions:
            return {'nodes': [], 'edges': []}
        
        # Calculate mean interaction strengths and filter for top interactions
        edge_candidates = []
        all_proteins = set()
        
        for (p1, p2), strengths in protein_interactions.items():
            mean_strength = np.mean(strengths)
            n_observations = len(strengths)
            
            all_proteins.add(p1)
            all_proteins.add(p2)
            edge_candidates.append({
                'source': p1,
                'target': p2,
                'weight': mean_strength,
                'n_colocalizations': n_observations
            })
        
        # Select top interactions but prioritize biologically meaningful ones
        top_n_edges = 8  # Use 8 for consistency with other visualizations
        edge_candidates.sort(key=lambda x: x['weight'], reverse=True)
        
        # Prioritize known biological interactions
        biological_edges = []
        other_edges = []
        
        for edge in edge_candidates:
            edge_key = tuple(sorted([edge['source'], edge['target']]))
            if edge_key in self.biological_interactions:
                edge['process'] = self.biological_interactions[edge_key]['process']
                edge['description'] = self.biological_interactions[edge_key]['description'] 
                biological_edges.append(edge)
            else:
                edge['process'] = 'default'
                edge['description'] = 'Other interaction'
                other_edges.append(edge)
        
        # Take top biological + fill with others
        edges = biological_edges[:top_n_edges] + other_edges[:max(0, top_n_edges - len(biological_edges))]
        edges = edges[:top_n_edges]
        
        # Only include proteins that have at least one top interaction
        proteins_with_edges = set()
        for edge in edges:
            proteins_with_edges.add(edge['source'])
            proteins_with_edges.add(edge['target'])
        
        nodes = [{'id': protein, 'color': self.protein_colors.get(protein, self.protein_colors['default'])} 
                for protein in sorted(proteins_with_edges)]
        
        return {'nodes': nodes, 'edges': edges}
    
    def _draw_biological_network(self, ax: plt.Axes, network: Dict, panel: Dict):
        """Draw clean protein network."""
        
        ax.set_title(f"{panel['title']}\\n{panel['description']}", 
                    fontsize=11, fontweight='bold')
        ax.axis('off')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        
        if not network['nodes']:
            ax.text(0, 0, 'No data', ha='center', va='center', fontsize=12, color='gray')
            return
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for node in network['nodes']:
            G.add_node(node['id'])
        
        # Add edges with biological annotation
        for edge in network['edges']:
            G.add_edge(edge['source'], edge['target'], 
                      weight=edge['weight'], 
                      n_colocalizations=edge['n_colocalizations'],
                      process=edge.get('process', 'default'),
                      description=edge.get('description', 'Other interaction'))
        
        # Biologically-informed layout
        if G.number_of_nodes() > 1:
            pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
        else:
            pos = {list(G.nodes())[0]: (0, 0)}
        
        # Constant node sizes for clarity (no confusing degree-based scaling)
        node_sizes = [500] * G.number_of_nodes()
        
        # Simplified edge properties - binary thickness
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        edge_processes = [G[u][v]['process'] for u, v in G.edges()]
        
        if edge_weights:
            median_weight = np.median(edge_weights)
            edge_widths = [4 if w >= median_weight else 2 for w in edge_weights]
            edge_alphas = [0.8 if w >= median_weight else 0.5 for w in edge_weights]
        else:
            edge_widths = [3]
            edge_alphas = [0.7]
        
        # Draw edges with biological process colors
        for i, (u, v) in enumerate(G.edges()):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Color by biological process
            process_color = self.process_colors.get(edge_processes[i], self.process_colors['default'])
            
            ax.plot([x1, x2], [y1, y2], 
                   color=process_color, 
                   linewidth=edge_widths[i],
                   alpha=edge_alphas[i],
                   solid_capstyle='round')
        
        # Draw nodes with force-graph styling
        node_colors = [self.protein_colors.get(node, self.protein_colors['default']) 
                      for node in G.nodes()]
        
        # Draw node borders (darker)
        border_colors = [self._darken_color(color) for color in node_colors]
        nx.draw_networkx_nodes(G, pos, ax=ax,
                              node_color=border_colors,
                              node_size=[s + 40 for s in node_sizes],  # Slightly larger for border
                              alpha=1.0)
        
        # Draw main nodes
        nx.draw_networkx_nodes(G, pos, ax=ax,
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.9)
        
        # Draw labels with better positioning
        for node in G.nodes():
            x, y = pos[node]
            ax.text(x, y, node, 
                   ha='center', va='center',
                   fontsize=7, fontweight='bold',
                   color='white',  # White text on colored nodes
                   bbox=dict(boxstyle='round,pad=0.1', 
                           facecolor='black', 
                           alpha=0.7, 
                           edgecolor='none'))
    
    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color for borders."""
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # Darken by 30%
        darkened = tuple(int(c * 0.7) for c in rgb)
        # Convert back to hex
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
    
    def _normalize_contact_pair(self, contact: str) -> str:
        """Normalize contact pairs to handle undirected interactions."""
        if ' ↔ ' in contact:
            domain1, domain2 = contact.split(' ↔ ')
            # Sort domains to ensure consistent ordering
            sorted_domains = sorted([domain1.strip(), domain2.strip()])
            return f"{sorted_domains[0]} ↔ {sorted_domains[1]}"
        return contact
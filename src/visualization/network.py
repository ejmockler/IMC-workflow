#!/usr/bin/env python3
"""
Clean, principled network visualization implementation.
Replaced broken implementation with working version.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class NetworkVisualizer:
    """Network visualization that actually works - no bullshit."""
    
    def __init__(self, config_path: str = "config.json"):
        # Load config but use our clean implementation
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except:
            self.config = {}
            
        # Clean, simple color scheme matching the ideal
        self.colors = {
            'kidney_inflammation': '#E74C3C',  # Red
            'kidney_repair': '#2ECC71',        # Green  
            'kidney_vasculature': '#3498DB',   # Blue
            'structural_controls': '#95A5A6',  # Gray
            'unknown': '#7F8C8D'               # Dark gray
        }
    
    def create_faceted_network_grid(self, results: List[Dict], output_path: str):
        """Main method called by imc_visualizer - create the 3x4 network grid."""
        return self.create_network_grid(results, output_path)
        
    def create_network_grid(self, results: List[Dict], output_path: str):
        """Create the 3x4 network grid matching the ideal image."""
        
        print("\n=== CLEAN NETWORK VISUALIZATION ===")
        
        # 1. Define panels using actual metadata fields present in the data
        # Based on dataset: conditions (Sham/Injury), mice (MS1/MS2), timepoints (0,1,3,7)
        panels = [
            # Row 1: Overview + injury conditions
            {'filter': {'region': None, 'condition': None, 'replicate': None},
             'title': 'All ROIs\nComplete Dataset\nKidney Injury Networks'},
            {'filter': {'region': None, 'condition': 'Sham', 'replicate': None},
             'title': 'Sham Controls\nBaseline Networks\nHealthy Kidney'},
            {'filter': {'region': None, 'condition': 'Injury', 'replicate': None},
             'title': 'Injury Response\nDamage & Repair\nRecovery Networks'},
            {'filter': {'region': None, 'condition': None, 'replicate': None, 'timepoint': 1},
             'title': 'Day 1 Post-Injury\nAcute Response\nInflammation Peak'},

            # Row 2: Recovery timeline (key timepoints)
            {'filter': {'region': None, 'condition': None, 'replicate': None, 'timepoint': 3},
             'title': 'Day 3 Post-Injury\nEarly Recovery\nRepair Initiation'},
            {'filter': {'region': None, 'condition': None, 'replicate': None, 'timepoint': 7},
             'title': 'Day 7 Post-Injury\nLate Recovery\nTissue Remodeling'},
            {'filter': {'region': None, 'condition': None, 'replicate': 'MS1'},
             'title': 'Mouse MS1\nReplicate Analysis\nIndividual Networks'},
            {'filter': {'region': None, 'condition': None, 'replicate': 'MS2'},
             'title': 'Mouse MS2\nReplicate Analysis\nIndividual Networks'},

            # Row 3: Tissue regions (cortex vs medulla)
            {'filter': {'region': 'Cortex', 'condition': None, 'replicate': None},
             'title': 'Renal Cortex\nGlomerular Region\nFiltration Networks'},
            {'filter': {'region': 'Medulla', 'condition': None, 'replicate': None},
             'title': 'Renal Medulla\nTubular Region\nConcentration Networks'},
            {'filter': {'region': 'Cortex', 'condition': 'Injury', 'replicate': None},
             'title': 'Injured Cortex\nGlomerular Damage\nRepair Networks'},
            {'filter': {'region': 'Medulla', 'condition': 'Injury', 'replicate': None},
             'title': 'Injured Medulla\nTubular Damage\nRepair Networks'},
        ]
        
        # 2. Build networks for each panel
        networks = []
        for i, panel in enumerate(panels):
            net_data = self._build_network(results, panel['filter'])
            networks.append(net_data)
            
            if net_data and net_data['graph']:
                G = net_data['graph']
                print(f"  Panel {i+1}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            else:
                print(f"  Panel {i+1}: No data")
        
        # 3. Create the figure with proper layout
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('High-Granularity Regional Networks: Region × Condition × Mouse Detail', 
                     fontsize=14, fontweight='bold')
        
        # 4. Draw each panel
        for idx, (panel, net_data) in enumerate(zip(panels, networks)):
            ax = plt.subplot(3, 4, idx + 1)
            self._draw_network(ax, net_data, panel['title'])
        
        # 5. Add global colorbar
        self._add_colorbar(fig)
        
        # 6. Save with high quality
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Clean network saved to: {output_path}")
        plt.close()
        
    def _build_network(self, results: List[Dict], filters: Dict) -> Optional[Dict]:
        """Build a network from filtered results."""
        
        # Aggregate colocalization data
        coloc_data = self._aggregate_colocalization(results, filters)
        
        if not coloc_data:
            return None
            
        # Build graph with lower threshold to show more connections
        threshold = 0.05  # Lower threshold to capture more interactions
        G = nx.Graph()
        
        # Use all pairs - remove biological filtering that may be too restrictive
        bio_pairs = coloc_data
        
        # Convert domain pairs to protein-protein interactions
        # Each domain pair like "CD31+CD34 ↔ CD11b+CD44" represents colocalization
        # We want proteins as nodes, with edge weights = sum of colocalization events
        protein_interactions = {}
        
        for pair_key, score in bio_pairs.items():
            if score >= threshold and ' ↔ ' in pair_key:
                # Parse domain pair: "CD31+CD34 ↔ CD11b+CD44"
                domain1, domain2 = pair_key.split(' ↔ ')
                
                # Extract proteins from each domain
                proteins1 = domain1.split('+')
                proteins2 = domain2.split('+') 
                
                # Create protein-protein interactions
                for p1 in proteins1:
                    for p2 in proteins2:
                        if p1 != p2:  # No self-loops
                            # Sort to make undirected
                            edge_key = tuple(sorted([p1, p2]))
                            if edge_key not in protein_interactions:
                                protein_interactions[edge_key] = []
                            protein_interactions[edge_key].append(score)
        
        # Add edges with aggregated weights
        for (p1, p2), scores in protein_interactions.items():
            # Use mean colocalization strength
            weight = sum(scores) / len(scores)
            G.add_edge(p1, p2, weight=weight, n_colocalizations=len(scores))
        
        if G.number_of_nodes() == 0:
            return None
            
        # Add node attributes
        for node in G.nodes():
            # Determine functional group from node name
            group = self._get_functional_group(node)
            G.nodes[node]['group'] = group
            G.nodes[node]['label'] = node
            
        return {'graph': G, 'coloc_data': coloc_data}
    
    def _aggregate_colocalization(self, results: List[Dict], filters: Dict) -> Dict:
        """Unified colocalization data aggregator handling both pipeline and domain data."""
        
        aggregated = {}
        count = {}
        
        for result in results:
            # Check if this result matches filters
            meta = result.get('metadata', {})
            
            # Map metadata fields correctly (dataset provides condition, mouse_id, timepoint)
            if hasattr(meta, 'get'):
                # Dict format
                region = meta.get('tissue_region')
                condition = meta.get('condition')
                replicate = meta.get('mouse_id')
                timepoint = meta.get('timepoint')
            else:
                # Metadata object format
                region = meta.tissue_region
                condition = meta.condition
                replicate = meta.mouse_replicate
                timepoint = meta.injury_day
            
            # Apply filters
            if filters['region'] and region != filters['region']:
                continue
            if filters['condition'] and condition != filters['condition']:
                continue
            if filters['replicate'] and replicate != filters['replicate']:
                continue
            if 'timepoint' in filters and filters['timepoint'] is not None and timepoint != filters['timepoint']:
                continue
                
            # Get both domain contacts and pipeline colocalization
            canonical_contacts = result.get('canonical_contacts', {})
            
            # Also try to load pipeline colocalization data
            pipeline_coloc = {}
            if 'filename' in result:
                pipeline_file = result['filename'].replace('.txt', '_pipeline_analysis.json')
                pipeline_path = Path('data/241218_IMC_Alun') / pipeline_file
                if pipeline_path.exists():
                    try:
                        import json
                        with open(pipeline_path, 'r') as f:
                            pipeline_data = json.load(f)
                        pipeline_coloc = pipeline_data.get('colocalization', {})
                    except:
                        pass
            
            # Process both data sources
            all_pairs = {}
            
            # 1. Add canonical contacts (domain-domain)
            for pair, freq in canonical_contacts.items():
                # Convert domain pair to protein pair (CD31+CD34 → CD31↔CD34)
                if '+' in pair:
                    proteins = sorted(pair.split('+'))
                    normalized_pair = '↔'.join(proteins)
                    all_pairs[normalized_pair] = float(freq)
            
            # 2. Add pipeline colocalization (protein-protein Pearson r) 
            for pair_key, data in pipeline_coloc.items():
                if isinstance(data, dict):
                    score = data.get('colocalization_score', 0)
                    if score > 0:  # Only include positive correlations
                        # Normalize protein names: CD31(Sm154Di)_CD34(Er166Di) → CD31↔CD34
                        p1 = data.get('protein_1', '').split('(')[0]
                        p2 = data.get('protein_2', '').split('(')[0]
                        if p1 and p2:
                            proteins = sorted([p1, p2])
                            normalized_pair = '↔'.join(proteins)
                            # Use max score if both sources provide data for same pair
                            all_pairs[normalized_pair] = max(all_pairs.get(normalized_pair, 0), score)
            
            # Aggregate all pairs
            for pair, score in all_pairs.items():
                if pair not in aggregated:
                    aggregated[pair] = 0
                    count[pair] = 0
                    
                aggregated[pair] += score
                count[pair] += 1
        
        # Average the scores
        final = {}
        for pair in aggregated:
            if count[pair] > 0:
                final[pair] = aggregated[pair] / count[pair]
                
        return final
    
    def _ensure_min_separation(self, pos: Dict, min_dist: float = 0.15) -> Dict:
        """Ensure minimum separation between nodes to prevent overlap."""
        import numpy as np
        
        nodes = list(pos.keys())
        coords = np.array([pos[n] for n in nodes])
        
        # Iteratively push overlapping nodes apart
        for _ in range(50):
            moved = False
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    diff = coords[j] - coords[i]
                    dist = np.linalg.norm(diff)
                    
                    if dist < min_dist and dist > 0:
                        # Push apart
                        push = (min_dist - dist) / 2
                        direction = diff / dist
                        coords[i] -= direction * push
                        coords[j] += direction * push
                        moved = True
            
            if not moved:
                break
        
        # Update positions
        new_pos = {}
        for i, node in enumerate(nodes):
            new_pos[node] = tuple(coords[i])
        
        return new_pos
    
    def _filter_biologically_interesting_pairs(self, coloc_data: Dict) -> Dict:
        """Filter to kidney injury/repair relevant protein pairs only."""
        
        # Define kidney injury/repair relevant interactions
        kidney_pairs = {
            # Vascular repair and regeneration
            'CD31↔CD34',     # Endothelial-progenitor
            'CD31↔CD140b',   # Endothelial-pericyte
            'CD34↔CD140b',   # Progenitor-pericyte
            
            # Immune resolution and repair
            'CD45↔CD206',    # Leukocyte-M2 macrophage
            'CD11b↔CD206',   # Myeloid-M2 macrophage  
            'CD45↔CD44',     # Immune-ECM remodeling
            
            # Tissue remodeling and healing
            'CD44↔CD206',    # ECM-repair macrophage
            'CD44↔CD31',     # ECM-endothelial
            'CD206↔CD140b',  # Repair macrophage-pericyte
            
            # Immune-vascular interactions (key in kidney injury)
            'CD45↔CD31',     # Immune-endothelial
            'CD11b↔CD31',    # Myeloid-endothelial
            'CD45↔CD34',     # Immune-progenitor
            'CD11b↔CD34',    # Myeloid-progenitor
            
            # Additional repair interactions
            'CD44↔CD140b',   # ECM-pericyte
            'CD44↔CD34'      # ECM-progenitor
        }
        
        filtered = {}
        for pair_key, score in coloc_data.items():
            # Normalize to standard format
            if '↔' in pair_key:
                normalized_pair = pair_key
            elif '+' in pair_key:
                proteins = sorted(pair_key.split('+'))
                normalized_pair = '↔'.join(proteins)
            else:
                # Handle other delimiters
                proteins = pair_key.replace('_', ' ').split()
                if len(proteins) >= 2:
                    proteins = sorted(proteins[:2])
                    normalized_pair = '↔'.join(proteins)
                else:
                    continue
                    
            # Check if this pair is kidney injury/repair relevant
            if normalized_pair in kidney_pairs:
                filtered[pair_key] = score
                
        # Return top 12 pairs to manage cognitive load
        sorted_pairs = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_pairs[:12])
    
    def _get_edge_color(self, u: str, v: str) -> str:
        """Get edge color based on specific protein interaction type."""
        
        # Create standardized pair key
        proteins = sorted([u, v])
        pair = '↔'.join(proteins)
        
        # Color by kidney injury/repair interaction type
        vascular_repair = {'CD31↔CD34', 'CD31↔CD140b', 'CD34↔CD140b'}
        immune_repair = {'CD45↔CD206', 'CD11b↔CD206', 'CD45↔CD44', 'CD44↔CD206'}
        immune_vascular = {'CD45↔CD31', 'CD11b↔CD31', 'CD45↔CD34', 'CD11b↔CD34'}
        tissue_remodeling = {'CD44↔CD31', 'CD206↔CD140b', 'CD44↔CD140b', 'CD44↔CD34'}
        
        if pair in vascular_repair:
            return '#3498DB'  # Blue for vascular repair
        elif pair in immune_repair:
            return '#E74C3C'  # Red for immune/inflammation resolution
        elif pair in immune_vascular:
            return '#FF8C00'  # Orange for immune-vascular cross-talk
        elif pair in tissue_remodeling:
            return '#2ECC71'  # Green for tissue remodeling/ECM
        else:
            # Fallback to functional group colors
            u_group = self._get_functional_group(u)
            v_group = self._get_functional_group(v)
            if u_group == v_group:
                return self.colors.get(u_group, '#95A5A6')
            else:
                return '#95A5A6'  # Gray for other combinations
    
    def _get_functional_group(self, node_name: str) -> str:
        """Determine functional group from node name."""
        
        # Map proteins to functional groups
        if any(p in node_name for p in ['CD45', 'CD11b', 'Ly6G']):
            return 'kidney_inflammation'
        elif any(p in node_name for p in ['CD206', 'CD44']):
            return 'kidney_repair'
        elif any(p in node_name for p in ['CD31', 'CD34', 'CD140a', 'CD140b']):
            return 'kidney_vasculature'
        elif any(p in node_name for p in ['DNA1', 'DNA2']):
            return 'structural_controls'
        else:
            return 'unknown'
    
    def _draw_network(self, ax: plt.Axes, net_data: Optional[Dict], title: str):
        """Draw a single network panel with proper spacing."""
        
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axis('off')
        
        if not net_data or not net_data['graph']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            return
            
        G = net_data['graph']
        
        # Add edge count to title
        n_edges = G.number_of_edges()
        current_title = ax.get_title()
        ax.set_title(f"{current_title} ({n_edges} edges)", fontsize=9)
        
        # 1. Go back to shell layout but with tighter packing for small nodes
        degrees = dict(G.degree())
        if len(degrees) > 0:
            degree_values = list(degrees.values())
            high_threshold = np.percentile(degree_values, 75)
            med_threshold = np.percentile(degree_values, 25)
            
            high_degree = [n for n, d in degrees.items() if d >= high_threshold]
            med_degree = [n for n, d in degrees.items() if med_threshold <= d < high_threshold]
            low_degree = [n for n, d in degrees.items() if d < med_threshold]
            
            # Build shells with high-degree nodes in center
            shells = []
            if high_degree:
                shells.append(high_degree)
            if med_degree:
                shells.append(med_degree)  
            if low_degree:
                shells.append(low_degree)
                
            # Use clean spring layout to avoid hub clustering
            pos = nx.spring_layout(G, k=3.0, iterations=100, 
                                 weight=None, seed=42)
        else:
            pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42)
        
        # 2. Proper separation for medium nodes
        pos = self._ensure_min_separation(pos, min_dist=0.08)
        
        # 3. Draw nodes with proper sizing and colors
        node_colors = [self.colors.get(G.nodes[n].get('group', 'unknown'), '#7F8C8D') 
                      for n in G.nodes()]
        
        # Medium-sized nodes that are actually visible
        node_size = 300  # Reasonable fucking size
        
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=node_size,
                              alpha=0.9,
                              ax=ax)
        
        # 3. Draw edges with functional group colors and reasonable limits
        edges = list(G.edges())
        weights = [G[u][v]['weight'] for u, v in edges]
        
        if weights and len(edges) <= 15:  # Only draw if manageable number of edges
            min_w, max_w = min(weights), max(weights)
            if max_w > min_w:
                # Sort edges by weight and take top 10 most significant
                edge_weights = [(edge, G[edge[0]][edge[1]]['weight']) for edge in edges]
                edge_weights.sort(key=lambda x: x[1], reverse=True)
                top_edges = edge_weights[:min(10, len(edge_weights))]
                
                # Draw edges with biological interaction colors
                for (u, v), w in top_edges:
                    alpha = 0.4 + 0.6 * ((w - min_w) / (max_w - min_w))
                    width = 1.5 + 3.5 * ((w - min_w) / (max_w - min_w))
                    
                    # Color based on biological interaction type
                    edge_color = self._get_edge_color(u, v)
                    
                    nx.draw_networkx_edges(G, pos, [(u, v)],
                                          width=width,
                                          alpha=alpha,
                                          edge_color=edge_color,
                                          ax=ax)
        elif weights:
            # Too many edges - just show summary text
            ax.text(0.02, 0.98, f'{len(edges)} connections\n(simplified view)', 
                   transform=ax.transAxes, fontsize=8, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # 4. Draw node labels INSIDE nodes with simplified text
        for node, (x, y) in pos.items():
            # Simplify label - just show the first protein
            if '+' in node:
                parts = node.split('+')
                label = parts[0] if len(parts[0]) <= 6 else parts[0][:6]
            else:
                label = node[:6] if len(node) > 6 else node
            
            # Determine text color based on node color
            node_color = self.colors.get(G.nodes[node].get('group', 'unknown'), '#7F8C8D')
            text_color = 'white' if self._is_dark(node_color) else 'black'
            
            # Readable font for medium nodes
            ax.text(x, y, label, 
                   ha='center', va='center',
                   fontsize=5, fontweight='bold',
                   color=text_color)
        
        # 5. Draw edge labels only for top 3 strongest connections to avoid clutter
        if weights and len(edges) <= 15:
            # Show labels for only the strongest connections
            edge_data = [(e, G[e[0]][e[1]]['weight']) for e in edges]
            edge_data.sort(key=lambda x: x[1], reverse=True)
            top_3_edges = edge_data[:min(3, len(edge_data))]
            
            for (u, v), weight in top_3_edges:
                if weight > 0.3:  # Only show very strong connections
                    # Calculate edge midpoint
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    
                    # Small offset perpendicular to edge
                    dx = y2 - y1
                    dy = x1 - x2
                    length = (dx**2 + dy**2)**0.5
                    if length > 0:
                        dx /= length
                        dy /= length
                        offset = 0.08
                        mid_x += dx * offset
                        mid_y += dy * offset
                    
                    # Draw edge label with clear styling
                    ax.text(mid_x, mid_y, f'{weight:.2f}',
                           ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='yellow',
                                   edgecolor='black',
                                   alpha=0.9))
    
    def _is_dark(self, color: str) -> bool:
        """Check if a color is dark."""
        if color.startswith('#'):
            color = color[1:]
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)  
        b = int(color[4:6], 16)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return luminance < 0.5
    
    def _add_colorbar(self, fig):
        """Add a colorbar for edge weights."""
        from matplotlib import cm
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize
        
        # Add colorbar axis
        ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        
        # Create colorbar
        norm = Normalize(vmin=0, vmax=1)
        cb = ColorbarBase(ax, cmap=cm.viridis, norm=norm, orientation='vertical')
        cb.set_label('Colocalization Strength (r)', fontsize=10)
        
        # Add legend for node colors
        ax2 = fig.add_axes([0.92, 0.05, 0.08, 0.08])
        ax2.axis('off')
        
        legend_elements = []
        for group, color in [
            ('Inflammation', self.colors['kidney_inflammation']),
            ('Repair', self.colors['kidney_repair']),
            ('Vasculature', self.colors['kidney_vasculature'])
        ]:
            from matplotlib.patches import Patch
            legend_elements.append(Patch(facecolor=color, label=group))
        
        ax2.legend(handles=legend_elements, loc='center', fontsize=8)
    
    def plot_colocalization_discovery(self, results: List[Dict], output_path: str):
        """Legacy method for compatibility - just calls create_network_grid"""
        return self.create_network_grid(results, output_path)
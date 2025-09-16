"""Network visualization for protein colocalization analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import networkx as nx

from src.config import Config


class NetworkVisualizer:
    """Creates protein colocalization network visualizations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_network_figure(self, results: List[Dict]) -> plt.Figure:
        """Create network analysis figure."""
        # Import the network analyzer from analysis module
        from src.analysis.network import NetworkAnalyzer
        
        # Build network from results
        analyzer = NetworkAnalyzer(self.config)
        network = analyzer.build_colocalization_network(results)
        
        if network is None or len(network.nodes()) == 0:
            return self._create_empty_network_figure()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Protein Colocalization Network Analysis', fontsize=16, fontweight='bold')
        
        # Main network plot
        self._plot_network_layout(axes[0, 0], network)
        
        # Network metrics
        self._plot_network_metrics(axes[0, 1], network)
        
        # Degree distribution
        self._plot_degree_distribution(axes[1, 0], network)
        
        # Community analysis
        self._plot_communities(axes[1, 1], network)
        
        plt.tight_layout()
        return fig
    
    def _plot_network_layout(self, ax, network):
        """Plot the main network layout with functional group colors."""
        pos = nx.spring_layout(network, k=1, iterations=50)
        
        # Get node colors by functional group
        node_colors = []
        protein_groups = self.config.proteins.get('functional_groups', {})
        group_to_color = {
            'immune': '#FF6B6B',
            'vascular': '#4ECDC4', 
            'structural': '#45B7D1',
            'metabolic': '#96CEB4',
            'signaling': '#FFEAA7'
        }
        
        for node in network.nodes():
            # Find which group this protein belongs to
            node_group = 'other'
            for group, proteins in protein_groups.items():
                if node in proteins:
                    node_group = group
                    break
            node_colors.append(group_to_color.get(node_group, '#95A5A6'))
        
        # Draw network
        nx.draw_networkx_nodes(network, pos, ax=ax, node_color=node_colors, 
                              node_size=300, alpha=0.8)
        nx.draw_networkx_edges(network, pos, ax=ax, edge_color='gray', 
                              alpha=0.5, width=0.5)
        nx.draw_networkx_labels(network, pos, ax=ax, font_size=8)
        
        ax.set_title('Protein Colocalization Network', fontweight='bold')
        ax.axis('off')
    
    def _plot_network_metrics(self, ax, network):
        """Plot key network metrics."""
        metrics = {
            'Nodes': len(network.nodes()),
            'Edges': len(network.edges()),
            'Density': nx.density(network),
            'Avg Clustering': nx.average_clustering(network),
        }
        
        # Add centrality metrics for top nodes
        betweenness = nx.betweenness_centrality(network)
        if betweenness:
            top_node = max(betweenness, key=betweenness.get)
            metrics[f'Hub: {top_node}'] = betweenness[top_node]
        
        # Plot as bar chart
        names = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(range(len(names)), values, color='skyblue', alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_title('Network Metrics', fontweight='bold')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_degree_distribution(self, ax, network):
        """Plot degree distribution."""
        degrees = [network.degree(n) for n in network.nodes()]
        
        if degrees:
            ax.hist(degrees, bins=max(1, len(set(degrees))), alpha=0.7, color='lightcoral')
            ax.set_xlabel('Degree')
            ax.set_ylabel('Count')
            ax.set_title('Degree Distribution', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No connections', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Degree Distribution', fontweight='bold')
    
    def _plot_communities(self, ax, network):
        """Plot community structure."""
        try:
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(network))
            
            # Create community mapping
            community_map = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = i
            
            # Plot community sizes
            comm_sizes = [len(comm) for comm in communities]
            ax.bar(range(len(comm_sizes)), comm_sizes, color='lightgreen', alpha=0.7)
            ax.set_xlabel('Community')
            ax.set_ylabel('Size')
            ax.set_title(f'Communities (n={len(communities)})', fontweight='bold')
            
        except ImportError:
            ax.text(0.5, 0.5, 'Community detection\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Communities', fontweight='bold')
    
    def _create_empty_network_figure(self):
        """Create figure for when no network data is available."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No significant protein colocalization\nnetwork detected', 
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Protein Colocalization Network', fontweight='bold')
        ax.axis('off')
        return fig


# Alias for backwards compatibility
CleanNetworkVisualizer = NetworkVisualizer
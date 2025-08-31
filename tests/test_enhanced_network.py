#!/usr/bin/env python3
"""Test enhanced network visualization with biologically-informed nonlinear spacing."""

import json
import sys
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.visualization.network import NetworkVisualizer

def create_test_network():
    """Create a test network with various centrality patterns"""
    G = nx.Graph()
    
    # Add nodes with functional groups and centrality patterns
    proteins = {
        'CD45': {'functional_group': 'kidney_inflammation', 'clean_name': 'CD45'},
        'CD11b': {'functional_group': 'kidney_inflammation', 'clean_name': 'CD11b'},
        'CD206': {'functional_group': 'kidney_repair', 'clean_name': 'CD206'},
        'CD44': {'functional_group': 'kidney_repair', 'clean_name': 'CD44'},
        'CD31': {'functional_group': 'kidney_vasculature', 'clean_name': 'CD31'},
        'CD34': {'functional_group': 'kidney_vasculature', 'clean_name': 'CD34'},
        'CD140b': {'functional_group': 'kidney_vasculature', 'clean_name': 'CD140b'},
    }
    
    for node, attrs in proteins.items():
        G.add_node(node, **attrs)
    
    # Add edges with weights to create hub/peripheral patterns
    # CD44 as major hub (high connectivity)
    G.add_edge('CD44', 'CD45', weight=0.85)
    G.add_edge('CD44', 'CD11b', weight=0.72)
    G.add_edge('CD44', 'CD206', weight=0.65)
    G.add_edge('CD44', 'CD31', weight=0.58)
    G.add_edge('CD44', 'CD34', weight=0.45)
    
    # CD31 as moderate hub
    G.add_edge('CD31', 'CD34', weight=0.78)
    G.add_edge('CD31', 'CD140b', weight=0.52)
    G.add_edge('CD31', 'CD45', weight=0.41)
    
    # Some peripheral connections
    G.add_edge('CD45', 'CD11b', weight=0.33)
    G.add_edge('CD34', 'CD140b', weight=0.28)
    G.add_edge('CD206', 'CD11b', weight=0.25)
    
    return G

def test_enhanced_layout():
    """Test the enhanced layout system"""
    print("Testing enhanced network layout with biologically-informed spacing...")
    
    # Create test network
    G = create_test_network()
    print(f"Created test network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Initialize visualizer
    visualizer = NetworkVisualizer()
    
    # Test the enhanced layout computation
    pos, xlim, ylim = visualizer._compute_reference_layout(G)
    
    # Calculate centrality for verification
    centrality = nx.betweenness_centrality(G, weight='weight')
    degree_centrality = dict(G.degree(weight='weight'))
    max_degree = max(degree_centrality.values()) if degree_centrality.values() else 1
    degree_centrality = {k: v/max_degree for k, v in degree_centrality.items()}
    
    print("\nNode positioning results:")
    for node in G.nodes():
        x, y = pos[node]
        radius = (x**2 + y**2)**0.5
        bet_cent = centrality.get(node, 0)
        deg_cent = degree_centrality.get(node, 0)
        print(f"  {node:>6}: radius={radius:.3f}, betweenness={bet_cent:.3f}, degree={deg_cent:.3f}")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Node colors by functional group
    group_colors = {
        'kidney_inflammation': '#FF6B6B',
        'kidney_repair': '#4ECDC4', 
        'kidney_vasculature': '#45B7D1'
    }
    
    node_colors = [group_colors.get(G.nodes[node]['functional_group'], 'gray') for node in G.nodes()]
    
    # Node sizes by combined centrality
    max_centrality = max(centrality.values()) if centrality.values() else 1
    combined_centrality = {}
    for node in G.nodes():
        bet = centrality.get(node, 0) / max(max_centrality, 1e-6)
        deg = degree_centrality.get(node, 0)
        combined_centrality[node] = (bet * deg)**0.5 if bet > 0 and deg > 0 else 0
    
    max_combined = max(combined_centrality.values()) if combined_centrality.values() else 1
    node_sizes = [400 + 1200 * (combined_centrality.get(node, 0) / max_combined) for node in G.nodes()]
    
    # Edge weights and styling
    edges = list(G.edges())
    edge_weights = [G[u][v]['weight'] for u, v in edges]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.8, ax=ax)
    
    # Draw all edges with alpha based on weight
    for (u, v), weight in zip(edges, edge_weights):
        alpha = 0.1 + 0.9 * ((weight - min_weight) / (max_weight - min_weight))
        width = 0.5 + 3.0 * ((weight - min_weight) / (max_weight - min_weight))
        nx.draw_networkx_edges(G, pos, [(u, v)], width=width, alpha=alpha, 
                             edge_color='gray', ax=ax)
    
    # Add internal node labels with contrast detection
    for node, (x, y) in pos.items():
        node_color = group_colors.get(G.nodes[node]['functional_group'], 'gray')
        # Simple contrast detection
        text_color = 'white' if node_color in ['#FF6B6B', '#45B7D1'] else 'black'
        ax.text(x, y, G.nodes[node]['clean_name'], ha='center', va='center',
               fontsize=10, fontweight='bold', color=text_color)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title('Enhanced Network Layout\nBiologically-Informed Nonlinear Spacing', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, label=group.replace('kidney_', '').title())
                      for group, color in group_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig('results/test_enhanced_network.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Enhanced network test saved to: results/test_enhanced_network.png")
    
    return True

if __name__ == "__main__":
    try:
        test_enhanced_layout()
        print("✅ Enhanced network layout test completed successfully")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
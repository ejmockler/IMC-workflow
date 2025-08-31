"""
Advanced Network Analysis for IMC Spatial Data
Provides mechanistic discovery through protein colocalization networks
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from pathlib import Path


@dataclass
class NetworkMetrics:
    """Network topology metrics for publication analysis"""
    modularity: float
    clustering_coefficient: float
    average_path_length: float
    density: float
    centrality_scores: Dict[str, float]
    hub_proteins: List[str]
    community_structure: Dict[str, int]


@dataclass 
class SpatialNetwork:
    """Spatial protein colocalization network"""
    graph: nx.Graph
    node_attributes: Dict[str, Dict]
    edge_attributes: Dict[Tuple[str, str], Dict] 
    metrics: NetworkMetrics
    biological_context: Dict[str, any]


class NetworkAnalyzer(ABC):
    """Abstract base for network analysis strategies"""
    
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.functional_groups = self.config['proteins']['functional_groups']
        self.annotations = self.config['proteins']['annotations']
    
    @abstractmethod
    def analyze(self, colocalization_data: Dict, metadata: Dict) -> SpatialNetwork:
        pass
    
    def _build_base_network(self, coloc_data: Dict, threshold: float = 0.3) -> nx.Graph:
        """Build base network from colocalization data"""
        G = nx.Graph()
        
        # Add nodes with functional group information
        proteins = set()
        for pair_key in coloc_data.keys():
            p1, p2 = self._parse_protein_pair(pair_key)
            proteins.update([p1, p2])
        
        for protein in proteins:
            group = self._get_functional_group(protein)
            annotation = self.annotations.get(self._clean_protein_name(protein), "Unknown")
            G.add_node(protein, 
                      functional_group=group,
                      annotation=annotation,
                      clean_name=self._clean_protein_name(protein))
        
        # Add edges for significant colocalization
        for pair_key, data in coloc_data.items():
            if isinstance(data, dict) and 'colocalization_score' in data:
                score = data['colocalization_score']
                if score >= threshold:
                    p1, p2 = self._parse_protein_pair(pair_key)
                    G.add_edge(p1, p2, 
                             weight=score,
                             score_std=data.get('score_std', 0),
                             n_measurements=data.get('n_measurements', 0))
        
        return G
    
    def _parse_protein_pair(self, pair_key: str) -> Tuple[str, str]:
        """Parse protein pair from various formats"""
        if ' ↔ ' in pair_key:
            return tuple(pair_key.split(' ↔ '))
        elif '_' in pair_key:
            return tuple(pair_key.split('_', 1))
        else:
            raise ValueError(f"Cannot parse protein pair: {pair_key}")
    
    def _clean_protein_name(self, protein: str) -> str:
        """Extract clean protein name from channel format"""
        if '(' in protein:
            return protein.split('(')[0]
        return protein
    
    def _get_functional_group(self, protein: str) -> str:
        """Get functional group for protein"""
        clean_name = self._clean_protein_name(protein)
        for group, proteins in self.functional_groups.items():
            if clean_name in proteins:
                return group
        # Debug: Print unmatched proteins (commented out after fixing CD140a)
        # if clean_name not in ['DNA1', 'DNA2']:  # Skip DNA controls
        #     print(f"  Debug: Protein '{clean_name}' (from '{protein}') not found in functional groups")
        return "unknown"


class SpatialCommunicationAnalyzer(NetworkAnalyzer):
    """Analyze spatial communication patterns between cell types"""
    
    def analyze(self, colocalization_data: Dict, metadata: Dict) -> SpatialNetwork:
        G = self._build_base_network(colocalization_data, threshold=0.25)
        
        # Identify communication hubs
        centrality = nx.betweenness_centrality(G, weight='weight')
        degree_centrality = nx.degree_centrality(G)
        
        # Find inter-group connections (cross-functional communication)
        inter_group_edges = []
        for edge in G.edges(data=True):
            p1, p2 = edge[0], edge[1]
            g1 = G.nodes[p1]['functional_group']
            g2 = G.nodes[p2]['functional_group'] 
            if g1 != g2 and g1 != 'unknown' and g2 != 'unknown':
                inter_group_edges.append((p1, p2, edge[2]))
        
        # Calculate network metrics
        metrics = self._calculate_metrics(G, centrality)
        
        # Biological context
        context = {
            'inter_group_communication': len(inter_group_edges),
            'immune_vascular_crosstalk': self._count_specific_crosstalk(inter_group_edges),
            'communication_hubs': [p for p, c in centrality.items() if c > 0.1],
            'network_type': 'spatial_communication'
        }
        
        return SpatialNetwork(
            graph=G,
            node_attributes={n: d for n, d in G.nodes(data=True)},
            edge_attributes={(u, v): d for u, v, d in G.edges(data=True)},
            metrics=metrics,
            biological_context=context
        )
    
    def _count_specific_crosstalk(self, inter_group_edges: List) -> int:
        """Count immune-vascular specific interactions"""
        count = 0
        for p1, p2, data in inter_group_edges:
            if (('immune_activation' in [self._get_functional_group(p1), self._get_functional_group(p2)]) and
                ('vascular_remodeling' in [self._get_functional_group(p1), self._get_functional_group(p2)])):
                count += 1
        return count
    
    def _calculate_metrics(self, G: nx.Graph, centrality: Dict) -> NetworkMetrics:
        """Calculate comprehensive network metrics"""
        try:
            clustering = nx.average_clustering(G, weight='weight')
        except:
            clustering = 0.0
            
        try:
            path_length = nx.average_shortest_path_length(G, weight='weight')
        except:
            path_length = float('inf')
        
        density = nx.density(G)
        
        # Find hub proteins (top 20% by centrality)
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        n_hubs = max(1, len(sorted_centrality) // 5)
        hubs = [p for p, _ in sorted_centrality[:n_hubs]]
        
        # Community detection
        try:
            communities = nx.community.greedy_modularity_communities(G, weight='weight')
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            modularity = nx.community.modularity(G, communities, weight='weight')
        except:
            community_map = {node: 0 for node in G.nodes()}
            modularity = 0.0
        
        return NetworkMetrics(
            modularity=modularity,
            clustering_coefficient=clustering,
            average_path_length=path_length,
            density=density,
            centrality_scores=centrality,
            hub_proteins=hubs,
            community_structure=community_map
        )


class TemporalNetworkAnalyzer(NetworkAnalyzer):
    """Analyze network evolution across timepoints"""
    
    def __init__(self, config_path: str = "config.json"):
        super().__init__(config_path)
        self.config_path = config_path
    
    def analyze(self, timepoint_data: Dict[str, Dict], metadata: Dict) -> Dict[str, SpatialNetwork]:
        """Analyze networks across multiple timepoints"""
        networks = {}
        
        for timepoint, coloc_data in timepoint_data.items():
            G = self._build_base_network(coloc_data, threshold=0.2)
            centrality = nx.betweenness_centrality(G, weight='weight')
            metrics = SpatialCommunicationAnalyzer(self.config_path)._calculate_metrics(G, centrality)
            
            # Temporal-specific context
            context = {
                'timepoint': timepoint,
                'network_size': len(G.nodes()),
                'edge_count': len(G.edges()),
                'network_type': 'temporal_evolution'
            }
            
            networks[timepoint] = SpatialNetwork(
                graph=G,
                node_attributes={n: d for n, d in G.nodes(data=True)},
                edge_attributes={(u, v): d for u, v, d in G.edges(data=True)},
                metrics=metrics,
                biological_context=context
            )
        
        return networks
    
    def analyze_network_evolution(self, networks: Dict[str, SpatialNetwork]) -> Dict:
        """Analyze how networks change over time"""
        timepoints = sorted(networks.keys())
        evolution_metrics = {}
        
        for i in range(len(timepoints) - 1):
            t1, t2 = timepoints[i], timepoints[i + 1]
            G1, G2 = networks[t1].graph, networks[t2].graph
            
            # Edge changes
            edges1 = set(G1.edges())
            edges2 = set(G2.edges())
            gained_edges = edges2 - edges1
            lost_edges = edges1 - edges2
            
            # Hub changes
            hubs1 = set(networks[t1].metrics.hub_proteins)
            hubs2 = set(networks[t2].metrics.hub_proteins)
            
            evolution_metrics[f"{t1}_to_{t2}"] = {
                'edges_gained': len(gained_edges),
                'edges_lost': len(lost_edges),
                'hub_stability': len(hubs1.intersection(hubs2)) / len(hubs1.union(hubs2)),
                'modularity_change': networks[t2].metrics.modularity - networks[t1].metrics.modularity
            }
        
        return evolution_metrics


class NetworkDiscovery:
    """Main interface for network analysis with robust design patterns"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.analyzers = {
            'spatial_communication': SpatialCommunicationAnalyzer(config_path),
            'temporal_evolution': TemporalNetworkAnalyzer(config_path)
        }
    
    def discover_spatial_networks(self, analysis_results: List[Dict]) -> Dict:
        """Main entry point for network discovery"""
        # Group by condition and timepoint
        grouped_data = self._group_by_condition_timepoint(analysis_results)
        
        results = {
            'networks_by_condition': {},
            'temporal_networks': {},
            'comparative_analysis': {}
        }
        
        # Analyze by condition
        for condition, coloc_data in grouped_data.items():
            if condition != 'temporal':
                analyzer = self.analyzers['spatial_communication']
                network = analyzer.analyze(coloc_data, {'condition': condition})
                results['networks_by_condition'][condition] = network
        
        # Temporal analysis
        if 'temporal' in grouped_data:
            temporal_analyzer = self.analyzers['temporal_evolution']
            temporal_networks = temporal_analyzer.analyze(grouped_data['temporal'], {})
            results['temporal_networks'] = temporal_networks
            results['comparative_analysis']['evolution'] = temporal_analyzer.analyze_network_evolution(temporal_networks)
        
        return results
    
    def _group_by_condition_timepoint(self, analysis_results: List[Dict]) -> Dict:
        """Group analysis results by experimental condition and timepoint"""
        grouped = {}
        temporal_data = {}
        
        for result in analysis_results:
            metadata = result.get('metadata', {})
            colocalization = result.get('colocalization', {})
            
            condition = metadata.get('condition', 'unknown')
            timepoint = metadata.get('timepoint')
            
            # Group by condition
            if condition not in grouped:
                grouped[condition] = {}
            
            # Merge colocalization data
            for pair, data in colocalization.items():
                if pair not in grouped[condition]:
                    grouped[condition][pair] = []
                grouped[condition][pair].append(data.get('colocalization_score', 0))
            
            # Group temporal data
            if timepoint is not None:
                tp_key = f"D{timepoint}" if timepoint > 0 else "Sham"
                if tp_key not in temporal_data:
                    temporal_data[tp_key] = {}
                
                for pair, data in colocalization.items():
                    if pair not in temporal_data[tp_key]:
                        temporal_data[tp_key][pair] = []
                    temporal_data[tp_key][pair].append(data.get('colocalization_score', 0))
        
        # Average the scores
        for condition in grouped:
            for pair in grouped[condition]:
                scores = grouped[condition][pair]
                grouped[condition][pair] = {
                    'colocalization_score': np.mean(scores),
                    'score_std': np.std(scores),
                    'n_measurements': len(scores)
                }
        
        # Average temporal scores
        for tp in temporal_data:
            for pair in temporal_data[tp]:
                scores = temporal_data[tp][pair]
                temporal_data[tp][pair] = {
                    'colocalization_score': np.mean(scores),
                    'score_std': np.std(scores),
                    'n_measurements': len(scores)
                }
        
        if temporal_data:
            grouped['temporal'] = temporal_data
        
        return grouped
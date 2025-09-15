"""
Region Graph Networks for Tissue Organization Analysis
Constructs and analyzes graphs of tissue regions without cell segmentation
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity
try:
    import community as community_louvain
except ImportError:
    community_louvain = None
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RegionGraphResult:
    """Container for region graph analysis results"""
    graph: nx.Graph
    node_features: Dict[int, np.ndarray]  # Region ID -> expression features
    communities: Dict[int, int]  # Region ID -> community ID
    centrality: Dict[int, float]  # Region ID -> centrality score
    flow_matrix: np.ndarray  # Simulated protein flow between regions
    modularity: float
    n_communities: int
    hub_regions: List[int]


class RegionGraphBuilder:
    """Builds graphs from tissue regions (superpixels or tiles)"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize graph builder
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.tile_size = self.config.get('tile_size', 50)
        self.adjacency = self.config.get('adjacency', 'queen')  # 'queen' or 'rook'
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
    
    def build_from_superpixels(self, superpixel_result) -> nx.Graph:
        """
        Build graph from superpixel parcellation
        
        Args:
            superpixel_result: SuperpixelResult from tissue parcellation
            
        Returns:
            NetworkX graph with regions as nodes
        """
        G = nx.Graph()
        
        # Add nodes with expression features
        for seg_id in range(superpixel_result.n_segments):
            G.add_node(seg_id,
                      expression=superpixel_result.mean_expressions[seg_id],
                      centroid=superpixel_result.centroids[seg_id],
                      size=superpixel_result.sizes[seg_id])
        
        # Add edges based on adjacency
        for seg_id, neighbors in superpixel_result.adjacency.items():
            for neighbor_id in neighbors:
                if seg_id < neighbor_id and neighbor_id < superpixel_result.n_segments:  # Avoid duplicates and bounds
                    # Compute edge weight based on expression similarity
                    expr1 = superpixel_result.mean_expressions[seg_id]
                    expr2 = superpixel_result.mean_expressions[neighbor_id]
                    
                    similarity = self._compute_similarity(expr1, expr2)
                    
                    if similarity > self.similarity_threshold:
                        # Also consider spatial distance
                        spatial_dist = np.linalg.norm(
                            superpixel_result.centroids[seg_id] - 
                            superpixel_result.centroids[neighbor_id]
                        )
                        
                        G.add_edge(seg_id, neighbor_id,
                                 weight=similarity,
                                 spatial_distance=spatial_dist)
        
        return G
    
    def build_from_grid(self, coords: np.ndarray, 
                       values: np.ndarray) -> nx.Graph:
        """
        Build graph from regular grid tiles
        
        Args:
            coords: Spatial coordinates
            values: Expression values
            
        Returns:
            NetworkX graph with grid tiles as nodes
        """
        # Create grid
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        x_bins = np.arange(x_min, x_max + self.tile_size, self.tile_size)
        y_bins = np.arange(y_min, y_max + self.tile_size, self.tile_size)
        
        # Assign points to tiles
        x_idx = np.digitize(coords[:, 0], x_bins) - 1
        y_idx = np.digitize(coords[:, 1], y_bins) - 1
        
        n_x = len(x_bins) - 1
        n_y = len(y_bins) - 1
        
        # Create graph
        G = nx.Graph()
        
        # Compute tile features
        tile_features = {}
        tile_counts = {}
        
        for i in range(len(coords)):
            if 0 <= x_idx[i] < n_x and 0 <= y_idx[i] < n_y:
                tile_id = y_idx[i] * n_x + x_idx[i]
                
                if tile_id not in tile_features:
                    tile_features[tile_id] = []
                    tile_counts[tile_id] = 0
                
                tile_features[tile_id].append(values[i])
                tile_counts[tile_id] += 1
        
        # Add nodes
        for tile_id, features in tile_features.items():
            if len(features) > 0:
                mean_expression = np.mean(features, axis=0)
                y_tile = tile_id // n_x
                x_tile = tile_id % n_x
                
                centroid = [
                    x_bins[x_tile] + self.tile_size / 2,
                    y_bins[y_tile] + self.tile_size / 2
                ]
                
                G.add_node(tile_id,
                         expression=mean_expression,
                         centroid=centroid,
                         size=tile_counts[tile_id])
        
        # Add edges based on adjacency
        for tile_id in G.nodes():
            y_tile = tile_id // n_x
            x_tile = tile_id % n_x
            
            # Check neighbors
            neighbors = []
            
            if self.adjacency == 'queen':
                # 8-connectivity
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        neighbor_x = x_tile + dx
                        neighbor_y = y_tile + dy
                        
                        if 0 <= neighbor_x < n_x and 0 <= neighbor_y < n_y:
                            neighbor_id = neighbor_y * n_x + neighbor_x
                            if neighbor_id in G.nodes():
                                neighbors.append(neighbor_id)
            else:  # 'rook'
                # 4-connectivity
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor_x = x_tile + dx
                    neighbor_y = y_tile + dy
                    
                    if 0 <= neighbor_x < n_x and 0 <= neighbor_y < n_y:
                        neighbor_id = neighbor_y * n_x + neighbor_x
                        if neighbor_id in G.nodes():
                            neighbors.append(neighbor_id)
            
            # Add edges with weights
            for neighbor_id in neighbors:
                if tile_id < neighbor_id:  # Avoid duplicates
                    expr1 = G.nodes[tile_id]['expression']
                    expr2 = G.nodes[neighbor_id]['expression']
                    
                    similarity = self._compute_similarity(expr1, expr2)
                    
                    G.add_edge(tile_id, neighbor_id, weight=similarity)
        
        return G
    
    def _compute_similarity(self, expr1: np.ndarray, 
                          expr2: np.ndarray) -> float:
        """Compute similarity between expression profiles"""
        # Cosine similarity
        similarity = cosine_similarity(
            expr1.reshape(1, -1),
            expr2.reshape(1, -1)
        )[0, 0]
        
        # Ensure positive similarity
        return (similarity + 1) / 2


class RegionGraphAnalyzer:
    """Analyzes region graphs for tissue organization patterns"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize analyzer
        
        Args:
            config: Analysis configuration
        """
        self.config = config or {}
    
    def analyze(self, graph: nx.Graph) -> RegionGraphResult:
        """
        Comprehensive analysis of region graph
        
        Args:
            graph: Region graph from builder
            
        Returns:
            RegionGraphResult with analysis
        """
        # Extract node features
        node_features = {
            node: data['expression'] 
            for node, data in graph.nodes(data=True)
        }
        
        # Community detection
        communities, modularity = self._detect_communities(graph)
        
        # Centrality analysis
        centrality = self._compute_centrality(graph)
        
        # Identify hub regions
        hub_regions = self._identify_hubs(centrality)
        
        # Simulate protein flow
        flow_matrix = self._simulate_flow(graph)
        
        return RegionGraphResult(
            graph=graph,
            node_features=node_features,
            communities=communities,
            centrality=centrality,
            flow_matrix=flow_matrix,
            modularity=modularity,
            n_communities=len(set(communities.values())),
            hub_regions=hub_regions
        )
    
    def _detect_communities(self, graph: nx.Graph) -> Tuple[Dict[int, int], float]:
        """Detect communities using Louvain method or fallback"""
        # Ensure graph has edges
        if graph.number_of_edges() == 0:
            # All nodes in single community
            communities = {node: 0 for node in graph.nodes()}
            return communities, 0.0
        
        if community_louvain is not None:
            # Apply Louvain algorithm
            partition = community_louvain.best_partition(graph)
            modularity = community_louvain.modularity(partition, graph)
        else:
            # Fallback to NetworkX greedy modularity
            from networkx.algorithms.community import greedy_modularity_communities
            
            communities_list = list(greedy_modularity_communities(graph))
            partition = {}
            for i, community in enumerate(communities_list):
                for node in community:
                    partition[node] = i
            
            # Compute modularity
            modularity = nx.algorithms.community.modularity(graph, communities_list)
        
        return partition, modularity
    
    def _compute_centrality(self, graph: nx.Graph) -> Dict[int, float]:
        """Compute node centrality scores"""
        if graph.number_of_edges() == 0:
            # No edges, all nodes have equal centrality
            return {node: 1.0 / graph.number_of_nodes() 
                   for node in graph.nodes()}
        
        # Combine multiple centrality measures
        degree_cent = nx.degree_centrality(graph)
        
        # Check if graph is connected for betweenness
        if nx.is_connected(graph):
            between_cent = nx.betweenness_centrality(graph)
            close_cent = nx.closeness_centrality(graph)
        else:
            # Compute for each component
            between_cent = {}
            close_cent = {}
            
            for component in nx.connected_components(graph):
                subgraph = graph.subgraph(component)
                between_sub = nx.betweenness_centrality(subgraph)
                close_sub = nx.closeness_centrality(subgraph)
                
                between_cent.update(between_sub)
                close_cent.update(close_sub)
        
        # Weighted PageRank
        try:
            pagerank = nx.pagerank(graph, weight='weight')
        except:
            pagerank = {node: 1.0 / graph.number_of_nodes() 
                       for node in graph.nodes()}
        
        # Combine scores
        combined_centrality = {}
        for node in graph.nodes():
            combined_centrality[node] = (
                0.25 * degree_cent.get(node, 0) +
                0.25 * between_cent.get(node, 0) +
                0.25 * close_cent.get(node, 0) +
                0.25 * pagerank.get(node, 0)
            )
        
        return combined_centrality
    
    def _identify_hubs(self, centrality: Dict[int, float],
                      threshold_percentile: float = 90) -> List[int]:
        """Identify hub regions based on centrality"""
        if not centrality:
            return []
        
        scores = list(centrality.values())
        threshold = np.percentile(scores, threshold_percentile)
        
        hubs = [node for node, score in centrality.items() 
               if score >= threshold]
        
        return hubs
    
    def _simulate_flow(self, graph: nx.Graph) -> np.ndarray:
        """
        Simulate protein diffusion/signaling flow
        
        Uses random walk to model protein movement
        """
        n_nodes = graph.number_of_nodes()
        node_list = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        # Initialize flow matrix
        flow_matrix = np.zeros((n_nodes, n_nodes))
        
        if graph.number_of_edges() == 0:
            return flow_matrix
        
        # Create transition matrix based on edge weights
        for i, node_i in enumerate(node_list):
            neighbors = list(graph.neighbors(node_i))
            
            if len(neighbors) > 0:
                # Get edge weights
                weights = []
                for node_j in neighbors:
                    edge_data = graph.get_edge_data(node_i, node_j)
                    weight = edge_data.get('weight', 1.0) if edge_data else 1.0
                    weights.append(weight)
                
                # Normalize weights
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                # Fill flow matrix
                for j, (node_j, weight) in enumerate(zip(neighbors, weights)):
                    j_idx = node_to_idx[node_j]
                    flow_matrix[i, j_idx] = weight
        
        # Add self-loops for stability
        diagonal_weight = 0.1
        flow_matrix = flow_matrix * (1 - diagonal_weight)
        np.fill_diagonal(flow_matrix, diagonal_weight)
        
        # Normalize rows
        row_sums = flow_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        flow_matrix = flow_matrix / row_sums[:, np.newaxis]
        
        return flow_matrix


class TissueMessagePassing:
    """Message passing on region graphs for signaling simulation"""
    
    def __init__(self, graph: nx.Graph, node_features: Dict[int, np.ndarray]):
        """
        Initialize message passing
        
        Args:
            graph: Region graph
            node_features: Initial node features (expression)
        """
        self.graph = graph
        self.node_features = node_features
        self.messages = {}
        self.updated_features = node_features.copy()
    
    def propagate(self, n_iterations: int = 10,
                 damping: float = 0.5) -> Dict[int, np.ndarray]:
        """
        Propagate messages through graph
        
        Args:
            n_iterations: Number of message passing iterations
            damping: Damping factor for feature updates
            
        Returns:
            Updated node features after message passing
        """
        for iteration in range(n_iterations):
            new_features = {}
            
            for node in self.graph.nodes():
                # Collect messages from neighbors
                neighbor_messages = []
                
                for neighbor in self.graph.neighbors(node):
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    weight = edge_data.get('weight', 1.0) if edge_data else 1.0
                    
                    # Message is weighted neighbor feature
                    message = self.updated_features[neighbor] * weight
                    neighbor_messages.append(message)
                
                # Aggregate messages
                if neighbor_messages:
                    aggregated = np.mean(neighbor_messages, axis=0)
                else:
                    aggregated = np.zeros_like(self.node_features[node])
                
                # Update features with damping
                new_features[node] = (
                    damping * self.node_features[node] +
                    (1 - damping) * aggregated
                )
            
            self.updated_features = new_features
        
        return self.updated_features
    
    def identify_signaling_paths(self, source: int, 
                                target: int) -> List[List[int]]:
        """
        Find signaling paths between regions
        
        Args:
            source: Source region ID
            target: Target region ID
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=5
            ))
            
            # Sort by total weight (higher is better)
            path_weights = []
            for path in paths:
                weight = 0
                for i in range(len(path) - 1):
                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                    weight += edge_data.get('weight', 1.0) if edge_data else 1.0
                path_weights.append(weight)
            
            # Sort paths by weight
            sorted_paths = [path for _, path in sorted(
                zip(path_weights, paths), reverse=True
            )]
            
            return sorted_paths[:5]  # Return top 5 paths
            
        except nx.NetworkXNoPath:
            return []


def analyze_tissue_organization(coords: np.ndarray,
                               values: np.ndarray,
                               method: str = 'superpixel') -> RegionGraphResult:
    """
    Complete tissue organization analysis pipeline
    
    Args:
        coords: Spatial coordinates
        values: Expression values
        method: 'superpixel' or 'grid'
        
    Returns:
        RegionGraphResult
    """
    builder = RegionGraphBuilder()
    
    if method == 'superpixel':
        # First create superpixels
        from src.analysis.superpixel import create_tissue_parcellation
        superpixel_result = create_tissue_parcellation(coords, values)
        graph = builder.build_from_superpixels(superpixel_result)
    else:
        graph = builder.build_from_grid(coords, values)
    
    # Analyze graph
    analyzer = RegionGraphAnalyzer()
    result = analyzer.analyze(graph)
    
    return result
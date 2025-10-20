"""
Graph-Based Clustering Baseline for IMC Pipeline

Provides comprehensive graph-based community detection algorithms as baseline
comparison against the existing spatial clustering approach. Focuses purely on
protein expression similarity graphs without spatial constraints, enabling
direct comparison of graph-based vs spatial-aware clustering methodologies.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, csc_matrix
import warnings

# Optional dependencies with graceful fallbacks
try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    warnings.warn("Leiden algorithm not available. Install with: pip install leidenalg python-igraph")

try:
    from sklearn.cluster import DBSCAN
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not available. Install with: pip install hdbscan")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Install with: pip install networkx")


class GraphBuilder:
    """
    Construct protein expression similarity graphs using various methods.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def build_knn_graph(
        self,
        feature_matrix: np.ndarray,
        k_neighbors: int = 15,
        metric: str = 'euclidean',
        include_self: bool = False,
        mode: str = 'connectivity'
    ) -> csr_matrix:
        """
        Build k-nearest neighbors graph from protein expression features.
        
        Args:
            feature_matrix: N x P feature matrix (samples x proteins)
            k_neighbors: Number of nearest neighbors
            metric: Distance metric ('euclidean', 'cosine', 'manhattan')
            include_self: Whether to include self-connections
            mode: 'connectivity', 'distance', or 'hybrid'
            
        Returns:
            Sparse adjacency matrix
        """
        n_samples = feature_matrix.shape[0]
        
        if k_neighbors >= n_samples:
            k_neighbors = max(1, n_samples - 1)
            warnings.warn(f"k_neighbors reduced to {k_neighbors} (n_samples - 1)")
        
        # Build kNN graph
        if mode == 'hybrid':
            # Use distance-weighted connectivity
            knn_graph = kneighbors_graph(
                feature_matrix, 
                n_neighbors=k_neighbors,
                mode='distance',
                metric=metric,
                include_self=include_self
            )
            # Convert distances to similarities: exp(-distance^2)
            knn_graph.data = np.exp(-knn_graph.data**2)
        else:
            knn_graph = kneighbors_graph(
                feature_matrix,
                n_neighbors=k_neighbors,
                mode=mode,
                metric=metric,
                include_self=include_self
            )
        
        # Make symmetric for undirected graph
        knn_graph = (knn_graph + knn_graph.T) / 2
        
        return knn_graph.tocsr()
    
    def build_radius_graph(
        self,
        feature_matrix: np.ndarray,
        radius: float = 1.0,
        metric: str = 'euclidean'
    ) -> csr_matrix:
        """
        Build radius neighbors graph (ε-neighborhood graph).
        
        Args:
            feature_matrix: N x P feature matrix
            radius: Neighborhood radius
            metric: Distance metric
            
        Returns:
            Sparse adjacency matrix
        """
        radius_graph = radius_neighbors_graph(
            feature_matrix,
            radius=radius,
            mode='connectivity',
            metric=metric,
            include_self=False
        )
        
        # Make symmetric
        radius_graph = (radius_graph + radius_graph.T) / 2
        
        return radius_graph.tocsr()
    
    def build_correlation_graph(
        self,
        feature_matrix: np.ndarray,
        threshold: float = 0.5,
        method: str = 'pearson'
    ) -> csr_matrix:
        """
        Build graph based on protein expression correlations.
        
        Args:
            feature_matrix: N x P feature matrix
            threshold: Correlation threshold for edge creation
            method: 'pearson' or 'spearman'
            
        Returns:
            Sparse adjacency matrix
        """
        if method == 'pearson':
            correlation_matrix = np.corrcoef(feature_matrix)
        elif method == 'spearman':
            from scipy.stats import spearmanr
            correlation_matrix, _ = spearmanr(feature_matrix, axis=1)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Apply threshold
        adjacency = np.abs(correlation_matrix) >= threshold
        np.fill_diagonal(adjacency, False)  # Remove self-connections
        
        return csr_matrix(adjacency.astype(float))
    
    def build_coexpression_graph(
        self,
        feature_matrix: np.ndarray,
        protein_names: List[str],
        coexpression_threshold: float = 0.7,
        local_neighborhood_size: int = 5
    ) -> csr_matrix:
        """
        Build graph based on protein co-expression patterns.
        
        Connects samples with similar protein co-expression profiles.
        
        Args:
            feature_matrix: N x P feature matrix
            protein_names: List of protein names
            coexpression_threshold: Threshold for co-expression similarity
            local_neighborhood_size: Size of local neighborhoods for comparison
            
        Returns:
            Sparse adjacency matrix
        """
        n_samples, n_proteins = feature_matrix.shape
        adjacency = np.zeros((n_samples, n_samples))
        
        # For each sample, find local protein co-expression pattern
        for i in range(n_samples):
            # Get k-nearest neighbors in expression space
            nn = NearestNeighbors(n_neighbors=min(local_neighborhood_size, n_samples-1))
            nn.fit(feature_matrix)
            _, neighbor_indices = nn.kneighbors([feature_matrix[i]])
            
            # Compute co-expression pattern in local neighborhood
            local_features = feature_matrix[neighbor_indices[0]]
            local_coexp = np.corrcoef(local_features.T)  # Protein x Protein correlation
            
            # Compare against global co-expression patterns
            for j in range(i+1, n_samples):
                # Get j's local neighborhood
                _, j_neighbor_indices = nn.kneighbors([feature_matrix[j]])
                j_local_features = feature_matrix[j_neighbor_indices[0]]
                j_local_coexp = np.corrcoef(j_local_features.T)
                
                # Similarity between co-expression patterns
                coexp_similarity = np.corrcoef(
                    local_coexp.ravel(), 
                    j_local_coexp.ravel()
                )[0, 1]
                
                if not np.isnan(coexp_similarity) and coexp_similarity >= coexpression_threshold:
                    adjacency[i, j] = adjacency[j, i] = coexp_similarity
        
        return csr_matrix(adjacency)
    
    def add_spatial_weights(
        self,
        graph: csr_matrix,
        spatial_coords: np.ndarray,
        spatial_weight: float = 0.3,
        spatial_sigma: float = 20.0
    ) -> csr_matrix:
        """
        Add spatial proximity weights to existing graph.
        
        Args:
            graph: Existing graph adjacency matrix
            spatial_coords: N x 2 spatial coordinates
            spatial_weight: Weight for spatial component (0=no spatial, 1=only spatial)
            spatial_sigma: Spatial Gaussian kernel bandwidth
            
        Returns:
            Spatially-weighted graph
        """
        if spatial_coords is None or spatial_weight == 0:
            return graph
        
        n_samples = graph.shape[0]
        
        # Compute spatial distances
        spatial_distances = pdist(spatial_coords, metric='euclidean')
        spatial_dist_matrix = squareform(spatial_distances)
        
        # Convert to similarities using Gaussian kernel
        spatial_similarities = np.exp(-spatial_dist_matrix**2 / (2 * spatial_sigma**2))
        np.fill_diagonal(spatial_similarities, 0)  # Remove self-connections
        
        # Combine expression and spatial graphs
        combined_graph = (1 - spatial_weight) * graph.toarray() + spatial_weight * spatial_similarities
        
        return csr_matrix(combined_graph)


class GraphCommunityDetection:
    """
    Community detection algorithms for protein expression graphs.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def leiden_clustering(
        self,
        graph: csr_matrix,
        resolution: float = 1.0,
        n_iterations: int = -1,
        randomness: float = 0.001
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Leiden algorithm for community detection.
        
        Args:
            graph: Sparse adjacency matrix
            resolution: Resolution parameter (higher = more communities)
            n_iterations: Maximum iterations (-1 = until convergence)
            randomness: Randomness parameter
            
        Returns:
            Tuple of (cluster_labels, algorithm_info)
        """
        if not LEIDEN_AVAILABLE:
            raise ImportError("Leiden algorithm requires leidenalg and python-igraph")
        
        # Convert to igraph
        sources, targets = graph.nonzero()
        weights = graph.data
        edges = list(zip(sources, targets))
        
        g = ig.Graph(edges=edges, directed=False)
        if len(weights) > 0:
            g.es['weight'] = weights
        
        # Run Leiden algorithm
        if len(weights) > 0:
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                n_iterations=n_iterations,
                seed=self.random_state
            )
        else:
            partition = leidenalg.find_partition(
                g,
                leidenalg.CPMVertexPartition,
                resolution_parameter=resolution,
                n_iterations=n_iterations,
                seed=self.random_state
            )
        
        labels = np.array(partition.membership)
        
        info = {
            'algorithm': 'leiden',
            'resolution': resolution,
            'modularity': partition.modularity,
            'n_communities': len(set(labels)),
            'n_iterations_actual': partition.n_iterations if hasattr(partition, 'n_iterations') else -1
        }
        
        return labels, info
    
    def louvain_clustering(
        self,
        graph: csr_matrix,
        resolution: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Louvain algorithm for community detection.
        
        Args:
            graph: Sparse adjacency matrix
            resolution: Resolution parameter
            
        Returns:
            Tuple of (cluster_labels, algorithm_info)
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("Louvain algorithm requires networkx")
        
        # Convert to NetworkX graph
        G = nx.from_scipy_sparse_array(graph)
        
        # Run Louvain algorithm
        try:
            import community.community_louvain as community_louvain
            partition = community_louvain.best_partition(G, resolution=resolution, random_state=self.random_state)
            labels = np.array([partition[i] for i in range(len(partition))])
            modularity = community_louvain.modularity(partition, G)
        except ImportError:
            # Fallback to networkx communities
            communities = nx.community.greedy_modularity_communities(G, resolution=resolution)
            labels = np.zeros(graph.shape[0], dtype=int)
            for i, community in enumerate(communities):
                for node in community:
                    labels[node] = i
            modularity = nx.community.modularity(G, communities)
        
        info = {
            'algorithm': 'louvain',
            'resolution': resolution,
            'modularity': modularity,
            'n_communities': len(set(labels))
        }
        
        return labels, info
    
    def spectral_clustering(
        self,
        graph: csr_matrix,
        n_clusters: int = 8,
        eigen_solver: str = 'arpack'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Spectral clustering on graph Laplacian.
        
        Args:
            graph: Sparse adjacency matrix
            n_clusters: Number of clusters
            eigen_solver: Eigenvalue solver ('arpack', 'lobpcg', 'amg')
            
        Returns:
            Tuple of (cluster_labels, algorithm_info)
        """
        from sklearn.cluster import SpectralClustering
        
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            eigen_solver=eigen_solver,
            random_state=self.random_state
        )
        
        labels = spectral.fit_predict(graph.toarray())
        
        info = {
            'algorithm': 'spectral',
            'n_clusters': n_clusters,
            'eigen_solver': eigen_solver
        }
        
        return labels, info
    
    def graph_dbscan(
        self,
        graph: csr_matrix,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        DBSCAN clustering using graph distances.
        
        Args:
            graph: Sparse adjacency matrix
            eps: Maximum distance between samples in same neighborhood
            min_samples: Minimum samples in neighborhood to form core point
            
        Returns:
            Tuple of (cluster_labels, algorithm_info)
        """
        # Convert graph to distance matrix
        # For connected components, use shortest path distances
        from scipy.sparse.csgraph import shortest_path
        
        # Convert similarities to distances (if needed)
        distance_matrix = shortest_path(graph, directed=False)
        
        # Replace infinite distances with large value
        distance_matrix[np.isinf(distance_matrix)] = 1000
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
        
        info = {
            'algorithm': 'graph_dbscan',
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'n_noise': np.sum(labels == -1)
        }
        
        return labels, info


class GraphClusteringBaseline:
    """
    Complete graph-based clustering baseline for IMC analysis.
    
    Provides pure graph-based clustering without spatial constraints
    for comparison against spatial-aware methods.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.graph_builder = GraphBuilder(random_state)
        self.community_detector = GraphCommunityDetection(random_state)
        
    def cluster_protein_expression(
        self,
        feature_matrix: np.ndarray,
        protein_names: List[str],
        spatial_coords: Optional[np.ndarray] = None,
        graph_method: str = 'knn',
        clustering_method: str = 'leiden',
        graph_params: Optional[Dict] = None,
        clustering_params: Optional[Dict] = None,
        standardize_features: bool = True
    ) -> Dict[str, Any]:
        """
        Perform graph-based clustering on protein expression data.
        
        Args:
            feature_matrix: N x P feature matrix (samples x proteins)
            protein_names: List of protein names
            spatial_coords: Optional N x 2 spatial coordinates (for spatial weighting)
            graph_method: Graph construction method ('knn', 'radius', 'correlation', 'coexpression')
            clustering_method: Clustering algorithm ('leiden', 'louvain', 'spectral', 'graph_dbscan')
            graph_params: Parameters for graph construction
            clustering_params: Parameters for clustering algorithm
            standardize_features: Whether to standardize features
            
        Returns:
            Dictionary with clustering results and metadata
        """
        if feature_matrix.shape[0] == 0:
            return self._empty_results()
        
        # Set default parameters
        graph_params = graph_params or {}
        clustering_params = clustering_params or {}
        
        # Standardize features if requested
        if standardize_features:
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        else:
            feature_matrix_scaled = feature_matrix.copy()
            scaler = None
        
        # Build graph
        graph = self._build_graph(
            feature_matrix_scaled, graph_method, spatial_coords, graph_params
        )
        
        # Perform clustering
        cluster_labels, clustering_info = self._perform_clustering(
            graph, clustering_method, clustering_params
        )
        
        # Compute graph-based metrics
        graph_metrics = self._compute_graph_metrics(graph, cluster_labels)
        
        # Compute cluster centroids
        cluster_centroids = self._compute_centroids(
            feature_matrix_scaled, cluster_labels, protein_names
        )
        
        # Clustering quality metrics
        quality_metrics = self._compute_quality_metrics(
            feature_matrix_scaled, cluster_labels, spatial_coords
        )
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centroids': cluster_centroids,
            'graph_adjacency': graph,
            'graph_metrics': graph_metrics,
            'clustering_info': clustering_info,
            'quality_metrics': quality_metrics,
            'scaler': scaler,
            'parameters': {
                'graph_method': graph_method,
                'clustering_method': clustering_method,
                'graph_params': graph_params,
                'clustering_params': clustering_params,
                'standardize_features': standardize_features
            },
            'metadata': {
                'n_samples': feature_matrix.shape[0],
                'n_proteins': feature_matrix.shape[1],
                'protein_names': protein_names,
                'n_clusters': len(np.unique(cluster_labels[cluster_labels >= 0])),
                'n_noise': np.sum(cluster_labels == -1) if -1 in cluster_labels else 0
            }
        }
    
    def _build_graph(
        self,
        feature_matrix: np.ndarray,
        graph_method: str,
        spatial_coords: Optional[np.ndarray],
        graph_params: Dict
    ) -> csr_matrix:
        """Build graph using specified method."""
        if graph_method == 'knn':
            graph = self.graph_builder.build_knn_graph(
                feature_matrix,
                k_neighbors=graph_params.get('k_neighbors', 15),
                metric=graph_params.get('metric', 'euclidean'),
                mode=graph_params.get('mode', 'connectivity')
            )
        elif graph_method == 'radius':
            graph = self.graph_builder.build_radius_graph(
                feature_matrix,
                radius=graph_params.get('radius', 1.0),
                metric=graph_params.get('metric', 'euclidean')
            )
        elif graph_method == 'correlation':
            graph = self.graph_builder.build_correlation_graph(
                feature_matrix,
                threshold=graph_params.get('threshold', 0.5),
                method=graph_params.get('method', 'pearson')
            )
        elif graph_method == 'coexpression':
            graph = self.graph_builder.build_coexpression_graph(
                feature_matrix,
                protein_names=graph_params.get('protein_names', []),
                coexpression_threshold=graph_params.get('coexpression_threshold', 0.7),
                local_neighborhood_size=graph_params.get('local_neighborhood_size', 5)
            )
        else:
            raise ValueError(f"Unknown graph method: {graph_method}")
        
        # Add spatial weights if requested
        spatial_weight = graph_params.get('spatial_weight', 0.0)
        if spatial_weight > 0 and spatial_coords is not None:
            graph = self.graph_builder.add_spatial_weights(
                graph, spatial_coords, spatial_weight,
                spatial_sigma=graph_params.get('spatial_sigma', 20.0)
            )
        
        return graph
    
    def _perform_clustering(
        self,
        graph: csr_matrix,
        clustering_method: str,
        clustering_params: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """Perform clustering using specified method."""
        if clustering_method == 'leiden':
            return self.community_detector.leiden_clustering(
                graph,
                resolution=clustering_params.get('resolution', 1.0),
                n_iterations=clustering_params.get('n_iterations', -1)
            )
        elif clustering_method == 'louvain':
            return self.community_detector.louvain_clustering(
                graph,
                resolution=clustering_params.get('resolution', 1.0)
            )
        elif clustering_method == 'spectral':
            return self.community_detector.spectral_clustering(
                graph,
                n_clusters=clustering_params.get('n_clusters', 8),
                eigen_solver=clustering_params.get('eigen_solver', 'arpack')
            )
        elif clustering_method == 'graph_dbscan':
            return self.community_detector.graph_dbscan(
                graph,
                eps=clustering_params.get('eps', 0.5),
                min_samples=clustering_params.get('min_samples', 5)
            )
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
    
    def _compute_graph_metrics(
        self,
        graph: csr_matrix,
        cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute graph-based clustering quality metrics."""
        metrics = {}
        
        # Graph connectivity metrics
        n_edges = graph.nnz
        n_nodes = graph.shape[0]
        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        
        metrics['graph_density'] = density
        metrics['n_edges'] = n_edges
        metrics['n_nodes'] = n_nodes
        
        # Modularity (if available)
        try:
            if NETWORKX_AVAILABLE:
                G = nx.from_scipy_sparse_array(graph)
                unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
                if len(unique_labels) > 1:
                    communities = [set(np.where(cluster_labels == label)[0]) for label in unique_labels]
                    modularity = nx.community.modularity(G, communities)
                    metrics['modularity'] = modularity
        except Exception:
            pass
        
        # Conductance for each cluster
        conductances = []
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        
        for label in unique_labels:
            cluster_nodes = np.where(cluster_labels == label)[0]
            if len(cluster_nodes) > 1:
                # Internal edges
                subgraph = graph[cluster_nodes][:, cluster_nodes]
                internal_edges = subgraph.nnz / 2  # Undirected graph
                
                # External edges
                cluster_degree = graph[cluster_nodes].sum()
                external_edges = cluster_degree - 2 * internal_edges
                
                # Conductance = external_edges / min(internal_edges, total_edges - internal_edges)
                total_edges = graph.nnz / 2
                conductance = external_edges / min(internal_edges + external_edges, 
                                                 total_edges - internal_edges) if internal_edges + external_edges > 0 else 1.0
                conductances.append(conductance)
        
        if conductances:
            metrics['mean_conductance'] = np.mean(conductances)
            metrics['std_conductance'] = np.std(conductances)
        
        return metrics
    
    def _compute_centroids(
        self,
        feature_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        protein_names: List[str]
    ) -> Dict[int, Dict[str, float]]:
        """Compute cluster centroids."""
        centroids = {}
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_features = feature_matrix[mask]
            
            centroid = {}
            for i, protein_name in enumerate(protein_names):
                centroid[protein_name] = float(np.mean(cluster_features[:, i]))
            
            centroids[int(label)] = centroid
        
        return centroids
    
    def _compute_quality_metrics(
        self,
        feature_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        spatial_coords: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Compute clustering quality metrics."""
        metrics = {}
        
        # Silhouette score
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        if len(unique_labels) > 1 and len(unique_labels) < len(cluster_labels):
            try:
                # Only use non-noise points for silhouette score
                non_noise_mask = cluster_labels >= 0
                if np.sum(non_noise_mask) > 1:
                    silhouette = silhouette_score(
                        feature_matrix[non_noise_mask],
                        cluster_labels[non_noise_mask]
                    )
                    metrics['silhouette_score'] = silhouette
            except Exception:
                pass
        
        # Spatial coherence (if spatial coordinates available)
        if spatial_coords is not None:
            try:
                from .spatial_clustering import compute_spatial_coherence
                spatial_coherence = compute_spatial_coherence(cluster_labels, spatial_coords)
                metrics['spatial_coherence'] = spatial_coherence
            except Exception:
                pass
        
        # Cluster size statistics
        unique_labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
        if len(counts) > 0:
            metrics['mean_cluster_size'] = float(np.mean(counts))
            metrics['std_cluster_size'] = float(np.std(counts))
            metrics['min_cluster_size'] = int(np.min(counts))
            metrics['max_cluster_size'] = int(np.max(counts))
        
        return metrics
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results for edge cases."""
        return {
            'cluster_labels': np.array([]),
            'cluster_centroids': {},
            'graph_adjacency': csr_matrix((0, 0)),
            'graph_metrics': {},
            'clustering_info': {},
            'quality_metrics': {},
            'scaler': None,
            'parameters': {},
            'metadata': {
                'n_samples': 0,
                'n_proteins': 0,
                'protein_names': [],
                'n_clusters': 0,
                'n_noise': 0
            }
        }
    
    def parameter_optimization(
        self,
        feature_matrix: np.ndarray,
        protein_names: List[str],
        spatial_coords: Optional[np.ndarray] = None,
        graph_method: str = 'knn',
        clustering_method: str = 'leiden',
        param_ranges: Optional[Dict] = None,
        n_trials: int = 20,
        optimization_metric: str = 'silhouette'
    ) -> Dict[str, Any]:
        """
        Optimize graph clustering parameters using grid search or random search.
        
        Args:
            feature_matrix: N x P feature matrix
            protein_names: List of protein names
            spatial_coords: Optional spatial coordinates
            graph_method: Graph construction method
            clustering_method: Clustering algorithm
            param_ranges: Parameter ranges for optimization
            n_trials: Number of parameter combinations to try
            optimization_metric: Metric to optimize ('silhouette', 'modularity', 'spatial_coherence')
            
        Returns:
            Dictionary with optimization results
        """
        if param_ranges is None:
            param_ranges = self._get_default_param_ranges(graph_method, clustering_method)
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges, n_trials)
        
        best_score = -np.inf
        best_params = None
        best_results = None
        all_results = []
        
        for i, (graph_params, clustering_params) in enumerate(param_combinations):
            try:
                results = self.cluster_protein_expression(
                    feature_matrix, protein_names, spatial_coords,
                    graph_method, clustering_method,
                    graph_params, clustering_params
                )
                
                # Extract optimization metric
                if optimization_metric == 'silhouette':
                    score = results['quality_metrics'].get('silhouette_score', -1)
                elif optimization_metric == 'modularity':
                    score = results['graph_metrics'].get('modularity', -1)
                elif optimization_metric == 'spatial_coherence':
                    score = results['quality_metrics'].get('spatial_coherence', -1)
                else:
                    score = -1
                
                if score > best_score:
                    best_score = score
                    best_params = (graph_params, clustering_params)
                    best_results = results
                
                all_results.append({
                    'graph_params': graph_params,
                    'clustering_params': clustering_params,
                    'score': score,
                    'n_clusters': results['metadata']['n_clusters']
                })
                
            except Exception as e:
                warnings.warn(f"Parameter combination {i} failed: {e}")
                continue
        
        return {
            'best_score': best_score,
            'best_params': best_params,
            'best_results': best_results,
            'all_results': all_results,
            'optimization_metric': optimization_metric
        }
    
    def _get_default_param_ranges(self, graph_method: str, clustering_method: str) -> Dict:
        """Get default parameter ranges for optimization."""
        param_ranges = {}
        
        # Graph construction parameters
        if graph_method == 'knn':
            param_ranges['graph'] = {
                'k_neighbors': [5, 10, 15, 20, 30],
                'metric': ['euclidean', 'cosine'],
                'spatial_weight': [0.0, 0.1, 0.3, 0.5]
            }
        elif graph_method == 'radius':
            param_ranges['graph'] = {
                'radius': [0.5, 1.0, 1.5, 2.0],
                'metric': ['euclidean', 'cosine'],
                'spatial_weight': [0.0, 0.1, 0.3, 0.5]
            }
        elif graph_method == 'correlation':
            param_ranges['graph'] = {
                'threshold': [0.3, 0.5, 0.7, 0.8],
                'method': ['pearson', 'spearman']
            }
        
        # Clustering parameters
        if clustering_method in ['leiden', 'louvain']:
            param_ranges['clustering'] = {
                'resolution': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
            }
        elif clustering_method == 'spectral':
            param_ranges['clustering'] = {
                'n_clusters': [3, 5, 8, 10, 12, 15]
            }
        elif clustering_method == 'graph_dbscan':
            param_ranges['clustering'] = {
                'eps': [0.3, 0.5, 0.8, 1.0],
                'min_samples': [3, 5, 8, 10]
            }
        
        return param_ranges
    
    def _generate_param_combinations(self, param_ranges: Dict, n_trials: int) -> List[Tuple[Dict, Dict]]:
        """Generate parameter combinations for optimization."""
        import itertools
        
        graph_ranges = param_ranges.get('graph', {})
        clustering_ranges = param_ranges.get('clustering', {})
        
        # Generate all combinations
        graph_combinations = []
        if graph_ranges:
            keys = list(graph_ranges.keys())
            values = list(graph_ranges.values())
            for combination in itertools.product(*values):
                graph_combinations.append(dict(zip(keys, combination)))
        else:
            graph_combinations = [{}]
        
        clustering_combinations = []
        if clustering_ranges:
            keys = list(clustering_ranges.keys())
            values = list(clustering_ranges.values())
            for combination in itertools.product(*values):
                clustering_combinations.append(dict(zip(keys, combination)))
        else:
            clustering_combinations = [{}]
        
        # Combine graph and clustering parameters
        all_combinations = []
        for graph_params in graph_combinations:
            for clustering_params in clustering_combinations:
                all_combinations.append((graph_params, clustering_params))
        
        # Limit to n_trials
        if len(all_combinations) > n_trials:
            indices = np.random.choice(len(all_combinations), n_trials, replace=False)
            all_combinations = [all_combinations[i] for i in indices]
        
        return all_combinations
    
    def compare_with_spatial_clustering(
        self,
        feature_matrix: np.ndarray,
        protein_names: List[str],
        spatial_coords: np.ndarray,
        spatial_clustering_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare graph-based clustering with spatial clustering results.
        
        Args:
            feature_matrix: N x P feature matrix
            protein_names: List of protein names
            spatial_coords: N x 2 spatial coordinates
            spatial_clustering_results: Results from spatial clustering
            
        Returns:
            Comparison metrics and analysis
        """
        # Run graph-based clustering
        graph_results = self.cluster_protein_expression(
            feature_matrix, protein_names, spatial_coords,
            graph_method='knn', clustering_method='leiden'
        )
        
        spatial_labels = spatial_clustering_results.get('cluster_labels', np.array([]))
        graph_labels = graph_results['cluster_labels']
        
        comparison = {}
        
        # Agreement metrics
        if len(spatial_labels) == len(graph_labels) and len(spatial_labels) > 0:
            # Remove noise points for comparison
            valid_mask = (spatial_labels >= 0) & (graph_labels >= 0)
            if np.sum(valid_mask) > 0:
                ari = adjusted_rand_score(spatial_labels[valid_mask], graph_labels[valid_mask])
                nmi = normalized_mutual_info_score(spatial_labels[valid_mask], graph_labels[valid_mask])
                
                comparison['adjusted_rand_index'] = ari
                comparison['normalized_mutual_info'] = nmi
        
        # Cluster count comparison
        n_spatial_clusters = len(np.unique(spatial_labels[spatial_labels >= 0]))
        n_graph_clusters = len(np.unique(graph_labels[graph_labels >= 0]))
        
        comparison['n_spatial_clusters'] = n_spatial_clusters
        comparison['n_graph_clusters'] = n_graph_clusters
        comparison['cluster_count_ratio'] = n_graph_clusters / max(n_spatial_clusters, 1)
        
        # Quality metrics comparison
        spatial_quality = spatial_clustering_results.get('quality_metrics', {})
        graph_quality = graph_results['quality_metrics']
        
        comparison['spatial_silhouette'] = spatial_quality.get('silhouette_score', np.nan)
        comparison['graph_silhouette'] = graph_quality.get('silhouette_score', np.nan)
        comparison['spatial_coherence'] = spatial_quality.get('spatial_coherence', np.nan)
        comparison['graph_spatial_coherence'] = graph_quality.get('spatial_coherence', np.nan)
        
        # Store full results for further analysis
        comparison['graph_results'] = graph_results
        comparison['spatial_results'] = spatial_clustering_results
        
        return comparison


def create_graph_clustering_baseline(
    feature_matrix: np.ndarray,
    protein_names: List[str],
    spatial_coords: Optional[np.ndarray] = None,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Factory function to create graph-based clustering baseline results.
    
    Provides a simple interface for comparison with existing spatial clustering.
    
    Args:
        feature_matrix: N x P feature matrix
        protein_names: List of protein names
        spatial_coords: Optional spatial coordinates
        config: Optional configuration dictionary
        
    Returns:
        Graph clustering results in format compatible with spatial clustering
    """
    baseline = GraphClusteringBaseline()
    
    # Extract parameters from config if provided
    if config:
        graph_method = config.get('graph_method', 'knn')
        clustering_method = config.get('clustering_method', 'leiden')
        graph_params = config.get('graph_params', {})
        clustering_params = config.get('clustering_params', {})
    else:
        graph_method = 'knn'
        clustering_method = 'leiden'
        graph_params = {}
        clustering_params = {}
    
    # Run clustering
    results = baseline.cluster_protein_expression(
        feature_matrix, protein_names, spatial_coords,
        graph_method, clustering_method,
        graph_params, clustering_params
    )
    
    # Reformat for compatibility with spatial clustering interface
    return {
        'cluster_labels': results['cluster_labels'],
        'cluster_centroids': results['cluster_centroids'],
        'clustering_info': results['clustering_info'],
        'quality_metrics': results['quality_metrics'],
        'graph_metrics': results['graph_metrics'],
        'method': f"graph_{clustering_method}",
        'parameters': results['parameters'],
        'metadata': results['metadata']
    }
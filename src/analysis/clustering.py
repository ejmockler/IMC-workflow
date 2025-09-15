"""
Advanced Clustering Methods for Spatial Analysis
Provides multiple state-of-the-art clustering algorithms via Strategy Pattern
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ClusteringResult:
    """Encapsulates clustering results with metadata"""
    labels: np.ndarray
    n_clusters: int
    algorithm: str
    parameters: Dict[str, Any]
    converged: bool = True
    iterations: Optional[int] = None
    inertia: Optional[float] = None


class SpatialClusterer(ABC):
    """Abstract base for clustering strategies"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @abstractmethod
    def fit_predict(self, data: np.ndarray, 
                   n_clusters: Optional[int] = None,
                   **kwargs) -> ClusteringResult:
        """
        Cluster data and return results
        
        Args:
            data: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters (optional for some algorithms)
            **kwargs: Algorithm-specific parameters
            
        Returns:
            ClusteringResult with labels and metadata
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return clusterer name for reporting"""
        pass
    
    def estimate_n_clusters(self, data: np.ndarray) -> int:
        """
        Estimate optimal number of clusters if not provided
        Default implementation for algorithms that require n_clusters
        """
        # Simple heuristic: sqrt(n/2)
        n_samples = len(data)
        estimated = int(np.sqrt(n_samples / 2))
        return max(2, min(estimated, 100))  # Reasonable bounds


class KMeansClusterer(SpatialClusterer):
    """
    Wrapper around sklearn KMeans (current baseline)
    Fast, deterministic, requires n_clusters
    """
    
    def __init__(self, n_init: int = 10, max_iter: int = 300, 
                 random_state: int = 42):
        super().__init__()
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
    
    def name(self) -> str:
        return "kmeans"
    
    def fit_predict(self, data: np.ndarray, 
                   n_clusters: Optional[int] = None,
                   **kwargs) -> ClusteringResult:
        from sklearn.cluster import KMeans
        
        if n_clusters is None:
            n_clusters = self.estimate_n_clusters(data)
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        labels = kmeans.fit_predict(data)
        
        return ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            algorithm=self.name(),
            parameters={
                'n_init': self.n_init,
                'max_iter': self.max_iter
            },
            converged=kmeans.n_iter_ < self.max_iter,
            iterations=int(kmeans.n_iter_),
            inertia=float(kmeans.inertia_)
        )


class MiniBatchKMeansClusterer(SpatialClusterer):
    """
    MiniBatch KMeans for large datasets
    3-10x faster than standard KMeans with slight accuracy tradeoff
    """
    
    def __init__(self, batch_size: int = 1000, n_init: int = 3,
                 max_iter: int = 100, random_state: int = 42):
        super().__init__()
        self.batch_size = batch_size
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
    
    def name(self) -> str:
        return "minibatch_kmeans"
    
    def fit_predict(self, data: np.ndarray,
                   n_clusters: Optional[int] = None,
                   **kwargs) -> ClusteringResult:
        from sklearn.cluster import MiniBatchKMeans
        
        if n_clusters is None:
            n_clusters = self.estimate_n_clusters(data)
        
        clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(self.batch_size, len(data)),
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        labels = clusterer.fit_predict(data)
        
        return ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            algorithm=self.name(),
            parameters={
                'batch_size': self.batch_size,
                'n_init': self.n_init
            },
            converged=clusterer.n_iter_ < self.max_iter,
            iterations=int(clusterer.n_iter_),
            inertia=float(clusterer.inertia_)
        )


class PhenoGraphClusterer(SpatialClusterer):
    """
    PhenoGraph: Graph-based community detection
    Automatically determines optimal cluster number
    Best for high-dimensional single-cell data
    """
    
    def __init__(self, k_neighbors: int = 30, resolution: float = 1.0,
                 min_cluster_size: int = 10):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.resolution = resolution
        self.min_cluster_size = min_cluster_size
        self._phenograph = None
    
    def name(self) -> str:
        return "phenograph"
    
    def _import_phenograph(self):
        """Lazy import to avoid forcing dependency"""
        if self._phenograph is None:
            try:
                import phenograph
                self._phenograph = phenograph
            except ImportError:
                raise ImportError(
                    "PhenoGraph not installed. Run: pip install phenograph"
                )
        return self._phenograph
    
    def fit_predict(self, data: np.ndarray,
                   n_clusters: Optional[int] = None,
                   **kwargs) -> ClusteringResult:
        phenograph = self._import_phenograph()
        
        # PhenoGraph determines its own cluster number
        # k_neighbors should be adjusted based on expected cluster size
        k = min(self.k_neighbors, len(data) - 1)
        
        communities, graph, Q = phenograph.cluster(
            data, 
            k=k,
            resolution_parameter=self.resolution,
            min_cluster_size=self.min_cluster_size
        )
        
        n_clusters = len(np.unique(communities))
        
        return ClusteringResult(
            labels=communities,
            n_clusters=n_clusters,
            algorithm=self.name(),
            parameters={
                'k_neighbors': k,
                'resolution': self.resolution,
                'modularity': float(Q)
            },
            converged=True
        )


class LeidenClusterer(SpatialClusterer):
    """
    Leiden algorithm: Improved version of Louvain
    Fast, stable community detection
    """
    
    def __init__(self, resolution: float = 1.0, n_iterations: int = 2,
                 random_state: int = 42):
        super().__init__()
        self.resolution = resolution
        self.n_iterations = n_iterations
        self.random_state = random_state
        self._leidenalg = None
        self._igraph = None
    
    def name(self) -> str:
        return "leiden"
    
    def _import_dependencies(self):
        """Lazy import of leidenalg and igraph"""
        if self._leidenalg is None:
            try:
                import leidenalg
                import igraph
                self._leidenalg = leidenalg
                self._igraph = igraph
            except ImportError:
                raise ImportError(
                    "Leiden dependencies not installed. "
                    "Run: pip install leidenalg igraph"
                )
        return self._leidenalg, self._igraph
    
    def fit_predict(self, data: np.ndarray,
                   n_clusters: Optional[int] = None,
                   **kwargs) -> ClusteringResult:
        leidenalg, igraph = self._import_dependencies()
        
        # Build kNN graph
        from sklearn.neighbors import kneighbors_graph
        
        k = min(30, len(data) - 1)
        adjacency = kneighbors_graph(data, k, mode='connectivity', 
                                    include_self=False)
        
        # Convert to igraph
        sources, targets = adjacency.nonzero()
        edges = list(zip(sources.tolist(), targets.tolist()))
        g = igraph.Graph(edges=edges, directed=False)
        
        # Ensure connected graph
        if not g.is_connected():
            g = g.clusters().giant()
        
        # Run Leiden
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.resolution,
            n_iterations=self.n_iterations,
            seed=self.random_state
        )
        
        # Extract labels
        labels = np.zeros(len(data), dtype=int)
        for i, community in enumerate(partition):
            for vertex in community:
                if vertex < len(data):
                    labels[vertex] = i
        
        n_clusters = len(partition)
        
        return ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            algorithm=self.name(),
            parameters={
                'resolution': self.resolution,
                'n_iterations': self.n_iterations,
                'modularity': partition.modularity
            },
            converged=True
        )


class FlowSOMClusterer(SpatialClusterer):
    """
    FlowSOM: Self-organizing maps + hierarchical clustering
    Fast, hierarchical, good for flow/mass cytometry data
    Custom implementation as no good Python package exists
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (10, 10),
                 learning_rate: float = 0.5, n_iter: int = 100):
        super().__init__()
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.n_iter = n_iter
    
    def name(self) -> str:
        return "flowsom"
    
    def fit_predict(self, data: np.ndarray,
                   n_clusters: Optional[int] = None,
                   **kwargs) -> ClusteringResult:
        """
        Simplified FlowSOM implementation
        Real FlowSOM would use minisom or custom SOM implementation
        """
        from sklearn.cluster import AgglomerativeClustering
        from scipy.spatial.distance import cdist
        
        # Step 1: Create SOM grid
        n_nodes = self.grid_size[0] * self.grid_size[1]
        
        # Initialize node weights (simplified - random sample from data)
        indices = np.random.choice(len(data), min(n_nodes, len(data)), 
                                 replace=False)
        node_weights = data[indices].copy()
        
        # If we need more nodes than data points, interpolate
        if n_nodes > len(data):
            node_weights = np.vstack([
                node_weights,
                np.random.randn(n_nodes - len(node_weights), data.shape[1])
            ])
        
        # Step 2: Train SOM (simplified - just assign to nearest nodes)
        # In real implementation, would update weights iteratively
        distances = cdist(data, node_weights)
        node_assignments = np.argmin(distances, axis=1)
        
        # Step 3: Update node weights based on assigned data
        for node_id in range(n_nodes):
            mask = node_assignments == node_id
            if np.any(mask):
                node_weights[node_id] = data[mask].mean(axis=0)
        
        # Step 4: Metaclustering - cluster the SOM nodes
        if n_clusters is None:
            n_clusters = min(20, n_nodes // 2)  # Heuristic
        
        if n_clusters >= n_nodes:
            # Each node is its own cluster
            metaclusters = np.arange(n_nodes)
        else:
            # Hierarchical clustering on node weights
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            metaclusters = hierarchical.fit_predict(node_weights)
        
        # Step 5: Map data points to metaclusters
        labels = metaclusters[node_assignments]
        
        return ClusteringResult(
            labels=labels,
            n_clusters=len(np.unique(labels)),
            algorithm=self.name(),
            parameters={
                'grid_size': self.grid_size,
                'n_nodes': n_nodes
            },
            converged=True,
            iterations=self.n_iter
        )


class ClustererFactory:
    """Factory for creating clustering algorithms from configuration"""
    
    # Registry of available clusterers
    _clusterers = {
        'kmeans': KMeansClusterer,
        'minibatch': MiniBatchKMeansClusterer,
        'phenograph': PhenoGraphClusterer,
        'leiden': LeidenClusterer,
        'flowsom': FlowSOMClusterer
    }
    
    @classmethod
    def create(cls, algorithm: str, config: Dict[str, Any] = None) -> SpatialClusterer:
        """
        Create clusterer instance from algorithm name
        
        Args:
            algorithm: Name of clustering algorithm
            config: Algorithm-specific configuration
            
        Returns:
            SpatialClusterer instance
        """
        if algorithm not in cls._clusterers:
            raise ValueError(f"Unknown algorithm: {algorithm}. "
                           f"Available: {list(cls._clusterers.keys())}")
        
        clusterer_class = cls._clusterers[algorithm]
        
        # Extract algorithm-specific config
        if config and algorithm in config:
            algo_config = config[algorithm]
            return clusterer_class(**algo_config)
        else:
            return clusterer_class()
    
    @classmethod
    def list_algorithms(cls) -> list:
        """List available clustering algorithms"""
        return list(cls._clusterers.keys())
    
    @classmethod
    def register(cls, name: str, clusterer_class: type):
        """Register a new clustering algorithm"""
        if not issubclass(clusterer_class, SpatialClusterer):
            raise TypeError("Clusterer must inherit from SpatialClusterer")
        cls._clusterers[name] = clusterer_class


def auto_select_clusterer(data: np.ndarray, 
                         config: Dict[str, Any] = None) -> str:
    """
    Automatically select best clustering algorithm based on data characteristics
    
    Args:
        data: Feature matrix
        config: Optional configuration
        
    Returns:
        Algorithm name
    """
    n_samples, n_features = data.shape
    
    # Decision tree for algorithm selection
    if n_samples > 100000:
        return 'minibatch'  # Scalability priority
    elif n_samples < 1000:
        return 'leiden'     # Works well on small data
    elif n_features > 50:
        return 'phenograph'  # Good for high-dimensional
    elif n_samples > 10000:
        return 'flowsom'    # Fast and effective
    else:
        return 'kmeans'     # Reliable default


def benchmark_clusterers(data: np.ndarray,
                        algorithms: Optional[list] = None,
                        n_clusters: Optional[int] = None) -> Dict[str, Dict]:
    """
    Benchmark multiple clustering algorithms on the same data
    
    Returns:
        Dictionary with timing and quality metrics for each algorithm
    """
    import time
    from src.analysis.validation import SilhouetteValidator
    
    if algorithms is None:
        algorithms = ['kmeans', 'minibatch']  # Fast ones for testing
    
    results = {}
    validator = SilhouetteValidator()
    
    for algo_name in algorithms:
        try:
            clusterer = ClustererFactory.create(algo_name)
            
            start_time = time.time()
            result = clusterer.fit_predict(data, n_clusters=n_clusters)
            elapsed = time.time() - start_time
            
            # Validate
            val_result = validator.validate(data, result.labels)
            
            results[algo_name] = {
                'time_seconds': elapsed,
                'n_clusters': result.n_clusters,
                'silhouette_score': val_result.score,
                'converged': result.converged,
                'parameters': result.parameters
            }
        except ImportError as e:
            results[algo_name] = {
                'error': f"Not installed: {str(e)}"
            }
        except Exception as e:
            results[algo_name] = {
                'error': str(e)
            }
    
    return results
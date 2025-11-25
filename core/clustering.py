"""
Advanced Clustering for RAPTOR Tree Building
Supports multiple clustering algorithms with automatic selection
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class ClusteringMethod(Enum):
    """Available clustering methods"""
    KMEANS = "kmeans"                    # Fast, good for known k
    HIERARCHICAL = "hierarchical"        # Produces natural tree structure
    DBSCAN = "dbscan"                    # Density-based, auto-detects k
    GMM = "gmm"                          # Soft clustering (probabilistic)
    HDBSCAN = "hdbscan"                  # Hierarchical DBSCAN (best for varying density)
    ADAPTIVE = "adaptive"                # Automatically selects best method


@dataclass
class ClusteringConfig:
    """Configuration for clustering behavior"""
    method: ClusteringMethod = ClusteringMethod.KMEANS  # KMeans is fast and predictable
    target_cluster_size: int = 10        # Target chunks per cluster
    min_cluster_size: int = 3            # Minimum cluster size
    max_cluster_size: int = 20           # Maximum cluster size
    similarity_threshold: float = 0.75   # Threshold for similarity-based grouping
    branching_factor: int = 5            # Target children per node in tree
    use_silhouette: bool = True          # Use silhouette score to optimize k
    random_state: int = 42               # For reproducibility


@dataclass
class ClusterResult:
    """Result from clustering operation"""
    labels: np.ndarray                   # Cluster assignment for each item
    n_clusters: int                      # Number of clusters found
    centroids: Optional[np.ndarray] = None  # Cluster centers
    silhouette_score: Optional[float] = None
    method_used: str = "unknown"
    metadata: Dict = field(default_factory=dict)


class SmartClusterer:
    """
    Intelligent clustering for RAPTOR tree building.
    Automatically selects best method based on data characteristics.
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self._sklearn_available = self._check_sklearn()
        self._hdbscan_available = self._check_hdbscan()
    
    def _check_sklearn(self) -> bool:
        """Check if sklearn is available"""
        try:
            from sklearn.cluster import KMeans
            return True
        except ImportError:
            return False
    
    def _check_hdbscan(self) -> bool:
        """Check if hdbscan is available"""
        try:
            import hdbscan
            return True
        except ImportError:
            return False
    
    def _compute_optimal_k(self, embeddings: np.ndarray) -> int:
        """
        Compute optimal number of clusters using elbow method + silhouette.
        """
        n_samples = len(embeddings)
        
        # Heuristic bounds
        k_min = max(2, n_samples // self.config.max_cluster_size)
        k_max = min(n_samples // self.config.min_cluster_size, n_samples // 2, 50)
        
        if k_max <= k_min:
            return max(2, n_samples // self.config.target_cluster_size)
        
        if not self._sklearn_available:
            return max(2, n_samples // self.config.target_cluster_size)
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        best_k = k_min
        best_score = -1
        
        for k in range(k_min, min(k_max + 1, k_min + 10)):  # Check up to 10 values
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception:
                continue
        
        return best_k
    
    def _cluster_kmeans(self, embeddings: np.ndarray, k: Optional[int] = None) -> ClusterResult:
        """KMeans clustering"""
        if not self._sklearn_available:
            raise ImportError("sklearn required for KMeans clustering")
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        if k is None:
            k = self._compute_optimal_k(embeddings)
        
        kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        sil_score = None
        if len(set(labels)) > 1:
            sil_score = silhouette_score(embeddings, labels)
        
        return ClusterResult(
            labels=labels,
            n_clusters=k,
            centroids=kmeans.cluster_centers_,
            silhouette_score=sil_score,
            method_used="kmeans"
        )
    
    def _cluster_hierarchical(self, embeddings: np.ndarray) -> ClusterResult:
        """Agglomerative hierarchical clustering - natural tree structure"""
        if not self._sklearn_available:
            raise ImportError("sklearn required for hierarchical clustering")
        
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        # Compute optimal k
        n_samples = len(embeddings)
        k = max(2, n_samples // self.config.target_cluster_size)
        k = min(k, n_samples - 1)
        
        clustering = AgglomerativeClustering(
            n_clusters=k,
            linkage='ward'  # Ward minimizes variance within clusters
        )
        labels = clustering.fit_predict(embeddings)
        
        # Compute centroids manually
        centroids = []
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                centroids.append(embeddings[mask].mean(axis=0))
            else:
                centroids.append(np.zeros(embeddings.shape[1]))
        centroids = np.array(centroids)
        
        sil_score = None
        if len(set(labels)) > 1:
            sil_score = silhouette_score(embeddings, labels)
        
        return ClusterResult(
            labels=labels,
            n_clusters=k,
            centroids=centroids,
            silhouette_score=sil_score,
            method_used="hierarchical"
        )
    
    def _cluster_dbscan(self, embeddings: np.ndarray) -> ClusterResult:
        """DBSCAN - density based, auto-detects clusters"""
        if not self._sklearn_available:
            raise ImportError("sklearn required for DBSCAN clustering")
        
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import NearestNeighbors
        
        # Auto-compute eps using k-nearest neighbors
        n_neighbors = min(self.config.min_cluster_size, len(embeddings) - 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        
        # Use knee point of sorted k-distances
        k_distances = np.sort(distances[:, -1])
        eps = np.percentile(k_distances, 90)  # 90th percentile as eps
        
        dbscan = DBSCAN(eps=eps, min_samples=self.config.min_cluster_size)
        labels = dbscan.fit_predict(embeddings)
        
        # Handle noise points (-1 label) by assigning to nearest cluster
        noise_mask = labels == -1
        if noise_mask.any() and (labels >= 0).any():
            # Find centroids of existing clusters
            unique_labels = [l for l in set(labels) if l >= 0]
            centroids = []
            for l in unique_labels:
                centroids.append(embeddings[labels == l].mean(axis=0))
            centroids = np.array(centroids)
            
            # Assign noise to nearest centroid
            noise_embeddings = embeddings[noise_mask]
            for i, emb in enumerate(noise_embeddings):
                distances = np.linalg.norm(centroids - emb, axis=1)
                nearest_cluster = unique_labels[np.argmin(distances)]
                labels[np.where(noise_mask)[0][i]] = nearest_cluster
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Compute centroids
        centroids = []
        for i in range(n_clusters):
            mask = labels == i
            if mask.sum() > 0:
                centroids.append(embeddings[mask].mean(axis=0))
        centroids = np.array(centroids) if centroids else None
        
        sil_score = None
        if n_clusters > 1:
            sil_score = silhouette_score(embeddings, labels)
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            centroids=centroids,
            silhouette_score=sil_score,
            method_used="dbscan",
            metadata={'eps': eps}
        )
    
    def _cluster_gmm(self, embeddings: np.ndarray) -> ClusterResult:
        """Gaussian Mixture Model - soft/probabilistic clustering"""
        if not self._sklearn_available:
            raise ImportError("sklearn required for GMM clustering")
        
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import silhouette_score
        
        k = self._compute_optimal_k(embeddings)
        
        gmm = GaussianMixture(
            n_components=k, 
            random_state=self.config.random_state,
            covariance_type='full'
        )
        labels = gmm.fit_predict(embeddings)
        
        sil_score = None
        if len(set(labels)) > 1:
            sil_score = silhouette_score(embeddings, labels)
        
        return ClusterResult(
            labels=labels,
            n_clusters=k,
            centroids=gmm.means_,
            silhouette_score=sil_score,
            method_used="gmm",
            metadata={'converged': gmm.converged_, 'n_iter': gmm.n_iter_}
        )
    
    def _cluster_hdbscan(self, embeddings: np.ndarray) -> ClusterResult:
        """HDBSCAN - hierarchical DBSCAN, best for varying density"""
        if not self._hdbscan_available:
            # Fallback to DBSCAN
            return self._cluster_dbscan(embeddings)
        
        import hdbscan
        from sklearn.metrics import silhouette_score
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=2,
            cluster_selection_method='eom'  # Excess of Mass
        )
        labels = clusterer.fit_predict(embeddings)
        
        # Handle noise (-1)
        noise_mask = labels == -1
        if noise_mask.any() and (labels >= 0).any():
            unique_labels = [l for l in set(labels) if l >= 0]
            centroids = []
            for l in unique_labels:
                centroids.append(embeddings[labels == l].mean(axis=0))
            centroids = np.array(centroids)
            
            noise_embeddings = embeddings[noise_mask]
            for i, emb in enumerate(noise_embeddings):
                distances = np.linalg.norm(centroids - emb, axis=1)
                nearest_cluster = unique_labels[np.argmin(distances)]
                labels[np.where(noise_mask)[0][i]] = nearest_cluster
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Compute centroids
        centroids = []
        for i in sorted(set(labels)):
            if i >= 0:
                mask = labels == i
                centroids.append(embeddings[mask].mean(axis=0))
        centroids = np.array(centroids) if centroids else None
        
        sil_score = None
        if n_clusters > 1:
            sil_score = silhouette_score(embeddings, labels)
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            centroids=centroids,
            silhouette_score=sil_score,
            method_used="hdbscan"
        )
    
    def _cluster_adaptive(self, embeddings: np.ndarray) -> ClusterResult:
        """
        Automatically select best clustering method based on data.
        
        Selection criteria:
        - Small dataset (<50): Hierarchical (natural tree)
        - Medium dataset (50-500): KMeans with silhouette optimization
        - Large dataset (500-5000): HDBSCAN or GMM
        - Very large (>5000): KMeans (speed)
        """
        n_samples = len(embeddings)
        
        if n_samples < 10:
            # Too small - just group into 2-3 clusters
            return self._cluster_kmeans(embeddings, k=min(3, n_samples // 2 + 1))
        
        elif n_samples < 50:
            # Small - hierarchical works well
            return self._cluster_hierarchical(embeddings)
        
        elif n_samples < 500:
            # Medium - try multiple methods, pick best silhouette
            results = []
            
            try:
                results.append(self._cluster_kmeans(embeddings))
            except Exception:
                pass
            
            try:
                results.append(self._cluster_hierarchical(embeddings))
            except Exception:
                pass
            
            try:
                results.append(self._cluster_gmm(embeddings))
            except Exception:
                pass
            
            if not results:
                return self._cluster_kmeans(embeddings)
            
            # Pick best silhouette score
            best = max(results, key=lambda r: r.silhouette_score or -1)
            return best
        
        elif n_samples < 5000:
            # Large - HDBSCAN is good for varying density
            if self._hdbscan_available:
                return self._cluster_hdbscan(embeddings)
            else:
                return self._cluster_kmeans(embeddings)
        
        else:
            # Very large - KMeans for speed
            return self._cluster_kmeans(embeddings)
    
    def cluster(self, embeddings: np.ndarray, 
                method: Optional[ClusteringMethod] = None) -> ClusterResult:
        """
        Main clustering method.
        
        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            method: Override default method (optional)
        
        Returns:
            ClusterResult with labels, centroids, and metrics
        """
        if len(embeddings) < 2:
            return ClusterResult(
                labels=np.array([0] * len(embeddings)),
                n_clusters=1,
                centroids=embeddings.mean(axis=0, keepdims=True) if len(embeddings) > 0 else None,
                method_used="single"
            )
        
        method = method or self.config.method
        
        if method == ClusteringMethod.KMEANS:
            return self._cluster_kmeans(embeddings)
        elif method == ClusteringMethod.HIERARCHICAL:
            return self._cluster_hierarchical(embeddings)
        elif method == ClusteringMethod.DBSCAN:
            return self._cluster_dbscan(embeddings)
        elif method == ClusteringMethod.GMM:
            return self._cluster_gmm(embeddings)
        elif method == ClusteringMethod.HDBSCAN:
            return self._cluster_hdbscan(embeddings)
        elif method == ClusteringMethod.ADAPTIVE:
            return self._cluster_adaptive(embeddings)
        else:
            return self._cluster_adaptive(embeddings)
    
    def build_cluster_hierarchy(self, embeddings: np.ndarray, 
                                 texts: List[str],
                                 max_levels: int = 4) -> Dict[int, List[Dict]]:
        """
        Build hierarchical clustering for RAPTOR tree.
        
        Returns dict mapping level -> list of cluster info
        Level 0 = leaf clusters (groups of chunks)
        Level 1+ = higher-level clusters (groups of clusters)
        """
        hierarchy = defaultdict(list)
        current_embeddings = embeddings
        current_items = [{'text': t, 'embedding': e, 'original_idx': i} 
                        for i, (t, e) in enumerate(zip(texts, embeddings))]
        
        for level in range(max_levels):
            if len(current_embeddings) < self.config.min_cluster_size:
                break
            
            # Cluster current level
            result = self.cluster(current_embeddings)
            
            # Group items by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(result.labels):
                clusters[label].append(current_items[i])
            
            # Store cluster info
            next_items = []
            for cluster_id, items in clusters.items():
                cluster_embeddings = np.array([item['embedding'] for item in items])
                centroid = cluster_embeddings.mean(axis=0)
                
                cluster_info = {
                    'cluster_id': cluster_id,
                    'level': level,
                    'centroid': centroid,
                    'member_count': len(items),
                    'members': items
                }
                
                hierarchy[level].append(cluster_info)
                
                # Prepare for next level
                next_items.append({
                    'text': f"[Cluster {cluster_id} summary]",  # Will be replaced by LLM summary
                    'embedding': centroid,
                    'original_idx': cluster_id,
                    'cluster_info': cluster_info
                })
            
            # If only 1 cluster, we're done
            if len(clusters) <= 1:
                break
            
            current_items = next_items
            current_embeddings = np.array([item['embedding'] for item in current_items])
            
            print(f"  Level {level}: {result.n_clusters} clusters (method: {result.method_used}, silhouette: {result.silhouette_score:.3f if result.silhouette_score else 'N/A'})")
        
        return dict(hierarchy)


def cluster_embeddings(embeddings: np.ndarray,
                       method: str = "adaptive",
                       target_cluster_size: int = 10,
                       min_cluster_size: int = 3) -> ClusterResult:
    """
    Convenience function for clustering embeddings.
    
    Args:
        embeddings: numpy array of embeddings
        method: One of "kmeans", "hierarchical", "dbscan", "gmm", "hdbscan", "adaptive"
        target_cluster_size: Target number of items per cluster
        min_cluster_size: Minimum cluster size
    
    Returns:
        ClusterResult with labels and centroids
    """
    method_map = {
        "kmeans": ClusteringMethod.KMEANS,
        "hierarchical": ClusteringMethod.HIERARCHICAL,
        "dbscan": ClusteringMethod.DBSCAN,
        "gmm": ClusteringMethod.GMM,
        "hdbscan": ClusteringMethod.HDBSCAN,
        "adaptive": ClusteringMethod.ADAPTIVE
    }
    
    config = ClusteringConfig(
        method=method_map.get(method, ClusteringMethod.ADAPTIVE),
        target_cluster_size=target_cluster_size,
        min_cluster_size=min_cluster_size
    )
    
    clusterer = SmartClusterer(config)
    return clusterer.cluster(embeddings)


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CLUSTERING METHODS")
    print("=" * 60)
    
    # Generate sample embeddings (random for testing)
    np.random.seed(42)
    n_samples = 100
    n_features = 384  # Same as all-MiniLM-L6-v2
    
    # Create 5 distinct clusters
    centers = np.random.randn(5, n_features)
    embeddings = []
    for i in range(n_samples):
        center = centers[i % 5]
        point = center + np.random.randn(n_features) * 0.3
        embeddings.append(point)
    embeddings = np.array(embeddings)
    
    config = ClusteringConfig(target_cluster_size=20)
    clusterer = SmartClusterer(config)
    
    for method in ClusteringMethod:
        try:
            print(f"\n{method.value.upper()}:")
            result = clusterer.cluster(embeddings, method)
            print(f"  Clusters: {result.n_clusters}")
            sil_str = f"{result.silhouette_score:.3f}" if result.silhouette_score else "N/A"
            print(f"  Silhouette: {sil_str}")
            print(f"  Method used: {result.method_used}")
        except Exception as e:
            print(f"  Error: {e}")

"""
Memory bank and coreset subsampling for PatchCore.

Implements greedy coreset subsampling and memory bank with nearest neighbor
search following the PatchCore paper (Roth et al., CVPR 2022).
"""

import numpy as np
import torch
from typing import Optional, Tuple
from tqdm import tqdm


class GreedyCoresetSubsampling:
    """
    Greedy Coreset Subsampling for memory bank reduction.
    
    Implements Algorithm 1 from PatchCore paper using minimax facility location
    to select a representative subset of patches.
    
    Args:
        target_ratio: Fraction of patches to retain (e.g., 0.01 for 1%)
        projection_dim: Dimensionality for random projection speedup (None=no projection)
    """
    
    def __init__(
        self,
        target_ratio: float = 0.01,
        projection_dim: Optional[int] = 128
    ):
        self.target_ratio = target_ratio
        self.projection_dim = projection_dim
        self.projection_matrix = None
    
    def _random_projection(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Johnson-Lindenstrauss random projection for speedup.
        
        Args:
            features: Feature vectors (N, D)
            
        Returns:
            Projected features (N, d*) where d* << D
        """
        N, D = features.shape
        
        if self.projection_dim is None or self.projection_dim >= D:
            return features
        
        # Create or reuse projection matrix
        if self.projection_matrix is None:
            self.projection_matrix = torch.randn(
                D, self.projection_dim,
                device=features.device
            ) / np.sqrt(self.projection_dim)
        
        return features @ self.projection_matrix
    
    def sample(self, features: torch.Tensor) -> np.ndarray:
        """
        Execute greedy coreset sampling.
        
        Iteratively selects patches that maximize minimum distance to
        already selected patches (furthest-point strategy).
        
        Args:
            features: Patch features to subsample (N, D)
            
        Returns:
            Array of selected indices (target_size,)
        """
        N = features.shape[0]
        target_size = max(1, int(N * self.target_ratio))
        
        # Apply random projection for speed
        features_proj = self._random_projection(features)
        features_proj = features_proj.cpu().numpy()
        
        # Initialize
        selected_indices = []
        remaining_indices = set(range(N))
        
        # First point: random selection
        first_idx = np.random.randint(N)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Track minimum distances for each point
        min_distances = np.full(N, np.inf)
        
        # Greedy selection loop
        with tqdm(total=target_size-1, desc="Coreset sampling", leave=False) as pbar:
            for _ in range(target_size - 1):
                # Last selected point
                last_selected = selected_indices[-1]
                last_feat = features_proj[last_selected]
                
                # Update minimum distances
                for idx in remaining_indices:
                    dist = np.linalg.norm(features_proj[idx] - last_feat)
                    min_distances[idx] = min(min_distances[idx], dist)
                
                # Select point with maximum minimum distance (furthest point)
                remaining_list = list(remaining_indices)
                remaining_dists = min_distances[remaining_list]
                furthest_idx_rel = np.argmax(remaining_dists)
                furthest_idx = remaining_list[furthest_idx_rel]
                
                selected_indices.append(furthest_idx)
                remaining_indices.remove(furthest_idx)
                
                pbar.update(1)
        
        return np.array(selected_indices)


class MemoryBank:
    """
    Memory Bank for PatchCore with nearest neighbor search and reweighting.
    
    Stores nominal patch features and provides methods for:
    - Fast nearest neighbor search (uses FAISS if available)
    - Anomaly scoring with density-based reweighting
    
    Args:
        features: Patch features (N, D) as numpy array
        use_faiss: Whether to use FAISS for fast NN search
        n_neighbors: Number of neighbors for reweighting (default: 9)
    """
    
    def __init__(
        self,
        features: Optional[np.ndarray] = None,
        use_faiss: bool = True,
        n_neighbors: int = 9
    ):
        self.features = features
        self.use_faiss = use_faiss
        self.n_neighbors = n_neighbors
        self.index = None
        
        if features is not None:
            self._build_index()
    
    def add_features(self, features: torch.Tensor) -> None:
        """
        Add features to memory bank.
        
        Args:
            features: Tensor (N, D) or numpy array
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        if self.features is None:
            self.features = features
        else:
            self.features = np.vstack([self.features, features])
        
        self._build_index()
    
    def _build_index(self) -> None:
        """Build FAISS index for fast nearest neighbor search."""
        if not self.use_faiss or self.features is None:
            return
        
        try:
            import faiss
            N, D = self.features.shape
            
            # Use exact L2 search
            self.index = faiss.IndexFlatL2(D)
            self.index.add(self.features.astype(np.float32))
        except ImportError:
            print("Warning: FAISS not available, using numpy fallback")
            self.use_faiss = False
    
    def search_nearest(
        self,
        query_features: torch.Tensor,
        k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search k nearest neighbors for each query.
        
        Args:
            query_features: Query features (M, D)
            k: Number of neighbors
            
        Returns:
            Tuple of (distances, indices):
            - distances: L2 distances (M, k)
            - indices: Indices in memory bank (M, k)
        """
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.cpu().numpy()
        
        query_features = query_features.astype(np.float32)
        
        if self.use_faiss and self.index is not None:
            distances, indices = self.index.search(query_features, k)
        else:
            # Numpy fallback
            M = query_features.shape[0]
            distances = np.zeros((M, k))
            indices = np.zeros((M, k), dtype=int)
            
            for i in range(M):
                dists = np.linalg.norm(
                    self.features - query_features[i], axis=1
                )
                sorted_idx = np.argsort(dists)[:k]
                distances[i] = dists[sorted_idx]
                indices[i] = sorted_idx
        
        return distances, indices
    
    def compute_anomaly_scores(
        self,
        query_features: torch.Tensor,
        apply_reweighting: bool = True
    ) -> np.ndarray:
        """
        Compute anomaly scores with density-based reweighting.
        
        Implements Eq. 6-7 from PatchCore paper:
        1. Find nearest neighbor for each query
        2. Compute base score as L2 distance
        3. Apply reweighting based on local k-NN density
        
        Args:
            query_features: Features to score (M, D)
            apply_reweighting: Whether to apply density reweighting
            
        Returns:
            Anomaly scores (M,)
        """
        # Search k+1 neighbors (1 for base score, k for reweighting)
        k_search = self.n_neighbors if apply_reweighting else 1
        distances, indices = self.search_nearest(query_features, k=k_search)
        
        # Base anomaly score: distance to nearest neighbor
        base_scores = distances[:, 0]
        
        if not apply_reweighting or k_search == 1:
            return base_scores
        
        # Reweighting based on local density (Eq. 7)
        # weight = 1 - exp(d_nearest) / sum(exp(d_k))
        exp_distances = np.exp(-distances)
        
        numerator = exp_distances[:, 0]
        denominator = np.sum(exp_distances, axis=1)
        
        weights = 1.0 - (numerator / (denominator + 1e-8))
        
        # Final score: base_score * weight
        reweighted_scores = base_scores * weights
        
        return reweighted_scores
    
    def __len__(self) -> int:
        """Return number of patches in memory bank."""
        return len(self.features) if self.features is not None else 0
    
    def save(self, filepath: str) -> None:
        """Save memory bank to disk."""
        np.save(filepath, self.features)
    
    @classmethod
    def load(cls, filepath: str, **kwargs) -> 'MemoryBank':
        """Load memory bank from disk."""
        features = np.load(filepath)
        return cls(features=features, **kwargs)

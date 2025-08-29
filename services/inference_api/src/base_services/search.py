import asyncio
import faiss
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any

from ..config import settings , PROJECT_ROOT
from ..utils import logger, measure_performance

class SimilarityService:
    def __init__(self):
        self.faiss_index_path = Path(settings.faiss_artifacts_path).resolve() / settings.faiss_index_path
        self.ss_nprobe_neighbors = settings.ss_nprobe_neighbors
        self.faiss_ivf_min_vectors = settings.faiss_ivf_min_vectors
        self.faiss_nlist = settings.faiss_nlist
        self.faiss_max_nlist = settings.faiss_max_nlist
        self.faiss_nlist_factor = settings.faiss_nlist_factor
        self.faiss_m = settings.faiss_m
        self.faiss_nbits = settings.faiss_nbits
        self.ss_top_k_neighbors = settings.ss_top_k_neighbors
        self.ss_max_distance_for_relevant_context = settings.ss_max_distance_for_relevant_context

        self.index = None
        self.dimension = None

    def _load_index(self):
        if self.index is not None:
            return

        index_path = self.faiss_index_path
        logger.info(f"Attempting to load FAISS index from: {index_path.resolve()}")

        if not index_path.exists():
            logger.error(
                f"FAISS index file not found at {index_path.resolve()} \n"
                f"Please Build the index using the command `python scripts/build_index.py`"
            )
            raise FileNotFoundError("Could not find FAISS index at the specified path.")

        try:
            self.index = faiss.read_index(str(index_path))
            self.dimension = self.index.d

            if hasattr(self.index, "nprobe"):
                self.index.nprobe = self.ss_nprobe_neighbors

            logger.info(
                f"FAISS index loaded successfully. "
                f"Dimension={self.dimension}, "
                f"Total vectors={self.index.ntotal}, "
                f"nprobe={getattr(self.index, 'nprobe', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the FAISS index from {index_path}.")
            raise RuntimeError(f"Could not initialize SimilarityService: {e}")

    def _build_index(self, vectors_arr: np.ndarray) -> faiss.Index:
        num_vectors, dimension = vectors_arr.shape
        if num_vectors < self.faiss_ivf_min_vectors:
            index = faiss.IndexFlatL2(dimension)
        else:
            nlist = min(self.faiss_max_nlist, max(1, self.faiss_nlist_factor * int(num_vectors ** 0.5)))
            nlist = min(nlist, num_vectors)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, self.faiss_m, self.faiss_nbits)
            index.train(vectors_arr)
        
        index.add(vectors_arr)
        return index

    def build_and_save_index(self, vectors: List[List[float]]):
        logger.info("--- SimilarityService: Building and saving new FAISS index ---")
        vectors_arr = np.asarray(vectors, dtype=np.float32)
        index = self._build_index(vectors_arr)
        loc = Path(settings.faiss_artifacts_path).resolve() / settings.faiss_index_path
        os.makedirs(loc.parent , exist_ok=True)
        faiss.write_index(index, str(loc))
        logger.info(f"FAISS index saved successfully to {self.faiss_index_path}")

    def _perform_search(self, query_arr: np.ndarray) -> Dict[str, List[Any]]:
        if self.index.ntotal == 0:
            return {"distances": [], "indices": []}
        k = min(self.ss_top_k_neighbors, self.index.ntotal)
        distances, indices = self.index.search(query_arr, k)
        unique_results = {}
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                dist, idx = distances[i][j], int(indices[i][j])
                if idx != -1 and dist <= self.ss_max_distance_for_relevant_context:
                    if idx not in unique_results or dist < unique_results[idx]:
                        unique_results[idx] = dist
        
        if not unique_results and self.index.ntotal > 0:
            return {"distances": distances[0].tolist(), "indices": indices[0].tolist()}
        
        sorted_results = sorted(unique_results.items(), key=lambda item: item[1])
        return {"distances": [item[1] for item in sorted_results], "indices": [item[0] for item in sorted_results]}

    @measure_performance()
    async def search(self, query_vectors: List[List[float]]) -> Dict[str, List[Any]]:
        if self.index is None:
            self._load_index()

        query_arr = np.asarray(query_vectors, dtype=np.float32)
        if query_arr.ndim != 2 or query_arr.shape[1] != self.dimension:
            raise ValueError(f"Invalid query shape. Expected (n_queries, {self.dimension}), got {query_arr.shape}")

        return await asyncio.to_thread(self._perform_search, query_arr)
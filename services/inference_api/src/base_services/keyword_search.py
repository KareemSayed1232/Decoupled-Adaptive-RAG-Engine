import os
import pickle
from rank_bm25 import BM25Okapi
from ..config import settings
from ..utils import logger, measure_performance
from pathlib import Path
class KeywordSearchService:
    def __init__(self):
        self._bm25_index = None

    def _load_index(self):
        if self._bm25_index is not None:
            return
        
        index_path = Path(settings.faiss_artifacts_path).resolve() / 'bm25.index'
        logger.info(f"Attempting to load BM25 index from: {index_path}")
        
        if not os.path.exists(index_path):
            logger.error(f"BM25 index file not found at {index_path}. Please build the artifacts first.")
            raise FileNotFoundError("Required artifact 'bm25.index' not found.")
        
        with open(index_path, 'rb') as f:
            self._bm25_index = pickle.load(f)
        
        logger.info("BM25 index loaded successfully.")

    @measure_performance(unit="ms")
    def search(self, query: str, top_k: int) -> list[int]:
        # ➕ The check is added here. The index is loaded on the first search.
        if self._bm25_index is None:
            self._load_index()

        tokenized_query = query.split(" ")
        
        doc_scores = self._bm25_index.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        return top_indices
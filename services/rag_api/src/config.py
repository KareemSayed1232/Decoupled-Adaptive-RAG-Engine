from pydantic import ConfigDict
from pydantic_settings import BaseSettings

class RagApiSettings(BaseSettings):
    
    reranker_rejection_threshold: float
    summarization_min_docs: int

    
    ss_top_k_neighbors: int
    ss_max_distance_for_relevant_context: float
    ss_nprobe_neighbors: int
    
    
    faiss_ivf_min_vectors: int
    faiss_nlist: int
    faiss_max_nlist: int
    faiss_nlist_factor: int
    faiss_m: int
    faiss_nbits: int
    
    
    faiss_index_path: str = ""
    faiss_chunks_path: str = ""
    base_context_file: str = ""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra="ignore"
    )

settings = RagApiSettings()
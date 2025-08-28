import torch
from typing import List
from pathlib import Path
from pydantic import ConfigDict 
from pydantic_settings import BaseSettings

class ProjectSettings(BaseSettings):
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embedding_model : str
    reranker_model : str
    llm_model_path : str


    
    faiss_index_path: str
    faiss_nlist: int     
    faiss_max_nlist: int     
    faiss_nlist_factor: int
    faiss_m: int      
    faiss_nbits: int    
    faiss_ivf_min_vectors: int
    faiss_chunks_path: str    
       

    
    server_port: int
    server_host: str
    base_context_file: str
    complete_context_file: str
    api_client_timeout:int

    
    chunking_max_characters: int
    chunking_combine_text_under_n_chars: int
    chunking_overlap: int

    
    gen_max_tokens: int
    gen_temperature: float
    gen_top_p: float
    gen_stop_words: List[str]
    gen_prompt_file_name : str 

    
    hyde_model_path: str
    hyde_max_new_tokens: int
    hyde_temperature: float
    hyde_top_p: float
    hyde_stop_words: List[str]
    hyde_prompt_file_name: str

    
    summarization_model_name: str
    summarize_max_tokens: int
    summarization_chunk_size: int
    summarization_chunk_overlap: int
    summarization_max_length: int
    summarization_min_length: int
    summarization_min_docs: int

    
    reranker_batch_size : int
    reranker_top_k:int
    reranker_tokenizer_max_length : int
    reranker_rejection_threshold: float

    
    ss_max_distance_for_relevant_context : float
    ss_top_k_neighbors : int
    ss_nprobe_neighbors: int


    model_config = ConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent.parent / ".env",
        extra="allow"
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
settings = ProjectSettings()
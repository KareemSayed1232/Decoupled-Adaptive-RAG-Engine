from sentence_transformers import SentenceTransformer  
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from mxbai_rerank import MxbaiRerankV2

from pathlib import Path
from ..config import settings , PROJECT_ROOT
from llama_cpp import Llama
from ..utils import logger
from transformers import AutoTokenizer 




class ModelLoader:
    _embedding_model = None
    _llm_model = None
    _hyde_llm_model = None 
    _summarization_model = None
    _summarization_tokenizer = None

    _reranker_model = None

    @classmethod
    def get_embedding_model(cls):
        logger.info("Loading Embedding Model")
        if cls._embedding_model is None:
            cls._embedding_model = SentenceTransformer(
                settings.embedding_model,
                
                device=settings.device
            )
        return cls._embedding_model


    @classmethod
    def get_llm_model(cls):
        logger.info("Loading Main Generation LLM")
        absolute_model_path = PROJECT_ROOT / settings.llm_model_path
        if cls._llm_model is None:
            cls._llm_model = Llama(
                model_path=str(absolute_model_path),
                n_gpu_layers=-1,
                flash_attn=True,
                chat_format='chatml',
                verbose=False, 
                n_ctx=2048,
                n_batch=512,
                n_threads=10,
            )
        return cls._llm_model


    @classmethod
    def get_hyde_model(cls):
        logger.info("Loading HyDE Generation LLM ")
        absolute_model_path = PROJECT_ROOT / settings.hyde_model_path
        if cls._hyde_llm_model is None:
            cls._hyde_llm_model = Llama(
                model_path=str(absolute_model_path),
                n_gpu_layers=-1,  
                flash_attn=True,
                chat_format='chatml',
                verbose=False, 
                n_ctx=3072,       
                n_batch=256,     
                n_threads=10,
            )
        return cls._hyde_llm_model

    @classmethod
    def get_summarization_model_and_tokenizer(cls):
        logger.info("Loading Summarization Model and Tokenizer")
        if cls._summarization_model is None:
            cls._summarization_tokenizer = AutoTokenizer.from_pretrained(settings.summarization_model_name)
            cls._summarization_model = AutoModelForSeq2SeqLM.from_pretrained(settings.summarization_model_name)
            cls._summarization_model.to(settings.device)
            cls._summarization_model.eval()
        return cls._summarization_model, cls._summarization_tokenizer
    
    @classmethod
    def get_reranking_model(cls):
        logger.info("Loading Reranking Model")
        if cls._reranker_model is None:
            cls._reranker_model = MxbaiRerankV2(settings.reranker_model,disable_transformers_warnings=True,
                            tokenizer_kwargs={"max_length": settings.reranker_tokenizer_max_length,
                                               "truncation": True , 
                                               })
        return cls._reranker_model



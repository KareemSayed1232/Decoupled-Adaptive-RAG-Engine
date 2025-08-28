
from fastapi import FastAPI
from contextlib import asynccontextmanager
from shared_models.models import *
import json

from .base_services import (
    EmbeddingService, RerankingService, LLMService, 
    SummarizationService, ChunkingService, SimilarityService,
    KeywordSearchService                                                                                                                                                
)

ml_services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("INFERENCE-API: Loading all ML models...")

    ml_services["chunking"] = ChunkingService()
    ml_services["embedding"] = EmbeddingService()
    ml_services["reranking"] = RerankingService()
    ml_services["llm"] = LLMService()
    ml_services["summarization"] = SummarizationService()
    ml_services["similarity"] = SimilarityService()
    ml_services["keyword"] = KeywordSearchService()
    print("INFERENCE-API: Models loaded and ready.")
    yield
    ml_services.clear()

app = FastAPI(title="Inference API", lifespan=lifespan)

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(request: PreprocessRequest):
    text = await ml_services["chunking"].preprocess_text(request.text)
    return PreprocessResponse(text=text)

@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    embeddings = await ml_services["embedding"].encode(request.texts)
    return EmbeddingResponse(embeddings=embeddings)

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    result = await ml_services["reranking"].rerank(request.question, request.extracted_docs)
    return RerankResponse(**result)

@app.post("/vector_search", response_model=SimilaritySearchResponse)
async def vector_search(request: SimilaritySearchRequest):
     result = await ml_services["similarity"].search(request.query)
     return SimilaritySearchResponse(**result)

@app.post("/keyword_search", response_model=KeywordSearchResponse)
async def keyword_search(request: KeywordSearchRequest):
    # This uses a hardcoded top_k for simplicity.
    # Your config logic will need to be adapted.
    from .config import settings
    indices = ml_services["keyword"].search(request.query, top_k=settings.ss_top_k_neighbors)
    return KeywordSearchResponse(indices=indices)

@app.post("/generate_hyde", response_model=HydeGenerationResponse)
async def generate_hyde(request: HydeGenerationRequest):
    return ml_services["llm"].generate_hyde(request.question)

@app.post("/summarize", response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest):
    return ml_services["summarization"].summarize(request.context_to_summarize)

@app.post("/generate_stream")
def generate_stream(request: GenerationStreamRequest):
    from fastapi.responses import StreamingResponse
    def event_generator():
        for token_chunk in ml_services["llm"].generate_stream(request.question, request.context):
            yield f"data: {json.dumps({'type': 'chunk', 'content': token_chunk})}\n\n"
        yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

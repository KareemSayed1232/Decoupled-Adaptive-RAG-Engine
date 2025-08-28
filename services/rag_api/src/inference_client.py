
import httpx
import os
import json
from shared_models.models import *

INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8001")

class InferenceAPIClient:
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=INFERENCE_API_URL, timeout=60.0)

    async def _post(self, endpoint: str, request_model, response_model):
        res = await self.client.post(endpoint, json=request_model.model_dump())
        res.raise_for_status()
        return response_model(**res.json())

    async def preprocess_text(self, req: PreprocessRequest) -> PreprocessResponse: return await self._post("/preprocess", req, PreprocessResponse)
    async def embed_texts(self, req: EmbeddingRequest) -> EmbeddingResponse: return await self._post("/embed", req, EmbeddingResponse)
    async def rerank(self, req: RerankRequest) -> RerankResponse: return await self._post("/rerank", req, RerankResponse)
    async def generate_hyde(self, req: HydeGenerationRequest) -> HydeGenerationResponse: return await self._post("/generate_hyde", req, HydeGenerationResponse)
    async def summarize_context(self, req: SummarizationRequest) -> SummarizationResponse: return await self._post("/summarize", req, SummarizationResponse)
    async def search_vectors(self, req: SimilaritySearchRequest) -> SimilaritySearchResponse: return await self._post("/vector_search", req, SimilaritySearchResponse)
    async def search_keywords(self, req: KeywordSearchRequest) -> KeywordSearchResponse: return await self._post("/keyword_search", req, KeywordSearchResponse)

    async def generate_stream(self, req: GenerationStreamRequest):
        async with self.client.stream("POST", "/generate_stream", json=req.model_dump(), timeout=None) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

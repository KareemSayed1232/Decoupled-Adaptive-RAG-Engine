
import asyncio
import os
import json
from typing import Dict, Any, AsyncGenerator, List, Optional
import aiofiles

from .inference_client import InferenceAPIClient
from shared_models.models import *
from .utils import logger, measure_performance

from pathlib import Path
from .config import settings , PROJECT_ROOT

class ContextBuilder:
    def __init__(self, client: InferenceAPIClient, base_context: str):
        self.client = client
        self.base_context = base_context
        self.summarization_min_docs = settings.summarization_min_docs
        self.reranker_rejection_threshold = settings.reranker_rejection_threshold

    async def build(self, reranked_results: List[RerankedResult]) -> str:
        if not reranked_results: return ""
        top_score = reranked_results[0].score
        logger.info(f"Top rank is :{top_score}")
        if top_score >= self.reranker_rejection_threshold:
            docs = [item.doc for item in reranked_results]
            return self.base_context + "\n\n" + "\n\n".join(docs)
        if top_score < self.reranker_rejection_threshold /2:
            return ''

        if len(reranked_results) >= self.summarization_min_docs:
            return await self._summarize_context(reranked_results)

        if reranked_results:
            return self.base_context + "\n\n" + reranked_results[0].doc
        return ""

    async def _summarize_context(self, results: List[RerankedResult]) -> str:
        docs_to_summarize = [item.doc for item in results[:self.summarization_min_docs]]
        summary_input = "\n\n---\n\n".join(docs_to_summarize)
        try:
            req = SummarizationRequest(context_to_summarize=summary_input)
            res = await self.client.summarize_context(req)
            return self.base_context + "\n\n" + res.summary
        except Exception as e:
            logger.error(f"Context summarization failed: {e}", exc_info=True)
            return self.base_context + "\n\n" + results[0].doc

class RAGOrchestrator:
    def __init__(self, client: InferenceAPIClient, context_builder: ContextBuilder, chunks: List[str]):
        self.client = client
        self.context_builder = context_builder
        self.chunks = chunks
        logger.info("RAGOrchestrator initialized.")

    @measure_performance()
    async def ask_question(self, question: str) -> AsyncGenerator[Dict, Any]:
        if not question.strip():
            yield {"type": "error", "content": "Question cannot be empty."}; return

        try:
            preprocess_req = PreprocessRequest(text=question)
            processed_question = (await self.client.preprocess_text(preprocess_req)).text

            context = await self._get_context(processed_question)
            if not context:
                yield {"type": "error", "content": "Sorry but looks like your question is out of my scope please try to ask question within my knowledge"}
                yield {"type": "stream_end"}
                return

            gen_req = GenerationStreamRequest(question=processed_question, context=context)
            async for event in self.client.generate_stream(gen_req):
                yield event

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}", exc_info=True)
            yield {"type": "error", "content": str(e)}
            yield {"type": "stream_end"}

    async def _get_context(self, question: str) -> str:
        hyde_req = HydeGenerationRequest(question=question)
        hyde_res = await self.client.generate_hyde(hyde_req)

        texts_to_embed = [question]
        if hyde_res and hyde_res.hypothetical_document:
            texts_to_embed.append(hyde_res.hypothetical_document)

        embed_req = EmbeddingRequest(texts=texts_to_embed)
        embeddings_res = await self.client.embed_texts(embed_req)

        sim_req = SimilaritySearchRequest(query=embeddings_res.embeddings)
        sim_res = await self.client.search_vectors(sim_req)

        key_req = KeywordSearchRequest(query=question)
        key_res = await self.client.search_keywords(key_req)

        indices = list(dict.fromkeys(sim_res.indices + key_res.indices))
        if not indices: return ""

        retrieved_chunks = [self.chunks[i] for i in indices if 0 <= i < len(self.chunks) and self.chunks[i].strip()]
        if not retrieved_chunks: return ""

        rerank_req = RerankRequest(question=question, extracted_docs=retrieved_chunks)
        rerank_res = await self.client.rerank(rerank_req)

        return await self.context_builder.build(rerank_res.reranked_results)

class AppState:
    rag_orchestrator: Optional[RAGOrchestrator] = None

app_state = AppState()


async def initialize_rag_system():
    artifact_path = os.path.join(Path(PROJECT_ROOT) , Path(settings.faiss_artifacts_path ))
    
    chunks_path = os.path.join(Path(artifact_path), Path(settings.faiss_chunks_path))
    base_context_path =  os.path.join(Path(artifact_path), Path(settings.faiss_chunks_path))
    logger.info(base_context_path)
    
    async with aiofiles.open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.loads(await f.read())
    async with aiofiles.open(base_context_path, 'r', encoding='utf-8') as f:
        base_context = await f.read()

    client = InferenceAPIClient()
    context_builder = ContextBuilder(client=client, base_context=base_context)
    
    app_state.rag_orchestrator = RAGOrchestrator(
        client=client,
        context_builder=context_builder,
        chunks=chunks,

    )

def get_rag_orchestrator() -> RAGOrchestrator:
    if not app_state.rag_orchestrator: raise RuntimeError("System not initialized.")
    return app_state.rag_orchestrator

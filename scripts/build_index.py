import asyncio
import json
import os
import sys
import pickle
from rank_bm25 import BM25Okapi
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from services.inference_api.src.config import settings as inference_settings
from services.inference_api.src.base_services.chunking import ChunkingService
from services.inference_api.src.base_services.embedding import EmbeddingService
from services.inference_api.src.base_services.search import SimilarityService
from services.inference_api.src.utils import logger

ARTIFACT_PATH = PROJECT_ROOT / "services" / "inference_api" / "artifacts"


async def create_and_save_chunks(chunking_service: ChunkingService):
    logger.info("--- Step 1: Processing and Saving Document Chunks ---")

    complete_context_path = PROJECT_ROOT / inference_settings.complete_context_file
    chunks_path = ARTIFACT_PATH / "chunks.json"

    with open(complete_context_path, 'r', encoding='utf-8') as f:
        complete_context = f.read()

    processed_context = await chunking_service.preprocess_text(complete_context)
    chunks = await chunking_service.chunk(content=processed_context)

    if not chunks:
        raise ValueError("No valid chunks were produced.")

    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    logger.info(f"Successfully created and saved {len(chunks)} chunks to {chunks_path}")
    return chunks

async def main():
    logger.info("Starting offline RAG artifact build process...")
    try:
        chunking_service = ChunkingService()
        embedding_service = EmbeddingService()

        similarity_service = SimilarityService()

        # --- Step 1: Create Chunks ---
        chunks = await create_and_save_chunks(chunking_service)

        logger.info("--- Step 2: Generating Embeddings ---")
        vectors = await embedding_service.encode(sentences=chunks)

        similarity_service.build_and_save_index(vectors)

        logger.info("--- Step 3: Building BM25 Keyword Index ---")
        tokenized_corpus = [doc.split(" ") for doc in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        bm25_index_path = ARTIFACT_PATH / 'bm25.index'
        with open(bm25_index_path, 'wb') as f:
            pickle.dump(bm25, f)

        logger.info(f"BM25 index built and saved successfully to {bm25_index_path}")
        logger.info("✅ RAG artifact build process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the build process: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
import re
import asyncio
from ..loader import ModelLoader
from fastapi import HTTPException
from ..config import settings
from ..utils import logger, measure_performance
import traceback
class RerankingService:
    def __init__(self):
            self.model = ModelLoader.get_reranking_model()
            self.top_k = settings.reranker_top_k
            self.batch_size = settings.reranker_batch_size
    @measure_performance()
    async def rerank(self, query, documents):
            
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.model.rank(
                    query=query,
                    documents=documents,
                    batch_size=self.batch_size,
                    top_k=self.top_k,
                ),
            )
            return {'reranked_results':[
                {"score":res.score,"doc":res.document}
                for res in result
            ]}
        
        except Exception as e:
            logger.error("Exception during rerank", exc_info=True)

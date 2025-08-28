from ..loader import ModelLoader
from ..utils import measure_performance
import asyncio

class EmbeddingService:
    def __init__(self):
        self.embedder = ModelLoader.get_embedding_model()

    @measure_performance()
    async def encode(self, sentences):
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.embedder.encode(sentences, precision="float32")
        )
        return embeddings.tolist()
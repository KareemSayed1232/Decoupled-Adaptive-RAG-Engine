from ..loader import ModelLoader
from ..utils import logger, measure_performance
from shared_models.models import SummarizationResponse
from typing import Optional, List
from ..config import settings
import torch

class SummarizationService:
    def __init__(self):
        self.model, self.tokenizer = ModelLoader.get_summarization_model_and_tokenizer()
        
        self.chunk_size = int(settings.summarization_chunk_size)
        self.chunk_overlap = int(settings.summarization_chunk_overlap)
        self.max_length = int(settings.summarization_max_length)
        self.min_length = int(settings.summarization_min_length)

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        start_index = 0
        while start_index < len(text):
            end_index = start_index + self.chunk_size
            chunks.append(text[start_index:end_index])
            start_index += self.chunk_size - self.chunk_overlap
        return chunks

    @measure_performance(unit="ms")
    def summarize(self, context_to_summarize: str) -> Optional[SummarizationResponse]:
        logger.info(f"Summarizing {len(context_to_summarize)} characters of context with low-level T5 generate...")

        try:
            docs = self._split_text(context_to_summarize)
            docs_to_summarize = [doc for doc in docs if len(doc) > self.min_length]
            if not docs_to_summarize:
                return SummarizationResponse(summary=context_to_summarize)

            logger.info(f"Split context into {len(docs_to_summarize)} chunks for summarization.")
            

            inputs = self.tokenizer(
                docs_to_summarize, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.model.device)

            with torch.no_grad():
                output_tokens = self.model.generate(
                    **inputs,
                    max_length=self.max_length, 
                    min_length=self.min_length,
                    do_sample=False
                )

            individual_summaries = self.tokenizer.batch_decode(
                output_tokens, 
                skip_special_tokens=True
            )

            combined_summary = "\n".join(individual_summaries)

            if not combined_summary:
                logger.warning("Summarization model returned an empty string.")
                return None
            
            return SummarizationResponse(summary=combined_summary)

        except Exception as e:
            logger.error(f"An unexpected error occurred during summarization: {e}", exc_info=True)
            return None
import re
from typing import List
from ..utils import logger
from ..config import settings

class ChunkingService:
    def __init__(self):
        self.chunk_size = settings.chunking_max_characters
        self.chunk_overlap = settings.chunking_overlap

        self.separators = ["\n\n", "\n", " ", ""]
        logger.info(f"ChunkingService initialized with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        
        final_chunks = []
        
        
        if len(text) <= self.chunk_size:
            return [text]
        
        
        current_separator = separators[0]
        next_separators = separators[1:]
        
        
        splits = text.split(current_separator)
        
        buffer = ""
        for i, split in enumerate(splits):
            
            if i > 0:
                split = current_separator + split

            
            if len(buffer) + len(split) <= self.chunk_size:
                buffer += split
            
            else:
                if buffer:  
                    final_chunks.append(buffer)
                
                if len(split) > self.chunk_size:
                    if not next_separators: 
                         final_chunks.append(split)
                    else:
                         final_chunks.extend(self._recursive_split(split, next_separators))
                else: 
                    buffer = split
        
        
        if buffer:
            final_chunks.append(buffer)
            
        return final_chunks

    async def chunk(self, content: str) -> List[str]:
        
        logger.info(f"Starting recursive text chunking on {len(content)} characters...")
        
        
        chunks = self._recursive_split(content, self.separators)
        
        
        
        final_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
            
        logger.info(f"Successfully created {len(final_chunks)} chunks.")
        return final_chunks

    async def preprocess_text(self, text: str) -> str:
        
        logger.info('Preprocessing text')
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r'[\t ]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
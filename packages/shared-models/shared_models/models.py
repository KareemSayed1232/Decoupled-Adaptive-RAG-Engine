from typing import List, Annotated, Optional , Union
from pydantic import BaseModel, Field, field_validator , model_validator , ConfigDict


class ChunkingRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, description="Text to be chunked.")]

    @field_validator('text')
    @classmethod
    def text_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('text must not be empty or just whitespace')
        return v
    
class ChunkingResponse(BaseModel):
    texts: Annotated[List[str], Field(min_length=1, description="List of chunked text pieces.")]











class PreprocessRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, description="Text to preprocess.")]

    @field_validator('text')
    @classmethod
    def text_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('text must not be empty or just whitespace')
        return v

class PreprocessResponse(BaseModel):
    text: str











class EmbeddingRequest(BaseModel):
    texts: Annotated[List[str], Field(min_length=1, description="List of texts to embed.")]

    @field_validator('texts')
    @classmethod
    def texts_must_not_contain_empty_strings(cls, v: List[str]) -> List[str]:
        if any(not s.strip() for s in v):
            raise ValueError('strings in the list must not be empty or just whitespace')
        return v

class EmbeddingResponse(BaseModel):
    embeddings: Annotated[List[List[float]], Field(min_length=1, description="List of embedding vectors.")]
    










class HydeGenerationRequest(BaseModel):
    question: Annotated[str, Field(min_length=1)]

    @field_validator('question')
    @classmethod
    def text_fields_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('field must not be empty or just whitespace')
        return v

class HydeGenerationResponse(BaseModel):
    hypothetical_document: Annotated[str, Field(
        min_length=1,
        description="A single, generated document that plausibly answers the user's question."
    )]











class GenerationStreamRequest(BaseModel):
    question: Annotated[str, Field(min_length=1)]
    context: Annotated[str, Field(min_length=1)]

    @field_validator('question', 'context')
    @classmethod
    def text_fields_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('field must not be empty or just whitespace')
        return v

class StreamValidationError(BaseModel):
    error: str
    raw_text: str
    

class LayoutDetailed(BaseModel):
    model_config = ConfigDict(extra="ignore")
    main_idea: str
    text: str
    key_points: List[str]

class LayoutTextOnly(BaseModel):
    model_config = ConfigDict(extra="ignore")
    main_idea: str
    text: str

class LayoutBrief(BaseModel):
    model_config = ConfigDict(extra="ignore")
    main_idea: str

class GeneratedAnswerContainer(BaseModel):
    main_idea: Optional[str] = None
    text: Optional[str] = None
    points: Optional[list[str]] = None
    details: Optional[dict] = None
    def get_chosen_layout(self):
        for field_name in type(self).model_fields:
            if getattr(self, field_name) is not None:
                return self
        return None

GeneratedAnswer = Union[LayoutDetailed, LayoutTextOnly, LayoutBrief]












class RerankRequest(BaseModel):
    question: Annotated[str, Field(min_length=1)]
    extracted_docs: Annotated[List[str], Field(min_length=1)]

    @field_validator('question')
    @classmethod
    def question_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('question must not be empty or just whitespace')
        return v

    @field_validator('extracted_docs')
    @classmethod
    def docs_must_not_contain_empty_strings(cls, v: List[str]) -> List[str]:
        if any(not s.strip() for s in v):
            raise ValueError('strings in extracted_docs must not be empty or just whitespace')
        return v

class RerankedResult(BaseModel):
    score: float
    doc: Annotated[str, Field(min_length=1)]

class RerankResponse(BaseModel):
    reranked_results: Annotated[List[RerankedResult], Field(min_length=0)]












class SimilaritySearchRequest(BaseModel):
    query: Annotated[List[List[float]], Field(min_length=1, description="List of query vectors for similarity search.")]
    
class SimilaritySearchResponse(BaseModel):
    distances: Annotated[List[float], Field(min_length=1, description="Distances from query to each returned vector.")]
    indices: Annotated[List[int], Field(min_length=1, description="Indices of the matching vectors in the FAISS index.")]












class KeywordSearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1)]

class KeywordSearchResponse(BaseModel):
    indices: Annotated[List[int], Field(min_length=0)]












class SummarizationRequest(BaseModel):

    context_to_summarize: Annotated[str, Field(
        min_length=1,
        description="A string containing one or more document chunks"
    )]

    @field_validator('context_to_summarize')
    @classmethod
    def context_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('context_to_summarize must not be empty or just whitespace')
        return v

class SummarizationResponse(BaseModel):

    summary: Annotated[str, Field(
        min_length=1,
        description="A concise, factually-grounded summary"
    )]
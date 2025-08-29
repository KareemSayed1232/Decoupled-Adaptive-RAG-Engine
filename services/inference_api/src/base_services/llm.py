from typing import Optional, Iterator, Dict, Any, Type, TypeVar
from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel, ValidationError
from ..loader import ModelLoader, PromptLoader
from ..config import settings
from ..utils import logger, measure_performance
from shared_models.models import HydeGenerationResponse , GeneratedAnswerContainer , HydeResponse
import json


PydanticModel = TypeVar("PydanticModel", bound=BaseModel)

class LLMService:
    def __init__(self):
        self.llm: Llama = ModelLoader.get_llm_model()

        self.hyde_llm: Llama = ModelLoader.get_hyde_model() 

        self.prompt_loader = PromptLoader()
        self.gen_prompt_template = self.prompt_loader.load(settings.gen_prompt_file_name)
        self.hyde_prompt_template = self.prompt_loader.load(settings.hyde_prompt_file_name)

        self.gen_temperature = settings.gen_temperature
        self.gen_max_tokens = settings.gen_max_tokens
        self.gen_top_p = settings.gen_top_p
        self.gen_stop_words = settings.gen_stop_words

        self.hyde_temperature = settings.hyde_temperature
        self.hyde_max_new_tokens = settings.hyde_max_new_tokens
        self.hyde_top_p = settings.hyde_top_p
        self.hyde_stop_words = settings.hyde_stop_words
        
        logger.info("LLMService initialized with Main and HyDE models.")

    def _prepare_grammar(self, model: Type[BaseModel]) -> Optional[LlamaGrammar]:
        try:
            schema_json = json.dumps(model.model_json_schema())
            return LlamaGrammar.from_json_schema(schema_json)
        except Exception as e:
            logger.error(f"Error preparing grammar for model {model.__name__}: {e}", exc_info=True)
            return None

    def _format_messages(self, prompt_template: dict, **kwargs) -> list[dict]:
        formatted_messages = []
        try:
            for message_template in prompt_template["messages"]:
                content = message_template["content"]
                for key, value in kwargs.items():
                    content = content.replace(f"{{{key}}}", str(value))
                formatted_messages.append({
                    "role": message_template["role"],
                    "content": content
                })
            return formatted_messages
        except KeyError as e:
            logger.error(f"Missing a key in prompt template for formatting: {e}", exc_info=True)
            raise
    

    def _create_structured_output(
            self,
            llama_instance: Llama,
            messages: list[dict],
            response_model: Type[PydanticModel],
            temperature: float,
            max_tokens: int,
            top_p: float,
            stop: list[str] | None = None,
        ) -> Optional[PydanticModel]:
            
            grammar = self._prepare_grammar(response_model)
            if not grammar:
                return None

            raw_output = "" 
            try:
                response = llama_instance.create_chat_completion(
                    messages=messages,
                    grammar=grammar,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                )
                raw_output = response['choices'][0]['message']['content']
                logger.info(f"Raw structured output received from LLM: {raw_output}")
                
                return response_model.model_validate_json(raw_output)

            except (ValidationError, json.JSONDecodeError) as e:
                logger.error(f"Pydantic validation failed for {response_model.__name__}: {e}\nRaw output: {raw_output}", exc_info=True)
                return None
            except Exception as e:
                logger.error(f"An unexpected error occurred during LLM call for {response_model.__name__}: {e}", exc_info=True)
                return None


    @measure_performance(unit="ms")
    def generate_hyde(self, question: str) -> Optional[HydeGenerationResponse]:
        logger.info(f"Generating HyDE document for question: '{question[:50]}...'")
        
        try:

            
            schema_str = json.dumps(HydeResponse.model_json_schema())
            messages = self._format_messages(
                self.hyde_prompt_template,
                question=question,
                schema=schema_str 
            )
        except KeyError:
            return None 

        structured_response = self._create_structured_output(
            llama_instance=self.hyde_llm,
            messages=messages,
            response_model=HydeResponse,
            temperature=self.hyde_temperature,
            max_tokens=self.hyde_max_new_tokens,
            top_p=self.hyde_top_p,
            stop=self.hyde_stop_words
        )
        if structured_response:
             return HydeGenerationResponse(hypothetical_document=structured_response.hypothetical_document)
        return None



    @measure_performance()
    def generate_stream(self, question: str, context: str) -> Iterator[str]:


        
        logger.info(f"Starting DYNAMIC TOKEN stream for question: '{question[:50]}...'")
        
        grammar = self._prepare_grammar(GeneratedAnswerContainer)
        if not grammar:
            raise ValueError("Failed to prepare the dynamic response container grammar.")

        try:
            schema_str = json.dumps(GeneratedAnswerContainer.model_json_schema(), indent=2)
            
            messages = self._format_messages(
                self.gen_prompt_template,
                schema=schema_str,
                context=context,
                question=question
            )
        except KeyError as e:
            logger.error(f"Missing key in prompt template: {e}", exc_info=True)
            raise ValueError(f"Failed to format dynamic generation prompt: {e}")
            
        stream = self.llm.create_chat_completion(
            messages=messages,
            grammar=grammar,
            temperature=self.gen_temperature,
            max_tokens=self.gen_max_tokens,
            stream=True,
        )

        for chunk in stream:
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content:
                yield content
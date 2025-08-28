import os
import json
from ..utils import logger
from pathlib import Path
class PromptLoader:
    def __init__(self, base_path: str = 'prompts'):
        self.base_path = Path(__file__).parent.parent / base_path

    def load(self, file_name: str):
        file_path = os.path.join(self.base_path, file_name)
        _, extension = os.path.splitext(file_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if extension == '.json':
                    return json.load(f)
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from {file_path}")
            raise

    
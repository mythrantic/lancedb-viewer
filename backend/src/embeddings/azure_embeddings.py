import logging
from typing import List, Union
import numpy as np
from lancedb.embeddings import TextEmbeddingFunction
from lancedb.embeddings.registry import register

@register("azure_openai")
class AzureOpenAIEmbeddings(TextEmbeddingFunction):
    def __init__(self):
        try:
            from routes.setup import AzureOpenAiConfig
            from azure.identity import DefaultAzureCredential
            from openai import AzureOpenAI
            
            self.config = AzureOpenAiConfig()
            self.credentials = DefaultAzureCredential()
            self.client = AzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_version=self.config.api_version,
                api_key=self.config.get_openai_api_key(self.credentials),
                max_retries=5,
            )
            self.model_name = self.config.text_embedder_lagre.deployment_name
            self._dim = self.config.text_embedder_lagre.ndims
            
        except Exception as e:
            raise ImportError(f"Failed to initialize Azure OpenAI: {e}")

    def ndims(self) -> int:
        return self._dim

    def generate_embeddings(self, texts: Union[List[str], np.ndarray]) -> List[np.array]:
        try:
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            return [v.embedding for v in response.data]
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise

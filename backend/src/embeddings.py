import logging
from typing import List, Union
import numpy as np
from functools import cached_property
from openai import AzureOpenAI
from lancedb.embeddings import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from azure.identity import DefaultAzureCredential
from routes.setup import AzureOpenAiConfig

openai_config = AzureOpenAiConfig()

@register("azure_openai")
class AzureOpenAIEmbeddings(TextEmbeddingFunction):
    """Azure OpenAI embeddings implementation"""
    
    name: str = openai_config.text_embedder_lagre.deployment_name
    
    def __init__(self):
        try:
            from routes.setup import AzureOpenAiConfig
            from azure.identity import DefaultAzureCredential
            
            self.config = AzureOpenAiConfig()
            self.credentials = DefaultAzureCredential()
            self._setup_azure()
        except ImportError as e:
            raise ImportError("Azure dependencies not available") from e

    def _setup_azure(self):
        """Set up Azure-specific configuration"""
        self.name = self.config.text_embedder_lagre.deployment_name
        self.azure_api_key = self.config.get_openai_api_key(self.credentials)
        self.azure_endpoint = self.config.endpoint
        self.azure_deployment = self.config.text_embedder_lagre.deployment_name
        self.azure_api_version = self.config.text_embedder_lagre.api_version

    def ndims(self) -> int:
        return openai_config.text_embedder_lagre.ndims

    def generate_embeddings(self, texts: Union[List[str], np.ndarray]) -> List[np.array]:
        try:
            response = self._azure_openai_client.embeddings.create(
                input=texts, 
                model=self.name
            )
            return [v.embedding for v in response.data]
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise

    @cached_property
    def _azure_openai_client(self):
        return AzureOpenAI(
            azure_endpoint=openai_config.endpoint,
            api_version=openai_config.api_version,
            api_key=self.azure_api_key,
            max_retries=5,
        )

@register("simple")
class SimpleEmbeddings(TextEmbeddingFunction):
    """Simple embeddings for testing - uses hash of text"""
    
    def __init__(self):
        super().__init__()
        self._dimensions = 64  # Small dimension for testing
        
    def ndims(self) -> int:
        return self._dimensions

    def embed(self, data: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for input text(s)"""
        if isinstance(data, str):
            data = [data]
        return np.array(self.generate_embeddings(data))

    def generate_embeddings(self, texts: Union[List[str], np.ndarray]) -> List[np.array]:
        """Generate simple embeddings using hash of text"""
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
            
        embeddings = []
        for text in texts:
            # Create a bounded hash value (between 0 and 2**32 - 1)
            hash_val = hash(text) & 0xFFFFFFFF  # Mask to 32 bits
            np.random.seed(hash_val)
            embedding = np.random.normal(0, 1, self._dimensions)
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
            
        return embeddings

def get_embedder(provider="simple"):
    """Get embedder instance based on provider"""
    from lancedb.embeddings import get_registry
    
    if provider == "azure":
        try:
            from .azure_embeddings import AzureOpenAIEmbeddings
            return get_registry().get("azure_openai").create()
        except Exception as e:
            logging.warning(f"Azure OpenAI embedder not available: {e}")
            logging.info("Falling back to simple embedder")
            return get_registry().get("simple").create()
    
    return get_registry().get("simple").create()

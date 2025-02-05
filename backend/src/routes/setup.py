from dataclasses import dataclass
import os
from functools import cached_property
from pydantic import BaseModel

# Optional Azure imports
try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from azure.keyvault.secrets import SecretClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Optional OpenAI imports
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


KEYVAULT_URL = "https://***.vault.azure.net/"
AZURE_OPENAI_ENDPOINT = "https://****.openai.azure.com/"
AZURE_COGNITIVE_SERVICE_SCOPE = "https://cognitiveservices.azure.com/.default"
API_VERSION = "2024-08-01-preview"


@dataclass
class OpenAiModelConfig:
    '''! also part of old code.
    '''
    deployment_name: str
    api_version: str = "2024-02-01"
    ndims: int = 1024
    
@dataclass
class AzureOpenAiConfig:
    def __init__(self, **kwargs):
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure dependencies not installed. Install with: "
                "pip install azure-identity azure-keyvault-secrets"
            )
        self.keyvault_url: str = KEYVAULT_URL
        self.endpoint: str = AZURE_OPENAI_ENDPOINT
        self.credential_scope: str = AZURE_COGNITIVE_SERVICE_SCOPE
        self.api_version: str = API_VERSION
        self.llm: str = "gpt-4o-2"
        self.embedding_model: str = "text-embedding-3-large"
        self.gpt4_o = OpenAiModelConfig(
            deployment_name="gpt-4o", api_version="2024-05-13", ndims=128_000)
        self.gpt4_turbo = OpenAiModelConfig(
            deployment_name="gpt-4-turbo",
            api_version="turbo-2024-04-09",
            ndims=128_000
        )
        self.ada_002_embedding = OpenAiModelConfig(
            deployment_name="aez-dev-ada-002", api_version="2", ndims=1536)
        self.text_embedder_lagre = OpenAiModelConfig(
            deployment_name="text-embedding-3-large",
            api_version="2024-04-09",
            # Number of dimensions of the embeddings (depends on the model)
            ndims=3072
        )

    def get_token_provider(self, credential):
        if not AZURE_AVAILABLE:
            raise ImportError("Azure dependencies not installed")
        return get_bearer_token_provider(credential, self.credential_scope)

    def get_openai_api_key(self, credential):
        if not AZURE_AVAILABLE:
            raise ImportError("Azure dependencies not installed")
        secret_client = SecretClient(
            vault_url=self.keyvault_url, credential=credential)
        secret = secret_client.get_secret("azure-openai-api-key")
        return secret.value


@dataclass
class StorageConfig:
    """Configuration for storage settings"""
    provider: str = "local"
    connection_string: str = None
    container_name: str = None
    local_path: str = "lancedb_data"
    credentials: dict = None

@dataclass
class DatabaseConfig:
    """Configuration for the database connection"""
    storage: StorageConfig
    table_name: str = "default"
    embedder_provider: str = "simple"  # Default to simple embedder
    
@dataclass
class AppConfig:
    """Main application configuration"""
    database: DatabaseConfig = DatabaseConfig(
        storage=StorageConfig()
    )

    @classmethod
    def from_environment(cls):
        """Create configuration from environment variables"""
        storage_provider = os.getenv("STORAGE_PROVIDER", "local")
        embedder_provider = os.getenv("EMBEDDER_PROVIDER", "simple")
        
        storage_config = StorageConfig(
            provider=storage_provider,
            local_path=os.getenv("LOCAL_DB_PATH", "lancedb_data")
        )
        
        if storage_provider == "azure":
            # Only import and configure Azure if explicitly requested
            try:
                from azure.identity import DefaultAzureCredential
                storage_config.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                storage_config.container_name = os.getenv("AZURE_STORAGE_CONTAINER")
            except ImportError:
                raise ImportError("Azure dependencies required for Azure storage")
                
        elif storage_provider == "s3":
            storage_config.credentials = {
                "bucket": os.getenv("AWS_S3_BUCKET"),
                "access_key": os.getenv("AWS_ACCESS_KEY_ID"),
                "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY")
            }
            
        return cls(
            database=DatabaseConfig(
                storage=storage_config,
                table_name=os.getenv("DB_TABLE_NAME", "default"),
                embedder_provider=embedder_provider
            )
        )


class LLM:
    """A wrapper around AzureOpenAI service, to simplify the functionality that the backend needs."""

    def __init__(
        self,
        credential: DefaultAzureCredential,
        config: AzureOpenAiConfig = AzureOpenAiConfig(),
    ):
        """_summary_

        Args:
            credential (DefaultAzureCredential): _description_
            config (AzureOpenAiConfig, optional): _description_. Defaults to AzureOpenAiConfig().
        """

        assert isinstance(config, AzureOpenAiConfig)
        self.config = config
        self.openai = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_ad_token_provider=get_bearer_token_provider(
                credential,
                config.credential_scope
            ),
            api_version=API_VERSION
        )

    def complete(self, query: str, system_message: str = "", max_tokens: int = 500):
        """
        Generates a completion for the given query using the OpenAI API.
        Args:
            query (str): The input text for which the completion is to be generated.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.
        Returns:
            dict: The response from the OpenAI API containing the generated completion.
        """

        messages = [
            {"role": "user", "content": query},
            {"role": "system", "content": system_message},
        ]

        result = self.openai.chat.completions.create(
            model=self.config.llm,
            temperature=0.7,
            messages=messages,
            max_tokens=max_tokens,
        )

        return result.choices[0].message.content

    def embed(self, input: list[str]):

        embeddings = self._azure_openai_client.embeddings.create(
            model=self.config.embedding_model, input=input)
        
        result = []

        for i,embedding in enumerate(embeddings.data):
            result.append({
                "text": input[i],
                "vector": embedding.embedding
            })

        return result

    @cached_property
    def _azure_openai_client(self):
        if not os.environ.get("OPENAI_API_KEY") and not self.config.get_openai_api_key(DefaultAzureCredential()):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return AzureOpenAI(
            azure_endpoint=self.config.endpoint,
            # azure_ad_token_provider=token_provdier,
            api_version=self.config.api_version,
            api_key=self.config.get_openai_api_key(DefaultAzureCredential()),
            max_retries=5,
        )
    

    def structured_response(self, query: str, pydantic_model: BaseModel):
        """
        Generates a structured response from the OpenAI API based on the provided query and pydantic model.

        Args:
            query (str): The input query to be processed by the OpenAI API.
            pydantic_model (BaseModel): The Pydantic model to structure the response.
        Returns:
            pydantic_model: An instance of the provided Pydantic model populated with the structured response.
        """
        response = self.openai.beta.chat.completions.parse(
            model=self.config.llm,
            temperature=0,
            messages=[
                {"role": "system", "content": "Extract the relevant information."},
                {"role": "user", "content": query},
            ],
            response_format=pydantic_model,
        )

        return response.choices[0].message.parsed

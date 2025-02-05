from dataclasses import dataclass
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI
from pydantic import BaseModel
from functools import cached_property
import os


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
    # Contains the necessary configuration for the Azure OpenAI API
    keyvault_url: str = KEYVAULT_URL
    endpoint: str = AZURE_OPENAI_ENDPOINT
    credential_scope: str = AZURE_COGNITIVE_SERVICE_SCOPE
    api_version: str = API_VERSION
    llm: str = "gpt-4o-2"
    embedding_model: str = "text-embedding-3-large"
    
    #! TODO from old code this below could removed. but you have to check it is not used in other places
    gpt4_o = OpenAiModelConfig(
        deployment_name="gpt-4o", api_version="2024-05-13", ndims=128_000)
    gpt4_turbo = OpenAiModelConfig(
        deployment_name="gpt-4-turbo",
        api_version="turbo-2024-04-09",
        ndims=128_000
    )
    ada_002_embedding = OpenAiModelConfig(
        deployment_name="aez-dev-ada-002", api_version="2", ndims=1536)
    text_embedder_lagre = OpenAiModelConfig(
        deployment_name="text-embedding-3-large",
        api_version="2024-04-09",
        # Number of dimensions of the embeddings (depends on the model)
        ndims=3072
    )

    def get_token_provider(self, credential: DefaultAzureCredential):
        return get_bearer_token_provider(credential, self.credential_scope)

    def get_openai_api_key(self, credential: DefaultAzureCredential):
        secret_client = SecretClient(
            vault_url=self.keyvault_url, credential=credential)
        secret = secret_client.get_secret("azure-openai-api-key")
        return secret.value


@dataclass
class StorageConfig:
    """Configuration for storage settings"""
    provider: str
    connection_string: str = None
    container_name: str = None
    local_path: str = None
    credentials: dict = None


@dataclass
class DatabaseConfig:
    """Configuration for the database connection"""
    storage: StorageConfig
    table_name: str = "default"
    
@dataclass
class AppConfig:
    """Main application configuration"""
    openai: AzureOpenAiConfig = AzureOpenAiConfig()
    database: DatabaseConfig = DatabaseConfig(
        storage=StorageConfig(
            provider="local",
            local_path="lancedb_data"
        )
    )

    @classmethod
    def from_environment(cls):
        """Create configuration from environment variables"""
        storage_provider = os.getenv("STORAGE_PROVIDER", "local")
        
        if storage_provider == "azure":
            storage_config = StorageConfig(
                provider="azure",
                connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
                container_name=os.getenv("AZURE_STORAGE_CONTAINER")
            )
        elif storage_provider == "s3":
            storage_config = StorageConfig(
                provider="s3",
                credentials={
                    "bucket": os.getenv("AWS_S3_BUCKET"),
                    "access_key": os.getenv("AWS_ACCESS_KEY_ID"),
                    "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY")
                }
            )
        else:
            storage_config = StorageConfig(
                provider="local",
                local_path=os.getenv("LOCAL_DB_PATH", "lancedb_data")
            )
            
        return cls(
            database=DatabaseConfig(
                storage=storage_config,
                table_name=os.getenv("DB_TABLE_NAME", "default")
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

    def embed(self, input: str | list[str]):

        if isinstance(input,str):
            input = [input]

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

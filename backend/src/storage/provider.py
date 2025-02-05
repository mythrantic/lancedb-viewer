from abc import ABC, abstractmethod
from dataclasses import dataclass
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import os
from urllib.parse import urlparse

@dataclass
class StorageConfig:
    provider: str  # 'local', 'azure', 's3'
    connection_string: str = None
    container_name: str = None
    local_path: str = None
    credentials: dict = None
    uri: str = None  # Add support for direct URIs
    
    def __post_init__(self):
        """Parse URI if provided"""
        if self.uri:
            parsed = urlparse(self.uri)
            if parsed.scheme == 'file':
                self.provider = 'local'
                self.local_path = parsed.path
            elif parsed.scheme == 'az':
                self.provider = 'azure'
                self.container_name = parsed.netloc
            elif parsed.scheme == 's3':
                self.provider = 's3'
                self.credentials = {'bucket': parsed.netloc}

class StorageProvider(ABC):
    @abstractmethod
    def get_uri(self) -> str:
        """Return the URI for LanceDB to connect to"""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Test if the connection is valid"""
        pass

class LocalStorageProvider(StorageProvider):
    def __init__(self, config: StorageConfig):
        self.path = config.local_path or "lancedb_data"
        os.makedirs(self.path, exist_ok=True)

    def get_uri(self) -> str:
        return self.path
    
    def validate_connection(self) -> bool:
        return os.path.exists(self.path)

class AzureBlobStorageProvider(StorageProvider):
    def __init__(self, config: StorageConfig):
        if config.connection_string:
            self.client = BlobServiceClient.from_connection_string(config.connection_string)
        else:
            self.client = BlobServiceClient(
                account_url=config.credentials.get('account_url'),
                credential=DefaultAzureCredential()
            )
        self.container_name = config.container_name

    def get_uri(self) -> str:
        return f"az://{self.container_name}"
    
    def validate_connection(self) -> bool:
        try:
            self.client.get_container_client(self.container_name).get_container_properties()
            return True
        except Exception:
            return False

class S3StorageProvider(StorageProvider):
    def __init__(self, config: StorageConfig):
        self.bucket = config.credentials.get('bucket')
        # Add S3 client initialization here

    def get_uri(self) -> str:
        return f"s3://{self.bucket}"
    
    def validate_connection(self) -> bool:
        # Implement S3 connection validation
        return True

def create_storage_provider(config: StorageConfig) -> StorageProvider:
    providers = {
        'local': LocalStorageProvider,
        'azure': AzureBlobStorageProvider,
        's3': S3StorageProvider
    }
    provider_class = providers.get(config.provider.lower())
    if not provider_class:
        raise ValueError(f"Unsupported storage provider: {config.provider}")
    return provider_class(config)

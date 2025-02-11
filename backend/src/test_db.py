import pandas as pd
from routes.setup import AppConfig, DatabaseConfig
from storage.provider import StorageConfig
from routes.manager import LanceDBManager
import pyarrow as pa
import numpy as np

def test_local_db():
    """Test local database operations"""
    # Connect to local database
    config = AppConfig(
        database=DatabaseConfig(
            storage=StorageConfig(
                provider="local",
                local_path="lancedb_data"
            )
        )
    )
    
    db = LanceDBManager(config)
    embedder = db._get_embedder()
    
    # Define schema for the test table including vector column
    schema = pa.schema([
        ('id', pa.int32()),
        ('text', pa.string()),
        ('category', pa.string()),
        ('vector', pa.list_(pa.float32(), embedder.ndims()))  # Vector column
    ])
    
    # Create sample data
    base_data = [
        {"id": 1, "text": "Hello world", "category": "greeting"},
        {"id": 2, "text": "How are you?", "category": "question"},
        {"id": 3, "text": "Goodbye world", "category": "farewell"}
    ]
    # Add vector embeddings to the data
    data = []
    for item in base_data:
        vector = embedder.generate_embeddings([item['text']])[0]
        item['vector'] = vector
        data.append(item)
    
    try:
        # Create table with schema
        db.create_table("test_table", schema=schema, overwrite=True)
        print("Created table successfully")
        
        # Add data
        db.add_data("test_table", data, unique_field="id")
        print("Added data successfully")
        
        # Test fetching data
        results = db.fetch_data("test_table", as_pandas=True, columns_to_exclude=['vector'])
        print("\nFetched data:")
        print(results)
        
        # Test vector search
        search_results = db.vector_search("test_table", "hello", limit=2, columns_to_exclude=['vector'])
        print("\nVector search results:")
        print(search_results)
        
    except Exception as e:
        print(f"Error during test: {e}")
        raise
    
    return db

if __name__ == "__main__":
    db = test_local_db()

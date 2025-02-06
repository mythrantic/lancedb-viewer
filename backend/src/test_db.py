import pandas as pd
from routes.setup import AppConfig, DatabaseConfig
from storage.provider import StorageConfig
from routes.manager import LanceDBManager

def test_local_db():
    """Test local database operations"""
    # Connect to local database
    config = AppConfig(
        database=DatabaseConfig(
            storage=StorageConfig(
                provider="local",
                local_path="test_db"
            )
        )
    )
    
    db = LanceDBManager(config)
    
    # Create sample data
    data = [
        {"id": 1, "text": "Hello world", "category": "greeting"},
        {"id": 2, "text": "How are you?", "category": "question"},
        {"id": 3, "text": "Goodbye world", "category": "farewell"}
    ]
    
    # Create table and add data
    db.create_table("test_table", schema=None, overwrite=True)
    db.add_data("test_table", data, unique_field="id")
    
    # Test fetching data
    results = db.fetch_data("test_table", as_pandas=True)
    print("\nFetched data:")
    print(results)
    
    # Test vector search
    search_results = db.vector_search("test_table", "hello", limit=2)
    print("\nVector search results:")
    print(search_results)
    
    return db

if __name__ == "__main__":
    db = test_local_db()

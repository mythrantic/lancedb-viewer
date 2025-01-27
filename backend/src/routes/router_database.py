import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from manager import LanceDBManager  # Import LanceDBManager
import hashlib
import numpy as np

router = APIRouter()

# Initialize LanceDBManager
# Use default database URL and Azure credentials and embedder
db_manager = LanceDBManager()


@router.post("/api/add-data/", tags=["Database"])
async def add_data(request: Request):
    """
    Adds data to the specified table.

    Args:

        request (Request): Body: 
            {
                "table": "table_name", 
                "data": [
                    {
                        "field1": "value1", 
                        "field2": "value2", 
                        ...
                    }
                ]
            }

    Returns:
        dict: Success message. and the result of the operation.

    Raises:
        HTTPException: If an error occurs while adding data.
    """
    try:
        data = await request.json()
        table = data["table"]
        records = data["data"]

        if table == "user":
            for record in records:
                record["user_id"] = hashlib.sha256(
                    (record["usename"] + record["email"]).encode('utf-8')).hexdigest()
            no_of_items_added = db_manager.add_data("user", records, unique_field="user_id")
        else:
            raise HTTPException(status_code=400, detail="Invalid table name")

        if no_of_items_added == 0:
            return {"message": f"All items already exist in the {table} table", "no_of_items_added": 0, "success": False}
        else:
            return {
                "success": True,
                "message": f"{no_of_items_added} records added to {table} table successfully",
                "no_of_items_added": no_of_items_added
            }

    except Exception as e:
        logging.exception("Exception occurred in add_data: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/update-data/", tags=["Database"])
async def update_data(request: Request):
    """
    updates data to the specified table.

    Args:

        request (Request): Body: 
            {
                "table": "table_name", 
                "data": [
                    {
                        "field1": "value1", 
                        "field2": "value2", 
                        ...
                    }
                ]
            }

    Returns:
        dict: Success message. and the result of the operation.

    Raises:
        HTTPException: If an error occurs while adding data.
    """
    try:
        data = await request.json()
        table = data["table"]
        records = data["data"]

        if table == "user":
            for record in records:
                record["user_id"] = hashlib.sha256(
                    (record["usename"] + record["email"]).encode('utf-8')).hexdigest()
            no_of_items_added = db_manager.update_data("user", records, unique_field="user_id")
        else:
            raise HTTPException(status_code=400, detail="Invalid table name")

        if no_of_items_added == 0:
            return {"message": f"All items is the exact same in the {table} table", "no_of_items_added": 0, "success": False}
        else:
            return {
                "success": True,
                "message": f"{no_of_items_added} records updated to {table} table successfully",
                "no_of_items_added": no_of_items_added
            }

    except Exception as e:
        logging.exception("Exception occurred in update_data: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/fetch-data/{table}/", tags=["Database"])
def fetch_data(table: str, columns_to_exclude: str = "", page: int = 1, per_page: int = 10, filter: str = None) -> JSONResponse:
    """
    Fetches data from the specified table with pagination and optional filtering.

    Args:
        table (str): The name of the table to fetch data from.
        page (int): Page number for pagination.
        per_page (int): Number of items per page.
        filter (str): SQL filter expression. Example: these are the filters that can be used - https://lancedb.github.io/lancedb/sql/#sql-filters
        columns_to_exclude (str): Comma-separated list of columns to exclude from the fetched data.

    Returns:
        dict: The fetched data.

    Raises:
        HTTPException: If an error occurs while fetching data.
    """
    try:
        # as_pandas=True returns a DataFrame
        data = db_manager.fetch_data(table, as_pandas=True, page=page, per_page=per_page, filter=filter, columns_to_exclude=columns_to_exclude.split(","))
        data_json = data.map(lambda x: x.tolist() if isinstance(
            x, np.ndarray) else x).to_dict(orient="records")
        return JSONResponse(content={
            "page": page,
            "per_page": per_page,
            "total": len(data_json),
            "data": data_json
        })
    except Exception as e:
        logging.exception("Exception occurred in fetch_data: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/vector-search/", tags=["Database"])
async def vector_search(request: Request):
    """
    Performs a vector search on the specified table.

    Args:
        request (Request): Body: {"table": "table_name", "query": "search_query", "limit": 50, "columns_to_exclude": "vector,_rowid"}

    Returns:
        dict: The search results.

    Raises:
        HTTPException: If an error occurs while performing the vector search.
    """
    try:
        data = await request.json()
        table = data["table"]
        query = data["query"]
        limit = data.get("limit", 50)
        columns_to_exclude = data.get("columns_to_exclude", "")

        results = db_manager.vector_search(table, query, limit, columns_to_exclude=columns_to_exclude.split(","))
        data_json = results.map(lambda x: x.tolist() if isinstance(
            x, np.ndarray) else x).to_dict(orient="records")
        return {
            "total": len(data_json),
            "data": data_json
        }
    except Exception as e:
        logging.exception("Exception occurred in vector_search: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

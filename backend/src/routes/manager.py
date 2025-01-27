import logging
import sys
import os

from typing import List, Dict, Any, Union
from lancedb.embeddings.utils import api_key_not_found_help
from lancedb.embeddings.registry import register
from lancedb.embeddings import TextEmbeddingFunction, get_registry
import numpy as np
from functools import cached_property
from openai import AzureOpenAI
from setup import AzureOpenAiConfig
from azure.identity import DefaultAzureCredential
import lancedb
import pandas as pd 

# add the root directory to the path so we can import the modules not in this directory
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.getcwd()), ".")))

openai_config = AzureOpenAiConfig()
try:
    credentials = DefaultAzureCredential()
    token_provdier = openai_config.get_token_provider(DefaultAzureCredential())
except Exception as e:
    logging.warning(
        "Please sure to run 'az login' in the container.")



# make the custom embedder function caus ethe one provided by lancedb is not working
@register("azure_openai")
class AzureOpenAIEmbeddings(TextEmbeddingFunction):
    """
    An embedding function that uses the Azure OpenAI API
    """

    name: str = openai_config.text_embedder_lagre.deployment_name
    azure_api_key: str = openai_config.get_openai_api_key(
        credential=credentials)
    azure_endpoint: str = openai_config.endpoint
    azure_deployment: str = openai_config.text_embedder_lagre.deployment_name
    azure_api_version: str = openai_config.text_embedder_lagre.api_version

    def ndims(self):
        return self._ndims

    @cached_property
    def _ndims(self):
        if self.name == openai_config.text_embedder_lagre.deployment_name:
            return openai_config.text_embedder_lagre.ndims
        else:
            raise ValueError(f"Unknown model name {self.name}")

    # def ndims(self):
    #     """
    #     Return the dimensionality of the embeddings.
    #     """
    #     if self.name == openai_config.text_embedder_lagre.deployment_name:
    #         return self._ndims
    #     else:
    #         raise ValueError(f"Unknown model name {self.name}")

    def generate_embeddings(
        self, texts: Union[List[str], np.ndarray]
    ) -> List[np.array]:
        """
        Get the embeddings for the given texts

        Parameters
        ----------
        texts: list[str] or np.ndarray (of str)
            The texts to embed
        """
        # TODO retry, rate limit, token limit
        if self.name == openai_config.text_embedder_lagre.deployment_name:
            rs = self._azure_openai_client.embeddings.create(
                input=texts, model=self.name
            )
        else:
            rs = self._azure_openai_client.embeddings.create(
                input=texts, model=self.name, dimensions=self.ndims()
            )
        return [v.embedding for v in rs.data]

    @cached_property
    def _azure_openai_client(self):
        if not os.environ.get("OPENAI_API_KEY") and not self.azure_api_key:
            api_key_not_found_help("openai")
        return AzureOpenAI(
            azure_endpoint=openai_config.endpoint,
            # azure_ad_token_provider=token_provdier,
            api_version=openai_config.api_version,
            api_key=openai_config.get_openai_api_key(DefaultAzureCredential()),
            max_retries=5,
        )


embedder = get_registry().get("azure_openai").create()


def get_embedder():
    return get_registry().get("azure_openai").create()

# nest_asyncio.apply()

# # Create Azure Credentials
# credentials = DefaultAzureCredential()
# azure_credentials = {
#     "azure_storage_account_name": "****",
#     "azure_tenant_id": "****",
#     "azure_storage_token": credentials.get_token(
#         "https://storage.azure.com/.default"
#     ).token,
# }


def get_azure_storage_options(credentials: DefaultAzureCredential = DefaultAzureCredential()):
    return {
        "azure_storage_account_name": "account_name",
        "azure_tenant_id": "tenant_id",
        "azure_storage_token": credentials.get_token(
            "https://storage.azure.com/.default"
        ).token,
    }


class LanceDBManager:
    def __init__(
        self,
        database_url: str = "az://dev-vector-db",
        storage_options: Dict[str, Any] = get_azure_storage_options(),
        embedder: TextEmbeddingFunction = embedder,
        async_mode: bool = True,
    ):
        """
        Initialize LanceDBManager with the database URL, Azure credentials, and an embedder.

        Args:
            database_url (str): The path or URL to the LanceDB database.
            azure_credentials (dict): Azure storage credentials.
            embedder (TextEmbeddingFunction): Embedder instance for generating embeddings.
        """
        self.database_url = database_url
        self.storage_options = storage_options
        self.embedder = embedder
        self.async_mode = async_mode
        self.connection = lancedb.connect(
            uri=self.database_url, storage_options=self.storage_options
        )
        
    def _get_unique_ids(self, table, unique_field):
        existing_ids = table.search().limit(table.count_rows()).select([unique_field]).to_list()
        existing_ids = [item[unique_field] for item in existing_ids]
        return existing_ids
    
    def _format_input_data(self, data: Union[pd.DataFrame, List[Dict[str, Any]]])-> List[Dict[str, Any]]:
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")
        if isinstance(data, dict):
            data = [data]
        return data
    
    
        

    @property
    def table_names(self):
        """
        Retrieve a list of table names in the database.

        Returns:
            List of table names.
        """
        try:
            return self.connection.table_names()
        except Exception as e:
            logging.error(f"Error fetching table names: {e}")
            raise

    def get_table(self, table_name: str):
        """
        Get a table object from the database.

        Args:
            table_name (str): Name of the table to retrieve.

        Returns:
            Table: LanceDB table object.
        """
        try:
            return  self.connection.open_table(table_name)
        except Exception as e:
            logging.error(f"Error getting table '{table_name}': {e}")
            raise

    async def create_schema(self, table_name: str, schema: Any):
        """
        Create a schema-based table in LanceDB.

        Args:
            table_name (str): Name of the table.
            schema (Any): Schema of the table.
        """
        try:
            table = self.connection.create_table(
                table_name, schema=schema, exist_ok=True
            )
        except Exception as e:
            logging.error(f"Error creating schema for table '{
                          table_name}': {e}")

    def create_table(self, table_name: str, schema: Any, overwrite: bool = False):
        """
        Create a new table in LanceDB.

        Args:
            table_name (str): Name of the table.
            schema (Any): Schema of the table.
            overwrite (bool): Whether to overwrite the table if it exists.
        """
        try:
            mode = "overwrite" if overwrite else "create"
            self.connection.create_table(table_name, schema=schema, mode=mode)
            logging.info(f"Table '{table_name}' created successfully.")
        except Exception as e:
            logging.error(f"Error creating table '{table_name}': {e}")
            
            

    def add_data(self, table_name: str, data: List[Dict[str, Any]], unique_field: str):
        """
        Add data to a LanceDB table, avoiding duplicates based on specified unique field.

        Args:
            table_name (str): Name of the table.
            data (List[Dict[str, Any]]): List of data entries to add.
            unique_field (str): Field to use for uniqueness check.

        Returns:
            int: Number of rows added.
        """
        if not unique_field:
            raise ValueError("Unique field must be specified to check for duplicates.")

        try:
            table = self.connection.open_table(table_name)
            existing_ids = self._get_unique_ids(table, unique_field)
            data = self._format_input_data(data)
            new_ids = [item[unique_field] for item in data] 
            
            #Get the difference between the existing ids and the new ids using set
            new_ids = list(set(new_ids) - set(existing_ids))
    
            logging.info(f"Found {len(new_ids)} new entries to add to table '{table_name}'.")

            #Filter the data to only include the new ids
            new_data = [item for item in data if item[unique_field] in new_ids]
            if len(new_data) == 0:
                logging.info(f"No new entries to add to table '{table_name}'.")
                return
            
            else:
                table.add(new_data)
                logging.info(f"Added {len(new_data)} entries to table '{table_name}'.")
            
        except Exception as e:
            logging.error(f"Error adding data to table '{table_name}': {e}")
            

    def update_data(self, table_name: str, data: List[Dict[str, Any]], unique_field: str):
        """
        Update data in a LanceDB table based on specified unique field.

        Args:
            table_name (str): Name of the table.
            data (List[Dict[str, Any]]): List of data entries to update.
            unique_field (str): Field to use for matching records.

        Returns:
            int: Number of rows updated.
        """
        if not unique_field:
            raise ValueError("Unique field must be specified for updates.")

        try:
            
            table =  self.connection.open_table(table_name)
            data = self._format_input_data(data)
            
            update_count = 0
            for row in data:
                # Extract the unique field value and other fields to update
                unique_value = row[unique_field]
                updates = {k: v for k, v in row.items() if k != unique_field}
                
                # Create where clause for this row
                where_clause = f"{unique_field} = \"{unique_value}\""
                
                # Update the matching row using the correct parameter name 'updates'
                table.update(values=updates, where=where_clause)
                update_count += 1

            logging.info(f"Updated {update_count} entries in table '{table_name}'.")
            return update_count

        except Exception as e:
            logging.error(f"Error updating data in table '{table_name}': {e}")
            raise

    def fetch_data(
        self,
        table_name: str,
        as_pandas: bool = True,
        page: int = 1,
        per_page: int = 10,
        filter: str = None,
        columns_to_exclude: List[str] = [],
    ):
        """
        Fetch data from a LanceDB table with pagination and optional filtering.

        Args:
            table_name (str): Name of the table.
            as_pandas (bool): Whether to return data as a pandas DataFrame.
            page (int): Page number for pagination.
            per_page (int): Number of items per page. Use -1 to fetch all data.
            filter (str): SQL filter expression. these are the filters that can be used - https://lancedb.github.io/lancedb/sql/#sql-filters
            columns_to_exclude (List[str]): List of columns to exclude from the results.

        Returns:
            DataFrame or List[Dict]: Fetched data.
            List[Dict]: Fetched data as a list of dictionaries if as_pandas is set to False.
        """
        # docs used to make this function: https://lancedb.github.io/lancedb/sql/#pre-and-post-filtering
        try:
            table = self.connection.open_table(table_name)
            query = table.search()

            # dont include the vector column in the results .select(["title", "text", "_distance"]) is used to define the columns to be returned
            # !DANGER The paranthesis around async_table.to_pandas() is used to make sure that the head function is called on the dataframe and not coroutine

            columns_to_include = [
                col for col in table.schema.names if col not in columns_to_exclude
            ]

            # if _rowid is not included in the columns to include then it will not be returned
            query = query.select(columns_to_include) if ("_rowid" in columns_to_exclude) else query.select(columns_to_include).with_row_id(True) 

            # the "await async_table.count_rows()" is done like that beacause there is a bug in lancedb v0.17.0 that does not respect the limit(-1) when used with where clause
            # https://github.com/lancedb/lancedb/issues/1852
            if filter:
                query = (
                    query.where(filter).limit(
                        per_page).offset((page - 1) * per_page)
                    if per_page != -1
                    else query.where(filter).limit(table.count_rows())
                )
            else:
                query = (
                    query.limit(per_page).offset((page - 1) * per_page)
                    if per_page != -1
                    else query.limit(table.count_rows())
                )
            df = query.to_pandas()
            return df if as_pandas else df.to_dict(orient="records")
        except Exception as e:
            logging.error(f"Error fetching data from table '{
                          table_name}': {e}")
            raise
       

    def vector_search(
        self,
        table_name: str,
        query: str,
        limit: int = 5,
        as_pandas: bool = True,
        columns_to_exclude: List[str] = [],
    ):
        """
        Perform a vector search on a LanceDB table.

        Args:
            table_name (str): Name of the table.
            query (str): Query text to search for.
            limit (int): Number of search results to return.
            as_pandas (bool): Whether to return data as a pandas DataFrame.
            columns_to_exclude (List[str]): List of columns to exclude from the results.

        Returns:
            DataFrame: Search results.
            List[Dict]: Search results as a list of dictionaries. if as_pandas is set to False
        """
        try:
            table = self.connection.open_table(table_name)

            # dont include the vector column in the results .select(["title", "text", "_distance"]) is used to define the columns to be returned
            # !DANGER The paranthesis around async_table.to_pandas() is used to make sure that the head function is called on the dataframe and not coroutine

            columns_to_include = [
                col for col in table.schema.names if col not in columns_to_exclude
            ]

            # Generate embedding for the query
            embedding = self.embedder.generate_embeddings([query])[0]

            # Perform vector search
            # results = await async_table.vector_search(embedding).limit(limit).to_pandas()
            results = (
                table.search(query=embedding)
                .select(columns_to_include)
                .with_row_id(with_row_id=True)  
                .limit(limit)
                .to_pandas()
            )

            return results if as_pandas else results.to_dict(orient="records")
        except Exception as e:
            logging.error(
                f"Error performing vector search on table '{table_name}': {e}"
            )
            raise

    def delete_table(self, table_name: str):
        """
        Delete a table from LanceDB.

        Args:
            table_name (str): Name of the table to delete.
        """
        try:
            self.connection.drop_table(table_name)
            logging.info(f"Table '{table_name}' deleted successfully.")
            return True
        except Exception as e:
            logging.error(f"Error deleting table '{table_name}': {e}")
            raise

    def delete_rows(self, table_name: str, condition: str):
        """
        Delete rows from a table based on a condition.

        Args:
            table_name (str): Name of the table.
            condition (str): Condition to match rows for deletion.
        """
        try:
            table = self.connection.open_table(table_name)
            table.delete(where=condition)
            logging.info(
                f"Rows matching condition '{condition}' deleted from table '{
                    table_name
                }'."
            )
        except Exception as e:
            logging.error(f"Error deleting rows from table '{
                          table_name}': {e}")
            raise

    async def delete_duplicates(self, table_name: str, subset: List[str]):
        """
        Remove duplicate rows from a LanceDB table based on specified columns.

        Args:
            table_name (str): Name of the table.
            subset (List[str]): List of column names to check for duplicates.

        Returns:
            int: Number of duplicate rows removed.
        """
        try:
            table = self.connection.open_table(table_name)

            # Fetch the table's data
            df = table.to_pandas()

            # Drop duplicates based on the specified subset
            df_unique = df.drop_duplicates(subset=subset)
            duplicates_removed = len(df) - len(df_unique)

            if duplicates_removed > 0:
                # Overwrite the table with the unique data
                table.add(
                    df_unique.to_dict(orient="records"), mode="overwrite"
                )
                logging.info(
                    f"Removed {duplicates_removed} duplicate rows from table '{
                        table_name
                    }'."
                )
            else:
                logging.info(f"No duplicates found in table '{table_name}'.")

            return duplicates_removed
        except Exception as e:
            logging.error(f"Error deleting duplicates from table '{
                          table_name}': {e}")
            raise
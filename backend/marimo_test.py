

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Lancedb Viewer""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""This to test the code that is gonna be the lancedb-viewer backend. the goal is it is gonna be able to support lance db hosted at different places. starting with loca, azure blob, s3 etc""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "How is this app implemented?": """
        - The bible is tokenized into chunks, which are embedded using
        Jina's `jina-embeddings-v3`.
        - Your question is embedded using the same model.
        - We use an approximate k-nearest neighbor search on the PDF embeddings to
        retrieve relevant chunks.
        - The most relevant chunks are added to the context of your prompt, which
        is processed by a GPT model.
        """
    })
    return


@app.cell
def _():
    import os
    from groq import Groq
    from ollama import Client
    from dotenv import load_dotenv
    import marimo as mo

    load_dotenv(
        "/Users/gormery/Desktop/projects/bible/src/ai/lancedb/.env"
    )  # take environment variables from .env

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY") if os.environ.get("GROQ_API_KEY") else mo.ui.text(label="🤖 Groq Key", kind="password")
    JINA_API_KEY = os.environ.get("JINA_API_KEY") if os.environ.get("JINA_API_KEY") else mo.ui.text(label="🦾 Jina Key", kind="password")
    return Client, GROQ_API_KEY, Groq, JINA_API_KEY, mo


@app.cell(hide_code=True)
def _(GROQ_API_KEY, JINA_API_KEY, mo):
    config = mo.hstack([GROQ_API_KEY, JINA_API_KEY])
    mo.accordion(
        {
            "⚙️ Config -  here is all thats needed for the notebook to run and function correctly. both of these are FREE!! baby!!": config
        }
    )
    return


@app.cell
def _(Client, GROQ_API_KEY, Groq):
    client = Groq(
        api_key=GROQ_API_KEY
    ) # uses the default api key in the environment

    ollamaClient = Client(
      host='https://ollama.valiantlynx.com'
    )
    return client, ollamaClient


@app.cell
def _(mo, ollamaClient):

    ollamaResponse = ollamaClient.chat(
        model='gemma3:latest', messages=[
          {
            'role': 'user',
            'content': 'Why is the sky blue? in three brief and short technichal sentences',
          }
        ]
    )


    mo.md(ollamaResponse.message.content)
    return


@app.cell(hide_code=True)
def _(mo):
    SYSTEM = """
        You will recieve verses of the bible.
        Answer the subsequent question using that context.
        If you dont know just say you dont know
    """
    mo.md(
        f"""
        We set a system message to determine how our agent Model will behave.

        This is the system message:
        **{SYSTEM}**
        """
    )
    return (SYSTEM,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        f"""
        ⚡️✨Test this chat below to see ✨⚡️
        """
    )
    return


@app.cell
def _(BibleSchema, SYSTEM, bible_query_model, bible_table, mo, ollamaClient):
    # Define the function to query the Bible table and ollama
    def query_model(messages, config):
        """
        Custom RAG (Retrieval-Augmented Generation) model for querying the Bible.

        Args:
            messages (List[ChatMessage]): The chat history, including the user question.
            config (ChatModelConfig): The configuration for the LLM.

        Returns:
            str: The LLM-generated response.
        """
        # Extract the latest user message
        user_message = messages[-1].content

        # Helper function to extract and sort context from LanceDB rows
        def extract_context(rows):
            return sorted(
                [{"full_text": r.full_text, "verse_id": r.verse_id} for r in rows],
                key=lambda x: x["verse_id"],
            )

        # Query the Bible table for context
        rows = bible_table.search(user_message).limit(100).to_pydantic(BibleSchema)
        context = extract_context(rows)

        if not context:
            return "No relevant context found in the database."

        # Prepare the context and question for the Groq model
        ollama_messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Question: {user_message}"},
        ]

        # Query the Groq model
        response = ollamaClient.chat.completions.create(
            messages=ollama_messages,
            model="llama3.1",
            stream=False,
        )

        # Stream and collect the response chunk by chunk
        # print("**Groq Model Response:**\n")
        # response_text = ""
        # for chunk in stream:
        #    if hasattr(chunk, "choices") and chunk.choices:
        #        for choice in chunk.choices:
        #            if choice.delta and choice.delta.content:
        #                print(choice.delta.content, end="", flush=True)
        #                response_text += choice.delta.content

        return response.choices[0].message.content

    # Configure the Marimo chat UI with the Bible query model
    ollama_chat = mo.ui.chat(bible_query_model)

    # Display the chat UI
    ollama_chat
    return


@app.cell
def _(BibleSchema, SYSTEM, bible_table, client, mo):
    # Define the function to query the Bible table and Groq model
    def bible_query_model(messages, config):
        """
        Custom RAG (Retrieval-Augmented Generation) model for querying the Bible.

        Args:
            messages (List[ChatMessage]): The chat history, including the user question.
            config (ChatModelConfig): The configuration for the LLM.

        Returns:
            str: The LLM-generated response.
        """
        # Extract the latest user message
        user_message = messages[-1].content

        # Helper function to extract and sort context from LanceDB rows
        def extract_context(rows):
            return sorted(
                [{"full_text": r.full_text, "verse_id": r.verse_id} for r in rows],
                key=lambda x: x["verse_id"],
            )

        # Query the Bible table for context
        rows = bible_table.search(user_message).limit(100).to_pydantic(BibleSchema)
        context = extract_context(rows)

        if not context:
            return "No relevant context found in the database."

        # Prepare the context and question for the Groq model
        groq_messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Question: {user_message}"},
        ]

        # Query the Groq model
        response = client.chat.completions.create(
            messages=groq_messages,
            model="llama3-8b-8192",
            stream=False,
        )

        # Stream and collect the response chunk by chunk
        # print("**Groq Model Response:**\n")
        # response_text = ""
        # for chunk in stream:
        #    if hasattr(chunk, "choices") and chunk.choices:
        #        for choice in chunk.choices:
        #            if choice.delta and choice.delta.content:
        #                print(choice.delta.content, end="", flush=True)
        #                response_text += choice.delta.content

        return response.choices[0].message.content

    # Configure the Marimo chat UI with the Bible query model
    chat = mo.ui.chat(bible_query_model)

    # Display the chat UI
    chat
    return (bible_query_model,)


@app.cell
def _(mo):
    import pandas as pd
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    from lancedb.embeddings import EmbeddingFunctionRegistry
    from tqdm import tqdm
    import requests


    # Initialize the Jina embedder through LanceDB
    registry = EmbeddingFunctionRegistry.get_instance()
    jina_embedder = registry.get("jina").create() # uses the api key in the environment
    ollama_embedder  = registry.get("ollama").create(name="nomic-embed-text", host="https://ollama.valiantlynx.com")

    # Confirm embedding dimensions from Jina
    EMBEDDING_DIM = ollama_embedder.ndims()
    print(f"Using embedding dimension: {EMBEDDING_DIM}")

    # Define schema for Bible embeddings
    class BibleSchema(LanceModel):
        text: str = ollama_embedder.SourceField()
        embedding: Vector(EMBEDDING_DIM) = ollama_embedder.VectorField()
        verse_id: int
        book_name: str
        chapter: int
        verse: int
        full_text: str

    # LanceDB setup
    db_path = "bible_lancedb_bob"
    db = lancedb.connect(db_path)
    bible_table = db.create_table("bible_embeddings", schema=BibleSchema, exist_ok=True)

    # Load Bible data
    bible_df = pd.read_csv(
        "/home/valiantlynx/projects/lancedb-viewer/backend/web.csv",
        names=["Verse ID", "Book Name", "Book Number", "Chapter", "Verse", "Text"],
        skiprows=1,  # Skip the header row since column names are explicitly defined
    )

    # Ensure correct data types and add concatenated metadata text
    bible_df["Verse ID"] = bible_df["Verse ID"].astype(int)
    bible_df["Book Number"] = bible_df["Book Number"].astype(int)
    bible_df["Chapter"] = bible_df["Chapter"].astype(int)
    bible_df["Verse"] = bible_df["Verse"].astype(int)
    bible_df["full_text"] = bible_df.apply(
        lambda row: f"{row['Book Name']} {row['Chapter']}:{row['Verse']} - {row['Text']}",
        axis=1,
    )

    # Log current database status
    existing_rows = bible_table.to_pandas()
    print(f"Total rows in the database: {len(existing_rows)}")
    existing_ids = set(existing_rows["verse_id"])
    print(f"Already embedded rows: {len(existing_ids)}")

    # Determine unembedded rows
    unembedded_bible_data = []
    for _, row in tqdm(bible_df.iterrows(), total=len(bible_df), desc="Checking embedding status"):
        if row["Verse ID"] not in existing_ids:
            unembedded_bible_data.append(
                {
                    "text": str(row["Text"]),
                    "verse_id": int(row["Verse ID"]),
                    "book_name": str(row["Book Name"]),
                    "chapter": int(row["Chapter"]),
                    "verse": int(row["Verse"]),
                    "full_text": str(row["full_text"]),
                }
            )

    print(f"Unembedded rows to process: {len(unembedded_bible_data)}")

    # Embed unembedded rows if there are any
    # Embed unembedded rows if there are any
    if unembedded_bible_data:
        print(f"Embedding {len(unembedded_bible_data)} new rows...")
    
        # Define the Ollama API endpoint (default local address)
        OLLAMA_API_URL = "https://ollama.valiantlynx.com/api/embeddings"
    
        for batch_start in tqdm(range(0, len(unembedded_bible_data), 100), desc="Embedding batches"):
            batch = unembedded_bible_data[batch_start:batch_start + 100]
            # Prepare texts for embedding
            texts = [row["full_text"] for row in batch]
        
            try:
                # Process each text individually through Ollama's Nomic embed model
                batch_embeddings = []
            
                for text in texts:
                    # Get embeddings from local Ollama using Nomic embed model
                    response = requests.post(
                        OLLAMA_API_URL,
                        json={
                            "model": "nomic-embed-text",
                            "prompt": text
                        }
                    )
                
                    if response.status_code != 200:
                        print(f"Error: {response.status_code}, Response: {response.text}")
                        continue
                
                    embedding = response.json().get("embedding", [])
                    batch_embeddings.append(embedding)
            
                # Add embeddings to LanceDB
                for idx, row in enumerate(batch):
                    if idx < len(batch_embeddings):
                        # Validate embedding dimensions
                        embedding = batch_embeddings[idx]
                        if len(embedding) != EMBEDDING_DIM:
                            print(f"Error embedding batch starting at index {batch_start}: Length of item not correct. Got {len(embedding)}, expected {EMBEDDING_DIM}")
                            continue
                        row["embedding"] = [float(e) for e in embedding]
                
                bible_table.add(batch)
            
            except Exception as e:
                print(f"Error embedding batch starting at index {batch_start}: {e}")
    else:
        print("No new rows to embed.")

    # Final database status
    final_rows = bible_table.to_pandas()
    print(f"Final rows in the database: {len(final_rows)}")
    print(f"Newly embedded rows: {len(final_rows) - len(existing_rows)}")

    mo.md(
        f"""
        This is the code for getting the datasett making embeddings out of all of them and saving them  to lacedb. ready to be used for quering
        """
    )
    return BibleSchema, bible_table, pd


@app.cell
def _(bible_table, mo, pd):
    import numpy as np
    from sklearn.decomposition import PCA

    # Query LanceDB for embeddings
    all_embeddings = np.array([row.embedding for row in bible_table.to_pandas().itertuples()])
    print(f"Embeddings shape: {all_embeddings.shape}")

    # PCA for dimensionality reduction
    pca = PCA(n_components=2, whiten=True)
    pca_result = pca.fit_transform(all_embeddings)

    # Visualization data
    embedding_plot = pd.DataFrame(
        {
            "x": pca_result[:, 0],
            "y": pca_result[:, 1],
            "full_text": [row.full_text for row in bible_table.to_pandas().itertuples()],
        }
    )

    mo.md(
        f"""
        Now to visualize the embedding we can use many thing. for example below we used PCA
        """
    )
    return (embedding_plot,)


@app.cell
def _(bible_table, mo):
    # Fetch all data from LanceDB and convert to a DataFrame
    db_data = bible_table.to_pandas()

    stuff_to_accord = mo.hstack([mo.ui.table(db_data)])
    mo.accordion({f"**This is how the data is like in the lancedb: total length: {len(db_data)} rows**": stuff_to_accord})
    return


@app.cell
def _(mo):
    mo.md(
        f"""
        Here's a PCA **embedding of bible verses**: each point represents a
        verse, with similar verses close to each other. The data is from the csv dataset. im thinking of indexing the entire bible.

        This notebook will automatically drill down into points you **select with
        your mouse**; try it!
        """
    )
    return


@app.cell
def _(chart, mo):
    table_ui = mo.ui.table(chart.value)
    return


@app.cell
def _(alt):
    def scatter(df):
        return (
            alt.Chart(df)
            .mark_circle()
            .encode(
                x=alt.X("x:Q").scale(),
                y=alt.Y("y:Q").scale(),
                color=alt.Color("full_text:N"),
            )
            .properties(width=500, height=500)
        )
    return (scatter,)


@app.cell
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install("altair")

    import altair as alt
    return (alt,)


@app.cell
def _(embedding_plot, mo, scatter):
    chart = mo.ui.altair_chart(scatter(embedding_plot))
    chart
    return (chart,)


if __name__ == "__main__":
    app.run()

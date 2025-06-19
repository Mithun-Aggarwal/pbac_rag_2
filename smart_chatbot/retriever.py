# smart_chatbot/retriever.py

import chromadb
from typing import List, Dict

def retrieve_relevant_chunks(
    query_embedding: List[float],
    collection: chromadb.Collection,
    config: dict
) -> Dict:
    """
    Queries ChromaDB to find the most relevant document chunks.

    Args:
        query_embedding (List[float]): The vectorized user query.
        collection (chromadb.Collection): The ChromaDB collection object.
        config (dict): Application configuration (for top_k).

    Returns:
        Dict: The results from ChromaDB, including documents, metadatas, and distances.
    """
    # Look for top_k in the vector_db config, defaulting to 5
    top_k = config.get("vector_db", {}).get("top_k_results", 5)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"]
    )
    return results
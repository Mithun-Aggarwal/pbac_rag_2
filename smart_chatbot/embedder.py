# embedder.py

import requests
import json

def embed_query(text, config):
    """
    Sends a query to Ollama's embedding API (e.g., nomic-embed-text).
    
    Args:
        text (str): The user's query
        config (dict): Loaded config.yaml

    Returns:
        List[float]: Embedding vector for the query
    """
    model = config.get("embedding_model", "nomic-embed-text")
    url = config.get("ollama_url", "http://localhost:11434/api/embeddings")
    
    payload = {
        "model": model,
        "prompt": text
    }

    response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    response.raise_for_status()

    data = response.json()
    return data.get("embedding", [])

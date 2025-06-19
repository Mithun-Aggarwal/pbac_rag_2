# smart_chatbot/embedder.py

import requests
import json
import google.generativeai as genai
import os

def embed_query(text: str, config: dict) -> list[float]:
    """
    Embeds a user query using the configured provider (Gemini or Ollama).
    """
    embedding_config = config.get('embedding', {})
    provider = embedding_config.get('provider')

    if provider == 'gemini':
        # --- CORRECT: Gemini Embedding Logic ---
        gemini_config = embedding_config.get('gemini', {})
        model_name = gemini_config.get('model', 'models/text-embedding-004')
        try:
            # Use the correct function: embed_content
            result = genai.embed_content(
                model=model_name,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error embedding query with Gemini: {e}")
            return []

    elif provider == 'ollama':
        # --- CORRECT: Ollama Embedding Logic ---
        ollama_config = embedding_config.get('ollama', {})
        model = ollama_config.get('model', 'nomic-embed-text')
        url = ollama_config.get('url', 'http://localhost:11434/api/embeddings')
        
        payload = {"model": model, "prompt": text}
        try:
            response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])
        except requests.RequestException as e:
            print(f"Error embedding query with Ollama: {e}")
            return []
    
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
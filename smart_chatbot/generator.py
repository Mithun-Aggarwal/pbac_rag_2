# smart_chatbot/generator.py

import requests
import json
from smart_chatbot.prompts import build_prompt

def generate_response(user_query: str, context_chunks: dict, config: dict) -> str:
    """
    Generates a grounded response using a generative LLM (via Ollama).

    Args:
        user_query (str): The question from the user.
        context_chunks (dict): The context snippets retrieved from ChromaDB.
        config (dict): Model and endpoint configuration.

    Returns:
        str: The answer from the LLM.
    """
    # Configuration for the generative model (e.g., llama3)
    gen_model_config = config.get('extraction', {}).get('local', {})
    model = gen_model_config.get('model', 'llama3:latest')
    url = gen_model_config.get('ollama_url')

    system_prompt = build_prompt()

    # Build a detailed context string from the ChromaDB results
    context_str = ""
    if context_chunks.get("documents") and context_chunks["documents"][0]:
        for i in range(len(context_chunks['documents'][0])):
            doc_text = context_chunks['documents'][0][i]
            meta = context_chunks['metadatas'][0][i]
            
            doc_title = meta.get('doc_title', 'N/A')
            doc_id = meta.get('doc_id', 'N/A')
            
            context_str += f"Source (doc_id: {doc_id}, title: {doc_title}):\n"
            context_str += f"Content: {doc_text}\n---\n"
    else:
        context_str = "No relevant context was found in the database."

    # Construct the final payload for the LLM
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nQUESTION: {user_query}"}
        ],
        "stream": False
    }

    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "Error: No content in response.")
        return content.strip()
    except requests.RequestException as e:
        return f"Error: Could not connect to the generative model. {e}"
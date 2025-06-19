# smart_chatbot/generator.py

import requests
import json
import google.generativeai as genai
from smart_chatbot.prompts import build_prompt

def generate_response(user_query: str, context_chunks: dict, config: dict) -> str:
    """
    Generates a grounded response using the configured generative LLM (Gemini or Ollama).
    """
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

    final_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION: {user_query}"
    
    extraction_config = config.get('extraction', {})
    provider = extraction_config.get('provider')

    try:
        if provider == 'gemini':
            # --- CORRECT: Gemini Generation Logic ---
            gemini_config = extraction_config.get('gemini', {})
            model_name = gemini_config.get('model', 'gemini-1.5-flash-latest')
            model = genai.GenerativeModel(
                model_name,
                system_instruction=system_prompt
            )
            response = model.generate_content(final_prompt)
            return response.text.strip()

        elif provider == 'local':
            # --- CORRECT: Ollama Generation Logic ---
            gen_model_config = extraction_config.get('local', {})
            model = gen_model_config.get('model', 'llama3:latest')
            url = gen_model_config.get('ollama_url')
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_prompt}
                ],
                "stream": False
            }
            response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "Error: No content in response.")
            return content.strip()
        
        else:
            raise ValueError(f"Unsupported extraction provider: {provider}")

    except Exception as e:
        return f"Error connecting to the generative model: {e}"
# pipeline/embedding_generator.py

"""
Embedding Generator Module
--------------------------
This script takes a validated JSON file, creates context-aware text chunks,
and generates vector embeddings using a configured provider (Ollama or Gemini).
"""

import os
import json
import requests
import google.generativeai as genai
from typing import Dict, List, Any, Iterator
import logging

# Configure the Gemini client if the API key is available
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini API has a limit of 100 documents per batch for embedding
GEMINI_BATCH_SIZE = 100

def _create_chunks_from_validated_json(validated_data: Dict[str, Any], chunk_size: int, chunk_overlap: int) -> Iterator[Dict[str, Any]]:
    """
    Creates context-aware chunks from the 'sections' of a validated JSON object.
    This function processes each section independently to preserve semantic boundaries.
    """
    doc_meta_context = {
        "doc_id": validated_data.get("doc_id"),
        "doc_title": validated_data.get("title"),
        "doc_type": validated_data.get("doc_type"),
        "drug_name": validated_data.get("drug_name"),
        "indication": validated_data.get("indication"),
        "outcome": validated_data.get("outcome")
    }
    
    chunk_id_counter = 0
    for section in validated_data.get("sections", []):
        section_heading = section.get("heading", "No Heading")
        section_text_raw = section.get("text", "")

        if isinstance(section_text_raw, list):
            section_text = " ".join(map(str, section_text_raw))
        else:
            section_text = section_text_raw

        if not section_text or len(section_text.split()) < 10:
            continue
            
        context_header = f"Document Title: {doc_meta_context['doc_title']}\nSection: {section_heading}\n\n"
        
        words = section_text.split()
        start_index = 0
        while start_index < len(words):
            end_index = start_index + chunk_size
            chunk_words = words[start_index:end_index]
            
            chunk_text_for_embedding = context_header + " ".join(chunk_words)

            yield {
                "chunk_id": f"{doc_meta_context['doc_id']}_{chunk_id_counter}",
                "text_for_embedding": chunk_text_for_embedding,
                "metadata": {
                    **doc_meta_context,
                    "section_heading": section_heading,
                    "page_start": section.get("page_start")
                }
            }
            chunk_id_counter += 1
            start_index += chunk_size - chunk_overlap
            if start_index >= len(words):
                break

def _embed_with_ollama(chunks: List[Dict[str, Any]], config: Dict, logger) -> List[Dict[str, Any]]:
    ollama_config = config['embedding']['ollama']
    logger.info(f"Embedding {len(chunks)} chunks using Ollama model: {ollama_config['model']}")
    
    for chunk in chunks:
        try:
            response = requests.post(
                ollama_config['url'],
                json={"model": ollama_config['model'], "prompt": chunk['text_for_embedding']}
            )
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if not embedding:
                raise ValueError("API returned an empty embedding vector.")
            
            chunk['embedding'] = embedding
        except Exception as e:
            logger.error(f"Ollama embedding failed for chunk {chunk['chunk_id']}: {e}")
            chunk['embedding'] = None
            chunk['error'] = str(e)
    return chunks

def _embed_with_gemini(chunks: List[Dict[str, Any]], config: Dict, logger) -> List[Dict[str, Any]]:
    """
    Corrected version that uses `embed_content` and processes in batches.
    """
    gemini_config = config['embedding']['gemini']
    model_name = gemini_config['model']
    logger.info(f"Embedding {len(chunks)} chunks using Gemini model: {model_name}")

    # Get the text from all chunks to be embedded
    texts_to_embed = [chunk['text_for_embedding'] for chunk in chunks]
    
    all_embeddings = []
    
    # Process the texts in batches to respect API limits
    for i in range(0, len(texts_to_embed), GEMINI_BATCH_SIZE):
        batch_texts = texts_to_embed[i:i + GEMINI_BATCH_SIZE]
        logger.info(f"Processing batch {i//GEMINI_BATCH_SIZE + 1}...")
        
        try:
            # CORRECTED: Use 'embed_content' and 'content' parameter
            result = genai.embed_content(
                model=model_name,
                content=batch_texts,
                task_type="retrieval_document" # Recommended for RAG
            )
            all_embeddings.extend(result['embedding'])
        except Exception as e:
            logger.error(f"Gemini embedding failed for a batch: {e}")
            # Add empty embeddings for the failed batch to maintain list size
            all_embeddings.extend([None] * len(batch_texts))
            
    # Assign the generated embeddings back to their corresponding chunks
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk['embedding'] = embedding
        if not embedding:
            chunk['error'] = "Batch embedding failed."
            
    return chunks

def generate_embeddings_for_document(validated_data: Dict[str, Any], config: Dict, logger) -> Dict[str, Any]:
    embedding_config = config['embedding']
    provider = embedding_config['provider']
    chunking_config = embedding_config['chunking']

    logger.info(f"Starting embedding generation with provider: '{provider}'")

    chunks = list(_create_chunks_from_validated_json(
        validated_data,
        chunk_size=chunking_config['size'],
        chunk_overlap=chunking_config['overlap']
    ))
    
    if not chunks:
        logger.warning(f"No chunks were created for doc_id: {validated_data.get('doc_id')}. Nothing to embed.")
        return {"doc_id": validated_data.get("doc_id"), "chunks": []}

    if provider == 'ollama':
        embedded_chunks = _embed_with_ollama(chunks, config, logger)
    elif provider == 'gemini':
        embedded_chunks = _embed_with_gemini(chunks, config, logger)
    else:
        raise ValueError(f"Unsupported embedding provider configured: '{provider}'")
        
    final_output = {
        "doc_id": validated_data.get("doc_id"),
        "embedding_provider": provider,
        "embedding_model": embedding_config.get(provider, {}).get('model'),
        "chunks": embedded_chunks
    }
    
    return final_output
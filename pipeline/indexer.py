# pipeline/indexer.py

"""
ChromaDB Indexer Module
-----------------------
This script reads the final embedding JSON files and populates a ChromaDB
vector store. The vector store is the core of the semantic search system,
allowing for fast and efficient retrieval of relevant document chunks.

The script is designed to be idempotent, meaning it can be run multiple
times without creating duplicate entries, thanks to ChromaDB's `upsert`
functionality and the unique IDs we generate for each chunk.

(Updated to handle non-primitive metadata types by serializing them to JSON strings)
"""

import os
import json
import chromadb
import argparse
import yaml
from tqdm import tqdm
from typing import Dict, List, Any

def resolve_paths(config: Dict):
    """Resolves path placeholders in the config using simple replacement."""
    paths = config['paths']
    output_base = paths.get('output_base', '')

    for key, val in list(paths.items()):
        if isinstance(val, str):
            paths[key] = val.replace('{paths.output_base}', output_base)
    return config

def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures all metadata values are of a type supported by ChromaDB.
    Converts lists, dicts, or other types into JSON strings.
    """
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif value is None:
            # ChromaDB can handle None, so we pass it through.
            continue
        else:
            # Convert lists, dicts, etc., to a JSON string representation.
            sanitized[key] = json.dumps(value)
    return sanitized

def index_documents(config: Dict[str, Any]):
    """
    Scans the embeddings directory, loads the data, and populates the
    ChromaDB vector store.
    """
    paths = config['paths']
    db_config = config['vector_db']
    
    embeddings_dir = paths['embeddings']
    db_path = paths['vector_store']
    collection_name = db_config['collection_name']

    if not os.path.isdir(embeddings_dir):
        print(f"❌ Error: Embeddings directory not found at '{embeddings_dir}'.")
        print("Please run the main document pipeline first to generate embeddings.")
        return

    # 1. Setup ChromaDB client and collection
    print(f"🚀 Initializing ChromaDB vector store at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    print(f"Collection '{collection_name}' loaded/created with {collection.count()} documents.")
    
    # 2. Find all embedding files to process
    embedding_files = [os.path.join(embeddings_dir, f) for f in os.listdir(embeddings_dir) if f.endswith('.json')]
    
    if not embedding_files:
        print("🤷 No new embedding files found to index.")
        return

    print(f"Found {len(embedding_files)} embedding files to process...")

    # 3. Process each file and add its chunks to the database
    for file_path in tqdm(embedding_files, desc="Indexing Files", unit="file"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = data.get("chunks", [])
        if not chunks:
            continue

        # Prepare data for batch insertion into ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for chunk in chunks:
            if not chunk.get("embedding"):
                continue

            ids.append(chunk['chunk_id'])
            embeddings.append(chunk['embedding'])
            
            # --- FIX: Sanitize metadata before adding it ---
            sanitized_meta = _sanitize_metadata(chunk['metadata'])
            metadatas.append(sanitized_meta)
            
            documents.append(chunk['text_for_embedding'])
            
        # 4. Upsert the data into ChromaDB in a single batch per file
        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

    print("\n✅ Indexing complete.")
    print(f"Collection '{collection_name}' now contains {collection.count()} documents.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Populate a ChromaDB vector store from embedding files.")
    parser.add_argument("--config", required=True, help="Path to the config.yaml file.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        
        config = resolve_paths(config)
        index_documents(config)

    except FileNotFoundError:
        print(f"❌ Error: Config file not found at '{args.config}'")
    except Exception as e:
        print(f"🔥 An unexpected error occurred: {e}")
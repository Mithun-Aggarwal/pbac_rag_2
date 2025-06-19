# smart_chatbot/runner.py

import argparse
import yaml
import os
import chromadb

# --- MODIFIED: Import the new central logger ---
from utils.logger import setup_logger
from smart_chatbot.embedder import embed_query
from smart_chatbot.retriever import retrieve_relevant_chunks
from smart_chatbot.generator import generate_response

def resolve_paths(config: dict):
    """Resolves path placeholders in the config."""
    paths = config['paths']
    output_base = paths.get('output_base', '')
    for key, val in list(paths.items()):
        if isinstance(val, str):
            paths[key] = val.replace('{paths.output_base}', output_base)
    return config

def main():
    parser = argparse.ArgumentParser(description="Ask questions to your document knowledge base.")
    parser.add_argument("--config", default="config.yaml", help="Path to the main config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config = resolve_paths(config)

    # --- MODIFIED: Setup the central logger ---
    log_path = os.path.join(config['paths']['logs'], 'chatbot.log')
    logger = setup_logger(name='chatbot', log_file=log_path)
    
    paths = config['paths']
    db_config = config['vector_db']
    try:
        client = chromadb.PersistentClient(path=paths['vector_store'])
        collection = client.get_collection(name=db_config['collection_name'])
        logger.info(f"Successfully connected to ChromaDB collection '{db_config['collection_name']}' with {collection.count()} documents.")
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB at {paths['vector_store']}. Error: {e}")
        print(f"❌ Could not connect to the vector database. Please ensure the path is correct and the store was indexed.")
        return

    print("✅ Smart Chatbot is ready. Ask questions about your documents.")
    print("   Type 'exit' or 'quit' to end the session.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("👋 Exiting chatbot.")
            break

        if not question:
            continue

        try:
            q_embedding = embed_query(question, config)
            top_chunks = retrieve_relevant_chunks(q_embedding, collection, config)
            response = generate_response(question, top_chunks, config)

            print(f"\n🤖 Answer:\n{response}")
            
            print("\n🔍 Sources Retrieved:")
            if top_chunks.get("documents") and top_chunks["documents"][0]:
                for i in range(len(top_chunks['documents'][0])):
                    meta = top_chunks['metadatas'][0][i]
                    distance = top_chunks['distances'][0][i]
                    print(f"  - Source {i+1} (distance: {distance:.4f})")
                    print(f"    - Title: {meta.get('doc_title', 'N/A')}")
                    print(f"    - Section: {meta.get('section_heading', 'N/A')}")

            logger.info(f"Question: {question}")
            logger.info(f"Answer: {response}")

        except Exception as e:
            logger.error(f"An error occurred during the query process: {e}", exc_info=True)
            print("⚠️ A critical error occurred. Please check the chatbot.log file for details.")

if __name__ == "__main__":
    main()
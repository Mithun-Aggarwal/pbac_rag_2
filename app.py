# app.py

# --- Hot-patch for sqlite3 version on Streamlit Cloud ---
import sys
import sqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End of hot-patch ---

import streamlit as st
import yaml
import os
import chromadb
import time

# Import your existing chatbot functions and utilities
from utils.logger import setup_logger
from smart_chatbot.embedder import embed_query
from smart_chatbot.retriever import retrieve_relevant_chunks
from smart_chatbot.generator import generate_response

# --- 1. Utility and Initialization Functions ---

def resolve_paths(config: dict) -> dict:
    """Resolves path placeholders in the config."""
    paths = config.get('paths', {})
    output_base = paths.get('output_base', '')
    for key, val in list(paths.items()):
        if isinstance(val, str):
            paths[key] = val.replace('{paths.output_base}', output_base)
    return config

@st.cache_resource
def load_backend_config():
    """Loads the main backend configuration."""
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    return resolve_paths(config)

@st.cache_resource
def load_ui_config():
    """Loads the UI-specific configuration."""
    with open("ui_config.yaml", 'r') as file:
        return yaml.safe_load(file)

@st.cache_resource
def load_database(config: dict):
    """Initializes a connection to the ChromaDB vector store."""
    client = chromadb.PersistentClient(path=config['paths']['vector_store'])
    collection_name = config['vector_db']['collection_name']
    
    # --- CHANGE THIS LINE ---
    # FROM:
    # collection = client.get_collection(name=collection_name)
    
    # TO:
    collection = client.get_or_create_collection(name=collection_name)
    
    print(f"Database loaded successfully. Using collection: {collection_name}")
    return collection


# --- 2. Load Configurations and Backend ---

config = load_backend_config()
ui_config = load_ui_config()
collection = load_database(config)

# --- 3. Page and Sidebar Configuration ---

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="🧠",
    layout="wide"
)

with st.sidebar:
    st.header("About")
    st.markdown(ui_config.get('about_text', "This chatbot finds answers in your documents using a RAG pipeline."))
    
    st.header("Configuration")
    embedding_provider = config.get('embedding', {}).get('provider', 'N/A')
    st.markdown(f"**Provider:** `{embedding_provider}`")
    # ... (other sidebar info)

    # --- NEW: Clear Conversation Button ---
    if st.button("Clear Conversation History"):
        st.session_state.messages = [
            {"role": "assistant", "content": ui_config.get('welcome_message', "Hello!")}
        ]
        st.rerun()

# --- 4. Chat Interface ---

st.title("🧠 AI Document Intelligence")
st.caption(f"Powered by `{embedding_provider.capitalize()}`. Currently searching `{collection.count()}` document chunks.")

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": ui_config.get('welcome_message', "Hello!")}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("View Retrieved Sources"):
                for source in message["sources"]:
                    with st.container(border=True):
                        st.markdown(f"**Source Document:** `{source.get('title', 'N/A')}`")
                        st.markdown(f"**Section:** `{source.get('section', 'N/A')}`")
                        st.markdown(f"**Relevance Score (Distance):** `{source.get('distance', 0.0):.4f}`")
                        st.caption(f"Retrieved Text Snippet:\n\n> {source.get('text', '').replace('%', ' ')}")

# --- NEW: Display Example Questions if chat is new ---
if len(st.session_state.messages) <= 1:
    st.info("Start by asking your own question below or try one of these examples.")
    cols = st.columns(len(ui_config.get('example_questions', [])))
    for i, question in enumerate(ui_config.get('example_questions', [])):
        with cols[i]:
            if st.button(question['title']):
                st.session_state.prompt_from_button = question['query'] # Use a temporary state to hold the query

# --- 5. Handle User Input and Generate Response ---

# Check if a button was clicked or if the user typed a prompt
if "prompt_from_button" in st.session_state:
    prompt = st.session_state.prompt_from_button
    del st.session_state.prompt_from_button # Clear the temporary state
else:
    prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        # --- NEW: Enhanced Status Indicator ---
        with st.status("Thinking...", expanded=False) as status:
            st.write("Embedding your question...")
            query_embedding = embed_query(prompt, config)
            time.sleep(0.5) # Small delay for better UX

            st.write("Searching knowledge base...")
            context_chunks = retrieve_relevant_chunks(query_embedding, collection, config)
            time.sleep(0.5)

            st.write("Generating final answer...")
            response_text = generate_response(prompt, context_chunks, config)
            status.update(label="Response generated!", state="complete", expanded=False)

        st.markdown(response_text)

        sources_for_display = []
        if context_chunks.get("documents") and context_chunks["documents"][0]:
            for i in range(len(context_chunks['documents'][0])):
                meta = context_chunks['metadatas'][0][i]
                sources_for_display.append({
                    "title": meta.get('doc_title', 'N/A'),
                    "section": meta.get('section_heading', 'N/A'),
                    "text": context_chunks['documents'][0][i],
                    "distance": context_chunks['distances'][0][i]
                })

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources_for_display
    })
    
    st.rerun()
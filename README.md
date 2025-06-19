# 🧠 AI_DATA_EXTRACTION_AND_SEARCH_V1

A modular, local-first pipeline that ingests and understands healthcare, regulatory, and research documents. Extracts clean text, summarizes, embeds locally using Ollama, and enables a private, grounded Q&A experience — all without needing OpenAI or cloud services.

---

## ✅ Key Features

### 🧾 Document Processing Pipeline
- Multi-format ingestion: `PDF`, `DOCX`, `TXT`, image scans
- OCR via Tesseract for image-based files
- Text extraction → normalization → optional LLM post-processing
- Configurable output in `JSON`, `TXT`, or `Markdown`
- Automatic refresh logic using file hashing
- Summarization, tagging, and classification via OpenAI, Gemini, or local LLM

### 💬 Smart Chatbot (RAG-Enabled)
- Local embedding via `nomic-embed-text` on Ollama
- Retrieval by cosine similarity over document chunks
- Response generation via `mistral` (also on Ollama)
- Prompts are grounded: no hallucinations, no unsupported claims
- Full logging and modular design for long-term scalability

---

## 📂 Project Structure

CURATED_INFORMATION/
├── documents/
│ ├── input_folder/ # Place raw source documents here
│ └── output_folder/ # JSON outputs with text + embeddings
├── logs/ # Log outputs and embeddings inspection
├── cache/ # Hashes for refresh detection
├── pipeline/ # Ingestion, extraction, normalization, embedding
│ ├── ingestion.py
│ ├── extract.py
│ ├── normalize.py
│ ├── embedding_generator.py
│ ├── output.py
│ ├── refresh.py
│ ├── doc_classifier.py # (planned) Auto-tagging of document types
│ └── logger.py
├── smart_chatbot/ # Chatbot + retrieval-augmented generation (RAG)
│ ├── runner.py
│ ├── embedder.py
│ ├── retriever.py
│ ├── generator.py
│ ├── prompts.py
│ ├── utils.py
│ └── logger.py
├── validate_embeddings.py # Validates embedding structure and similarity
├── config.yaml # Central configuration
├── .env # Ollama + model overrides
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## ⚙️ Setup Instructions

### 1. 🧪 Create and activate your virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
2. 📥 Install Python dependencies
bash
Copy
Edit
pip install --upgrade pip
pip install -r requirements.txt
3. 🔧 Install OCR & PDF tools (for Ubuntu)
bash
Copy
Edit
sudo apt install tesseract-ocr poppler-utils
🔐 Ollama Setup (Local LLMs & Embedding)
Make sure you have Ollama installed. Then:

bash
Copy
Edit
ollama run nomic-embed-text
ollama run mistral
🚀 Running the Document Pipeline
Place your files into documents/input_folder/, then run:

bash
Copy
Edit
python document_pipeline_main.py --config config.yaml
Outputs will be saved to documents/output_folder/ as .json.

💬 Running the Smart Local Chatbot
To chat with a processed document:

bash
Copy
Edit
python -m smart_chatbot.runner --config config.yaml --file documents/output_folder/your_doc.json
Chat will use nomic-embed-text to embed your question, retrieve top chunks, and answer using mistral.

📦 Example Output File
json
Copy
Edit
{
  "metadata": {
    "source_file": "cost-manual-2016.pdf",
    "detected_type": "COST_MANUAL",
    "pages": 18,
    "extracted_at": "2025-06-15T13:12:04"
  },
  "chunks": [
    {
      "id": 1,
      "text": "The unit cost of item X is derived from...",
      "embedding": [0.013, 0.248, ...]
    },
    ...
  ],
  "llm_output": {
    "summary": "This manual defines unit costs...",
    "tags": ["costing", "healthcare", "PBAC"],
    "classification": "COST_MANUAL"
  }
}
🔎 Validation Tools
Run embedding validation:
bash
Copy
Edit
python validate_embeddings.py --file documents/output_folder/your_doc.json --logdir logs
Verifies embedding shapes

Computes cosine similarity between chunks

Plots embeddings with PCA

🔧 Config Management
Configuration is loaded from config.yaml and can be overridden by .env.

Example .env:

env
Copy
Edit
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=mistral
OLLAMA_EMBED_URL=http://localhost:11434/api/embeddings
OLLAMA_CHAT_URL=http://localhost:11434/api/chat
TOP_K=3
CHUNK_SIZE=400
CHUNK_OVERLAP=50
LOG_DIR=logs
🧠 What's Next
 Implement doc_classifier.py to auto-detect document types

 Add FAISS or Chroma integration for scale

 Enable multi-document Q&A

 Build Streamlit or web UI

 Add citation-based answering

🤝 Credits
Crafted by Mithun + Bruce (ChatGPT), designed for real-world document intelligence at scale — in healthcare, regulation, and policy domains.

📝 License
MIT License – use, remix, and build your own local-AI-powered search!

yaml
Copy
Edit

---

Would you like this saved as a downloadable `.md` file as well? Or want a visual diagram (architecture flo# pbac_rag
# pbac_rag
# pbac_rag_2
# pbac_rag_2

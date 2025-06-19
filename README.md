# Public Policy RAG System

This system implements a Retrieval-Augmented Generation (RAG) pipeline for analyzing public policy documents. It uses LLaMA for question answering and ChromaDB for vector storage.

## Features

- PDF document ingestion and processing
- Automatic text chunking with overlap
- Vector embeddings using HuggingFace's sentence-transformers
- Persistent vector storage with ChromaDB
- Question answering using LLaMA model
- Interactive query interface

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download LLaMA Model:
   - Download the LLaMA 2 7B Chat GGUF model from HuggingFace
   - Create a `models` directory in the project root
   - Place the downloaded model file (e.g., `llama-2-7b-chat.gguf`) in the `models` directory

## Directory Structure

```
.
├── public_policies - Copy/    # Directory containing PDF files
├── policy_chroma/            # Vector store directory (created automatically)
├── models/                   # Directory for LLaMA model
├── policy_rag.py            # Main RAG implementation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Usage

1. Ensure your PDF files are in the `public_policies - Copy` directory
2. Run the RAG system:
```bash
python policy_rag.py
```

3. The system will:
   - Process all PDFs in the directory
   - Create embeddings and store them in ChromaDB
   - Initialize the LLaMA model
   - Start an interactive query session

4. Type your questions about the policies and press Enter
5. Type 'exit' to quit the program

## Example Questions

- "What are the key differences between Arizona and Florida medical policies?"
- "What are the coverage requirements for mental health services in Colorado?"
- "Compare the prescription drug coverage across all states"

## Notes

- The system uses the `all-MiniLM-L6-v2` model for embeddings
- Chunk size is set to 1000 characters with 200 character overlap
- The LLaMA model is configured with a 4096 token context window
- Vector store is persisted to disk for future use

## Requirements

- Python 3.8+
- Sufficient RAM for LLaMA model (minimum 8GB recommended)
- GPU recommended but not required 
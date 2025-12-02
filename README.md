📘 Base RAG — Retrieval Augmented Generation System

Base RAG is a lightweight Retrieval-Augmented Generation (RAG) pipeline designed to process documents, generate embeddings, store them in a vector database, and perform semantic search to answer user queries.
This project demonstrates a clean and modular RAG foundation suitable for expansion into a full production-grade AI assistant.

🚀 Features

Document ingestion (.txt, .pdf)

Sentence-level chunking for high retrieval accuracy

Vietnamese embedding model (SentenceTransformer-based)

Qdrant Embedded vector database

Semantic search with cosine similarity

Modular architecture:

embedding → Encode text

data_process → Load & preprocess documents

qdrant_setup → Vector DB operations

main → Build the vector index

qa → Question-answer retrieval loop

🏗 Project Structure
project/
 ├── data/                      # Input documents (.txt, .pdf)
 ├── vector_store/              # Qdrant Embedded storage (auto-generated)
 ├── src/
 │    ├── embedding.py          # Text → Embeddings
 │    ├── data_process.py       # Load, clean, chunk documents
 │    ├── qdrant_setup.py       # Qdrant init, upsert, search
 │    ├── main.py               # Build vector DB (indexing)
 │    └── qa.py                 # Query-answer interface
 ├── README.md
 └── requirements.txt

📥 Installation
1. Create virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.\.venv\Scripts\activate     # Windows

2. Install dependencies
pip install -r requirements.txt

3. Prepare NLTK models (required for sentence tokenization)
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

🧠 How It Works (RAG Pipeline)
1. Document Loading

The system loads .txt and .pdf files from the data/ directory.

2. Sentence Chunking

Each document is split into individual sentences using NLTK.
This improves retrieval performance for short, specific questions.

3. Embedding Generation

Each sentence is encoded using:

dangvantuan/vietnamese-document-embedding


This produces a 768-dimensional semantic vector.

4. Vector Database

Qdrant Embedded is used to store vectors locally.
No external server required.

5. Semantic Retrieval

Given a user query:

Encode the query → query embedding

Search Qdrant for nearest vectors

Return the most relevant sentences

6. Response

The system currently returns raw retrieved sentences.
It can be extended with an LLM for natural language generation.

📌 Usage
1. Build the Vector Index

This processes documents → chunks → embeddings → vector DB.

python src/main.py

2. Query the System

Interactive retrieval mode:

python src/qa.py


Example:

Question: Chính sách hoàn trả là gì?


Output:

[document] Chính sách hoàn trả được quy định tại...

🧩 Components Overview
embedding.py

Loads the embedding model

Encodes lists of sentences

Supports GPU acceleration when available

data_process.py

Loads .txt and .pdf documents

Performs sentence-level chunking

Outputs normalized document chunks

qdrant_setup.py

Initializes Qdrant local instance

Creates or recreates collections

Upserts sentence embeddings

Performs nearest-neighbor search

main.py

End-to-end data indexing pipeline

Generates and stores embeddings

qa.py

Interactive retrieval loop

User enters questions

System retrieves relevant chunks

🔧 Configuration

You can adjust:

Parameter	Location	Description
Chunk size	data_process.py	Sentence, paragraph, or window chunking
Embedding model	embedding.py	Replace with any SentenceTransformer model
Vector DB path	qdrant_setup.py	Default: ./vector_store
Top-K results	qa.py	Default: 3
🧱 Extending This Project

This repository offers a minimal RAG foundation.
It can be extended with:

✔ LLM Response Generation

Use GPT, Claude, or Llama to generate final answers from retrieved context.

✔ Re-Ranking (Improved Accuracy)

Add a Cross-Encoder model to reorder retrieved results.

✔ FastAPI Backend

Expose the RAG pipeline as an API.

✔ Frontend UI

Build a chatbot interface using React / Vue / Streamlit.

✔ Multi-file Formats

Add support for DOCX, MD, HTML, images (OCR), etc.

📄 License

MIT License
Free for personal and commercial use.

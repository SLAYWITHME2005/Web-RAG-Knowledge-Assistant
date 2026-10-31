# 🌐 Web-RAG Assistant — LangChain, FAISS & Groq

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C.svg?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorStore-007ACC.svg)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-yellow.svg?logo=huggingface)](https://huggingface.co/)
[![Groq](https://img.shields.io/badge/Groq-Cloud%20LLM-F55036.svg)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A dynamic **Retrieval-Augmented Generation (RAG)** application capable of scraping, indexing, and answering detailed questions about live web pages and documents in real time. Built with **LangChain**, **FAISS** vector store, **Hugging Face** embeddings, and **Groq Cloud LLM** with an intuitive **Gradio** web interface.

---

## 📸 Application Preview

<p align="center">
  <img width="95%" alt="Web RAG Gradio Interface" src="https://github.com/user-attachments/assets/44f9f401-28a4-47e9-9317-b77bd4dd14b4" />
</p>

---

## 🛠️ Retrieval Architecture

```mermaid
flowchart LR
    A["Target Web URL"] --> B["WebBaseLoader"]
    B --> C["RecursiveCharacterTextSplitter<br>(Chunk: 1000, Overlap: 200)"]
    C --> D["HuggingFace Embeddings<br>(all-MiniLM-L6-v2)"]
    D --> E["In-Memory FAISS VectorStore"]
    F["User Question"] --> G["Similarity Retriever (k=4)"]
    E --> G
    G --> H["Groq ChatGroq LLM + Prompt"]
    H --> I["Gradio Chat Interface Output"]
```

---

## 🌟 Key Features

- 🌐 **Instant URL Ingestion:** Scrapes web pages on the fly using `WebBaseLoader`.
- 🧩 **Semantic Chunking:** Splits lengthy web documents with `RecursiveCharacterTextSplitter` preserving semantic paragraph boundaries.
- 📐 **Dense Vector Search:** Embeds text chunks via `sentence-transformers/all-MiniLM-L6-v2` into a fast FAISS vector index.
- ⚡ **Groq Accelerated Inference:** Queries `ChatGroq` for rapid, grounded contextual answers with zero hallucination.

---

## 📂 Project Structure

```plaintext
GENAI_day2_project/
├── code1.py          # Complete LangChain RAG pipeline & Gradio UI
├── .gitignore        # Clean repository filter
└── README.md         # Documentation & demo screenshot
```

---

## ⚡ Quickstart

```bash
# Clone the repository
git clone https://github.com/SLAYWITHME2005/GENAI_day2_project.git
cd GENAI_day2_project

# Install dependencies
pip install langchain langchain-community langchain-groq langchain-huggingface faiss-cpu gradio sentence-transformers beautifulsoup4 python-dotenv

# Set your Groq API key
export GROQ_API_KEY="your_api_key_here"  # On Windows PowerShell: $env:GROQ_API_KEY="your_api_key_here"

# Launch the application
python code1.py
```

---

## 👩‍💻 Author
- **Gayathri Rajeev** ([@SLAYWITHME2005](https://github.com/SLAYWITHME2005))

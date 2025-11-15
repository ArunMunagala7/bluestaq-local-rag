# 🧠 Bluestaq Local RAG Challenge

This repository contains a local Retrieval-Augmented Generation (RAG) system intended to run without cloud dependencies. It pairs a quantized Llama model with hybrid retrieval (FAISS + BM25) and a CLI for interaction.

---

## 🚀 Overview

**Goal:** Build a self-contained RAG pipeline that runs locally and lets you query a document corpus with a small, quantized Llama model.

**Key components:**
- Quantized LLM (GGUF format, Llama 3.2 3B Instruct in `models/gguf`)
- Hybrid retrieval: FAISS (dense) + BM25 (sparse)
- CLI for single queries and chat (`app/app.py`)
- Ingestion pipeline to build corpus and FAISS index (`app/ingest.py`)
- Configurable via `config.yaml`

---

## 📦 Repository

**Repository:** https://github.com/ArunMunagala7/bluestaq-local-rag

Project layout (top-level):

- `app/` — application code and CLI
- `data/` — corpus, index and uploads
- `models/gguf/` — GGUF model files
- `config.yaml` — runtime configuration
- `requirements.txt` — Python dependencies

Key files inside `app/`:

- `app.py` — Typer CLI (commands: `query-basic`, `query-rag`, `chat`, `upload`, `bulk-upload`)
- `ingest.py` — file processing and FAISS index creation
- `retriever.py` — hybrid retrieval logic
- `rag.py` — RAG pipeline that combines retrieval + generation

---

## 🧠 Model & Config

- Default model path: `models/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf` (set in `config.yaml` under `model.gguf_path`).
- Make sure you have the GGUF model file present before running queries.
- Adjust runtime parameters in `config.yaml` (context size, threads, GPU layers, retrieval chunking, etc.).

---

## ⚙️ Setup

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify `config.yaml` points to your model path and desired settings.

---

## ▶️ Usage (CLI)

Run the Typer CLI via module execution. Example commands:

```bash
# Single-shot LLM only (no retrieval)
python -m app.app query-basic "What is Retrieval Augmented Generation?"

# RAG query (retrieval + generation)
python -m app.app query-rag "Summarize the key ideas from the uploaded paper."

# Interactive chat mode
python -m app.app chat

# Upload a single file into the corpus
python -m app.app upload path/to/file.pdf

# Bulk upload files from data/uploads and rebuild index
python -m app.app bulk-upload
```

Notes:
- If the FAISS index is missing or out of date, run `bulk-upload` to process `data/uploads` and rebuild the index.
- The CLI prints retrieved passages and simple relevance scores when using `query-rag`.

---

## 📂 Data

- `data/corpus/` — text chunks extracted from source PDFs and documents
- `data/index/` — FAISS index files (e.g. `faiss.index`)
- `data/uploads/` — drop files here for ingestion

There are sample text files under `data/corpus/` in this repo.

---

## 🧩 Notes & Tips

- The project is designed to run locally; ensure you have a compatible CPU/GPU setup and enough RAM for the chosen model and context window.
- For faster model inference, tune `n_threads` and `n_gpu_layers` in `config.yaml`.
- If you prefer running the model via SSH keys/remote or using a different model location, update `config.yaml` accordingly.

---

## 📚 References

- See `app/` for implementation details and `config.yaml` for runtime tuning.

If you want, I can also add a short Quick Start section with exact commands for common setups (CPU-only, macOS with MPS, or using GPU).
# 🧠 Bluestaq Local RAG Challenge

This repository contains the complete implementation of a **local Retrieval-Augmented Generation (RAG)** system designed to run entirely on a laptop — no cloud dependencies required.  
It integrates a quantized local Llama model with hybrid dense–sparse retrieval and a user-friendly Command-Line Interface (CLI) for natural language interaction.

---

## 🚀 Overview

**Goal:**  
To develop a robust, efficient, and fully local language model pipeline that augments generation through document retrieval, while maintaining sub-second response times on consumer hardware.

**Key Components:**
- 🧩 Quantized LLM (Llama 3.2 3B Instruct — GGUF)
- 🔍 Hybrid Retrieval (Dense + Sparse using FAISS and BM25)
- 💬 CLI-based Query and Chat Interface
- 🗂️ Local Corpus Management and PDF Ingestion
- ⚙️ Configurable Parameters via `config.yaml`

---

## 📦 Code Repository

**Repository:** [https://github.com/ArunMunagala7/local-rag](https://github.com/ArunMunagala7/local-rag)

All source code is modularized under `app/`, with separate scripts for:
- **`rag.py`** → retrieval pipeline integration  
- **`retriever.py`** → hybrid dense/sparse search  
- **`ingest.py`** → corpus creation and FAISS index building  
- **`app.py`** → Typer-based CLI entrypoint  
- **`config.yaml`** → centralized configuration and model tuning  

---

## 🧠 Model Files

The project uses a **quantized local model**:


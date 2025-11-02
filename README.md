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


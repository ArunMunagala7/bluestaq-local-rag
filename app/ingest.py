import os, pickle, faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def process_file(path, cfg):
    corpus_dir = cfg["paths"]["corpus_dir"]
    text = ""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif path.lower().endswith(".txt"):
        text = open(path, "r", encoding="utf8").read()
    else:
        raise ValueError("Unsupported file type.")
    fname = os.path.basename(path)
    with open(os.path.join(corpus_dir, fname.replace(" ", "_") + ".txt"), "w", encoding="utf8") as f:
        f.write(text)


def ingest_corpus(cfg):
    corpus_dir = cfg["paths"]["corpus_dir"]
    index_dir = cfg["paths"]["index_dir"]
    os.makedirs(index_dir, exist_ok=True)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts, titles = [], []
    for file in os.listdir(corpus_dir):
        path = os.path.join(corpus_dir, file)
        if not os.path.isfile(path): continue
        with open(path, "r", encoding="utf8") as f:
            texts.append(f.read())
        titles.append(file)

    # Chunking
    chunks, meta = [], []
    chunk_size = cfg["retrieval"]["chunk_tokens"]
    for t, title in zip(texts, titles):
        words = t.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
            meta.append({"title": title})

    # Dense embeddings
    X = embedder.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)
    faiss_index = faiss.IndexFlatIP(X.shape[1])
    faiss_index.add(X)
    faiss.write_index(faiss_index, os.path.join(index_dir, "faiss.index"))
    pickle.dump(chunks, open(os.path.join(index_dir, "docs.pkl"), "wb"))

    # BM25
    bm25 = BM25Okapi([c.split() for c in chunks])
    pickle.dump(bm25, open(os.path.join(index_dir, "bm25.pkl"), "wb"))
    print("✅ Corpus ingested & indexed successfully.")

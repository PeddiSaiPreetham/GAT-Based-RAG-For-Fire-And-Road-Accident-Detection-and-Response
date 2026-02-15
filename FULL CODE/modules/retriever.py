# modules/retriever.py
"""
FAISS-backed text retriever using SentenceTransformers.
 - Retriever(model_name='all-mpnet-base-v2', index_path=None)
 - build_index(docs_dir)  # optional helper
 - load_index(index_path, docs_dir=None)
 - query(text, k=5) -> list of dicts: {'doc_id':..., 'text':..., 'distance':...}
"""
import os
import pickle
import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

log = logging.getLogger("retriever")

class Retriever:
    def __init__(self, model_name="all-mpnet-base-v2", index_path=None):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.docs = []      # list of text strings
        self.doc_ids = []   # list of document ids (filenames)
        self.index_path = index_path

    def build_index(self, docs_dir):
        if not os.path.exists(docs_dir):
            raise FileNotFoundError(f"docs_dir not found: {docs_dir}")
        texts = []
        ids = []
        for fname in sorted(os.listdir(docs_dir)):
            p = os.path.join(docs_dir, fname)
            if not os.path.isfile(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                texts.append(f.read())
                ids.append(fname)
        if len(texts) == 0:
            raise RuntimeError("No docs found to build index.")
        embs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        d = embs.shape[1]
        self.index = faiss.IndexHNSWFlat(d, 32)
        self.index.hnsw.efConstruction = 200
        self.index.add(embs)
        self.docs = texts
        self.doc_ids = ids
        # write index + docs
        if self.index_path:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.index_path + ".pkl", "wb") as f:
                pickle.dump({"docs": self.docs, "doc_ids": self.doc_ids}, f)
            log.info("FAISS index saved to %s", self.index_path)

    def load_index(self, index_path, docs_dir=None):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        self.index = faiss.read_index(index_path)
        # try to load docs pickle next to index
        pkl = index_path + ".pkl"
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                meta = pickle.load(f)
                self.docs = meta.get("docs", [])
                self.doc_ids = meta.get("doc_ids", [])
                log.info("Loaded docs metadata from %s", pkl)
                return
        # else, reconstruct from docs_dir if provided
        if docs_dir:
            texts = []
            ids = []
            for fname in sorted(os.listdir(docs_dir)):
                p = os.path.join(docs_dir, fname)
                if not os.path.isfile(p): continue
                with open(p, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    ids.append(fname)
            self.docs = texts
            self.doc_ids = ids
            log.info("Loaded docs from docs_dir: %s", docs_dir)
            return
        # fallback: empty lists
        log.warning("Docs metadata not found. Retriever will return empty results unless docs are provided.")

    def query(self, text, k=5):
        if self.index is None:
            log.warning("No FAISS index loaded.")
            return []
        q = self.embedder.encode([text], convert_to_numpy=True)
        D, I = self.index.search(q, k)
        results = []
        for j, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            results.append({
                "doc_id": self.doc_ids[idx] if idx < len(self.doc_ids) else f"doc_{idx}",
                "text": self.docs[idx],
                "distance": float(D[0, j])
            })
        return results

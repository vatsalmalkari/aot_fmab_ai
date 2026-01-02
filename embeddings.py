import os
import json
from glob import glob
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from .config import EMBEDDING_MODEL, CHROMA_DB_PATH, JSON_DIR

# Setup ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="anime_data")

# Setup embedding model
device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("USE_GPU") else "cpu"
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

@lru_cache(maxsize=100)
def get_embedding(text: str):
    return embedding_model.encode(text).tolist()

def chunk_text(text, max_length=500):
    words = text.split()
    chunks, current = [], []
    for word in words:
        if len(" ".join(current + [word])) <= max_length:
            current.append(word)
        else:
            chunks.append(" ".join(current))
            current = [word]
    if current:
        chunks.append(" ".join(current))
    return chunks

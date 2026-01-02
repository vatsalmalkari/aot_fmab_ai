import os
import json
from glob import glob
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Config
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH = "./aot_fmab_db"
JSON_DIR = "./json_output"
BATCH_SIZE = 32
CHUNK_SIZE = 500  # words per chunk

# Setup ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="anime_data")

# embedding model
device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("USE_GPU") else "cpu"
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

# Cache for embeddings
@lru_cache(maxsize=100)
def get_embedding(text: str):
    return embedding_model.encode(text).tolist()

# Chunking function
def chunk_text(text, max_length=CHUNK_SIZE):
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

# Populate ChromaDB
def populate_chroma():
    if collection.count() > 0:
        print("ChromaDB already populated. Skipping embedding.")
        return

    print("Populating ChromaDB with embeddings...")

    documents, metadatas, ids = [], [], []

    for file in glob(os.path.join(JSON_DIR, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = {
            "name": data.get("name", ""),
            "anime": data.get("anime", ""),
            "type": data.get("type", ""),
            "source_file": os.path.basename(file)
        }

        if data.get("type") == "episode":
            metadata.update({
                "season": data.get("season", ""),
                "episode": data.get("episode_number", ""),
                "title": data.get("title", "")
            })

        # split into chunks
        content = data.get("content", "")
        for i, chunk in enumerate(chunk_text(content)):
            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(f"{os.path.basename(file)}_{i}")

    # embeddings in batches
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i+BATCH_SIZE]
        batch_metas = metadatas[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]

        embeddings = embedding_model.encode(batch_docs).tolist()

        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
            embeddings=embeddings
        )

    print("ChromaDB population complete!")
    print(f"Total documents in DB: {collection.count()}")

if __name__ == "__main__":
    populate_chroma()

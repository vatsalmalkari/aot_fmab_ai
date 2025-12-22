import os
import json
from sentence_transformers import SentenceTransformer
import chromadb
from functools import lru_cache

# config
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH = "./aot_fmab_db"
JSON_DIR = "./json_output"

# setup chromadb and model
model = SentenceTransformer(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="anime_data")

# cached embedding
@lru_cache(maxsize=100)
def get_embedding(text):
    return model.encode(text).tolist()

# helper to chunk text
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

# json population
def populate_chroma_if_empty():
    if collection.count() > 0:
        print("ChromaDB already populated. Skipping embedding.")
        return

    print("Populating ChromaDB with embeddings...")
    documents, metadatas, ids = [], [], []

    for file in os.listdir(JSON_DIR):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(JSON_DIR, file), 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = {
            "name": data.get("name", ""),
            "anime": data.get("anime", ""),
            "type": data.get("type", ""),
            "source_file": data.get("metadata", {}).get("source_file", "")
        }

        if data.get("type") == "episode":
            metadata.update({
                "season": data.get("season", ""),
                "episode": data.get("episode_number", ""),
                "title": data.get("title", "")
            })

        for i, chunk in enumerate(chunk_text(data.get("content", ""))):
            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(f"{file}_{i}")

    # Compute embeddings in batches
    batch_size = 32
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        embeddings = model.encode(batch_docs).tolist()

        collection.add(
            documents=batch_docs,
            metadatas=batch_metadatas,
            ids=batch_ids,
            embeddings=embeddings
        )

    print("ChromaDB population complete!")

# main execution
if __name__ == "__main__":
    populate_chroma_if_empty()
    print("Loading complete.")
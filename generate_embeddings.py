import os
import json
from glob import glob
from tqdm import tqdm
import time

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from sentence_transformers import SentenceTransformer
import chromadb

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH = "./anime_vector_db"
JSON_DIR = "./json_output"

BATCH_SIZE = 32
CHUNK_SIZE = 500  # characters per chunk

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="anime_data")

spark = SparkSession.builder \
    .appName("AnimeRAGEmbeddings") \
    .getOrCreate()

def chunk_text(text: str, max_length: int = CHUNK_SIZE):
    text = text.strip()
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def load_jsons_as_spark_df(json_dir: str):
    files = glob(os.path.join(json_dir, "*.json"))
    rows = []

    for file in tqdm(files, desc="Reading JSON files"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        filename = os.path.basename(file)

        # Default metadata
        anime = data.get("anime", "")
        typ = data.get("type", "")
        name = data.get("name", "")
        season = None
        episode_number = None
        title = None

        # Parse from filename if missing
        if filename.startswith("char_"):
            parts = filename.replace(".json","").split("_")
            if len(parts) >= 3:
                anime = anime or parts[1]
                name = name or "_".join(parts[2:])
        elif filename.startswith("ep_"):
            parts = filename.replace(".json","").split("_")
            if anime.lower() != "fmab":
                # AOT/JJK: ep_animename_Season_number_episode_number.json
                anime = anime or parts[1]
                season = int(parts[3])
                episode_number = int(parts[5])
                title = data.get("title","")
            else:
                # FMAB: ep_fmab_episode_number.json
                anime = anime or "fmab"
                episode_number = int(parts[3])
                title = data.get("title","")

        content = data.get("content","").strip()
        if not content:
            continue

        rows.append({
            "anime": anime or "",
            "type": typ or "",
            "name": name or "",
            "season": season if season is not None else 0,
            "episode_number": episode_number if episode_number is not None else 0,
            "title": title or "",
            "content": content,
            "source_file": filename
        })

    schema = StructType([
        StructField("anime", StringType(), True),
        StructField("type", StringType(), True),
        StructField("name", StringType(), True),
        StructField("season", IntegerType(), True),
        StructField("episode_number", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("content", StringType(), True),
        StructField("source_file", StringType(), True),
    ])
    return spark.createDataFrame(rows, schema=schema)

def embed_partition(partition):
    """Compute embeddings for a partition of rows"""
    import torch
    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    batch_docs, batch_metas, batch_ids = [], [], []

    for row in partition:
        chunks = chunk_text(row.content)
        for i, chunk in enumerate(chunks):
            batch_docs.append(chunk)
            batch_metas.append({
                "anime": row.anime,
                "type": row.type,
                "name": row.name,
                "season": row.season,
                "episode": row.episode_number,
                "title": row.title,
                "source_file": row.source_file
            })
            batch_ids.append(f"{row.source_file}_{i}")

            if len(batch_docs) >= BATCH_SIZE:
                embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()
                for doc, meta, eid, emb in zip(batch_docs, batch_metas, batch_ids, embeddings):
                    yield doc, meta, eid, emb
                batch_docs, batch_metas, batch_ids = [], [], []

    if batch_docs:
        embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()
        for doc, meta, eid, emb in zip(batch_docs, batch_metas, batch_ids, embeddings):
            yield doc, meta, eid, emb

def populate_chroma():
    if collection.count() > 0:
        print("ChromaDB already populated. Skipping embedding.")
        return

    df = load_jsons_as_spark_df(JSON_DIR)
    rdd = df.rdd

    start_time = time.perf_counter()
    for doc, meta, eid, emb in tqdm(rdd.mapPartitions(embed_partition).collect(), desc="Adding embeddings"):
        collection.add(
            documents=[doc],
            metadatas=[meta],
            ids=[eid],
            embeddings=[emb]
        )
    end_time = time.perf_counter()
    print("ChromaDB population complete!")
    print(f"Total chunks stored: {collection.count()}")
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    populate_chroma()
    spark.stop()

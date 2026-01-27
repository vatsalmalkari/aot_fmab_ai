import os
import json
from functools import lru_cache
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from google import genai
from dotenv import load_dotenv
from rapidfuzz import process

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load environment
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Models
FLASH_MODEL = "gemini-2.5-flash"
LITE_MODEL = "gemini-2.5-flash-lite"

# GenAI client
genai_client = genai.Client(api_key=API_KEY)

# Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333", check_compatibility=False)

COLLECTION_NAME = "anime_data"
VECTOR_DIR = "./anime_vector_db"  # precomputed vectors folder

# Create collection if not exists
if COLLECTION_NAME not in [c.name for c in qdrant_client.get_collections().collections]:
    # Load vector size from first file
    sample_file = next(f for f in os.listdir(VECTOR_DIR) if f.endswith(".json"))
    with open(os.path.join(VECTOR_DIR, sample_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    vector_size = len(data["vector"])
    
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

# Load precomputed embeddings into Qdrant
for file_name in os.listdir(VECTOR_DIR):
    if not file_name.endswith(".json"):
        continue
    path = os.path.join(VECTOR_DIR, file_name)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Expect each JSON: {"title": "...", "text": "...", "vector": [...]}
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[{
            "id": data["title"],
            "vector": data["vector"],
            "payload": {"title": data["title"], "text": data["text"]}
        }]
    )

# Fuzzy matching
all_titles_or_names = []

def load_titles_and_names():
    names = set()
    for file in os.listdir(VECTOR_DIR):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(VECTOR_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        names.add(data["title"])
    global all_titles_or_names
    all_titles_or_names = list(names)

def fix_typo(query: str, score_cutoff: int = 70) -> str:
    if not all_titles_or_names:
        return query
    match = process.extractOne(query, all_titles_or_names, score_cutoff=score_cutoff)
    return match[0] if match else query

# RAG
response_cache = {}

def retrieve_context(prompt: str, top_k: int = 5) -> str:
    # Use fuzzy title to pick vector
    corrected_prompt = fix_typo(prompt)
    
    # Fetch vector from Qdrant if title exists
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=None,  # Qdrant supports filtering by payload; here we fallback to all vectors
        query_filter={"must": [{"key": "title", "match": {"value": corrected_prompt}}]},
        limit=top_k
    )

    # Fallback: nearest neighbors if no exact match
    if not results:
        # search by vector similarity: pick first title's vector
        vector = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[corrected_prompt]
        )[0].vector
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k
        )

    documents = [hit.payload.get("text", "") for hit in results]
    return "\n\n".join(documents)

def generate_rag(prompt: str, use_lite: bool = True, temperature: float = 0.7, max_tokens: int = 512, mode: str = "trivia"):
    cache_key = (prompt, use_lite, temperature, max_tokens, mode)
    if cache_key in response_cache:
        return response_cache[cache_key]

    model_id = LITE_MODEL if use_lite else FLASH_MODEL
    context = retrieve_context(prompt)

    instructions = {
        "trivia": "Answer factually using only the provided context.",
        "fanfiction": "Write an immersive fanfiction scene grounded in the context.",
        "summary": "Write a concise, structured summary using the context."
    }

    full_prompt = f"""
{instructions.get(mode, '')}

CONTEXT:
{context}

USER QUESTION:
{prompt}
"""

    response = genai_client.models.generate_content(
        model=model_id,
        contents=full_prompt,
        config=genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    )

    output = response.text.strip()
    response_cache[cache_key] = output
    return output

# FastAPI app
app = FastAPI(title="Anime RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup_event():
    load_titles_and_names()

@app.get("/")
def health():
    return {"status": "online", "docs": "/docs"}

@app.get("/ask")
def ask(prompt: str, use_lite: bool = True, temperature: float = 0.7, max_tokens: int = 512, mode: str = "trivia"):
    response = generate_rag(prompt, use_lite, temperature, max_tokens, mode)
    return {"response": response}

if __name__ == "__main__":
    print("Running Anime RAG Web App + API")
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import json
from functools import lru_cache
from typing import List
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv
from rapidfuzz import process
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH = "./aot_fmab_db"
JSON_DATA_DIR = "./json_output"

# Google GenAI
google_client = genai.Client(api_key=API_KEY)
FLASH_MODEL = "gemini-2.5-flash"
LITE_MODEL = "gemini-2.5-flash-lite"

# Embeddings + ChromaDB
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="anime_data")


@lru_cache(maxsize=100)
def get_embedding(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()


# Fuzzy Matching for Titles
all_titles_or_names = []


def load_titles_and_names():
    global all_titles_or_names
    names = set()
    for file in os.listdir(JSON_DATA_DIR):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(JSON_DATA_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            names.add(data.get("name", ""))
            names.add(data.get("anime", ""))
            if data.get("type") == "episode":
                names.add(data.get("title", ""))
    all_titles_or_names = [n for n in names if n]


def fix_typo(query, choices=all_titles_or_names, score_cutoff=70):
    best = process.extractOne(query, choices, score_cutoff=score_cutoff)
    return best[0] if best else query


# Retrieval
def retrieve_context(prompt: str, top_k=5):
    corrected_prompt = fix_typo(prompt)
    embedding = get_embedding(corrected_prompt)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    chunks = results["documents"][0] if results["documents"] else []
    return "\n".join(chunks)


# RAG Generator
response_cache = {}


def generate_rag(prompt, use_lite, temp, max_tokens, mode):
    cache_key = (prompt, use_lite, temp, max_tokens, mode)
    if cache_key in response_cache:
        return response_cache[cache_key]

    model_id = LITE_MODEL if use_lite else FLASH_MODEL
    context = retrieve_context(prompt)

    instructions = {
        "trivia": "Answer using factual detail based on the context:",
        "fanfiction": "Write an immersive fanfiction scene based on the context:",
        "summary": "Provide a detailed but concise summary using the context:",
    }

    full_prompt = f"""{instructions.get(mode, '')}

CONTEXT:
{context}

USER:
{prompt}
"""

    response = google_client.models.generate_content(
        model=model_id,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_tokens
        )
    )

    output = response.text.strip()
    response_cache[cache_key] = output
    return output

app = FastAPI(title="Anime RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    """Health endpoint."""
    return {
        "status": "online",
        "docs": "/docs",
        "message": "Anime RAG API is active",
        "collection_count": collection.count()
    }

@app.get("/ask")
def api_ask(
    prompt: str,
    use_lite: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 512,
    mode: str = "trivia"
):
    """RAG endpoint for generating responses."""
    ans = generate_rag(prompt, use_lite, temperature, max_tokens, mode)
    return {"response": ans}

if __name__ == "__main__":
    load_titles_and_names()
    print("Running Anime RAG Web App + API")
    print("FastAPI: http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

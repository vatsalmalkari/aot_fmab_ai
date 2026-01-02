import os
import json
from functools import lru_cache
from typing import List
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from rapidfuzz import process
from chromadb.config import Settings

# Load env
load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH = "./aot_fmab_db"
JSON_DATA_DIR = "./json_output"

genai.configure(api_key=os.getenv("API_KEY"))

FLASH_MODEL = "gemini-2.5-flash"
LITE_MODEL = "gemini-2.5-flash-lite"

flash_model = genai.GenerativeModel(FLASH_MODEL)
lite_model = genai.GenerativeModel(LITE_MODEL)


# Choose model
def get_model(use_lite: bool = False):
    return lite_model if use_lite else flash_model

# Load embedding + ChromaDB
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.Client(Settings(
    persist_directory=CHROMA_DB_PATH,
    chroma_db_impl="duckdb+parquet"  ))
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


# RAG Response Generator
response_cache = {}

def generate_rag(prompt, use_lite, temp, max_tokens, mode):
    cache_key = (prompt, use_lite, temp, max_tokens, mode)
    if cache_key in response_cache:
        return response_cache[cache_key]

    model = get_model(use_lite)
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

    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temp,
            max_output_tokens=max_tokens
        )
    )

    output = response.text.strip()
    response_cache[cache_key] = output
    return output

# GRADIO UI
with gr.Blocks() as gr_app:
    with gr.Column():
        prompt_input = gr.Textbox(label="Your Anime Question/Prompt", lines=5)
        use_lite_checkbox = gr.Checkbox(label="Use Lite Model (Faster)", value=True)
        temp_slider = gr.Slider(0, 1.0, step=0.1, value=0.7, label="Creativity")
        token_slider = gr.Slider(50, 2048, step=50, value=512, label="Max Tokens")
        mode_dropdown = gr.Dropdown(
            choices=["trivia", "fanfiction", "summary"],
            value="trivia",
            label="Response Mode"
        )
        submit_btn = gr.Button("Generate Response")
        output_box = gr.Textbox(label="AI Response", lines=18)

    submit_btn.click(
        generate_rag,
        inputs=[prompt_input, use_lite_checkbox, temp_slider, token_slider, mode_dropdown],
        outputs=output_box
    )

if __name__ == "__main__":
    load_titles_and_names()

    print("Running Anime RAG Web App + API ")
    print("Gradio UI: http://127.0.0.1:7860")
    def start_gradio():
        gr_app.launch(server_name="0.0.0.0", server_port=7860, share=False)

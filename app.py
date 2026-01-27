import os
import json
from typing import List
from functools import lru_cache
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
import gradio as gr
from google import genai
from pydantic import BaseModel
from rapidfuzz import process
from sentence_transformers import SentenceTransformer
import uvicorn

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Config & Paths
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH = "./anime_vector_db"
JSON_DATA_DIR = "./json_output"

# Poster Paths
AOT_IMAGE = "images/aot_poster.png"
FMAB_IMAGE = "images/fmab_poster.png"
JJK_IMAGE = "images/jjk_poster.png"

FLASH_MODEL = "gemini-2.5-flash"
LITE_MODEL = "gemini-2.5-flash-lite"

app = FastAPI(title="Anime RAG API")
client = genai.Client(api_key=API_KEY)

# Initialize models
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="anime_data")

# Logic & Helper Functions
all_titles_or_names: List[str] = []

def load_titles_and_names():
    global all_titles_or_names
    names = set()
    if os.path.exists(JSON_DATA_DIR):
        for file in os.listdir(JSON_DATA_DIR):
            if file.endswith(".json"):
                with open(os.path.join(JSON_DATA_DIR, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key in ["name", "anime", "title"]:
                    if data.get(key):
                        names.add(data[key])
    all_titles_or_names = list(names)

@lru_cache(maxsize=256)
def get_embedding(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()

def retrieve_context(prompt: str, top_k: int = 5) -> str:
    corrected_prompt = prompt
    if all_titles_or_names:
        match = process.extractOne(prompt, all_titles_or_names, score_cutoff=70)
        corrected_prompt = match[0] if match else prompt
    
    embedding = get_embedding(corrected_prompt)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    documents = results.get("documents", [[]])[0]
    return "\n\n".join(documents)

def run_rag_pipeline(prompt, use_lite, temp, max_tokens, mode):
    context = retrieve_context(prompt)
    model_id = LITE_MODEL if use_lite else FLASH_MODEL

    instructions = {
        "trivia": "Answer factually using only the provided context.",
        "fanfiction": "Write an immersive fanfiction scene grounded in the context.",
        "summary": "Write a concise, structured summary using the context."
    }
    
    system_instruction = instructions.get(mode, "You are a helpful anime assistant.")
    user_content = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{prompt}"

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=user_content,
            config={
                "system_instruction": system_instruction,
                "temperature": temp,
                "max_output_tokens": max_tokens,
            }
        )
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

# FastAPI Endpoint
class QueryRequest(BaseModel):
    prompt: str
    use_lite: bool = True
    temp: float = 0.7
    max_tokens: int = 512
    mode: str = "trivia"

@app.post("/query")
async def api_query(req: QueryRequest):
    output = run_rag_pipeline(req.prompt, req.use_lite, req.temp, req.max_tokens, req.mode)
    return {"response": output}

# Gradio UI
# Helper to check if image exists to avoid Gradio resolution errors
def get_image_path(path):
    return path if os.path.exists(path) else None

with gr.Blocks(css=".poster {border: 3px solid #555; border-radius: 8px;}") as gr_app:
    gr.Markdown("# 🎌 Anime RAG Explorer")
    
    with gr.Row():
        gr.Image(value=get_image_path(AOT_IMAGE), label="Attack on Titan", interactive=False, elem_classes="poster")
        gr.Image(value=get_image_path(FMAB_IMAGE), label="FMAB", interactive=False, elem_classes="poster")
        gr.Image(value=get_image_path(JJK_IMAGE), label="Jujutsu Kaisen", interactive=False, elem_classes="poster")

    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(label="Ask about the anime lore...", lines=5, placeholder="e.g., Who is the Colossal Titan?")
            with gr.Row():
                use_lite = gr.Checkbox(label="Use Lite Model", value=True)
                mode = gr.Dropdown(["trivia", "fanfiction", "summary"], value="trivia", label="Response Mode")
            temp = gr.Slider(0, 1.0, value=0.7, label="Creativity/Temperature")
            max_tokens = gr.Slider(50, 2048, value=512, step=50, label="Max Tokens")
            submit_btn = gr.Button("Generate Response", variant="primary")
        
        with gr.Column(scale=4):
            output_box = gr.Textbox(label="AI Response", lines=18)

    submit_btn.click(
        run_rag_pipeline,
        inputs=[prompt_input, use_lite, temp, max_tokens, mode],
        outputs=output_box
    )

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, gr_app, path="/")

if __name__ == "__main__":
    load_titles_and_names()
    print("Server starting...")
    print("API Docs: http://localhost:8000/docs")
    print("Interface: http://localhost:8000/")
    uvicorn.run(app, host="0.0.0.0", port=8000)

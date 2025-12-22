import gradio as gr
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import os
import json
from typing import List, Dict
from functools import lru_cache
from dotenv import load_dotenv
from config import API_KEY, CHROMA_DB_PATH, EMBEDDING_MODEL
load_dotenv()  # loads from .env

# Initialize components
API_KEY = os.getenv("API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./aot_fmab_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-small-en-v1.5")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "anime_data")

# Initialize models and clientgenai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Embedding with caching
@lru_cache(maxsize=100)
def get_embedding(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()

# ChromaDB setup
JSON_DATA_DIR = "./json_output"
collection = chroma_client.get_or_create_collection(name="anime_data")
if collection.count() == 0:
    print("Populating collection")
    documents, metadatas, ids = [], [], []
    
    for root, _, files in os.walk(JSON_DATA_DIR):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                
                doc_id = f"{os.path.splitext(file)[0]}_{len(documents)}"
                meta = {
                    'source': file,
                    'anime': data.get('anime', 'unknown'),
                    'type': data.get('type', 'unknown'),
                    **{f"meta_{k}": str(v) for k, v in data.get('metadata', {}).items()}
                }

                documents.append(data['content'])
                metadatas.append(meta)
                ids.append(doc_id)

    embeddings = [get_embedding(doc) for doc in documents]
    collection.add(documents=documents, metadatas=metadatas, embeddings=embeddings, ids=ids)

# Helper functions
def _build_where_clause(prompt: str) -> Dict:
    prompt_lower = prompt.lower()
    where = {}

    if any(x in prompt_lower for x in ['aot', 'attack on titan']):
        where['anime'] = 'aot'
    elif any(x in prompt_lower for x in ['fmab', 'fullmetal']):
        where['anime'] = 'fmab'

    if 'episode' in prompt_lower:
        where['type'] = 'episode'
    elif 'character' in prompt_lower:
        where['type'] = 'character'

    return where if where else None

def _format_context(results: Dict) -> str:
    if not results or not results['documents']:
        return "No relevant context found."
    
    context = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        context.append(f"Source: {meta.get('source', 'unknown')}\nContent: {doc[:500]}...\n")
    return "\n".join(context)

def _build_prompt(prompt: str, mode: str, context: str) -> str:
    instructions = {
        'trivia': "Provide a concise, factual answer to the trivia question.",
        'summary': "Generate a comprehensive yet concise summary.",
        'fan_fiction': "Create an engaging story that respects canon lore."
    }
    return f"{instructions.get(mode, '')}\n\nContext:\n{context}\n\nQuestion: {prompt}"

def generate_response(prompt: str, mode: str, temperature: float, max_tokens: int) -> str:
    query_embedding = get_embedding(prompt)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where=_build_where_clause(prompt)
    )
    context = _format_context(results)
    response = gemini_model.generate_content(
        _build_prompt(prompt, mode, context),
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    )
    return response.text

# Gradio Interface
custom_css = """<keep your entire CSS as-is>"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:
    with gr.Column(elem_classes="welcome-panel"):
        gr.Markdown("""
        Your AOT & FMAB Expert
        The ultimate AI assistant for Attack on Titan and Fullmetal Alchemist Brotherhood
        """)
        
        with gr.Accordion("How to Use This GPT", open=True):
            gr.Markdown("""
            ### This assistant specializes in:
            -Trivia Answers: "Who is the Armored Titan? "
            -Episode Summaries: "What happened in FMAB Episode 19? "
            -Fan Fiction: "What if Eren and Edward met? "
            """)
            
            with gr.Row(elem_classes="image-row"):
                aot_img = gr.Image(value="images/aot_poster.png", show_label=False, elem_id="aot-logo")
                fmab_img = gr.Image(value="images/fmab_poster.png", show_label=False, elem_id="fmab-logo")
                gr.Markdown("""
                <div style="margin-left: 20px;">
                Simply type your question or story prompt below, select a response mode, 
                and adjust the settings to customize your experience!
                </div>
                """)

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Your Anime Question/Prompt", placeholder="e.g. 'Explain Zeke's euthanasia plan'", lines=4)
            mode = gr.Radio(
                choices=[("Trivia Mode", "trivia"), ("Summary Mode", "summary"), ("Fan Fiction Mode", "fan_fiction")],
                label="Response Type", value="trivia", elem_classes="radio-buttons"
            )

            with gr.Accordion("Advanced Settings", open=False):
                temp = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.7, info="0.0 = Strictly factual, 1.0 = Highly creative")
                tokens = gr.Slider(minimum=50, maximum=2048, step=50, value=512, info="1 token ≈ 1 word (~400 words for 512)")

            submit_btn = gr.Button("Generate Response", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(label="AI Response", placeholder="Your generated response will appear here...", lines=12)

    submit_btn.click(fn=generate_response, inputs=[prompt, mode, temp, tokens], outputs=output)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
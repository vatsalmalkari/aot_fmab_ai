import gradio as gr
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import os
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

API_KEY = "AIzaSyBW4QhiPWUloZ1rmgp0X14hO039IygsS1E"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  
CHROMA_DB_PATH = "./aot_fmab_db"
COLLECTION_NAME = "anime_data"
JSON_DATA_DIR = "./json_output"
MAX_CONTEXT_LENGTH = 1500 

def initialize_components():
    components = {}
    
    try:
        genai.configure(api_key=API_KEY)
        components['gemini_model'] = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"Gemini initialization error: {str(e)}")
        components['gemini_model'] = None

    try:
        components['embedding_model'] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Embedding model error: {str(e)}")
        components['embedding_model'] = None

    try:
        components['chroma_client'] = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    except Exception as e:
        print(f"ChromaDB client error: {str(e)}")
        components['chroma_client'] = None

    return components

components = initialize_components()

@lru_cache(maxsize=100)
def get_embedding(text: str) -> List[float]:
    if components['embedding_model']:
        return components['embedding_model'].encode(text).tolist()
    raise ValueError("Embedding model not available")

def setup_chroma_db():
    if not components['chroma_client']:
        raise RuntimeError("ChromaDB client not initialized")
    
    try:
        collection = components['chroma_client'].get_or_create_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Collection error: {str(e)}")
        return None

    if collection.count() == 0:
        print("Populating collection...")
        documents, metadatas, ids = [], [], []
        
        for root, _, files in os.walk(JSON_DATA_DIR):
            for file in files:
                if file.endswith('.json'):
                    try:
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
                    
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
        
        if documents:
            with ThreadPoolExecutor() as executor:
                embeddings = list(executor.map(get_embedding, documents))
            collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
    
    return collection

anime_collection = setup_chroma_db()


def _build_where_clause(prompt: str) -> Dict:
    prompt_lower = prompt.lower()
    where = {}
    
    # Anime detection
    if any(x in prompt_lower for x in ['aot', 'attack on titan']):
        where['anime'] = 'aot'
    elif any(x in prompt_lower for x in ['fmab', 'fullmetal']):
        where['anime'] = 'fmab'
    
    # Content type detection
    if 'episode' in prompt_lower:
        where['type'] = 'episode'
    elif 'character' in prompt_lower:
        where['type'] = 'character'
    
    return where if where else None

def generate_response(prompt: str, mode: str, temperature: float, max_tokens: int) -> str:
    """Generate response using RAG pipeline"""
    if not components['gemini_model']:
        return "Error: Model not initialized"
    
    try:
        # Enhanced query construction
        query_embedding = get_embedding(prompt)
        results = anime_collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where=_build_where_clause(prompt))
        
        context = _format_context(results)
        response = components['gemini_model'].generate_content(
            _build_prompt(prompt, mode, context),
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        return response.text
    
    except Exception as e:
        return f"Error: {str(e)}"

def _format_context(results: Dict) -> str:
    """Format context from ChromaDB results"""
    if not results or not results['documents']:
        return "No relevant context found."
    
    context = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        context.append(
            f"Source: {meta.get('source', 'unknown')}\n"
            f"Content: {doc[:500]}...\n"
        )
    return "\n".join(context)

def _build_prompt(prompt: str, mode: str, context: str) -> str:
    """Construct optimized prompt based on mode"""
    instructions = {
        'trivia': "Provide a concise, factual answer to the trivia question.",
        'summary': "Generate a comprehensive yet concise summary.",
        'fan_fiction': "Create an engaging story that respects canon lore."
    }
    return f"{instructions.get(mode, '')}\n\nContext:\n{context}\n\nQuestion: {prompt}"

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("Anime GPT: AOT & FMAB Expert")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Your Question", placeholder="Ask about characters, episodes, or request a story...")
            mode = gr.Radio(
                ["trivia", "summary", "fan_fiction"],
                label="Response Mode",
                value="trivia"
            )
            temp = gr.Slider(0, 1, value=0.7, label="Creativity Level")
            tokens = gr.Slider(50, 2048, value=512, step=50, label="Response Length")
            btn = gr.Button("Generate Response")
        
        with gr.Column():
            output = gr.Textbox(label="AI Response", lines=10)
    
    btn.click(
        fn=generate_response,
        inputs=[prompt, mode, temp, tokens],
        outputs=output
    )


custom_css = """
:root {
    --bg-color: #ffffff;
    --text-color: #333333;
    --card-bg: #f8f9fa;
    --border-color: #e0e0e0;
}

.dark {
    --bg-color: #1a1a1a;
    --text-color: #f0f0f0;
    --card-bg: #2d2d2d;
    --border-color: #444444;
}

.gradio-container {
    font-family: 'Helvetica', Arial, sans-serif;
    background: var(--bg-color);
    color: var(--text-color);
}

.welcome-panel {
    background: linear-gradient(135deg, var(--card-bg) 0%, #c3cfe2 100%);
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 25px;
    border: 1px solid var(--border-color);
}

.instructions {
    background: var(--card-bg);
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    border: 1px solid var(--border-color);
}

.param-card {
    background: var(--card-bg);
    padding: 18px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin: 12px 0;
    border: 1px solid var(--border-color);
}

#aot-logo, #fmab-logo {
    border-radius: 10px;
    border: 2px solid var(--border-color);
    margin-right: 15px;
    height: 220px !important;
    width: 220px !important;
    object-fit: contain;
}

.image-row {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
}

.dark #aot-logo,
.dark #fmab-logo {
    border-color: #555;
}
"""

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
                aot_img = gr.Image(
                    value="images/aot_poster.png",
                    show_label=False,
                    elem_id="aot-logo"
                )
                fmab_img = gr.Image(
                    value="images/fmab_poster.png",
                    show_label=False,
                    elem_id="fmab-logo"
                )
                gr.Markdown("""
                <div style="margin-left: 20px;">
                Simply type your question or story prompt below, select a response mode, 
                and adjust the settings to customize your experience!
                </div>
                """)
    # ========== MAIN INTERFACE ==========
    with gr.Row():
        with gr.Column(scale=3):
            # Input Section
            with gr.Group():
                prompt = gr.Textbox(
                    label="Your Anime Question/Prompt",
                    placeholder="e.g. 'Explain Zeke's euthanasia plan' or 'Write a story where Edward meets Levi'",
                    lines=4
                )
                
                mode = gr.Radio(
                    choices=[
                        ("Trivia Mode", "trivia"),
                        ("Summary Mode", "summary"),
                        ("Fan Fiction Mode", "fan_fiction")
                    ],
                    label="Response Type",
                    value="trivia",
                    elem_classes="radio-buttons"
                )
            
            # Parameters with explanations
            with gr.Accordion(" Advanced Settings", open=False):
                with gr.Column(elem_classes="param-card"):
                    gr.Markdown(" Creativity (Temperature)")
                    temp = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.1, value=0.7,
                        info="0.0 = Strictly factual, 1.0 = Highly creative"
                    )
                
                with gr.Column(elem_classes="param-card"):
                    gr.Markdown(" Response Length (Max Tokens)")
                    tokens = gr.Slider(
                        minimum=50, maximum=2048, step=50, value=512,
                        info="1 token ≈ 1 word (512 tokens = ~400 words)"
                    )
            
            submit_btn = gr.Button("Generate Response", variant="primary")

        with gr.Column(scale=2):
            # Output Section
            output = gr.Textbox(
                label="AI Response",
                placeholder="Your generated response will appear here...",
                lines=12,
                elem_id="output-box"
            )
            
            with gr.Accordion("Click me if you need help understanding the Settings", open=False):
                gr.Markdown("""
                Response Types:
                - Trivia: For quick facts and answers
                - Summary: For overview of episode/character
                - Fan Fiction: For creative stories and crossovers
                
                Advanced Settings:
                - Temperature: Controls randomness (lower=more factual)
                - Max Tokens: Limits response length (higher=longer answers)
                """)
    submit_btn.click(
        fn=generate_response,
        inputs=[prompt, mode, temp, tokens],
        outputs=output
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  
    )
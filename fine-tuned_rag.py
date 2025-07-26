import logging
import gradio as gr
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import os
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from fuzzywuzzy import process, fuzz
from functools import lru_cache

ALL_CHARACTER_NAMES = []
ALL_EPISODE_TITLES = []
ALL_ANIME_ARCS = []
ALL_CHARACTER_TRAITS = []

# Configuration
API_KEY = "AIzaSyBW4QhiPWUloZ1rmgp0X14hO039IygsS1E"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH = "./aot_fmab_db"
COLLECTION_NAME = "anime_data"
JSON_DATA_DIR = "./json_output"
MAX_CONTEXT_LENGTH = 1500

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_components():
    """Initialize all required components with error handling"""
    components = {}
    
    try:
        genai.configure(api_key=API_KEY)
        components['gemini_model'] = genai.GenerativeModel('gemini-1.5-flash')
        logging.info("Gemini model initialized successfully")
    except Exception as e:
        logging.error(f"Gemini initialization error: {str(e)}")
        components['gemini_model'] = None

    try:
        components['embedding_model'] = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info("Embedding model loaded successfully")
    except Exception as e:
        logging.error(f"Embedding model error: {str(e)}")
        components['embedding_model'] = None

    try:
        components['chroma_client'] = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        logging.info("ChromaDB client initialized successfully")
    except Exception as e:
        logging.error(f"ChromaDB client error: {str(e)}")
        components['chroma_client'] = None

    return components

components = initialize_components()

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> List[float]:
    """Get cached embedding for text"""
    if not components.get('embedding_model'):
        raise ValueError("Embedding model not available")
    return components['embedding_model'].encode(text).tolist()

def setup_chroma_db():
    """Initialize ChromaDB collection with data"""
    global ALL_CHARACTER_NAMES, ALL_EPISODE_TITLES, ALL_ANIME_ARCS, ALL_CHARACTER_TRAITS
    
    if not components.get('chroma_client'):
        raise RuntimeError("ChromaDB client not initialized")

    try:
        collection = components['chroma_client'].get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logging.info(f"Collection '{COLLECTION_NAME}' accessed")
    except Exception as e:
        logging.error(f"Collection error: {str(e)}")
        raise

    if collection.count() == 0:
        logging.info(f"Populating ChromaDB collection '{COLLECTION_NAME}'")
        documents, metadatas, ids = [], [], []
        temp_data = {
            'characters': set(),
            'episodes': set(),
            'arcs': set(),
            'traits': set()
        }

        for root, _, files in os.walk(JSON_DATA_DIR):
            for file in files:
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        doc_id = f"{os.path.splitext(file)[0]}_{len(documents)}"
                        meta = {
                            'source': file,
                            'anime': data.get('anime', 'unknown'),
                            'type': data.get('type', 'unknown'),
                            **{f"meta_{k}": str(v) for k, v in data.get('metadata', {}).items()}
                        }
                        
                        # Collect metadata
                        if data['type'] == 'character':
                            if 'name' in data:
                                temp_data['characters'].add(data['name'])
                            if 'aliases' in data:
                                temp_data['characters'].update(data['aliases'])
                            if 'metadata' in data and 'traits' in data['metadata']:
                                temp_data['traits'].update(data['metadata']['traits'])
                        elif data['type'] == 'episode':
                            if 'title' in data:
                                temp_data['episodes'].add(data['title'])
                            if 'arc' in data:
                                temp_data['arcs'].add(data['arc'])
                        
                        documents.append(data['content'])
                        metadatas.append(meta)
                        ids.append(doc_id)
                    
                    except Exception as e:
                        logging.error(f"Error processing {file}: {str(e)}")

        # Update global lists
        ALL_CHARACTER_NAMES = sorted(list(temp_data['characters']))
        ALL_EPISODE_TITLES = sorted(list(temp_data['episodes']))
        ALL_ANIME_ARCS = sorted(list(temp_data['arcs']))
        ALL_CHARACTER_TRAITS = sorted(list(temp_data['traits']))
        
        logging.info(f"Loaded {len(ALL_CHARACTER_NAMES)} characters, {len(ALL_EPISODE_TITLES)} episodes, {len(ALL_ANIME_ARCS)} arcs, {len(ALL_CHARACTER_TRAITS)} traits")

        # Add to ChromaDB
        if documents:
            with ThreadPoolExecutor() as executor:
                embeddings = list(executor.map(get_embedding, documents))
            collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            logging.info(f"Added {len(documents)} documents to collection")
    
    return collection

def _build_where_clause(prompt: str) -> Dict:
    """Construct optimized where clause based on prompt"""
    prompt_lower = prompt.lower()
    conditions = []
    
    # Anime detection
    anime_map = {
        'aot': ['attack on titan', 'shingeki no kyojin'],
        'fmab': ['fullmetal alchemist', 'hagane no renkinjutsushi']
    }
    for anime_id, terms in anime_map.items():
        if any(term in prompt_lower for term in terms):
            conditions.append({'anime': anime_id})
            break
    
    # Document type
    type_indicators = {
        'episode': ['episode', 'ep ', 'season'],
        'character': ['character', 'who is', 'tell me about']
    }
    for doc_type, indicators in type_indicators.items():
        if any(ind in prompt_lower for ind in indicators):
            conditions.append({'type': doc_type})
            break
    
    # Fuzzy matching
    match_config = [
        (ALL_CHARACTER_NAMES, 80, ['name', 'aliases']),
        (ALL_EPISODE_TITLES, 75, ['title']),
        (ALL_ANIME_ARCS, 70, ['arc'])
    ]
    
    for values, threshold, fields in match_config:
        match = process.extractOne(prompt_lower, values, scorer=fuzz.token_set_ratio)
        if match and match[1] >= threshold:
            conditions.append({'$or': [{f: match[0]} for f in fields]})
    
    return conditions[0] if len(conditions) == 1 else {'$and': conditions} if conditions else None

def generate_response(prompt: str, mode: str, temperature: float, max_tokens: int) -> str:
    """Generate response using RAG pipeline without source numbering"""
    try:
        if not components.get('gemini_model'):
            return "Error: Model not initialized"
        
        query_embedding = get_embedding(prompt)
        where_clause = _build_where_clause(prompt)
        
        results = anime_collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where=where_clause,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format context without source numbering
        context = "\n".join(
            f"{doc[:300]}..."  # Just show the content snippet without source info
            for doc in results['documents'][0]
        )
        
        response = components['gemini_model'].generate_content(
            f"Mode: {mode}\nContext:\n{context}\nQuestion: {prompt}",
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.9
            ),
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            }
        )
        return response.text
    
    except Exception as e:
        logging.error(f"Generation error: {str(e)}")
        return f"Error generating response: {str(e)}"

# Initialize ChromaDB
try:
    anime_collection = setup_chroma_db()
    logging.info("ChromaDB setup completed successfully")
except Exception as e:
    logging.error(f"Failed to initialize ChromaDB: {str(e)}")
    raise

# Gradio Interface
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
    # ========== HEADER SECTION ==========
    with gr.Column(elem_classes="welcome-panel"):
        gr.Markdown("""
        #  Anime GPT: AOT & FMAB Expert
        *Your ultimate assistant for Attack on Titan and Fullmetal Alchemist Brotherhood knowledge*
        """)
        
        with gr.Accordion("How to Use This GPT", open=False):
            gr.Markdown("""
            ###  This assistant specializes in:
            - **Trivia Answers**: "Who is the Armored Titan?"  
            - **Episode Summaries**: "What happened in FMAB Episode 19?"  
            - **Fan Fiction**: "What if Eren and Edward met?"  
            - **Character Analysis**: "Compare Levi and Roy Mustang's leadership styles"  
            - **Theories & Discussions**: "Explain the philosophical themes in AOT"  
            """)
            
            with gr.Row(elem_classes="image-row"):
                aot_img = gr.Image(
                    value="images/aot_poster.png",
                    show_label=False,
                    elem_id="aot-logo",
                    interactive=False
                )
                fmab_img = gr.Image(
                    value="images/fmab_poster.png",
                    show_label=False,
                    elem_id="fmab-logo",
                    interactive=False
                )
                
    # ========== MAIN INTERFACE ==========
    with gr.Row(equal_height=True):
        # ===== INPUT COLUMN =====
        with gr.Column(scale=2):
            # Input Section
            with gr.Group():
                prompt = gr.Textbox(
                    label=" Your Anime Question/Prompt",
                    placeholder="Examples:\n"
                    "- 'Explain Zeke's euthanasia plan'\n"
                    "- 'Write a story where Edward meets Levi'\n"
                    "- 'Compare the homunculi to the Titans'\n"
                    "- 'Summarize the Battle of Shiganshina'",
                    lines=5,
                    max_lines=8,
                    elem_id="prompt-box"
                )
                
            # Mode Selection
            with gr.Group():
                mode = gr.Radio(
                    choices=[
                        (" Trivia Mode (Factual Answers)", "trivia"),
                        (" Summary Mode (Episode/Character Overview)", "summary"),
                        (" Fan Fiction Mode (Creative Stories)", "fan_fiction")
                    ],
                    label=" Response Type",
                    value="trivia",
                    elem_classes="radio-buttons"
                )
            
            # Advanced Settings
            with gr.Accordion(" Advanced Settings", open=False):
                with gr.Row():
                    with gr.Column():
                        temp = gr.Slider(
                            label=" Creativity (Temperature)",
                            minimum=0.0, maximum=1.0, step=0.1, value=0.7,
                            info="Lower = More factual, Higher = More creative"
                        )
                    with gr.Column():
                        tokens = gr.Slider(
                            label=" Response Length (Max Tokens)",
                            minimum=50, maximum=2048, step=50, value=512,
                            info="1 token ≈ 1 word (512 ≈ 400 words)"
                        )
            
            submit_btn = gr.Button(
                "Generate Response ", 
                variant="primary",
                elem_id="submit-btn"
            )
            
            gr.Markdown("""
            <div style="text-align: center; margin-top: 10px;">
            <small>Tip: For best results, be specific with your questions!</small>
            </div>
            """)

        # ===== OUTPUT COLUMN =====
        with gr.Column(scale=3):
            # Output Section
            output = gr.Textbox(
                label="🤖 AI Response",
                placeholder="Your generated response will appear here...",
                lines=18,
                elem_id="output-box",
                show_copy_button=True
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                regenerate_btn = gr.Button("Regenerate", variant="secondary")
                copy_btn = gr.Button("Copy to Clipboard", variant="secondary")
            
            with gr.Accordion("💡 Response Examples", open=False):
                gr.Markdown("""
                **Trivia Example:**  
                *"The Armored Titan is the form taken by Reiner Braun, one of the Warriors from Marley..."*
                
                **Summary Example:**  
                *"Episode 19 of FMAB, titled 'Death of the Undying', focuses on the battle between the Elric brothers..."*
                
                **Fan Fiction Example:**  
                *"The meeting between Eren Yeager and Edward Elric would likely begin with mutual suspicion. Edward..."*
                """)
    
    # ========== EVENT HANDLERS ==========
    submit_btn.click(
        fn=generate_response,
        inputs=[prompt, mode, temp, tokens],
        outputs=output
    )
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[prompt, output]
    )
    regenerate_btn.click(
        fn=generate_response,
        inputs=[prompt, mode, temp, tokens],
        outputs=output
    )
    copy_btn.click(
        fn=None,
        inputs=output,
        js="() => {navigator.clipboard.writeText(document.getElementById('output-box').value);}"
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  
    )

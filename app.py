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

load_dotenv()  # Loads API key

# Config / Paths
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH = "./aot_fmab_db"
JSON_DATA_DIR = "./json_output"
genai.configure(api_key=os.getenv("API_KEY"))

# Initialize models
FLASH_MODEL = "gemini-2.5-flash"
LITE_MODEL = "gemini-2.5-flash-lite"

flash_model = genai.GenerativeModel(FLASH_MODEL)
lite_model = genai.GenerativeModel(LITE_MODEL)

def get_model(use_lite: bool = False):
    return lite_model if use_lite else flash_model

# Initialize embeddings & DB
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="anime_data")

@lru_cache(maxsize=100)
def get_embedding(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()

# Cache for fast repeated queries
response_cache = {}

# Helper: split text into chunks
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

# Populates ChromaDB
def process_json_files(json_dir: str):
    if collection.count() > 0:
        print("ChromaDB already populated.")
        return

    print("Populating ChromaDB with embeddings...")
    documents, metadatas, ids = [], [], []

    for file in os.listdir(json_dir):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(json_dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = {
            "name": data.get("name", ""),
            "anime": data.get("anime", ""),
            "type": data.get("type", ""),
            "source_file": data.get("metadata", {}).get("source_file", "")
        }

        for i, chunk in enumerate(chunk_text(data.get("content", ""))):
            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(f"{file}_{i}")

    batch_size = 32
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        embeddings = embedding_model.encode(batch_docs).tolist()
        collection.add(
            documents=batch_docs,
            metadatas=batch_metadatas,
            ids=batch_ids,
            embeddings=embeddings
        )
    print("ChromaDB population complete!")

# Fuzzy matching for typos
all_titles_or_names = []  #from JSON metadata
def load_titles_and_names():
    global all_titles_or_names
    names = set()
    for file in os.listdir(JSON_DATA_DIR):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(JSON_DATA_DIR, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            names.add(data.get("name", ""))
            names.add(data.get("anime", ""))
            if data.get("type") == "episode":
                names.add(data.get("title", ""))
    all_titles_or_names = [n for n in names if n]

def fix_typo(query, choices=all_titles_or_names, score_cutoff=70):
    best_match = process.extractOne(query, choices, score_cutoff=score_cutoff)
    return best_match[0] if best_match else query

# ------------------------------
# Retrieval
# ------------------------------
def retrieve_context(prompt, top_k=5):
    corrected_prompt = fix_typo(prompt)
    query_embedding = get_embedding(corrected_prompt)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    context_chunks = results['documents'][0] if results['documents'] else []
    return "\n".join(context_chunks)

# ------------------------------
# Generate response with RAG + continuation
# ------------------------------
def generate_response(prompt: str, use_lite: bool, temperature: float, max_tokens: int,
                      mode: str, max_iterations: int = 5) -> str:
    cache_key = (prompt, use_lite, temperature, max_tokens, mode)
    if cache_key in response_cache:
        return response_cache[cache_key]

    model = get_model(use_lite)
    context = retrieve_context(prompt, top_k=5)
    mode_instructions = {
        "trivia": "Answer with detailed trivia facts about the anime using the context below:",
        "fanfiction": "Write a detailed fanfiction scene based on the anime using the context below:",
        "summary": "Provide a concise summary of the anime content using the context below:"
    }
    prefix = mode_instructions.get(mode, "")
    full_prompt = f"{prefix}\n\nContext:\n{context}\n\nUser Prompt:\n{prompt}"

    output = ""
    current_prompt = full_prompt

    try:
        for _ in range(max_iterations):
            response = model.generate_content(
                current_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    top_p=0.9
                )
            )
            text_chunk = getattr(response, "text", "")
            if not text_chunk.strip():
                break
            output += text_chunk.strip() + "\n\n"
            # Continue
            current_prompt = f"Continue the text from before in the same style and context:\n{text_chunk}"

        output = output.strip()
        response_cache[cache_key] = output
        return output
    except Exception as e:
        return f"Error generating response: {e}"

# ------------------------------
# Test retrieval accuracy
# ------------------------------
def evaluate_retrieval_accuracy(top_k=5):
    test_queries = [
        {"query": "Who is Edward Elric?", "expected_doc": "Edward Elric is"},
        {"query": "Attack on Titan episode 1 summary", "expected_doc": "In episode 1"}
    ]
    correct = 0
    for item in test_queries:
        query = fix_typo(item["query"])
        embedding = get_embedding(query)
        results = collection.query(query_embeddings=[embedding], n_results=top_k)
        retrieved_docs = results['documents'][0] if results['documents'] else []
        if any(item["expected_doc"] in doc for doc in retrieved_docs):
            correct += 1
    accuracy = correct / len(test_queries)
    return accuracy

# ------------------------------
# Gradio UI
# ------------------------------
with gr.Blocks() as app:
    with gr.Column():
        prompt_input = gr.Textbox(label="Your Anime Question/Prompt", lines=5)
        use_lite_checkbox = gr.Checkbox(label="Use Lite Model (Faster)", value=True)
        temp_slider = gr.Slider(0.0, 1.0, step=0.1, value=0.7, label="Creativity (Temperature)")
        token_slider = gr.Slider(50, 2048, step=50, value=512, label="Response Length (Max Tokens)")
        mode_dropdown = gr.Dropdown(
            choices=["trivia", "fanfiction", "summary"],
            value="trivia",
            label="Response Mode"
        )
        submit_btn = gr.Button("Generate Response")
        output_box = gr.Textbox(label="AI Response", lines=18, show_copy_button=True)

    submit_btn.click(
        fn=generate_response,
        inputs=[prompt_input, use_lite_checkbox, temp_slider, token_slider, mode_dropdown],
        outputs=output_box
    )

# Main
if __name__ == "__main__":
    process_json_files(JSON_DATA_DIR)
    load_titles_and_names()
    print("ChromaDB loaded and names indexed.")

    # test retrieval accuracy
    acc = evaluate_retrieval_accuracy(top_k=5)
    print(f"Test retrieval accuracy: {acc*100:.2f}%")

    app.launch()

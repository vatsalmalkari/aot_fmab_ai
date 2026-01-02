---
title: Anime QA with RAG
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: true
short_description: A RAG model that answers AOT & FMAB questions
---

# AOT & FMAB AI — Anime QA System

This is an **Anime Question-Answering (QA) system** built with **Retrieval-Augmented Generation (RAG)**.  
It lets you ask questions about **Attack on Titan (AOT)** and **Fullmetal Alchemist: Brotherhood (FMAB)** and get **accurate, context-aware answers**.

### Response Modes

- **Trivia** — Quick factual answers about characters, episodes, or the anime world.  
- **Summary** — Summarizes episodes, characters, or the overall plot.  
- **Fanfiction** — Generates short fictional stories set in the anime universe.  

---

## How It Works (Workflow)

1. **Data Preparation** — JSON files containing episode summaries, character bios, and trivia are stored in `./json_output`.  
2. **Embedding Generation** — Text data is converted into **vector embeddings** using **SentenceTransformers**. These embeddings are stored in **ChromaDB** for semantic search.  
3. **Semantic Search** — User queries are matched against the database using embeddings. Typos are corrected automatically with **RapidFuzz**.  
4. **RAG Response Generation** — Retrieved context is fed into **Google Gemini** to generate a natural, informative response in the chosen mode (trivia, summary, fanfiction).  
5. **Delivery** — Users see the answer in either the **Gradio web app** or via **FastAPI endpoints**.

---

## Project Files and Execution Order

1. **`generate_embeddings.py`**  
   - Generates vector embeddings from your JSON anime data.  
   - Must be run **before** starting the app.  

2. **`fastapi.py`**  
   - Runs the **backend API**.  
   - Users can query using endpoints like `/ask?prompt=YourQuestion&mode=trivia`.  

3. **`gradio_app.py`** (or `app.py`)  
   - Launches the **interactive Gradio frontend**.  
   - Connects to the RAG pipeline to provide instant answers.  

**Recommended Order to Run:**
```bash
python generate_embeddings.py   # Step 1: Populate ChromaDB
python fastapi.py               # Step 2: Start backend API
python gradio_app.py            # Step 3: Launch Gradio UI

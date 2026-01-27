
---

# Spoiler-Controlled TV Show RAG Agent

Ever wanted to talk to a chatbot about a TV show **you’re currently watching** —
**without accidentally getting spoilers** about future episodes?

This project does exactly that.

It is a **Retrieval-Augmented Generation (RAG) system** that lets you ask questions about TV shows (currently **Attack on Titan**, **Jujutsu Kaisen and **Fullmetal Alchemist: Brotherhood**) while ensuring responses are **grounded only in approved context**.

No accidental spoilers.
No future-episode leaks.
Just what *you’ve already watched*.

---

##  What This System Does

* Answers questions using **only retrieved context**
* Prevents spoilers by **never generating beyond the stored data**
* Supports multiple response styles:

  * **Trivia** — factual answers
  * **Summary** — concise explanations
  * **Fanfiction** — creative but context-safe stories
* Works as:

  * **Interactive Gradio Web App**
  *  **FastAPI backend for production or frontend integration**

---

## How It Works (End-to-End Workflow)

### Data Preparation

* Anime episode summaries, character bios, and trivia are stored as JSON files in:

  ```
  ./json_output/
  ```
---

### Embedding Generation (One-Time Step)

* Each document is converted into vector embeddings using:

  * **SentenceTransformers**
* Embeddings are stored in a **persistent ChromaDB vector database**:

  ```
  ./anime_vector_db/
  ```

This step is **required once** (or whenever data changes).

---

### Semantic Retrieval

When a user asks a question:

* The query is embedded
* **ChromaDB** finds the most relevant documents
* **RapidFuzz** fixes typos (e.g. *“Erin Jaeger” → “Eren Yeager”*)

---

### RAG Generation (Spoiler-Safe)

* Retrieved context is injected into a prompt
* **Google Gemini (Flash / Flash-Lite)** generates the response
* The model is instructed to **use only the provided context**

If the information doesn’t exist, it won’t be generated.
---

### Delivery

* **Gradio UI** for interactive use
* **FastAPI** for programmatic access (apps, websites, mobile)

---

## Project Structure

```
project/
├── generate_embeddings.py   # Run once to build vector DB
├── anime_vector_db/         # Persistent ChromaDB
├── json_output/             # Source data
├── app.py                   # Gradio UI
├── fast_api_app.py          # FastAPI backend
├── requirements.txt
├── Dockerfile
└── .env
└── images                   # has images for graio
└── config.py                #configurations
└──  txt to json             # converts summaries to json
```

---

## How to Run 

### Clone the Repository

```bash
git clone https://github.com/your-username/spoiler-controlled-tv-rag.git
cd spoiler-controlled-tv-rag
```
---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Set Environment Variables

Create a `.env` file:

```env
API_KEY=your_google_gemini_api_key
```

---

### Generate Embeddings

```bash
python generate_embeddings.py
```

**Must be done before running app or API**

---

## A: Run Gradio App (UI)

```bash
python app.py
```
Open:

```
http://127.0.0.1:7860
```

Best for:

* Demos
* Portfolio
* Interactive exploration

---

## B: Run FastAPI Backend

```bash
python fast_api_app.py
```

Open API docs:

```
http://127.0.0.1:8000/docs
```

Example API call:

```http
GET /ask?prompt=Who is Levi Ackerman?&mode=trivia
```

Best for:
* Frontend integration
* Mobile apps
* Production deployments

---

## Docker (Production)

```bash
docker build -t spoiler-rag .
docker run -p 8000:8000 spoiler-rag
```
Embeddings must already exist before building the image.

### There are some test queries in test_queries.json feel free to try them!!
---

## API vs Gradio — When to Use?

| Use Case           | Choose           |
| ------------------ | ---------------- |
| Demo / UI          | Gradio           |
| Production backend | FastAPI          |
| Web / mobile app   | FastAPI          |
| Portfolio showcase | Gradio           |
| Cloud deployment   | FastAPI + Docker |

Both use the **same RAG pipeline and vector database**.

---

## How to add to this project

* Add new shows -> drop JSON files into `json_output/`
* Re-run embeddings
* Add episode-level spoiler control
* Add authentication / rate limiting
* Swap ChromaDB for Qdrant
* Add streaming responses
* Connect Gradio to FastAPI instead of direct calls
* Add more test queries to test the responses
---

## What I learned
> **Embeddings are data.
> Apps are consumers.
> Spoilers don’t exist if the data doesn’t exist.**

---

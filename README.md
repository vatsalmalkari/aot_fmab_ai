---
title: Anime QA with RAG
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: true
short_description: a rag model that answers aot & fmab questions
---
# AOT & FMAB AI

Hi everyone! This is an **Anime QA system** built with **Retrieval-Augmented Generation (RAG)**.

You can ask any question about the anime **Attack on Titan (AOT)** or **Fullmetal Alchemist: Brotherhood (FMAB)**.

There are three main response modes:

- **Trivia** — Get quick factual answers about characters, episodes, or the anime world.  
- **Summary** — Receive summaries of episodes, characters, or overall plot.  
- **Fanfiction** — Generate short fictional stories set in the anime universe.

# What This Project Does

This project is a **question-answering system for anime fans**. It uses a **RAG approach** to provide accurate, context-aware answers based on curated anime content like:

- Character bios  
- Episode summaries  
- Trivia facts  

Instead of relying purely on AI generation, the system first **retrieves relevant facts** from a pre-built dataset and then uses **Google Gemini** to generate natural, informative responses.

## Key Features

- **Semantic Search** — Finds the most relevant content using embeddings.  
- **Generative Answers** — Uses Google Gemini to produce rich, natural language replies.  
- **Multiple Modes** — Supports trivia, summaries, and fanfiction.  
- **Custom Metadata Filtering** — Filter searches by anime, type (character/episode), and more.  
- **Fuzzy Matching** — Corrects typos and handles approximate queries intelligently.  

## Getting Started

1. Populate the database with your JSON anime data (episode summaries, character bios, etc.).  
2. Launch the interactive **Gradio app**.  
3. Choose your mode (trivia, summary, fanfiction) and ask your question.  
4. Get accurate and creative AI responses based on your dataset!  

This project combines **semantic retrieval** and **generative AI** to give fans an interactive, fun, and informative anime experience.

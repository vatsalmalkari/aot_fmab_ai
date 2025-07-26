import json
import os
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize embedding model
EMBEDDING_MODEL = "bge-small-en-v1.5"  
model = SentenceTransformer("BAAI/bge-small-en-v1.5", use_auth_token=False)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./aot_fmab_db")
collection = client.get_or_create_collection(name="anime_data")

def chunk_text(text, max_length=500):
    """Split long text into smaller chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def process_json_files(json_dir: str):
    """ JSON files embeddings"""
    files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    
    documents = []
    metadatas = []
    ids = []
    
    for file in files:
        with open(os.path.join(json_dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        
   
        metadata = {
            "name": data.get("name", ""),
            "anime": data["anime"],
            "type": data["type"],
            "source_file": data["metadata"]["source_file"]
        }
        
        # Add episode/character-specific fields
        if data["type"] == "episode":
            metadata.update({
                "season": data["season"],
                "episode": data["episode_number"],
                "title": data["title"]
            })
        
        # Split content into chunks 
        content_chunks = chunk_text(data["content"])
        
        for i, chunk in enumerate(content_chunks):
            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(f"{file}_{i}")
    
    # embeddings in batches
    batch_size = 32
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        embeddings = model.encode(batch_docs, convert_to_tensor=False).tolist()
        
        # Add to ChromaDB
        collection.add(
            documents=batch_docs,
            metadatas=batch_metadatas,
            ids=batch_ids,
            embeddings=embeddings
        )

if __name__ == "__main__":
    JSON_DIR = "./json_output"  
    process_json_files(JSON_DIR)
    print("Embeddings generated and stored in ChromaDB!")
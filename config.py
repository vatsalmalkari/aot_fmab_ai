from dotenv import load_dotenv
import os

load_dotenv()
# Config
API_KEY = os.getenv("API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./aot_fmab_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-small-en-v1.5")

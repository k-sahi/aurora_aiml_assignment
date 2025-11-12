# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Initialize the app
app = FastAPI(title="Aurora AI/ML Assignment", version="1.0")

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "messages.json"
MODEL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    messages = json.load(f)
if not isinstance(messages, list):
    raise ValueError("Expected json array")
log.info(f"Loaded {len(messages)} messages from {DATA_PATH}")

embedder = SentenceTransformer(MODEL_EMBED)
message_texts = [m["message"] for m in messages]
user_ids = [m["user_id"] for m in messages]
TOP_K = 5
 
embeddings = embedder.encode(message_texts, convert_to_numpy=True)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(np.array(embeddings, dtype=np.float32))
log.info(f"FAISS index build with {len(message_texts)} messages.")

class Query(BaseModel):
    question: str
    top_k: int = TOP_K

@app.post("/ask")
def ask_question(query: Query):
    if not index or not messages:
        return {"error": "No messages are indexed yet."}

    q_emb = embedder.encode([query.question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(np.array(q_emb, dtype=np.float32), query.top_k)

    try:
        for i in indices[0]:
            return {
                "answer": message_texts[i]
            }
    except Exception as e:
        return {"no revlevant messages found"}

# Run this app with: uvicorn app.main:app --reload
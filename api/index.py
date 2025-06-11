import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import aiohttp
import asyncio
import logging
import base64
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from mangum import Mangum  # 🚀 Vercel-compatible handler

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

DB_PATH = "knowledge_base.db"
API_KEY = os.getenv("API_KEY")
SIMILARITY_THRESHOLD = 0.68
MAX_RESULTS = 10
MAX_CONTEXT_CHUNKS = 4

# --- FastAPI App ---
app = FastAPI()
handler = Mangum(app)  # 👈 required for Vercel

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# --- Models ---
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkInfo(BaseModel):
    url: str
    text: str

# --- Helper Functions ---
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    if np.all(vec1 == 0) or np.all(vec2 == 0): return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

async def get_embedding(text):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")
    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }
    async with aiohttp.ClientSession() as session:
        async with session.post("https://aipipe.org/openai/v1/embeddings", headers=headers, json=payload) as res:
            if res.status == 200:
                return (await res.json())["data"][0]["embedding"]
            raise HTTPException(status_code=res.status, detail=await res.text())

async def search_knowledge_base(query, limit=2):
    conn = get_db_connection()
    cursor = conn.cursor()
    embedding = await get_embedding(query)
    results = []

    for table, url_field in [("discourse_chunks", "url"), ("markdown_chunks", "original_url")]:
        cursor.execute(f"SELECT * FROM {table} WHERE embedding IS NOT NULL")
        for row in cursor.fetchall():
            try:
                emb = json.loads(row["embedding"])
                sim = cosine_similarity(embedding, emb)
                if sim >= SIMILARITY_THRESHOLD:
                    results.append({
                        "source": row[url_field],
                        "content": row["content"],
                        "similarity": sim
                    })
            except:
                continue
    conn.close()
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]

# --- Main API Route ---
@app.post("/api/")
async def answer_question(request: QueryRequest):
    try:
        print("API_KEY =", API_KEY)
        top_chunks = await search_knowledge_base(request.question, limit=3)
        answer = f"Best guess based on knowledge base for: '{request.question}'"
        links = [{"url": c["source"], "text": c["source"]} for c in top_chunks]
        return {"answer": answer, "links": links}
    except Exception as e:
        logger.error(str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sqlite3
import pandas as pd
import numpy as np
import re
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
base_dir         = os.path.dirname(os.path.abspath(__file__))
db_path          = os.path.join(base_dir, "papers_warehouse.db")
embeddings_path  = os.path.join(base_dir, "bert_embeddings.pkl")
df_path          = os.path.join(base_dir, "papers_df.pkl")

# ─────────────────────────────────────────
# STEP 1 — Load DataFrame
# First time → loads from DB → saves pkl
# Next time  → loads from pkl (5 secs!)
# ─────────────────────────────────────────
if os.path.exists(df_path):
    print("📦 Loading saved DataFrame...")
    with open(df_path, "rb") as f:
        df = pickle.load(f)
    print(f"DataFrame loaded! {len(df)} papers!")
else:
    print("📦 Loading from warehouse DB...")
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM papers", conn)
    conn.close()
    with open(df_path, "wb") as f:
        pickle.dump(df, f)
    print(f"DataFrame saved → {df_path}")
    print(f"{len(df)} papers loaded!")

# ─────────────────────────────────────────
# STEP 2 — Load BERT Model
# ─────────────────────────────────────────
print("🤖 Loading BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ BERT model ready!")

# ─────────────────────────────────────────
# STEP 3 — Load OR Build Embeddings
# First time → builds on Kaggle GPU → saves pkl
# Next time  → loads pkl (30 secs!)
# ─────────────────────────────────────────
if os.path.exists(embeddings_path):
    print("📦 Loading saved BERT embeddings...")
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    print(f"Embeddings loaded! Shape: {embeddings.shape}")
else:
    print("Building BERT embeddings...")
    print("First time only — takes 5-10 mins on GPU!")
    embeddings = model.encode(
        df["clean_text"].fillna("").tolist(),
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings built + saved → {embeddings_path}")

print(f"🚀 Server ready! {len(df)} papers indexed!")

# ─────────────────────────────────────────
# QUERY CLEANING
# Same cleaning as preprocessing.py!
# ─────────────────────────────────────────
def clean_query(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words)

def clean_link(val):
    v = str(val).strip()
    if v in ["nan", "none", "None", "", "NaN"]:
        return ""
    return v
def clean_subjects(val):
    if not val:
        return ""
    
    s = str(val)

    # remove brackets and quotes
    s = re.sub(r"[\[\]']", "", s)

    # split + clean + join
    parts = [x.strip() for x in s.split(",") if x.strip()]
    
    return ", ".join(parts)
# ─────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

# ─────────────────────────────────────────
# ROUTE 1 — Home Page
# ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    html_path = os.path.join(base_dir, "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# ─────────────────────────────────────────
# ROUTE 2 — Recommend API
# ─────────────────────────────────────────
@app.post("/recommend")
def recommend(req: QueryRequest):
    query = req.query.strip()
    if not query:
        return []

    # Clean query same way as preprocessing!
    cleaned_query = clean_query(query)
    print(f"Original : {query}")
    print(f"Cleaned  : {cleaned_query}")

    if not cleaned_query:
        return []

    # Encode with BERT
    query_embedding = model.encode(
        [cleaned_query],
        convert_to_numpy=True
    )

    # Cosine similarity against all embeddings
    scores      = cosine_similarity(
                    query_embedding, 
                    embeddings
                  ).flatten()
    top_indices = scores.argsort()[::-1][:10]

    results = []
    for idx in top_indices:
        row   = df.iloc[idx]
        score = float(scores[idx])
        if score < 0.01:
            continue
        results.append({
            "title":           str(row.get("title",           "N/A")),
            "authors":         str(row.get("authors",         "N/A")),
            "category":        str(row.get("category",        "N/A")),
            "primary_subject": str(row.get("primary_subject", "N/A")),
            "subjects":        str(row.get("subjects",        "N/A")),
            "date":            str(row.get("date",            "N/A")),
            "description":     str(row.get("description",     "")),
            "link":            clean_link(row.get("link",           "")),
            "link_of_paper":   clean_link(row.get("link_of_paper",  "")),
            "link_of_pdf":     clean_link(row.get("link_of_pdf",    "")),
            "score":           round(score, 4)
        })
    return results

# ─────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8001,
        reload=False
    )
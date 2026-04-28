from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sqlite3
import pandas as pd
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

base_dir = os.path.dirname(os.path.abspath(__file__))
db_path  = os.path.join(base_dir, "papers_warehouse.db")

# ─────────────────────────────────────────
# STEP 1 — Load data
# ─────────────────────────────────────────
print("Loading data...")
conn = sqlite3.connect(db_path)
df   = pd.read_sql("SELECT * FROM papers", conn)
conn.close()
print(f"✅ {len(df)} papers loaded!")

# ─────────────────────────────────────────
# STEP 2 — Build TF-IDF
# ─────────────────────────────────────────
print("Building TF-IDF model...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["clean_text"].fillna(""))
print("✅ TF-IDF ready!")

# ─────────────────────────────────────────
# CLEAN FUNCTIONS
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
    if v.lower() in ["nan", "none", ""]:
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
# ROUTE 1 — Home
# ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    html_path = os.path.join(base_dir, "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# ─────────────────────────────────────────
# ROUTE 2 — Recommend (TF-IDF)
# ─────────────────────────────────────────
@app.post("/recommend")
def recommend(req: QueryRequest):
    query = req.query.strip()
    if not query:
        return []

    cleaned_query = clean_query(query)
    if not cleaned_query:
        return []

    query_vec = vectorizer.transform([cleaned_query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = scores.argsort()[::-1][:10]

    results = []
    for idx in top_indices:
        row   = df.iloc[idx]
        score = float(scores[idx])

        if score < 0.01:
            continue

        results.append({
            "title":           str(row.get("title", "N/A")),
            "authors":         str(row.get("authors", "N/A")),
            "category":        str(row.get("category", "N/A")),
            "primary_subject": str(row.get("primary_subject", "N/A")),
            "subjects": clean_subjects(row.get("subjects", "")),
            "date":            str(row.get("date", "N/A")),
            "description":     str(row.get("description", "")),
            "link":            clean_link(row.get("link", "")),
            "link_of_paper":   clean_link(row.get("link_of_paper", "")),
            "link_of_pdf":     clean_link(row.get("link_of_pdf", "")),
            "score":           round(score, 4)
        })

    return results

# ─────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=False)
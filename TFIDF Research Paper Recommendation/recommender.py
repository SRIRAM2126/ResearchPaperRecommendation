import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
def load_data(db_path="papers_warehouse.db"):
    print("Loading data...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM papers", conn)
    conn.close()
    print(f"✅ Loaded {len(df)} papers")
    return df

# ─────────────────────────────────────────
# BUILD TF-IDF
# ─────────────────────────────────────────
def build_tfidf(df):
    print("Building TF-IDF matrix (this may take time for 2.7L rows)...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(df["clean_text"].fillna(""))

    print("✅ TF-IDF matrix ready")
    return vectorizer, tfidf_matrix

# ─────────────────────────────────────────
# RECOMMEND FUNCTION
# ─────────────────────────────────────────
def recommend(query, df, vectorizer, tfidf_matrix, top_n=5):
    query_vec = vectorizer.transform([query.lower()])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = scores.argsort()[::-1][:top_n]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "title": row.get("title", "N/A"),
            "authors": row.get("authors", "N/A"),
            "category": row.get("category", "N/A"),
            "date": row.get("date", "N/A"),
            "link_of_paper": row.get("link_of_paper", ""),
            "link_of_pdf": row.get("link_of_pdf", ""),
            "score": round(float(scores[idx]), 4)
        })
    return results

# ─────────────────────────────────────────
# MAIN PROGRAM (CLI)
# ─────────────────────────────────────────
def main():
    df = load_data()
    vectorizer, tfidf_matrix = build_tfidf(df)

    print("\n=== TF-IDF Research Paper Recommender ===")
    print("Type a keyword or topic. Type 'exit' to quit.\n")

    while True:
        query = input("Enter topic: ").strip()

        if query.lower() == "exit":
            print("Exiting...")
            break

        if not query:
            continue

        results = recommend(query, df, vectorizer, tfidf_matrix)

        print(f"\nTop {len(results)} results for '{query}':\n")

        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}")
            print(f"   Authors  : {r['authors']}")
            print(f"   Category : {r['category']} | Date: {r['date']}")
            print(f"   Paper    : {r['link_of_paper']}")
            print(f"   PDF      : {r['link_of_pdf']}")
            print(f"   Score    : {r['score']}")
            print()

# ─────────────────────────────────────────
if __name__ == "__main__":
    main()
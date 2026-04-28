# PaperLens — Research Paper Recommender

PaperLens is a lightweight research discovery engine designed to recommend relevant academic papers from arXiv based on a user’s query.

It combines **TF-IDF vectorization** and **cosine similarity** to retrieve top-N matching papers, and optionally enhances results using **BERT-based semantic search**.

---

## 📌 Overview

PaperLens processes a **free-text query** (e.g., *"graph neural networks"*) and returns:

- Top-N ranked research papers
- Cosine similarity scores
- Direct links to arXiv and PDFs

The system is built using:
- **Scrapy** (data extraction)
- **Pandas & Regex** (preprocessing)
- **scikit-learn** (TF-IDF)
- **FastAPI** (API serving)

---

## 🚀 Features

- 🔍 Keyword-based paper search (TF-IDF)
- 🧠 Semantic search (BERT enhancement)
- ⚡ Fast response using precomputed vectors
- 📄 Rich metadata (authors, category, abstract, PDF links)
- 📊 Scalable to ~300K+ research papers

---

## 🏗️ Project Architecture
PaperLens
│
├── scraping/
│ └── Scrapy spiders (data extraction)

│
├── preprocessing.py
│ └── Cleaning, deduplication, feature building

│
├── recommender.py
│ └── TF-IDF + cosine similarity logic

│
├── app.py
│ └── FastAPI (TF-IDF based)

│
├── app_bert.py
│ └── BERT-based semantic search

│
├── database/
│ └── SQLite DB / CSV data

│
└── cache/
├── papers_df.pkl
└── bert_embeddings.pkl


---

## 📥 Data Extraction (Scrapy)

Data is scraped from **arXiv.org** using Scrapy spiders.

### Spider Design

- `CrawlSpider` with rules:
  - `/abs/\d{4}.\d+` → parse research paper pages
  - `/archive/`, `/list/`, `/year/` → navigation

### Extracted Fields

- Title
- Authors
- Abstract (description)
- Category & Subjects
- Publication Date
- arXiv Link
- PDF Link

---

## 🧹 Data Preprocessing

Handled in `preprocessing.py`.

### Steps:

#### 1. Text Cleaning
- Lowercasing
- Regex removal: `[^a-z0-9\s]`
- Normalize whitespace

#### 2. Deduplication
- Remove duplicate `(title, description)`
- Remove duplicate titles
- Drop empty rows

#### 3. Date Processing
- Convert to datetime
- Extract year & month
- Fill missing values with median

#### 4. Feature Engineering

Combine all text fields into one:
- clean_text = title + description + primary_subject+category + authors + subjects


This becomes the **input to TF-IDF**.

---

## 🔑 Core Model: TF-IDF

TF-IDF is used as the main recommendation engine.

### Formula

TF-IDF(t, d) = TF(t, d) × IDF(t)

- TF = term frequency in document
- IDF = inverse document frequency

### Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english'
)

tfidf_matrix = vectorizer.fit_transform(
    df['clean_text'].fillna('')
)

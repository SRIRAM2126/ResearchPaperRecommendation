import pandas as pd
import re
import os
import ast

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess():
    # Auto-detect the folder where THIS script lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path  = os.path.join(base_dir, "raw_dataset.csv")
    output_path = os.path.join(base_dir, "papers_clean.csv")

    print(f"Looking for file at: {input_path}")   # ← will print exact path
    
    df = pd.read_csv(input_path,encoding="utf-8")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    before = len(df)
    df.drop_duplicates(
        subset=["title",'description'],  # title + link combo
        inplace=True
    )
    # Also remove if description is exactly same
    # df.drop_duplicates(
    #     subset=["description"],
    #     keep="first",
    #     inplace=True
    # )
    after = len(df)
    print(f"Duplicates removed: {before - after}")
    print(f"Total rows: {len(df)}")

    

    def parse_list_col(val):
        try:
            result = ast.literal_eval(str(val))
            if isinstance(result, list):
                return [str(x).strip() for x in result]
            return []
        except:
            return []

    # Apply
    df["authors_list"]  = df["authors"].apply(parse_list_col)
    df["subjects_list"] = df["subjects"].apply(parse_list_col)

    # Flatten for clean text
    df["authors_clean"]  = df["authors_list"].apply(lambda x: " ".join(x))
    df["subjects_clean"] = df["subjects_list"].apply(lambda x: " ".join(x))

    df.dropna(how="all", inplace=True)
    # Updated clean_text with authors + subjects included
    df["clean_text"] = (
        df["title"].apply(clean_text) + " " +
        df["description"].apply(clean_text) + " " +
        df["primary_subject"].apply(clean_text) + " " +
        df["category"].apply(clean_text) + " " +
        df["authors_clean"].apply(clean_text) + " " +    # ← ADD THIS
        df["subjects_clean"].apply(clean_text)            # ← ADD THIS
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
# What we should also do
    df["year"]  = df["date"].dt.year    # extract year for filtering
    df["month"] = df["date"].dt.month   # extract month
    # Papers with no date → fill with median year
    df["year"].fillna(df["year"].median(), inplace=True)

    before = len(df)
    df.drop_duplicates(subset=["title"], inplace=True)
    print(f"Duplicates removed: {before - len(df)}")

    df.to_csv(output_path, index=False)
    print(f"✅ Saved → {output_path}")

if __name__ == "__main__":
    preprocess()
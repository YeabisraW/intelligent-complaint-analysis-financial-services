import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import pickle

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "notebooks/data/processed/filtered_complaints.csv"
VECTOR_STORE_DIR = "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# -----------------------------
# Load cleaned data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Expected columns (adjust if names differ)
TEXT_COL = "Consumer complaint narrative"
PRODUCT_COL = "Product"
ID_COL = "Complaint ID"

# -----------------------------
# Stratified sampling (10k–15k)
# -----------------------------
sample_size = 12000

df_sample, _ = train_test_split(
    df,
    train_size=sample_size,
    stratify=df[PRODUCT_COL],
    random_state=42
)

# -----------------------------
# Text chunking
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

texts = []
metadata = []

for _, row in df_sample.iterrows():
    chunks = text_splitter.split_text(str(row[TEXT_COL]))
    for chunk in chunks:
        texts.append(chunk)
        metadata.append({
            "complaint_id": row[ID_COL],
            "product": row[PRODUCT_COL]
        })

print(f"Total chunks created: {len(texts)}")

# -----------------------------
# Embedding model
# -----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True
)

# -----------------------------
# FAISS indexing
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# -----------------------------
# Persist vector store
# -----------------------------
faiss.write_index(index, f"{VECTOR_STORE_DIR}/faiss.index")

with open(f"{VECTOR_STORE_DIR}/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Vector store saved successfully.")

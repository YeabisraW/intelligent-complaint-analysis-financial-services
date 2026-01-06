import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import pickle

# -----------------------------
# Data Loading
# -----------------------------
def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    required_cols = ["Consumer complaint narrative", "Product", "Complaint ID"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

# -----------------------------
# Stratified Sampling
# -----------------------------
def sample_data(df: pd.DataFrame, stratify_col: str, sample_size: int, random_state: int = 42) -> pd.DataFrame:
    if sample_size > len(df):
        raise ValueError("Sample size cannot be larger than dataset length")
    
    df_sample, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[stratify_col],
        random_state=random_state
    )
    return df_sample

# -----------------------------
# Text Chunking
# -----------------------------
def chunk_texts(df: pd.DataFrame, text_col: str, id_col: str, product_col: str,
                chunk_size: int = 500, chunk_overlap: int = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    texts, metadata = [], []
    for _, row in df.iterrows():
        chunks = text_splitter.split_text(str(row[text_col]))
        for chunk in chunks:
            texts.append(chunk)
            metadata.append({
                "complaint_id": row[id_col],
                "product": row[product_col]
            })
    return texts, metadata

# -----------------------------
# Embedding Generation
# -----------------------------
def generate_embeddings(texts: list, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                        batch_size: int = 64):
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True
    )
    return np.array(embeddings).astype("float32")

# -----------------------------
# FAISS Index
# -----------------------------
def build_faiss_index(embeddings: np.ndarray):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# -----------------------------
# Save Vector Store
# -----------------------------
def save_vector_store(index, metadata: list, vector_dir: str = "vector_store"):
    os.makedirs(vector_dir, exist_ok=True)

    faiss_index_path = os.path.join(vector_dir, "faiss.index")
    metadata_path = os.path.join(vector_dir, "metadata.pkl")

    try:
        faiss.write_index(index, faiss_index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
    except Exception as e:
        raise IOError(f"Failed to save vector store: {e}")

    print(f"✅ Vector store saved successfully at '{vector_dir}'")

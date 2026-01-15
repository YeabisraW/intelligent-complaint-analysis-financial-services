import os
import pickle
import numpy as np
import faiss
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Initialize Local LLM
# -----------------------------
# Note: Ensure you have run 'ollama pull llama3.2:1b' in your terminal
llm = ChatOllama(
    model="llama3.2:1b", 
    temperature=0
)

# Keep the same embedding model as Task 2
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Paths
VECTOR_STORE_DIR = "vector_store"
METADATA_FILE = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")
FAISS_INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss.index")

# -----------------------------
# 2. Vector Store Loader
# -----------------------------
def load_vector_store():
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("Vector store missing. Run Task 2 first.")
    
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# -----------------------------
# 3. Retriever Logic
# -----------------------------
def retrieve_top_k(question, index, metadata, k=5):
    q_vec = embedding_model.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_vec, k)
    
    retrieved_chunks = []
    for idx in indices[0]:
        if idx < len(metadata):
            retrieved_chunks.append(metadata[idx])
    return retrieved_chunks

# -----------------------------
# 4. Prompt Construction
# -----------------------------
def build_prompt(question, retrieved_chunks):
    context_list = []
    for i, chunk in enumerate(retrieved_chunks):
        content = chunk.get('text', 'No content found')
        context_list.append(f"Complaint {i+1}: {content}")
    
    context_text = "\n\n".join(context_list)
    
    # This "Instruction" part is the key to stopping the "I don't know" answers
    prompt = f"""
    You are a Senior Customer Insights Analyst at CrediTrust. 
    Analyze the customer complaints provided below to answer the user's question.
    
    If the complaints contain frustrations, declines, or fees, summarize them as 'Key Issues'.
    Even if the text sounds like a 'rant', identify the root cause of the customer's unhappiness.

    Question: {question}

    Customer Complaints:
    {context_text}

    Analysis:
    """
    return prompt

# -----------------------------
# 5. Generate Answer
# -----------------------------
# You'll need your original dataframe (df) for this
# 1. Update the function definition (Remove 'df')
def generate_answer(question, index, metadata):
    retrieved = retrieve_top_k(question, index, metadata)
    prompt = build_prompt(question, retrieved)
    # This call to Ollama is now fueled by the text in the metadata
    response = llm.invoke(prompt).content
    return response, retrieved

# 2. Update the execution block at the bottom
if __name__ == "__main__":
    try:
        index, metadata = load_vector_store()
        query = "Why are people unhappy with Credit Cards?"
        
        # Call the updated function without 'df'
        answer, sources = generate_answer(query, index, metadata)
        
        print(f"\n--- CREDYTRUST ANALYSIS ---\n{answer}")
        
    except Exception as e:
        print(f"Error: {e}")
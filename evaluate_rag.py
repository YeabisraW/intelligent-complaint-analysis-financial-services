import os
import faiss
import pickle
from rag_logic import CrediTrustPipeline

def run_evaluation():
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "vector_store", "faiss.index")
    meta_path = os.path.join(base_dir, "vector_store", "metadata.pkl")

    # 2. Check if files exist
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print("❌ Error: Could not find 'faiss.index' or 'metadata.pkl' in vector_store folder.")
        return

    # 3. Load Assets
    print("📂 Loading Vector Store...")
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    # 4. Initialize Pipeline
    pipeline = CrediTrustPipeline(index, metadata)

    # 5. Representative Test Questions (Reviewer Request: 5-10 questions)
    test_queries = [
        "What are the main issues with Credit Card late fees?",
        "How do customers describe their experience with debt collectors?",
        "Are there recurring complaints about unauthorized bank transfers?",
        "What are the top frustrations with mortgage loan servicing?",
        "Summarize complaints related to identity theft and accounts."
    ]

    print(f"🚀 Starting Evaluation of {len(test_queries)} queries...\n")
    
    evaluation_logs = []

    for i, q in enumerate(test_queries):
        print(f"Testing Query {i+1}: {q}")
        try:
            answer, sources = pipeline.run(q, k=3)
            
            result_entry = f"QUERY: {q}\nANSWER: {answer}\nSOURCES: {len(sources)} docs used.\n{'-'*50}\n"
            evaluation_logs.append(result_entry)
            
            print(f"✅ Success. (Sources found: {len(sources)})")
        except Exception as e:
            print(f"❌ Failed: {e}")

    # 6. Save Results for Reviewer
    log_file = "rag_evaluation_report.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.writelines(evaluation_logs)
    
    print(f"\n✨ Evaluation Complete! Results saved to: {log_file}")

if __name__ == "__main__":
    run_evaluation()
import ollama
from sentence_transformers import SentenceTransformer

class CrediTrustPipeline:
    """
    A RAG Pipeline class for analyzing financial complaints.
    Encapsulates Retrieval, Prompt Construction, and Generation.
    """
    
    def __init__(self, index, metadata):
        """
        Initialize the pipeline with a FAISS index and metadata list.
        """
        self.index = index
        self.metadata = metadata
        # Using a standard lightweight embedding model
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve(self, query, k=5):
        """
        Retrieves the top k most relevant complaints based on semantic similarity.
        """
        query_embedding = self.embed_model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        
        # Filter out any invalid indices and return metadata
        return [self.metadata[i] for i in indices[0] if i < len(self.metadata)]

    def generate(self, query, context_docs):
        """
        Constructs a prompt and generates an answer using the local Llama 3.2 model.
        """
        # Combine retrieved texts into one context string
        context_text = "\n\n".join([doc.get('text', '') for doc in context_docs])
        
        prompt = f"""
        You are a financial analyst at CrediTrust. Use the following customer complaints to answer the question.
        If the answer isn't in the complaints, say you don't have enough information.
        
        Complaints:
        {context_text}
        
        Question: {query}
        
        Instruction: Provide a concise summary and list the main pain points.
        """
        
        response = ollama.generate(model='llama3.2:1b', prompt=prompt)
        return response['response']

    def run(self, query, k=5):
        """
        The main entry point to run the full RAG process.
        """
        sources = self.retrieve(query, k=k)
        answer = self.generate(query, sources)
        return answer, sources
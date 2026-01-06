from graphviz import Digraph

# Figure 3: Complaint narrative chunking
dot1 = Digraph(comment='Complaint Narrative Chunking Process')
dot1.node('A', 'Raw Complaint Narrative')
dot1.node('B', 'Text Cleaning')
dot1.node('C', 'Chunking into Segments')
dot1.node('D', 'Ready for Embedding')
dot1.edges(['AB', 'BC', 'CD'])
dot1.render('figure3_chunking_process', format='png', cleanup=True)

# Figure 4: Embedding generation and vector storage
dot2 = Digraph(comment='Embedding Generation Pipeline')
dot2.node('A', 'Chunks from Figure 3')
dot2.node('B', 'Embedding Model (SentenceTransformer)')
dot2.node('C', 'Vector Store (FAISS / Pinecone)')
dot2.edges(['AB', 'BC'])
dot2.render('figure4_embedding_pipeline', format='png', cleanup=True)

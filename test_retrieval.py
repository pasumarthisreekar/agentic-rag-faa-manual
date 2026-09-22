# test_retrieval.py
import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("Loading local embedding model and Qdrant index...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
client = qdrant_client.QdrantClient(path="qdrant_data")

vector_store = QdrantVectorStore(
    client=client, 
    collection_name="faa_maintenance_hybrid", 
    enable_hybrid=True
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

# Test the exact query that failed in Test 1
query = "What are the standard tie-down procedures for securing a heavy aircraft?"
print(f"\nRunning test query: '{query}'\n")

retriever = index.as_retriever(
    similarity_top_k=5, 
    sparse_top_k=5, 
    vector_store_query_mode="hybrid"
)
nodes = retriever.retrieve(query)

print(f"==============================================")
print(f"Retrieved {len(nodes)} raw nodes from Qdrant:")
print(f"==============================================\n")

for i, node in enumerate(nodes, 1):
    print(f"--- NODE {i} (Relevance Score: {node.score}) ---")
    print(node.node.text + "...\n")  # Prints the first 400 characters of each chunk
import os


os.environ["FASTMCP_LOG_LEVEL"] = "ERROR"
os.environ["PREFECT_LOGGING_LEVEL"] = "ERROR"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastmcp import FastMCP
import threading

# 1. Initialize FastMCP instantly
mcp = FastMCP(name="FAA Technical Manual Server")

# Global cache and thread synchronization lock
_index = None
_index_lock = threading.Lock()

def get_index():
    """Lazily loads libraries, the embedding model, and Qdrant index safely using a thread lock."""
    global _index
    if _index is None:
        with _index_lock:
            # Double-checked locking pattern to prevent concurrent race conditions
            if _index is None:
                import torch
                torch.set_num_threads(1)  # Prevents Windows multi-threading deadlocks
                
                import qdrant_client
                from llama_index.core import VectorStoreIndex, StorageContext, Settings
                from llama_index.vector_stores.qdrant import QdrantVectorStore
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
                client = qdrant_client.QdrantClient(path="qdrant_data")
                vector_store = QdrantVectorStore(
                    client=client, 
                    collection_name="faa_maintenance_hybrid", 
                    enable_hybrid=True
                )
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                _index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    return _index

# 2. Define tools using the thread-safe lazy loader
@mcp.tool
def vector_search(query: str) -> str:
    """Searches the technical manual for general concepts, maintenance procedures, and system overviews."""
    
    with open("mcp_debug.log", "a") as f:
        f.write(f"\n[TOOL CALLED] vector_search invoked with query: '{query}'\n")

    try:
        retriever = get_index().as_retriever(similarity_top_k=10, sparse_top_k=10, vector_store_query_mode="hybrid")
        nodes = retriever.retrieve(query)
        result = "\n\n".join([n.node.text for n in nodes])
        
        with open("mcp_debug.log", "a") as f:
            f.write(f"[TOOL SUCCESS] Retrieved {len(nodes)} nodes. First node snippet: {nodes[0].node.text[:100]}\n")
            
        return result
    except Exception as e:
        with open("mcp_debug.log", "a") as f:
            f.write(f"[TOOL ERROR] Exception occurred: {str(e)}\n")
        raise e

@mcp.tool
def exact_keyword_search(part_number: str) -> str:
    """Use for exact keyword matching. Critical for searching specific alphanumeric part numbers, error codes, or torque specifications."""
    retriever = get_index().as_retriever(sparse_top_k=10, vector_store_query_mode="sparse")
    nodes = retriever.retrieve(part_number)
    return "\n\n".join([n.node.text for n in nodes])

@mcp.tool
def get_safety_warnings(component: str) -> str:
    """Retrieves ONLY the 'WARNING', 'CAUTION', or 'HAZARD' boxes associated with a specific vehicle or aircraft part."""
    retriever = get_index().as_retriever(similarity_top_k=10, sparse_top_k=10, vector_store_query_mode="hybrid")
    prompt = f"SAFETY WARNINGS CAUTIONS HAZARDS {component}"
    nodes = retriever.retrieve(prompt)
    return "\n\n".join([n.node.text for n in nodes])

if __name__ == "__main__":
    print("Pre-loading Qdrant index and embedding model into memory...")
    get_index()  # Forces synchronous initialization before server accepts requests
    print("Server ready. Starting FastMCP SSE transport on port 8000...")
    
    mcp.run(transport="sse", port=8000)
import os
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client

# -- NEW DOCLING IMPORTS --
from llama_index.readers.docling import DoclingReader
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

def ingest_faa_manual_hybrid(pdf_path: str, collection_name: str):
    print("Step 1: Parsing FAA PDF with Docling (PyPdfium2 Backend & OCR Disabled)...")
    
    # 1. Turn off OCR and prevent memory hoarding
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        generate_parsed_pages=False 
    )
    
    # 2. Inject the custom pipeline and the memory-safe PyPdfium2 backend
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend # <-- NEW LINE: Switches to the memory-safe backend
            )
        }
    )
    
    # 3. Pass the custom converter to LlamaIndex
    reader = DoclingReader(doc_converter=doc_converter)
    documents = reader.load_data(file_path=pdf_path)
    
    print("Step 2: Markdown Hierarchical Chunking...")
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Extracted {len(nodes)} logically grouped nodes.")
    
    print("Step 3: Initializing local Qdrant with HYBRID Search Enabled...")
    client = qdrant_client.QdrantClient(path="qdrant_data")
    
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=collection_name, 
        enable_hybrid=True 
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Step 4: Loading Local HuggingFace Embedding Model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Step 5: Generating Dense & Sparse Embeddings and Building Index...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    print(f"\nSuccess! Hybrid Index built and stored locally in /qdrant_data.")
    return index

if __name__ == "__main__":
    ingest_faa_manual_hybrid(
        pdf_path="data/raw/faa_maintenance_handbook.pdf",
        collection_name="faa_maintenance_hybrid"
    )
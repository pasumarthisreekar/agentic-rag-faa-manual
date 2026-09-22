# Agentic RAG System for FAA Technical Manual Q&A

A multi-tool agentic RAG system that answers FAA aircraft maintenance questions strictly grounded in an ingested technical manual. Built with LlamaIndex and Gemini, with retrieval tools exposed to the agent over the Model Context Protocol (MCP).

## What it does

- Ingests and hierarchically chunks a large FAA maintenance handbook PDF using **Docling**
- Indexes content into **Qdrant** with hybrid dense + sparse search using local HuggingFace embeddings
- Exposes three specialized retrieval tools over a **FastMCP** server: general hybrid vector search, exact keyword/part-number search, and a dedicated safety-warning retriever
- Runs a **LlamaIndex FunctionAgent** (Gemini) that dynamically selects between tools based on query type
- Enforces strict grounding via system-prompt design — the agent explicitly declines to answer when retrieval returns no relevant content, rather than hallucinating a procedure

## Results

Evaluated with a **RAGAS** pipeline that captures live tool-call outputs as retrieved context, scored against a held-out QA set with ground-truth answers:

| Metric | Score |
|---|---|
| Faithfulness | 0.91 |
| Answer Relevancy | 0.80 |
| Context Precision | ~1.00 |
| Context Recall | ~1.00 |

## Architecture

```
FAA Manual (PDF)
      │
      ▼
  Docling (parsing + chunking)
      │
      ▼
  Qdrant (hybrid dense + sparse index)
      │
      ▼
  FastMCP Server ── exposes 3 tools ──▶ LlamaIndex FunctionAgent (Gemini)
                                              │
                                              ▼
                                        Grounded answer
```

## Tech stack

Python, LlamaIndex, FastMCP, Qdrant, Docling, Gemini API, RAGAS, HuggingFace embeddings

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your own Gemini API key:
   ```bash
   cp .env.example .env
   ```
3. Place your source PDF at `data/raw/faa_maintenance_handbook.pdf` and run the ingestion pipeline:
   ```bash
   python ingest.py
   ```
4. In one terminal, start the MCP server:
   ```bash
   python mcp_server.py
   ```
5. In another terminal, run the agent:
   ```bash
   python agent.py
   ```

## Files

- `ingest.py` — parses and indexes the FAA manual into Qdrant
- `mcp_server.py` — FastMCP server exposing the three retrieval tools
- `agent.py` — LlamaIndex agent that connects to the MCP server and answers queries
- `eval_ragas.py` — RAGAS evaluation pipeline
- `test_retrieval.py` — standalone script for sanity-checking retrieval quality

## Notes

This project uses a publicly available FAA maintenance handbook as its knowledge source. No proprietary or sensitive data is included.

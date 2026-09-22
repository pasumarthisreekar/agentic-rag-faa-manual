import os
import asyncio
import pandas as pd
from datasets import Dataset

from dotenv import load_dotenv

load_dotenv()  # Loads variables from a local .env file (not committed to git)

# LlamaIndex Imports
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.tools import FunctionTool

# Ragas & LangChain Imports
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Helper to spy on and capture tool outputs
class ContextCollector:
    def __init__(self):
        self.contexts = []

    def add(self, text):
        self.contexts.append(str(text))

    def clear(self):
        self.contexts = []

async def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Please set your GOOGLE_API_KEY environment variable (see .env.example).")
        exit(1)

    print("Step 1: Connecting to Local FastMCP Server...")
    mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
    mcp_tool_spec = McpToolSpec(client=mcp_client)
    original_tools = await mcp_tool_spec.to_tool_list_async()

    # ---------------------------------------------------------
    # CONTEXT CAPTURE: Creating New Wrapped Tools
    # ---------------------------------------------------------
    collector = ContextCollector()
    wrapped_tools = []

    for tool in original_tools:
        def create_wrapped_tool(t):
            orig_async = t.async_fn
            orig_sync = t.fn

            async def wrapped_async(*args, **kwargs):
                result = await orig_async(*args, **kwargs)
                collector.add(result)
                return result

            def wrapped_sync(*args, **kwargs):
                result = orig_sync(*args, **kwargs)
                collector.add(result)
                return result

            return FunctionTool.from_defaults(
                fn=wrapped_sync,
                async_fn=wrapped_async,
                tool_metadata=t.metadata
            )

        wrapped_tools.append(create_wrapped_tool(tool))

    print("Step 2: Initializing Gemini LLM...")
    llm = GoogleGenAI(model="gemini-3.5-flash-lite", temperature=0.0)

    agent = FunctionAgent(
        tools=wrapped_tools,
        llm=llm,
        system_prompt="""You are an expert FAA technical maintenance assistant. 
1. You MUST use your provided tools to search the FAA manual before answering.
2. If the tools return no relevant results, you MUST state: 'I cannot find that specific procedure in the authorized FAA technical manual.' 
3. NEVER guess, NEVER provide generic procedures, and NEVER provide information not explicitly found in the manual. 
4. Always highlight SAFETY WARNINGS in bold."""
    )

    # ---------------------------------------------------------
    # SYNTHETIC DATASET (Ground Truths extracted from FAA Manual)
    # ---------------------------------------------------------
    evaluation_set = [
        {
            "query": "What are the three elements required for a fire to occur?",
            "ground_truth": "The three elements required for a fire to occur are fuel, heat, and oxygen."
        },
        {
            "query": "What types of materials are involved in a Class A fire?",
            "ground_truth": "Class A fires involve ordinary combustible materials, such as wood, cloth, paper, and upholstery materials."
        },
        {
            "query": "What types of substances fuel a Class B fire?",
            "ground_truth": "Class B fires involve flammable petroleum products, combustible liquids, greases, solvents, and paints."
        },
        {
            "query": "What defines a Class C fire?",
            "ground_truth": "A Class C fire involves energized electrical wiring and equipment."
        }
    ]

    data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": []
    }

    print("\n==============================================")
    print("Executing Agent & Collecting Contexts")
    print("==============================================\n")

    for i, test in enumerate(evaluation_set, 1):
        query = test["query"]
        print(f"Running Query {i}/{len(evaluation_set)}: {query}")

        collector.clear()
        response = await agent.run(user_msg=query)

        data["user_input"].append(query)
        data["response"].append(str(response))
        data["retrieved_contexts"].append(collector.contexts.copy())
        data["reference"].append(test["ground_truth"])

        await asyncio.sleep(2)

    print("\nStep 3: Formatting HuggingFace Dataset...")
    dataset = Dataset.from_dict(data)

    print("Step 4: Initializing Ragas Evaluator LLM...")
    eval_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)
    eval_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_emb = LangchainEmbeddingsWrapper(eval_embeddings)

    print("Step 5: Executing Ragas Evaluation Matrix...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall()
        ],
        llm=ragas_llm,
        embeddings=ragas_emb
    )

    print("\n==============================================")
    print("Final Ragas Evaluation Scores:")
    print("==============================================")
    df = result.to_pandas()

    if df.empty or "user_input" not in df.columns:
        print("Error: DataFrame is empty. The Google API jobs failed.")
        print(df)
    else:
        print(df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])
        df.to_csv("ragas_evaluation_report.csv", index=False)
        print("\nReport saved to 'ragas_evaluation_report.csv'. Project Finished!")

if __name__ == "__main__":
    asyncio.run(main())
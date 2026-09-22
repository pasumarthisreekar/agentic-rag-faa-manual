import os
import asyncio

from dotenv import load_dotenv

load_dotenv()  # Loads variables from a local .env file (not committed to git)

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

async def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Please set your GOOGLE_API_KEY environment variable (see .env.example).")
        exit(1)

    print("Step 1: Connecting to Local FastMCP Server via HTTP/SSE...")

    mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
    mcp_tool_spec = McpToolSpec(client=mcp_client)
    tools = await mcp_tool_spec.to_tool_list_async()
    print(f"Successfully loaded {len(tools)} tools from MCP server.")

    print("Step 2: Initializing Gemini LLM...")
    llm = GoogleGenAI(
        model="gemini-3.6-flash",
        temperature=0.1,
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    print("Step 3: Initializing Modern LlamaIndex FunctionAgent...")
    agent = FunctionAgent(
        tools=tools,
        llm=llm,
        system_prompt="""You are an expert FAA technical maintenance assistant. 
1. You MUST use your provided tools to search the FAA manual before answering.
2. If the tools return no relevant results, you MUST state: 'I cannot find that specific procedure in the authorized FAA technical manual.' 
3. NEVER guess, NEVER provide generic procedures, and NEVER provide information not explicitly found in the manual. 
4. Always highlight SAFETY WARNINGS in bold."""
    )

    print("\n==============================================")
    print("Testing the Modern MCP-Powered Agent Workflow")
    print("==============================================\n")

    test_queries = [
        "What are the standard tie-down procedures for securing a heavy aircraft?",
        "What is the standard position for a taxi signalman when guiding an aircraft?",
        "What is the exact torque specification for the alternator on a Cessna 140?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n=================== TEST {i} ===================")
        print(f"User Query: {query}\n")

        print(f"DEBUG: Loaded tools -> {[t.metadata.name for t in agent.tools]}")
        response = await agent.run(user_msg=query)

        print("\nFinal Synthesized Answer:")
        print("----------------------------------------------")
        print(str(response))
        print("==============================================\n")

if __name__ == "__main__":
    asyncio.run(main())
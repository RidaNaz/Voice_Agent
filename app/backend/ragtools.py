import re
from typing import Any
import google.generativeai as genai
from pinecone import Pinecone
from rtmt import RTMiddleTier, Tool, ToolResult, ToolResultDirection

_search_tool_schema = {
    "type": "function",
    "name": "search",
    "description": "Search the knowledge base. The knowledge base is in English, translate to and from English if " + \
                   "needed. Results are formatted as a source name first in square brackets, followed by the text " + \
                   "content, and a line with '-----' at the end of each result.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

_grounding_tool_schema = {
    "type": "function",
    "name": "report_grounding",
    "description": "Report use of a source from the knowledge base as part of an answer (effectively, cite the source). Sources " + \
                   "appear in square brackets before each knowledge base passage. Always use this tool to cite sources when responding " + \
                   "with information from the knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of source names from last statement actually used, do not include the ones not used to formulate a response"
            }
        },
        "required": ["sources"],
        "additionalProperties": False
    }
}

async def _search_tool(
    pinecone_index,
    genai_model,
    args: Any) -> ToolResult:
    print(f"Searching for '{args['query']}' in the knowledge base.")
    
    # Generate embedding for the query using Gemini
    embedding = genai_model.embed_content(
        content=args["query"],
        task_type="retrieval_query"
    )
    
    # Query Pinecone
    results = pinecone_index.query(
        vector=embedding["embedding"],
        top_k=5,
        include_metadata=True
    )
    
    result = ""
    for match in results.matches:
        result += f"[{match.id}]: {match.metadata['content']}\n-----\n"
    
    return ToolResult(result, ToolResultDirection.TO_SERVER)

KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_=\-]+$')

# TODO: move from sending all chunks used for grounding eagerly to only sending links to 
# the original content in storage, it'll be more efficient overall
async def _report_grounding_tool(pinecone_index, args: Any) -> None:
    sources = [s for s in args["sources"] if KEY_PATTERN.match(s)]
    list = " OR ".join(sources)
    print(f"Grounding source: {list}")
    # Use search instead of filter to align with how detailt integrated vectorization indexes
    # are generated, where chunk_id is searchable with a keyword tokenizer, not filterable 
    results = pinecone_index.query(
        vector=list,
        top_k=len(sources),
        include_metadata=True
    )
    
    # If your index has a key field that's filterable but not searchable and with the keyword analyzer, you can 
    # use a filter instead (and you can remove the regex check above, just ensure you escape single quotes)
    # search_results = await search_client.search(filter=f"search.in(chunk_id, '{list}')", select=["chunk_id", "title", "chunk"])

    docs = []
    for match in results.matches:
        docs.append({"chunk_id": match.id, "title": match.metadata['title'], "chunk": match.metadata['content']})
    return ToolResult({"sources": docs}, ToolResultDirection.TO_CLIENT)

def attach_rag_tools(rtmt: RTMiddleTier,
    pinecone_api_key: str,
    pinecone_environment: str,
    pinecone_index_name: str,
    gemini_api_key: str
    ) -> None:
    
    # Initialize Pinecone with new syntax
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    
    # Initialize Gemini
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    embedding_model = genai.GenerativeModel('embedding-001')

    rtmt.tools["search"] = Tool(
        schema=_search_tool_schema, 
        target=lambda args: _search_tool(index, embedding_model, args)
    )
    rtmt.tools["report_grounding"] = Tool(
        schema=_grounding_tool_schema, 
        target=lambda args: _report_grounding_tool(index, args)
    )
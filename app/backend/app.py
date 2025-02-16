import logging
import os
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv
import google.generativeai as genai

from ragtools import attach_rag_tools
from rtmt import RTMiddleTier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

class GeminiRTMiddleTier(RTMiddleTier):
    def __init__(self, model, voice_choice=None):
        self.model = model
        self.voice_choice = voice_choice
        self.tools = {}
        if voice_choice is not None:
            logger.info("Realtime voice choice set to %s", voice_choice)

async def create_app():
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        load_dotenv()

    app = web.Application()

    # Initialize Gemini
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-pro')

    rtmt = GeminiRTMiddleTier(
        model=model,
        voice_choice=os.environ.get("VOICE_CHOICE", "alloy")
    )
    
    rtmt.system_message = """
        You are a helpful assistant. Only answer questions based on information you searched in the knowledge base, accessible with the 'search' tool. 
        The user is listening to answers with audio, so it's *super* important that answers are as short as possible, a single sentence if at all possible. 
        Never read file names or source names or keys out loud. 
        Always use the following step-by-step instructions to respond: 
        1. Always use the 'search' tool to check the knowledge base before answering a question. 
        2. Always use the 'report_grounding' tool to report the source of information from the knowledge base. 
        3. Produce an answer that's as short as possible. If the answer isn't in the knowledge base, say you don't know.
    """.strip()

    # Attach RAG tools with Pinecone configuration
    attach_rag_tools(rtmt,
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
        pinecone_environment=os.environ["PINECONE_ENVIRONMENT"],
        pinecone_index_name=os.environ["PINECONE_INDEX_NAME"],
        gemini_api_key=os.environ["GEMINI_API_KEY"]
    )

    # Set up routes
    current_directory = Path(__file__).parent
    app.add_routes([web.get('/', lambda _: web.FileResponse(current_directory / 'static/index.html'))])
    app.router.add_static('/', path=current_directory / 'static', name='static')
    
    # Attach the websocket handler
    rtmt.attach_to_app(app, "/realtime")
    
    return app

if __name__ == "__main__":
    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", 8765))
    web.run_app(create_app(), host=host, port=port)
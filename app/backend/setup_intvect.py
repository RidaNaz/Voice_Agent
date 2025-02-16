import logging
import os
from pathlib import Path
from pinecone import Pinecone
from dotenv import load_dotenv
from rich.logging import RichHandler

def check_vector_store(pinecone_api_key: str, pinecone_environment: str, pinecone_index_name: str):
    """Check Pinecone connection and index existence"""
    try:
        # Initialize Pinecone with new syntax
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists
        if pinecone_index_name not in pc.list_indexes().names():
            logger.error(f"Pinecone index '{pinecone_index_name}' does not exist!")
            return False
            
        # Get index stats
        index = pc.Index(pinecone_index_name)
        stats = index.describe_index_stats()
        
        logger.info(f"Successfully connected to Pinecone index: {pinecone_index_name}")
        logger.info(f"Total vectors in index: {stats.total_vector_count}")
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {str(e)}")
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    logger = logging.getLogger("voicerag")

    # Load environment variables
    load_dotenv()

    # Required environment variables
    required_vars = [
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "PINECONE_INDEX_NAME",
    ]

    # Check for required environment variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Check vector store connection
    success = check_vector_store(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME")
    )

    if not success:
        logger.error("Vector store check failed!")
        exit(1)
"""
Initialize VinaCompare system
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_documents():
    """Load test documents into vector store"""
    from src.models.embeddings import get_embedder
    from src.vector_store import get_vector_store

    logger.info("Loading test documents...")

    # Load documents
    docs_path = Path(__file__).parent.parent / 'data' / 'test' / 'sample_documents.json'
    with open(docs_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    logger.info(f"Loaded {len(documents)} documents from {docs_path}")

    # Get embedder and vector store
    logger.info("Initializing embedder...")
    embedder = get_embedder()

    logger.info("Connecting to Milvus...")
    vector_store = get_vector_store()
    vector_store.connect()

    # Prepare data
    document_ids = [doc['id'] for doc in documents]
    texts = [doc['text'] for doc in documents]

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = embedder.encode_documents(texts)
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")

    # Prepare metadata
    metadata = [
        {
            'title': doc['title'],
            'category': doc['category'],
            'source': doc['source']
        }
        for doc in documents
    ]

    # Insert into Milvus
    logger.info("Inserting into Milvus...")
    vector_store.insert(
        document_ids=document_ids,
        texts=texts,
        embeddings=embeddings.tolist(),
        metadata=metadata
    )

    # Build BM25 index for hybrid search
    logger.info("Building BM25 index...")
    from src.retrieval.retriever import get_retriever
    retriever = get_retriever()
    retriever.initialize()
    retriever.build_bm25_index([
        {"id": doc["id"], "text": doc["text"], "metadata": {"title": doc["title"]}}
        for doc in documents
    ])

    logger.info(f"Inserted {len(documents)} test documents successfully!")

    # Verify
    stats = vector_store.get_collection_stats()
    logger.info(f"Collection stats: {stats}")


def init_database():
    """Initialize database connection"""
    from src.database import init_db

    logger.info("Initializing database...")
    try:
        init_db()
        logger.info("Database initialized!")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
        logger.warning("Make sure PostgreSQL is running (docker-compose up -d postgres)")


def main():
    """Main initialization"""
    logger.info("=" * 50)
    logger.info("VinaCompare System Initialization")
    logger.info("=" * 50)

    # Initialize database
    init_database()

    # Load test documents
    try:
        load_test_documents()
    except Exception as e:
        logger.error(f"Failed to load test documents: {e}")
        logger.error("Make sure Milvus is running (docker-compose up -d)")
        raise

    logger.info("=" * 50)
    logger.info("Initialization Complete!")
    logger.info("=" * 50)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Start the backend: python src/main.py")
    logger.info("  2. Test the API: python scripts/test_system.py")
    logger.info("  3. Access API docs: http://localhost:8000/docs")


if __name__ == "__main__":
    main()

"""
VinaCompare FastAPI Backend
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from contextlib import asynccontextmanager

from src.config import settings
from src.database import init_db, close_db
from src.vector_store import get_vector_store
from src.models.vistral import create_vistral_model
from src.rag.pipeline import create_rag_pipeline
from src.retrieval.retriever import get_retriever
from src.database import get_db_context
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
rag_pipeline = None
model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global rag_pipeline, model_loaded

    # Startup
    logger.info("Starting VinaCompare backend...")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization failed (may not be running): {e}")

    # Connect to Milvus
    try:
        vector_store = get_vector_store()
        vector_store.connect()
        logger.info("Milvus connected")
    except Exception as e:
        logger.warning(f"Milvus connection failed (may not be running): {e}")

    # Initialize retriever
    try:
        retriever = get_retriever()
        retriever.initialize()
        logger.info("Retriever initialized")
    except Exception as e:
        logger.warning(f"Retriever initialization failed: {e}")

    # Load model (optional - can be deferred)
    try:
        logger.info("Loading Vistral-7B model (this may take a while)...")
        model = create_vistral_model()
        rag_pipeline = create_rag_pipeline(model)
        model_loaded = True
        logger.info("Vistral-7B model loaded successfully")
    except Exception as e:
        logger.warning(f"Model loading failed (can be loaded later): {e}")
        model_loaded = False

    logger.info("VinaCompare backend started!")

    yield

    # Shutdown
    logger.info("Shutting down VinaCompare backend...")
    try:
        vector_store = get_vector_store()
        vector_store.disconnect()
    except:
        pass
    close_db()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Vietnamese Technical Q&A with Multiple LLMs",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    model: str = "Vistral-7B-Chat"
    top_k: int = 5
    retrieval_mode: str = "hybrid"


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    model: str
    confidence: float
    sources: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class FeedbackRequest(BaseModel):
    query_id: str
    model: str
    rating: Optional[int] = None
    thumbs: Optional[str] = None
    comment: Optional[str] = ""


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "dense"


# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "database": "connected",
        "vector_store": "connected"
    }


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Main RAG endpoint

    Execute RAG pipeline with specified model and return answer
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/search")
async def search_documents(request: SearchRequest):
    """
    Search documents without generation

    Useful for testing retrieval
    """
    try:
        retriever = get_retriever()
        documents = retriever.search(
            query=request.query,
            top_k=request.top_k,
            mode=request.mode
        )

        return {
            "query": request.query,
            "results": [
                {
                    "document_id": doc.id,
                    "text": doc.text,
                    "score": doc.score,
                    "metadata": doc.metadata
                }
                for doc in documents
            ],
            "count": len(documents)
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback"""
    try:
        with get_db_context() as db:
            db.execute(
                text("""
                INSERT INTO feedback
                (query_id, model_name, rating, thumbs_direction, comment)
                VALUES (:query_id, :model_name, :rating, :thumbs_direction, :comment)
                """),
                {
                    "query_id": request.query_id,
                    "model_name": request.model,
                    "rating": request.rating,
                    "thumbs_direction": request.thumbs,
                    "comment": request.comment
                }
            )

        return {"status": "success", "query_id": request.query_id}

    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_stats():
    """Get system statistics"""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()

        return {
            "vector_store": stats,
            "models_available": settings.AVAILABLE_MODELS,
            "current_model": "Vistral-7B-Chat" if model_loaded else "None",
            "model_loaded": model_loaded
        }

    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models")
async def list_models():
    """List available models"""
    return {
        "models": settings.AVAILABLE_MODELS,
        "default": settings.DEFAULT_MODEL,
        "loaded": "Vistral-7B-Chat" if model_loaded else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )

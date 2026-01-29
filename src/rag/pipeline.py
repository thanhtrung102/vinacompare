"""
RAG Pipeline - Retrieval Augmented Generation
"""
from typing import Dict, Any, List, Optional
import logging
import time
import uuid

from src.retrieval.retriever import get_retriever, Document
from src.models.base_llm import BaseVietnameseLLM, GenerationConfig
from src.database import get_db_context
from src.config import settings
from sqlalchemy import text

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline"""

    def __init__(self, model: BaseVietnameseLLM):
        self.model = model
        self.retriever = get_retriever()

    def query(
        self,
        question: str,
        top_k: int = None,
        retrieval_mode: str = None,
        generation_config: GenerationConfig = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Execute complete RAG pipeline

        Args:
            question: User question in Vietnamese
            top_k: Number of documents to retrieve
            retrieval_mode: "dense", "sparse", or "hybrid"
            generation_config: Generation configuration
            user_id: User identifier

        Returns:
            Dictionary with answer, sources, and metrics
        """
        query_id = str(uuid.uuid4())
        pipeline_start = time.time()

        try:
            # 1. Retrieval
            logger.info(f"Query {query_id}: Retrieving documents for: {question}")
            retrieval_start = time.time()

            documents = self.retriever.search(
                query=question,
                top_k=top_k or settings.RETRIEVAL_TOP_K,
                mode=retrieval_mode or settings.RETRIEVAL_MODE
            )

            retrieval_time_ms = int((time.time() - retrieval_start) * 1000)
            logger.info(f"Retrieved {len(documents)} documents in {retrieval_time_ms}ms")

            # 2. Generation
            logger.info(f"Query {query_id}: Generating answer")
            context_texts = [doc.text for doc in documents]

            result = self.model.generate(
                query=question,
                context=context_texts,
                config=generation_config
            )

            # 3. Post-processing
            total_time_ms = int((time.time() - pipeline_start) * 1000)

            # 4. Log to database
            self._log_query(
                query_id=query_id,
                question=question,
                result=result,
                documents=documents,
                retrieval_time_ms=retrieval_time_ms,
                total_time_ms=total_time_ms,
                user_id=user_id,
                retrieval_mode=retrieval_mode or settings.RETRIEVAL_MODE
            )

            # 5. Prepare response
            response = {
                "query_id": query_id,
                "question": question,
                "answer": result.answer,
                "model": result.model_name,
                "confidence": result.confidence,
                "sources": [
                    {
                        "document_id": doc.id,
                        "text": doc.text,
                        "score": doc.score,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ],
                "metrics": {
                    "retrieval_time_ms": retrieval_time_ms,
                    "generation_time_ms": result.latency_ms,
                    "total_time_ms": total_time_ms,
                    "tokens_used": result.tokens_used,
                    "num_sources": len(documents)
                }
            }

            logger.info(f"Query {query_id} completed in {total_time_ms}ms")
            return response

        except Exception as e:
            logger.error(f"RAG pipeline failed for query {query_id}: {e}")
            raise

    def _log_query(
        self,
        query_id: str,
        question: str,
        result,
        documents: List[Document],
        retrieval_time_ms: int,
        total_time_ms: int,
        user_id: Optional[str],
        retrieval_mode: str
    ):
        """Log query to database"""
        try:
            with get_db_context() as db:
                # Log query
                db.execute(
                    text("""
                    INSERT INTO query_logs (query_id, user_id, query_text)
                    VALUES (:query_id, :user_id, :query_text)
                    """),
                    {"query_id": query_id, "user_id": user_id, "query_text": question}
                )

                # Log model response
                db.execute(
                    text("""
                    INSERT INTO model_responses
                    (query_id, model_name, answer, confidence_score,
                     latency_ms, tokens_used)
                    VALUES (:query_id, :model_name, :answer, :confidence_score,
                            :latency_ms, :tokens_used)
                    """),
                    {
                        "query_id": query_id,
                        "model_name": result.model_name,
                        "answer": result.answer,
                        "confidence_score": result.confidence,
                        "latency_ms": result.latency_ms,
                        "tokens_used": result.tokens_used
                    }
                )

                # Log retrieval metrics
                db.execute(
                    text("""
                    INSERT INTO retrieval_logs
                    (query_id, search_time_ms, num_results, retrieval_mode)
                    VALUES (:query_id, :search_time_ms, :num_results, :retrieval_mode)
                    """),
                    {
                        "query_id": query_id,
                        "search_time_ms": retrieval_time_ms,
                        "num_results": len(documents),
                        "retrieval_mode": retrieval_mode
                    }
                )

        except Exception as e:
            logger.error(f"Failed to log query {query_id}: {e}")


def create_rag_pipeline(model: BaseVietnameseLLM) -> RAGPipeline:
    """Create RAG pipeline with model"""
    return RAGPipeline(model)

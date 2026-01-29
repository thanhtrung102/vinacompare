"""
Hybrid retrieval system combining dense and sparse search
"""
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np
from underthesea import word_tokenize
import logging

from src.models.embeddings import get_embedder
from src.vector_store import get_vector_store
from src.config import settings

logger = logging.getLogger(__name__)


class Document:
    """Document container"""

    def __init__(
        self,
        id: str,
        text: str,
        score: float,
        metadata: Dict[str, Any] = None
    ):
        self.id = id
        self.text = text
        self.score = score
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(id={self.id}, score={self.score:.4f})"


class HybridRetriever:
    """Hybrid retrieval combining dense (vector) and sparse (BM25) search"""

    def __init__(self):
        self.embedder = None
        self.vector_store = None
        self.bm25_index = None
        self.documents_cache = []
        self._initialized = False

    def initialize(self):
        """Initialize embedder and vector store"""
        if self._initialized:
            return
        self.embedder = get_embedder()
        self.vector_store = get_vector_store()
        self._initialized = True

    def search(
        self,
        query: str,
        top_k: int = None,
        mode: str = None,
        alpha: float = 0.5
    ) -> List[Document]:
        """
        Hybrid search

        Args:
            query: Search query in Vietnamese
            top_k: Number of results to return
            mode: "dense", "sparse", or "hybrid"
            alpha: Weight for dense search (1-alpha for sparse)

        Returns:
            List of Document objects ranked by relevance
        """
        if not self._initialized:
            self.initialize()

        top_k = top_k or settings.RETRIEVAL_TOP_K
        mode = mode or settings.RETRIEVAL_MODE

        try:
            if mode == "dense":
                return self._dense_search(query, top_k)
            elif mode == "sparse":
                return self._sparse_search(query, top_k)
            elif mode == "hybrid":
                return self._hybrid_search(query, top_k, alpha)
            else:
                raise ValueError(f"Unknown retrieval mode: {mode}")

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def _dense_search(self, query: str, top_k: int) -> List[Document]:
        """Dense vector search"""
        # Encode query
        query_embedding = self.embedder.encode(query)[0].tolist()

        # Search in Milvus
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # Convert to Document objects
        documents = [
            Document(
                id=r["document_id"],
                text=r["text"],
                score=r["score"],
                metadata=r.get("metadata", {})
            )
            for r in results
        ]

        return documents

    def _sparse_search(self, query: str, top_k: int) -> List[Document]:
        """Sparse BM25 search"""
        if self.bm25_index is None:
            logger.warning("BM25 index not built yet")
            return []

        # Tokenize query
        query_tokens = word_tokenize(query, format="text").split()

        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Create documents
        documents = [
            Document(
                id=self.documents_cache[idx]["id"],
                text=self.documents_cache[idx]["text"],
                score=float(scores[idx]),
                metadata=self.documents_cache[idx].get("metadata", {})
            )
            for idx in top_indices
            if scores[idx] > 0
        ]

        return documents

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        alpha: float
    ) -> List[Document]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF)

        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight for dense search

        Returns:
            Fused and ranked documents
        """
        # Get results from both methods
        dense_results = self._dense_search(query, top_k * 2)
        sparse_results = self._sparse_search(query, top_k * 2)

        # If sparse is empty, fall back to dense
        if not sparse_results:
            return dense_results[:top_k]

        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        doc_scores = {}

        # Add dense scores
        for rank, doc in enumerate(dense_results):
            doc_id = doc.id
            rrf_score = alpha * (1 / (k + rank + 1))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        # Add sparse scores
        for rank, doc in enumerate(sparse_results):
            doc_id = doc.id
            rrf_score = (1 - alpha) * (1 / (k + rank + 1))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Retrieve full document objects
        doc_map = {doc.id: doc for doc in dense_results + sparse_results}

        final_documents = []
        for doc_id, score in sorted_docs:
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                doc.score = score
                final_documents.append(doc)

        return final_documents

    def build_bm25_index(self, documents: List[Dict[str, Any]]):
        """
        Build BM25 index from documents

        Args:
            documents: List of dicts with 'id', 'text', 'metadata'
        """
        logger.info(f"Building BM25 index for {len(documents)} documents")

        # Cache documents
        self.documents_cache = documents

        # Tokenize all documents
        tokenized_docs = [
            word_tokenize(doc["text"], format="text").split()
            for doc in documents
        ]

        # Build BM25 index
        self.bm25_index = BM25Okapi(
            tokenized_docs,
            k1=settings.BM25_K1,
            b=settings.BM25_B
        )

        logger.info("BM25 index built successfully")


# Global retriever instance
_retriever = None


def get_retriever() -> HybridRetriever:
    """Get global retriever instance"""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

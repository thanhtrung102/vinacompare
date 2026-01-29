"""
Milvus vector store connection and operations
"""
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from typing import List, Dict, Any, Optional
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Milvus vector store manager"""

    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.collection: Optional[Collection] = None

    def connect(self):
        """Connect to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")

            # Load or create collection
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                logger.info(f"Loaded existing collection: {self.collection_name}")
            else:
                self._create_collection()
                logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _create_collection(self):
        """Create Milvus collection with schema"""

        # Define schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name="document_id",
                dtype=DataType.VARCHAR,
                max_length=100
            ),
            FieldSchema(
                name="chunk_text",
                dtype=DataType.VARCHAR,
                max_length=10000
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=settings.EMBEDDING_DIMENSION
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON
            )
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Vietnamese technical documents"
        )

        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        # Create index for vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        self.collection.load()

    def insert(
        self,
        document_ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Insert documents into collection

        Args:
            document_ids: List of document IDs
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadata: List of metadata dicts

        Returns:
            List of inserted IDs
        """
        try:
            data = [
                document_ids,
                texts,
                embeddings,
                metadata
            ]

            mr = self.collection.insert(data)
            self.collection.flush()

            logger.info(f"Inserted {len(document_ids)} documents into Milvus")
            return mr.primary_keys

        except Exception as e:
            logger.error(f"Failed to insert into Milvus: {e}")
            raise

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_expr: Optional filter expression

        Returns:
            List of search results with scores and metadata
        """
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 100}
            }

            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["document_id", "chunk_text", "metadata"]
            )

            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit.id,
                        "document_id": hit.entity.get("document_id"),
                        "text": hit.entity.get("chunk_text"),
                        "score": hit.score,
                        "metadata": hit.entity.get("metadata", {})
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search Milvus: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            if self.collection is None:
                return {"error": "Collection not loaded"}
            stats = self.collection.num_entities
            return {
                "collection_name": self.collection_name,
                "num_entities": stats,
                "index_type": "HNSW",
                "metric_type": "COSINE"
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def disconnect(self):
        """Disconnect from Milvus"""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")


# Global instance
vector_store = MilvusVectorStore()


def get_vector_store() -> MilvusVectorStore:
    """Get vector store instance"""
    return vector_store

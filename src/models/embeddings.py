"""
Embedding model for Vietnamese text
"""
from sentence_transformers import SentenceTransformer
from typing import List, Union
import torch
import logging
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class VietnameseEmbedder:
    """Vietnamese text embedder using multilingual-e5"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load_model(self):
        """Load embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            logger.info(f"Embedding model loaded on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress_bar: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Show progress bar
            normalize: Normalize embeddings to unit vectors

        Returns:
            numpy array of embeddings
        """
        if self.model is None:
            self.load_model()

        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # Add instruction prefix for multilingual-e5
        prefixed_texts = [f"query: {text}" for text in texts]

        try:
            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=batch_size or settings.EMBEDDING_BATCH_SIZE,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = None
    ) -> np.ndarray:
        """
        Encode documents (different prefix than queries)

        Args:
            texts: List of document texts
            batch_size: Batch size for encoding

        Returns:
            numpy array of embeddings
        """
        if self.model is None:
            self.load_model()

        # Add passage prefix for documents
        prefixed_texts = [f"passage: {text}" for text in texts]

        try:
            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=batch_size or settings.EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode documents: {e}")
            raise


# Global embedder instance
_embedder = None


def get_embedder() -> VietnameseEmbedder:
    """Get global embedder instance"""
    global _embedder
    if _embedder is None:
        _embedder = VietnameseEmbedder()
        _embedder.load_model()
    return _embedder

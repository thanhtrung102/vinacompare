"""
Base LLM class for Vietnamese models
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Generation configuration"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class GenerationResult:
    """Generation result container"""
    answer: str
    model_name: str
    latency_ms: int
    tokens_used: int
    confidence: float
    metadata: Dict[str, Any]


class BaseVietnameseLLM(ABC):
    """Base class for Vietnamese LLMs"""

    def __init__(self, model_name: str, model_path: str = None):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer"""
        pass

    @abstractmethod
    def generate(
        self,
        query: str,
        context: List[str],
        config: GenerationConfig = None
    ) -> GenerationResult:
        """
        Generate answer from query and context

        Args:
            query: User question in Vietnamese
            context: List of relevant context strings
            config: Generation configuration

        Returns:
            GenerationResult with answer and metadata
        """
        pass

    def _build_prompt(self, query: str, context: List[str]) -> str:
        """
        Build prompt for Vietnamese technical Q&A

        Args:
            query: User question
            context: Relevant context documents

        Returns:
            Formatted prompt string
        """
        context_str = "\n\n".join([
            f"[Tài liệu {i+1}]: {ctx}"
            for i, ctx in enumerate(context)
        ])

        prompt = f"""Dựa trên các tài liệu kỹ thuật sau, hãy trả lời câu hỏi một cách chính xác và chi tiết bằng tiếng Việt.

TÀI LIỆU THAM KHẢO:
{context_str}

CÂU HỎI: {query}

TRẢ LỜI:"""

        return prompt

    def _calculate_confidence(self, generation_output) -> float:
        """
        Calculate confidence score from generation output

        This is a simplified version. A more sophisticated approach
        would use token probabilities if available.
        """
        # Placeholder - return default confidence
        return 0.8

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None

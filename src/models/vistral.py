"""
Vistral-7B-Chat implementation
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import List
import time
import logging

from src.models.base_llm import BaseVietnameseLLM, GenerationConfig, GenerationResult
from src.config import settings

logger = logging.getLogger(__name__)


class Vistral7B(BaseVietnameseLLM):
    """Vistral-7B-Chat model implementation"""

    def __init__(self):
        super().__init__(
            model_name="Vistral-7B-Chat",
            model_path="Viet-Mistral/Vistral-7B-Chat"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load Vistral-7B model with 4-bit quantization"""
        try:
            logger.info(f"Loading {self.model_name} from {self.model_path}")

            # Configure 4-bit quantization for efficiency
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)

            self.model.eval()

            logger.info(f"{self.model_name} loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            raise

    def generate(
        self,
        query: str,
        context: List[str],
        config: GenerationConfig = None
    ) -> GenerationResult:
        """Generate answer using Vistral-7B"""

        if not self.is_loaded():
            self.load_model()

        config = config or GenerationConfig()
        start_time = time.time()

        try:
            # Build prompt
            prompt = self._build_prompt(query, context)

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    do_sample=config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            tokens_used = outputs.shape[1]

            # Extract answer (remove any remaining prompt)
            answer = generated_text.strip()

            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(outputs)

            return GenerationResult(
                answer=answer,
                model_name=self.model_name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                confidence=confidence,
                metadata={
                    "device": self.device,
                    "quantization": "4bit" if self.device == "cuda" else "none"
                }
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


# Factory function
def create_vistral_model() -> Vistral7B:
    """Create and return Vistral-7B model"""
    model = Vistral7B()
    model.load_model()
    return model

"""
LLM Engine Module

Handles local LLM inference using HuggingFace Transformers:
- Model loading with CPU optimization
- Inference with caching
- Memory management
"""

import os
import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
 """Result of text generation."""
 
 text: str
 input_tokens: int
 output_tokens: int
 generation_time_seconds: float
 model_name: str
 
 @property
 def tokens_per_second(self) -> float:
 """Calculate generation speed."""
 if self.generation_time_seconds > 0:
 return self.output_tokens / self.generation_time_seconds
 return 0.0


class LLMEngine:
 """
 Local LLM inference engine using HuggingFace Transformers.
 
 Features:
 - CPU-optimized inference
 - Model caching
 - Memory-efficient loading
 - Multiple model support
 """
 
 # Supported models with their configurations
 SUPPORTED_MODELS = {
 "microsoft/phi-2": {
 "type": "causal",
 "context_length": 2048,
 "description": "Fast, code-focused (2.7B params)",
 },
 "google/gemma-2b-it": {
 "type": "causal",
 "context_length": 8192,
 "description": "Balanced, instruction-tuned (2B params)",
 },
 "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
 "type": "causal",
 "context_length": 2048,
 "description": "Ultra-fast, lightweight (1.1B params)",
 },
 }
 
 def __init__(
 self,
 model_name: str = "microsoft/phi-2",
 device: str = "cpu",
 use_quantization: bool = True,
 ):
 """
 Initialize the LLM engine.
 
 Args:
 model_name: HuggingFace model identifier
 device: Device to use ('cpu' or 'cuda')
 use_quantization: Whether to use INT8 quantization for memory efficiency
 """
 self.model_name = model_name
 self.device = device
 self.use_quantization = use_quantization
 
 self.model = None
 self.tokenizer = None
 self._is_loaded = False
 
 # Response cache
 self._cache: Dict[str, str] = {}
 self._cache_max_size = 100
 
 def load_model(self) -> None:
 """
 Load the model and tokenizer.
 
 This is called automatically on first generation,
 but can be called explicitly to pre-load.
 """
 if self._is_loaded:
 return
 
 logger.info(f"Loading model: {self.model_name}")
 logger.info("This may take a few minutes on first run...")
 
 try:
 from transformers import AutoModelForCausalLM, AutoTokenizer
 
 # Load tokenizer
 self.tokenizer = AutoTokenizer.from_pretrained(
 self.model_name,
 trust_remote_code=True,
 )
 
 # Set padding token if not set
 if self.tokenizer.pad_token is None:
 self.tokenizer.pad_token = self.tokenizer.eos_token
 
 # Load model with memory optimizations
 model_kwargs = {
 "trust_remote_code": True,
 "low_cpu_mem_usage": True,
 }
 
 # Use quantization for memory efficiency on CPU
 if self.use_quantization and self.device == "cpu":
 try:
 model_kwargs["torch_dtype"] = torch.float32
 logger.info("Using float32 for CPU inference")
 except Exception:
 pass
 
 self.model = AutoModelForCausalLM.from_pretrained(
 self.model_name,
 **model_kwargs,
 )
 
 # Move to device
 self.model = self.model.to(self.device)
 self.model.eval()
 
 self._is_loaded = True
 logger.info(f"Model loaded successfully on {self.device}")
 
 except Exception as e:
 logger.error(f"Failed to load model: {e}")
 raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
 
 def generate(
 self,
 prompt: str,
 max_new_tokens: int = 512,
 temperature: float = 0.1,
 top_p: float = 0.95,
 use_cache: bool = True,
 ) -> GenerationResult:
 """
 Generate text from a prompt.
 
 Args:
 prompt: Input prompt
 max_new_tokens: Maximum tokens to generate
 temperature: Sampling temperature (lower = more deterministic)
 top_p: Nucleus sampling parameter
 use_cache: Whether to use cached results
 
 Returns:
 GenerationResult with generated text and metadata
 """
 import time
 
 # Check cache
 cache_key = f"{prompt}_{max_new_tokens}_{temperature}"
 if use_cache and cache_key in self._cache:
 return GenerationResult(
 text=self._cache[cache_key],
 input_tokens=0,
 output_tokens=0,
 generation_time_seconds=0.0,
 model_name=self.model_name,
 )
 
 # Ensure model is loaded
 if not self._is_loaded:
 self.load_model()
 
 start_time = time.time()
 
 # Tokenize input
 inputs = self.tokenizer(
 prompt,
 return_tensors="pt",
 truncation=True,
 max_length=self._get_context_length() - max_new_tokens,
 ).to(self.device)
 
 input_tokens = inputs["input_ids"].shape[1]
 
 # Generate
 with torch.no_grad():
 outputs = self.model.generate(
 **inputs,
 max_new_tokens=max_new_tokens,
 temperature=temperature if temperature > 0 else 1.0,
 top_p=top_p,
 do_sample=temperature > 0,
 pad_token_id=self.tokenizer.pad_token_id,
 eos_token_id=self.tokenizer.eos_token_id,
 )
 
 # Decode output (only new tokens)
 generated_tokens = outputs[0][input_tokens:]
 generated_text = self.tokenizer.decode(
 generated_tokens,
 skip_special_tokens=True,
 )
 
 generation_time = time.time() - start_time
 output_tokens = len(generated_tokens)
 
 # Update cache
 if use_cache:
 self._add_to_cache(cache_key, generated_text)
 
 return GenerationResult(
 text=generated_text.strip(),
 input_tokens=input_tokens,
 output_tokens=output_tokens,
 generation_time_seconds=generation_time,
 model_name=self.model_name,
 )
 
 def generate_with_template(
 self,
 template: str,
 variables: Dict[str, str],
 **kwargs,
 ) -> GenerationResult:
 """
 Generate using a template with variable substitution.
 
 Args:
 template: Prompt template with {variable} placeholders
 variables: Dictionary of variable values
 **kwargs: Additional arguments for generate()
 
 Returns:
 GenerationResult
 """
 # Format template
 try:
 prompt = template.format(**variables)
 except KeyError as e:
 raise ValueError(f"Missing template variable: {e}")
 
 return self.generate(prompt, **kwargs)
 
 def _get_context_length(self) -> int:
 """Get context length for current model."""
 model_info = self.SUPPORTED_MODELS.get(self.model_name, {})
 return model_info.get("context_length", 2048)
 
 def _add_to_cache(self, key: str, value: str) -> None:
 """Add to response cache with size limit."""
 if len(self._cache) >= self._cache_max_size:
 # Remove oldest entry
 oldest_key = next(iter(self._cache))
 del self._cache[oldest_key]
 self._cache[key] = value
 
 def clear_cache(self) -> None:
 """Clear the response cache."""
 self._cache.clear()
 
 def unload_model(self) -> None:
 """Unload model to free memory."""
 if self.model is not None:
 del self.model
 self.model = None
 if self.tokenizer is not None:
 del self.tokenizer
 self.tokenizer = None
 self._is_loaded = False
 
 # Force garbage collection
 import gc
 gc.collect()
 
 if torch.cuda.is_available():
 torch.cuda.empty_cache()
 
 logger.info("Model unloaded")
 
 @property
 def is_loaded(self) -> bool:
 """Check if model is loaded."""
 return self._is_loaded
 
 def get_model_info(self) -> Dict[str, Any]:
 """Get information about the current model."""
 model_info = self.SUPPORTED_MODELS.get(self.model_name, {})
 return {
 "name": self.model_name,
 "loaded": self._is_loaded,
 "device": self.device,
 "context_length": model_info.get("context_length", "unknown"),
 "description": model_info.get("description", ""),
 "cache_size": len(self._cache),
 }

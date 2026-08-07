"""
model_loader.py

Loads a Hugging Face causal LM + tokenizer, and provides a simple
generate() function with automatic prompt truncation to fit the
model's context window.

This is an independent implementation for this project (not a copy
of any other repo's code) — inspired by common patterns for loading
and running local HF models.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Conservative default context/output budgets per model family.
# Extend this as needed; falls back to a safe default otherwise.
TOKEN_LIMITS = {
    "Qwen/Qwen3-0.6B": (8192, 1024),
    "Qwen/Qwen3-8B": (32768, 2048),
    "meta-llama/Llama-3.2-1B-Instruct": (8192, 1024),
    "meta-llama/Llama-3.1-8B-Instruct": (32768, 2048),
}
DEFAULT_LIMITS = (4096, 1024)


@dataclass
class GenerationParams:
    max_new_tokens: int
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    # Greedy decoding by default: always picks the highest-probability
    # token, so the same input always produces the same output. This
    # matters for a comparative study -- with sampling on, re-running the
    # same file through the same model gives a DIFFERENT BLEU-3 score
    # each time, which is noise unrelated to which model is actually
    # better. Set do_sample=True only for exploratory/creative use.
    do_sample: bool = False


class ModelRunner:
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        load_dotenv()
        self.model_name = model_name
        self.input_tokens, self.output_tokens = TOKEN_LIMITS.get(model_name, DEFAULT_LIMITS)

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN not set. Add it to your .env file (see .env.example)."
            )

        cache_dir = cache_dir or os.getenv("MODEL_CACHE_DIR", str(Path.cwd() / ".hf_cache"))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=hf_token, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            cache_dir=cache_dir,
            torch_dtype="auto",
            device_map="auto",
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_chat_template = getattr(self.tokenizer, "chat_template", None) is not None
        logger.info(f"Model loaded (device={self.device}, chat_template={self.use_chat_template})")

    def _build_text(self, messages: List[Dict[str, str]]) -> str:
        if self.use_chat_template:
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if "qwen" in self.model_name.lower():
                kwargs["enable_thinking"] = False
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return self.tokenizer.apply_chat_template(messages, **kwargs)
        # Fallback plain-text format for models without a chat template
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        lines.append("Assistant:")
        return "\n\n".join(lines)

    def generate(self, messages: List[Dict[str, str]], params: Optional[GenerationParams] = None) -> str:
        if params is None:
            params = GenerationParams(max_new_tokens=min(2048, self.output_tokens))

        text = self._build_text(messages)
        tokenized = self.tokenizer([text], return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask")

        safety = 16
        allowed_len = max(1, self.input_tokens - params.max_new_tokens - safety)
        if input_ids.shape[1] > allowed_len:
            # Truncation strategy: keep the END of the prompt (most recent
            # context), since for both our tasks the tail of the prompt is
            # the most relevant part (e.g. the code/description right
            # before the point of interest).
            input_ids = input_ids[:, -allowed_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -allowed_len:]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature if params.do_sample else None,
            top_p=params.top_p if params.do_sample else None,
            repetition_penalty=params.repetition_penalty,
            do_sample=params.do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = input_ids.shape[1]
        generated = outputs[0][prompt_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

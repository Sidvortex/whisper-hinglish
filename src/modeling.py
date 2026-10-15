"""Whisper model / processor setup shared by training, evaluation and serving."""
from __future__ import annotations

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_processor(name: str, language: str, task: str) -> WhisperProcessor:
    proc = WhisperProcessor.from_pretrained(name)
    proc.tokenizer.set_prefix_tokens(language=language, task=task)
    return proc


def configure_generation(model, language: str, task: str, max_length: int = 225) -> None:
    gc = model.generation_config
    gc.language = language
    gc.task = task
    gc.forced_decoder_ids = None
    gc.max_length = max_length
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []


def load_for_inference(path_or_name: str, base_name: str | None, language: str, task: str,
                       adapter: str | None = None):
    """Load a fine-tuned model directory, or a base model plus a LoRA adapter."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(path_or_name, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    configure_generation(model, language, task)
    proc_src = base_name or path_or_name
    try:
        proc = load_processor(path_or_name, language, task)
    except OSError:
        proc = load_processor(proc_src, language, task)
    return model.to(device).eval(), proc

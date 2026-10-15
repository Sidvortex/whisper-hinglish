"""Decode audio -> log-mel, tokenize text, pad both. Runs inside DataLoader workers."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data import decode_audio


@dataclass
class WhisperCollator:
    processor: object
    sampling_rate: int = 16000
    max_audio_seconds: float = 30.0
    max_label_tokens: int = 440
    decoder_start_token_id: int | None = None

    def __call__(self, batch: list[dict]) -> dict:
        audios = [decode_audio(b["audio"], self.sampling_rate, self.max_audio_seconds) for b in batch]
        feats = self.processor.feature_extractor(
            audios, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_features

        tok = self.processor.tokenizer
        ids = [tok(b["text"]).input_ids[: self.max_label_tokens] for b in batch]
        labels = tok.pad({"input_ids": ids}, return_tensors="pt")
        lab = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        # the model prepends the decoder start token itself; drop it if the tokenizer added it
        if self.decoder_start_token_id is not None and (lab[:, 0] == self.decoder_start_token_id).all():
            lab = lab[:, 1:]
        return {"input_features": feats.to(torch.float32), "labels": lab}

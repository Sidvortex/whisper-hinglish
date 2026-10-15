"""Load one or more speech sources into a common (audio, text, group, source) schema.

Audio stays undecoded in the dataset (raw bytes or a path) and is decoded in the
collator. That avoids the torchcodec dependency of newer `datasets` versions and
keeps disk usage low (no precomputed log-mel features).
"""
from __future__ import annotations

import io
import unicodedata

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, Value, concatenate_datasets, load_dataset

SPLITS = ("train", "eval", "test")


def _load_source(src: dict, split_name: str) -> Dataset | None:
    split = src["splits"].get(split_name)
    if split is None:
        return None
    if src["type"] == "hf":
        ds = load_dataset(src["dataset"], src.get("subset"), split=split)
    elif src["type"] == "manifest":
        # jsonl lines: {"audio": "/path/clip.wav", "text": "...", "group": "hinglish"}
        ds = load_dataset("json", data_files=split, split="train")
    else:
        raise ValueError(f"unknown source type {src['type']}")

    audio_col = src.get("audio_column", "audio")
    text_col = src.get("text_column", "text")
    group_col = src.get("group_column")

    for col, allowed in (src.get("keep") or {}).items():
        allowed = set(allowed)
        ds = ds.filter(lambda v: v in allowed, input_columns=[col])
    ds = ds.filter(lambda t: isinstance(t, str) and t.strip() != "", input_columns=[text_col])

    ds = ds.cast_column(audio_col, Audio(decode=False))
    if audio_col != "audio":
        ds = ds.rename_column(audio_col, "audio")
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")
    if group_col and group_col in ds.column_names:
        if group_col != "group":
            if "group" in ds.column_names:
                ds = ds.remove_columns("group")
            ds = ds.rename_column(group_col, "group")
    else:
        ds = ds.add_column("group", [src["name"]] * len(ds))
    ds = ds.add_column("source", [src["name"]] * len(ds))
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("audio", "text", "group", "source")])
    ds = ds.cast_column("group", Value("string"))
    return ds


def load_corpus(dcfg: dict) -> dict[str, Dataset]:
    out = {}
    for split in SPLITS:
        parts = [p for p in (_load_source(s, split) for s in dcfg["sources"]) if p is not None]
        if parts:
            ds = concatenate_datasets(parts)
            out[split] = ds.shuffle(seed=dcfg["seed"]) if split == "train" else ds
    return out


def decode_audio(audio: dict, target_sr: int, max_seconds: float | None = None) -> np.ndarray:
    """audio = {"bytes": ..., "path": ...} -> mono float32 at target_sr."""
    if audio.get("bytes"):
        wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    else:
        wav, sr = sf.read(audio["path"], dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != target_sr:
        import soxr

        wav = soxr.resample(wav, sr, target_sr)
    if max_seconds:
        wav = wav[: int(max_seconds * target_sr)]
    return wav.astype(np.float32)


# ------------------------------------------------------------ text normalisation for WER

_KEEP = {"'"}


def normalize_text(text: str) -> str:
    """Script-preserving normaliser for code-mixed text.

    - Unicode NFC (so Devanagari matras compare correctly)
    - lowercase Latin
    - strip punctuation incl. the danda (।) but keep Devanagari marks
    - collapse whitespace
    """
    text = unicodedata.normalize("NFC", text).lower()
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in _KEEP or cat[0] in ("L", "M", "N") or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())

"""Transcribe audio files of any length with the fine-tuned model.

    python transcribe.py --model outputs/whisper-small-hinglish/final clip.wav meeting.mp3
"""
from __future__ import annotations

import argparse

import numpy as np

from eval_model import transcribe_batch
from src.data import decode_audio
from src.modeling import load_for_inference

SR = 16000


def chunks(wav: np.ndarray, seconds: int = 28, overlap: int = 2):
    step = (seconds - overlap) * SR
    for start in range(0, max(len(wav) - overlap * SR, 1), step):
        yield wav[start : start + seconds * SR]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--language", default="hindi")
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    model, proc = load_for_inference(args.model, None, args.language, "transcribe")
    for f in args.files:
        wav = decode_audio({"path": f, "bytes": None}, SR)
        parts = list(chunks(wav))
        texts = transcribe_batch(model, proc, parts, SR)
        print(f"{f}:\n{' '.join(t.strip() for t in texts)}\n")


if __name__ == "__main__":
    main()

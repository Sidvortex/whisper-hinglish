"""ASR server for the ESP32-S3 keyword-spotting pipeline (wake word on device -> audio here).

    MODEL_DIR=outputs/whisper-small-hinglish/final uvicorn serve:app --host 0.0.0.0 --port 8020

    POST /transcribe            multipart file upload (wav/flac/ogg/mp3)
    POST /transcribe/pcm16      raw little-endian 16-bit mono PCM body at 16 kHz
                                (what an ESP32 I2S mic stream naturally produces)
"""
from __future__ import annotations

import os
import time

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile

from eval_model import transcribe_batch
from src.data import decode_audio
from src.modeling import load_for_inference

SR = 16000
app = FastAPI(title="Whisper Hinglish ASR")
model, proc = load_for_inference(
    os.environ.get("MODEL_DIR", "outputs/whisper-small-hinglish/final"),
    os.environ.get("BASE_MODEL", "openai/whisper-small"),
    os.environ.get("LANGUAGE", "hindi"),
    "transcribe",
)


def _run(wav: np.ndarray) -> dict:
    t0 = time.perf_counter()
    text = transcribe_batch(model, proc, [wav[: 30 * SR]], SR)[0].strip()
    return {"text": text, "seconds_audio": round(len(wav) / SR, 2),
            "latency_ms": round(1000 * (time.perf_counter() - t0))}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "device": str(model.device)}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    return _run(decode_audio({"bytes": data, "path": None}, SR))


@app.post("/transcribe/pcm16")
async def transcribe_pcm(request: Request) -> dict:
    body = await request.body()
    wav = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    return _run(wav)

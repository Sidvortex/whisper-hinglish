# whisper-hinglish

Fine-tunes **OpenAI Whisper** on **Hindi-English code-mixed (Hinglish) speech**: sentences like "मैं office जा रहा हूँ, meeting के बाद call करता हूँ". Off-the-shelf Whisper handles these badly. It tends to force everything into one script or translate the English words.

The repo includes a test-set evaluation broken down by language, a long-audio transcriber, and an ASR server with a raw-PCM endpoint for an ESP32-S3 keyword-spotting front end (wake word on device, audio streamed here).

## Configs

| Config | Model | Method | T4 time |
|---|---|---|---|
| `whisper_small.yaml` | whisper-small (244M) | full fine-tune | ≈3–4 h / 4k steps |
| `whisper_medium_lora.yaml` | whisper-medium (769M) | LoRA r=32 | ≈4–5 h / 3k steps |

## Data

Default: [`addyo07/noisy-hinglish-asr`](https://huggingface.co/datasets/addyo07/noisy-hinglish-asr), which has train/val/test splits and a `language` column (`hinglish`, `hi`, `en`, `neutral`). Non-speech `neutral` clips are dropped, and the `language` column drives the per-language WER breakdown. Look at a few transcripts on the dataset page to confirm how it writes Hindi (Devanagari vs romanised) before reporting results.

Adding more data is the biggest lever for quality. [MUCS 2021](https://huggingface.co/papers/2104.00235) has a Hindi-English code-switched subset (≈90 h on OpenSLR). It ships as Kaldi folders; convert them and add a `manifest` source to the config:

```bash
python prepare_kaldi.py --kaldi-dir mucs/train --audio-root mucs --out data/mucs/train.jsonl
python prepare_kaldi.py --kaldi-dir mucs/test  --audio-root mucs --out data/mucs/test.jsonl
```

Any folder of clips works the same way: write a jsonl with `{"audio": "/path.wav", "text": "...", "group": "hinglish"}` per line.

## Run on Kaggle

Push to `github.com/Sidvortex/whisper-hinglish`, import `kaggle_notebook.ipynb`, enable GPU (T4) and Internet, and run all. The notebook does a 10-minute smoke run first.

```bash
python train.py --config configs/whisper_small.yaml
python eval_model.py --config configs/whisper_small.yaml --limit 800   # base vs fine-tuned
```

`test_results.json` reports WER and CER overall and per group (hinglish / hi / en). `test_samples.md` shows reference, base and fine-tuned transcripts side by side.

The text normaliser keeps both scripts (NFC Unicode, lowercase Latin, punctuation and danda stripped). CER is the fairer metric for Hindi, and WER for the English-heavy portions, so report both.

## Use it locally (CPU)

whisper-small runs at roughly real time on a laptop CPU.

```bash
pip install --break-system-packages -r requirements.txt
python transcribe.py --model outputs/whisper-small-hinglish/final clip.wav

MODEL_DIR=outputs/whisper-small-hinglish/final uvicorn serve:app --host 0.0.0.0 --port 8020
curl -F file=@clip.wav localhost:8020/transcribe
# ESP32: POST the raw 16 kHz 16-bit mono I2S buffer as the body
curl --data-binary @clip.pcm localhost:8020/transcribe/pcm16
```

For a faster server on a cheap box, convert to CTranslate2 and use faster-whisper:

```bash
pip install --break-system-packages ctranslate2 faster-whisper
ct2-transformers-converter --model outputs/whisper-small-hinglish/final --output_dir whisper-small-hinglish-ct2 --quantization int8
```

## Notes

- `language: hindi` is the decoder prompt. Training teaches the model to keep English words in Latin script inside Hindi sentences when the labels do.
- The audio column is decoded with `soundfile` and resampled with `soxr` inside the data collator, so there's no torchcodec or ffmpeg dependency and no huge precomputed feature cache.

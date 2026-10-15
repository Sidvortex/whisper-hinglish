"""Convert a Kaldi-style data directory into a jsonl manifest this repo can train on.

Many Indian speech corpora (e.g. the MUCS 2021 Hindi-English code-switching set on
OpenSLR) ship Kaldi folders containing:
    text        utt_id  transcript ...
    wav.scp     rec_id  /path/to/file.wav      (or utt_id -> path when there is no segments file)
    segments    utt_id  rec_id  start_sec  end_sec   (optional)

    python prepare_kaldi.py --kaldi-dir mucs/train --audio-root mucs --out data/mucs/train.jsonl
    python prepare_kaldi.py --kaldi-dir mucs/test  --audio-root mucs --out data/mucs/test.jsonl

Segmented utterances are cut into 16 kHz wav clips under <out_dir>/clips/.
Then add a `type: manifest` source to your config (see configs/whisper_small.yaml).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from src.data import decode_audio

SR = 16000


def read_map(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def resolve(path_field: str, audio_root: Path) -> Path:
    # wav.scp entries can be plain paths or pipes ("sox x.wav -t wav - |"); take the first file-like token
    token = next((t for t in path_field.split() if "." in t and "/" in t), path_field.split()[0])
    p = Path(token)
    return p if p.is_absolute() else audio_root / p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaldi-dir", required=True)
    ap.add_argument("--audio-root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default="hinglish")
    args = ap.parse_args()

    kd, root, out = Path(args.kaldi_dir), Path(args.audio_root), Path(args.out)
    text, wavscp, segments = read_map(kd / "text"), read_map(kd / "wav.scp"), read_map(kd / "segments")
    clips_dir = out.parent / "clips" / kd.name
    clips_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[str, object] = {}
    written = skipped = 0
    with out.open("w", encoding="utf-8") as f:
        for utt, transcript in text.items():
            try:
                if segments:
                    rec, start, end = segments[utt].split()
                    if rec not in cache:
                        cache.clear()  # keep one recording in memory at a time
                        cache[rec] = decode_audio({"path": str(resolve(wavscp[rec], root)), "bytes": None}, SR)
                    wav = cache[rec][int(float(start) * SR) : int(float(end) * SR)]
                    clip = clips_dir / f"{utt}.wav"
                    sf.write(clip, wav, SR)
                    audio_path = clip
                else:
                    audio_path = resolve(wavscp[utt], root)
                f.write(json.dumps({"audio": str(audio_path.resolve()), "text": transcript.strip(),
                                    "group": args.group}, ensure_ascii=False) + "\n")
                written += 1
            except (KeyError, RuntimeError, ValueError) as e:
                skipped += 1
                if skipped <= 5:
                    print(f"skip {utt}: {e}")
    print(f"wrote {written} utterances to {out} (skipped {skipped})")


if __name__ == "__main__":
    main()

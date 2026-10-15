"""WER / CER on the test split, overall and per group (hinglish / hi / en), base vs fine-tuned.

    python eval_model.py --config configs/whisper_small.yaml
    python eval_model.py --model outputs/whisper-small-hinglish/final --limit 500
    python eval_model.py --skip-base
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import jiwer
import torch
from tqdm import tqdm

from src.config import base_parser, load_config
from src.data import decode_audio, load_corpus, normalize_text
from src.modeling import load_for_inference


@torch.no_grad()
def transcribe_batch(model, proc, audios, sr: int) -> list[str]:
    feats = proc.feature_extractor(audios, sampling_rate=sr, return_tensors="pt").input_features
    feats = feats.to(model.device, dtype=model.dtype)
    ids = model.generate(input_features=feats)
    return proc.tokenizer.batch_decode(ids, skip_special_tokens=True)


def evaluate(model, proc, ds, sr: int, batch_size: int) -> tuple[dict, list[dict]]:
    rows = []
    for i in tqdm(range(0, len(ds), batch_size), desc="transcribing"):
        batch = ds[i : i + batch_size]
        audios = [decode_audio(a, sr, 30) for a in batch["audio"]]
        hyps = transcribe_batch(model, proc, audios, sr)
        for ref, hyp, grp in zip(batch["text"], hyps, batch["group"]):
            rows.append({"group": grp, "ref": ref, "hyp": hyp,
                         "ref_n": normalize_text(ref), "hyp_n": normalize_text(hyp)})

    def wer_cer(rs):
        rs = [r for r in rs if r["ref_n"]]
        if not rs:
            return None
        refs, hyps = [r["ref_n"] for r in rs], [r["hyp_n"] for r in rs]
        return {"n": len(rs), "wer": round(100 * jiwer.wer(refs, hyps), 2),
                "cer": round(100 * jiwer.cer(refs, hyps), 2)}

    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    metrics = {"overall": wer_cer(rows), **{g: wer_cer(rs) for g, rs in sorted(by_group.items())}}
    return metrics, rows


def main() -> None:
    parser = base_parser(__doc__, "configs/whisper_small.yaml")
    parser.add_argument("--model", default=None, help="fine-tuned model dir (default: <output_dir>/final)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    mcfg, dcfg = cfg["model"], cfg["data"]
    out = Path(cfg["train"]["output_dir"])
    tuned = args.model or str(out / "final")

    test = load_corpus(dcfg)["test"]
    if args.limit:
        test = test.shuffle(seed=dcfg["seed"]).select(range(min(args.limit, len(test))))
    print(f"test clips: {len(test)}")

    runs = [] if args.skip_base else [("base", mcfg["name"])]
    runs.append(("fine_tuned", tuned))
    results, samples = {}, {}
    for label, path in runs:
        model, proc = load_for_inference(path, mcfg["name"], mcfg["language"], mcfg["task"])
        metrics, rows = evaluate(model, proc, test, dcfg["sampling_rate"], args.batch_size)
        results[label] = metrics
        samples[label] = rows
        print(label, json.dumps(metrics, indent=2))
        del model
        torch.cuda.empty_cache()

    out.mkdir(parents=True, exist_ok=True)
    (out / "test_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    lines = ["# Test transcripts (first 30)\n"]
    for i in range(min(30, len(test))):
        r0 = samples[runs[-1][0]][i]
        lines.append(f"**[{r0['group']}] ref:** {r0['ref']}  ")
        for label, _ in runs:
            lines.append(f"**{label}:** {samples[label][i]['hyp']}  ")
        lines.append("")
    (out / "test_samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'test_results.json'} and {out / 'test_samples.md'}")


if __name__ == "__main__":
    main()

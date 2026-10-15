"""Fine-tune Whisper on Hindi-English code-mixed speech.

    python train.py --config configs/whisper_small.yaml
    python train.py --config configs/whisper_medium_lora.yaml
    python train.py --set train.max_steps=50 train.eval_steps=25 data.max_eval_samples=32   # smoke run
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import jiwer
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration

from src.collator import WhisperCollator
from src.config import base_parser, load_config
from src.data import load_corpus, normalize_text
from src.modeling import configure_generation, load_processor


def main() -> None:
    args = base_parser(__doc__, "configs/whisper_small.yaml").parse_args()
    cfg = load_config(args.config, args.set)
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    out = Path(tcfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(dcfg)
    train_ds = corpus["train"]
    eval_ds = corpus["eval"].shuffle(seed=dcfg["seed"]).select(
        range(min(dcfg["max_eval_samples"], len(corpus["eval"])))
    )
    print(f"train={len(train_ds)} eval(subset)={len(eval_ds)}")
    print("train groups:", dict(Counter(train_ds["group"])))

    proc = load_processor(mcfg["name"], mcfg["language"], mcfg["task"])
    model = WhisperForConditionalGeneration.from_pretrained(mcfg["name"])
    configure_generation(model, mcfg["language"], mcfg["task"], tcfg["generation_max_length"])
    model.config.use_cache = False

    if mcfg["method"] == "lora":
        from peft import LoraConfig, get_peft_model

        lc = cfg["lora"]
        model = get_peft_model(model, LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], bias="none",
        ))
        model.print_trainable_parameters()

    collator = WhisperCollator(
        processor=proc,
        sampling_rate=dcfg["sampling_rate"],
        max_audio_seconds=dcfg["max_audio_seconds"],
        max_label_tokens=dcfg["max_label_tokens"],
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    tok = proc.tokenizer

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tok.pad_token_id
        preds = [normalize_text(t) for t in tok.batch_decode(pred_ids, skip_special_tokens=True)]
        refs = [normalize_text(t) for t in tok.batch_decode(label_ids, skip_special_tokens=True)]
        pairs = [(r, p) for r, p in zip(refs, preds) if r]
        refs, preds = [r for r, _ in pairs], [p for _, p in pairs]
        return {"wer": 100 * jiwer.wer(refs, preds), "cer": 100 * jiwer.cer(refs, preds)}

    targs = Seq2SeqTrainingArguments(
        output_dir=str(out / "checkpoints"),
        max_steps=tcfg["max_steps"],
        learning_rate=tcfg["lr"],
        warmup_steps=tcfg["warmup_steps"],
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        per_device_eval_batch_size=tcfg["eval_batch_size"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=tcfg["generation_max_length"],
        logging_steps=tcfg["logging_steps"],
        dataloader_num_workers=tcfg["num_workers"],
        remove_unused_columns=False,   # the collator needs the raw audio column
        label_names=["labels"],
        report_to="none",
        seed=dcfg["seed"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=proc.feature_extractor,
    )
    trainer.train()

    final = out / "final"
    metrics = trainer.evaluate()  # best checkpoint is loaded at this point
    if mcfg["method"] == "lora":
        final_model = trainer.model.merge_and_unload()
    else:
        final_model = trainer.model
    final_model.config.use_cache = True  # fast generation at inference time
    final_model.save_pretrained(str(final))
    proc.save_pretrained(str(final))
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Fine-tuned model saved to {final}")


if __name__ == "__main__":
    main()

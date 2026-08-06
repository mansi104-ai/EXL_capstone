"""Fine-tune a DistilBERT multi-label classifier for the four drivers.

Trains on data/labeled/{train,val}.jsonl produced by scripts/generate_data.py
and saves to the configured model_dir. Requires torch + transformers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config, load_config
from .base import DRIVER_ORDER


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_examples(rows: list[dict]) -> tuple[list[str], list[list[float]]]:
    texts = [r["text"] for r in rows]
    labels = [[float(r["labels"][d.value]) for d in DRIVER_ORDER] for r in rows]
    return texts, labels


def train(cfg: Config | None = None, epochs: int = 8) -> str:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    cfg = cfg or load_config()
    labeled_dir = cfg.path("data", "labeled")
    model_dir = cfg.path(cfg.classifier["model_dir"])
    base_model = cfg.classifier["base_model"]
    max_length = int(cfg.classifier.get("max_length", 128))

    train_rows = _read_jsonl(labeled_dir / "train.jsonl")
    val_rows = _read_jsonl(labeled_dir / "val.jsonl")
    train_texts, train_labels = _to_examples(train_rows)
    val_texts, val_labels = _to_examples(val_rows)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_length)
        enc["labels"] = batch["labels"]
        return enc

    train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels}).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    val_ds = Dataset.from_dict({"text": val_texts, "labels": val_labels}).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    id2label = {i: d.value for i, d in enumerate(DRIVER_ORDER)}
    label2id = {d.value: i for i, d in enumerate(DRIVER_ORDER)}
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(DRIVER_ORDER),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        labels = labels.astype(int)
        tp = (preds & labels).sum()
        fp = (preds & (1 - labels)).sum()
        fn = ((1 - preds) & labels).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    model_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(model_dir / "_trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("Validation metrics:", {k: round(v, 3) for k, v in metrics.items() if isinstance(v, float)})

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Saved fine-tuned model to {model_dir}")
    return str(model_dir)

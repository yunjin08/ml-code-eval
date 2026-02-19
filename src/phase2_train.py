"""
Phase 2: Model Training and Validation
Fine-tune CodeBERT and train Random Forest baseline on DiverseVul train set.
Benchmark both on validation set; select the model with higher F1 for Phase 3.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

# Google Chat webhook for training logs (hardcoded)
CHAT_WEBHOOK_URL = (
    "https://chat.googleapis.com/v1/spaces/AAQAwJE8HMo/messages"
    "?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI"
    "&token=1D1W6woJWK8LDFSMJis-x3WrexFXgL1GmNt73c19MnA"
)


def send_chat_log(text: str) -> None:
    """POST a plain-text message to Google Chat. Failures are ignored (no crash)."""
    try:
        data = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            CHAT_WEBHOOK_URL,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # don't fail training if Chat is unreachable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_curated():
    """Load curated dataset and split indices."""
    csv_path = os.path.join(DATA_DIR, "curated_cpp.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Run phase1_dataset.py first. Missing {csv_path}")
    df = pd.read_csv(csv_path)
    with open(os.path.join(DATA_DIR, "splits.json")) as f:
        splits = json.load(f)
    train_df = df[df["id"].isin(splits["train_ids"])].copy()
    val_df = df[df["id"].isin(splits["validation_ids"])].copy()
    return train_df, val_df


def train_codebert(train_df, val_df, max_train=None, max_val=None, epochs=3, batch_size=16, use_cpu=False):
    """Fine-tune CodeBERT for binary classification; return model, tokenizer, validation F1."""
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from transformers import EvalPrediction

    class ChatLogCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None:
                return
            epoch = state.epoch if state.epoch is not None else 0
            f1 = metrics.get("eval_f1", 0)
            loss = metrics.get("eval_loss", 0)
            send_chat_log(
                f"[Phase 2] CodeBERT epoch {epoch:.1f} — eval_f1: {f1:.4f}, eval_loss: {loss:.4f}"
            )

    if max_train:
        train_df = train_df.sample(n=max_train, random_state=42)
    if max_val:
        val_df = val_df.sample(n=max_val, random_state=42)

    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # Avoid MPS (Apple GPU) OOM on M1/M2/M3 by using CPU when requested
    if use_cpu:
        model = model.to("cpu")
        print("Using CPU for CodeBERT (--cpu). Slower but avoids Metal OOM on Mac.")

    max_length = 512

    def tokenize(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    from datasets import Dataset

    train_ds = Dataset.from_pandas(train_df[["code", "label"]].rename(columns={"label": "labels"}))
    val_ds = Dataset.from_pandas(val_df[["code", "label"]].rename(columns={"label": "labels"}))
    train_ds = train_ds.map(
        tokenize,
        batched=True,
        remove_columns=["code"],
        desc="Tokenize train",
    )
    val_ds = val_ds.map(
        tokenize,
        batched=True,
        remove_columns=["code"],
        desc="Tokenize val",
    )
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # Class weight for imbalance (vulnerable minority)
    n0 = (train_df["label"] == 0).sum()
    n1 = (train_df["label"] == 1).sum()
    weight = torch.tensor([1.0, max(1.0, n0 / max(n1, 1))], dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = np.argmax(logits, axis=-1)
        return {
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
        }

    out_dir = os.path.join(MODELS_DIR, "codebert_checkpoint")
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=200,
        save_total_limit=2,
        report_to="none",
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[ChatLogCallback()],
    )
    n_train = len(train_ds)
    n_val = len(val_ds)
    send_chat_log(
        f"[Phase 2] CodeBERT training started — train: {n_train}, val: {n_val}, epochs: {epochs}, batch_size: {batch_size}"
    )
    trainer.train()
    eval_out = trainer.evaluate()
    trainer.save_model(os.path.join(MODELS_DIR, "codebert"))
    tokenizer.save_pretrained(os.path.join(MODELS_DIR, "codebert"))
    return model, tokenizer, eval_out.get("eval_f1", 0.0), eval_out


def train_random_forest(train_df, val_df, max_train=None, max_val=None):
    """Extract lizard features, train RF; return model, validation F1."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score

    sys.path.insert(0, SCRIPT_DIR)
    from utils.lizard_metrics import extract_metrics

    def features_for_df(df):
        rows = []
        for _, row in df.iterrows():
            m = extract_metrics(row["code"], language="c")
            rows.append(
                [
                    m["cyclomatic_complexity"],
                    m["nloc"],
                    m["token_count"],
                    m["parameter_count"],
                ]
            )
        return np.array(rows)

    if max_train:
        train_df = train_df.sample(n=max_train, random_state=42)
    if max_val:
        val_df = val_df.sample(n=max_val, random_state=42)

    X_train = features_for_df(train_df)
    y_train = train_df["label"].values
    X_val = features_for_df(val_df)
    y_val = val_df["label"].values

    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    f1 = f1_score(y_val, preds, zero_division=0)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)

    import joblib

    joblib.dump(rf, os.path.join(MODELS_DIR, "rf.pkl"))
    return rf, f1, {"eval_f1": f1, "eval_precision": prec, "eval_recall": rec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train", type=int, default=None, help="Cap train size for quick runs")
    parser.add_argument("--max_val", type=int, default=None, help="Cap val size for quick runs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--skip_codebert", action="store_true", help="Only train RF")
    parser.add_argument("--skip_rf", action="store_true", help="Only train CodeBERT")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for CodeBERT (avoids MPS OOM on Apple M1/M2/M3)")
    args = parser.parse_args()

    send_chat_log(
        f"[Phase 2] Training started — max_train: {args.max_train}, max_val: {args.max_val}, "
        f"epochs: {args.epochs}, batch_size: {args.batch_size}, cpu: {args.cpu}"
    )

    train_df, val_df = load_curated()
    report = {"codebert": None, "random_forest": None, "selected_model": None}

    try:
        if not args.skip_codebert:
            import torch

            print("Training CodeBERT...")
            _, _, cb_f1, cb_metrics = train_codebert(
                train_df, val_df,
                max_train=args.max_train, max_val=args.max_val,
                epochs=args.epochs, batch_size=args.batch_size,
                use_cpu=args.cpu,
            )
            report["codebert"] = cb_metrics
            print(f"CodeBERT validation F1: {cb_f1:.4f}")
            send_chat_log(f"[Phase 2] CodeBERT done — validation F1: {cb_f1:.4f}")
        else:
            cb_f1 = -1.0

        if not args.skip_rf:
            print("Training Random Forest...")
            _, rf_f1, rf_metrics = train_random_forest(
                train_df, val_df, max_train=args.max_train, max_val=args.max_val
            )
            report["random_forest"] = rf_metrics
            print(f"Random Forest validation F1: {rf_f1:.4f}")
            send_chat_log(f"[Phase 2] Random Forest done — validation F1: {rf_f1:.4f}")
        else:
            rf_f1 = -1.0

        if cb_f1 >= rf_f1 and not args.skip_codebert:
            report["selected_model"] = "codebert"
        else:
            report["selected_model"] = "random_forest"

        report_path = os.path.join(RESULTS_DIR, "phase2_validation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Selected model for Phase 3: {report['selected_model']}")
        print(f"Report saved to {report_path}")
        send_chat_log(
            f"[Phase 2] Finished — selected: {report['selected_model']}, report: {report_path}"
        )
    except Exception as e:
        send_chat_log(f"[Phase 2] Error — {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()

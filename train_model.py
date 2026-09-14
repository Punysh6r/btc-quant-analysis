"""Train a TF-IDF + LogisticRegression news-sentiment classifier.

Uses a strict temporal split (first 80% train / last 20% test, no
shuffle) so evaluation reflects forward-looking performance. Saves a
joblib bundle plus a JSON metrics file next to it.

Example:
    python train_model.py
    python train_model.py --data data/dataset.csv --out artifacts/model.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = "data/dataset.csv"
DEFAULT_OUT = "artifacts/model.joblib"
TEST_FRACTION = 0.20
MIN_ROWS = 200
THRESHOLD_PCT = 2.0
HORIZON = "24h"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF + LogisticRegression news classifier."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Labeled dataset CSV path, resolved relative to this script's "
        "directory (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output joblib path, resolved relative to this script's "
        "directory (default: %(default)s).",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=MIN_ROWS,
        help="Minimum labeled rows required to train (default: %(default)s). "
        "Lower it only for pipeline demos with tiny data.",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=THRESHOLD_PCT,
        help="Label threshold in percent, stored as metadata "
        "(default: %(default)s). Must match the --threshold used in build_dataset.py.",
    )
    return parser.parse_args(argv)


def resolve(path_str: str) -> Path:
    """Resolve a path relative to this script's directory."""
    path = Path(path_str)
    return path if path.is_absolute() else SCRIPT_DIR / path


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the labeled dataset sorted chronologically.

    Raises:
        RuntimeError: If the file is unreadable or misses required columns.
    """
    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        raise RuntimeError(f"Could not read dataset file {data_path}: {exc}") from exc
    for column in ("published_at", "text", "label"):
        if column not in df.columns:
            raise RuntimeError(
                f"Dataset file {data_path} must contain a '{column}' column."
            )
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at", "text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df.sort_values("published_at").reset_index(drop=True)


def build_pipeline():
    """Build the TF-IDF + LogisticRegression pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    return vectorizer, model, make_pipeline(vectorizer, model)


def main(argv: list[str] | None = None) -> int:
    """Train, evaluate, and persist the news-sentiment model."""
    args = parse_args(argv)
    data_path = resolve(args.data)
    out_path = resolve(args.out)
    if not data_path.exists():
        print(f"Dataset file not found: {data_path}", file=sys.stderr)
        return 1

    try:
        df = load_dataset(data_path)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if len(df) < args.min_rows:
        print(
            f"Not enough training data: {len(df)} rows (minimum {args.min_rows}). "
            "Extend the news backfill with a wider --since window or a "
            "larger --max-items cap, then rebuild the dataset "
            "(or pass --min-rows N for a demo).",
            file=sys.stderr,
        )
        return 1
    classes = sorted(df["label"].unique().tolist())
    if len(classes) < 2:
        print(
            f"Not enough label diversity: only {classes} present "
            "(need at least 2 of BUY / SELL / HOLD).",
            file=sys.stderr,
        )
        return 1

    split_at = int(len(df) * (1 - TEST_FRACTION))
    train_df = df.iloc[:split_at]
    test_df = df.iloc[split_at:]
    if train_df.empty or test_df.empty:
        print("Temporal split produced an empty train or test set.", file=sys.stderr)
        return 1
    if train_df["label"].nunique() < 2:
        print(
            "Training split contains a single class; cannot fit the classifier. "
            "Add more history so every regime appears in the first 80%.",
            file=sys.stderr,
        )
        return 1

    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )
    except ImportError:
        print(
            "The 'scikit-learn' package is not installed. "
            "Install it with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    try:
        import joblib
    except ImportError:
        print(
            "The 'joblib' package is not installed. "
            "Install it with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    vectorizer, model, pipeline = build_pipeline()
    pipeline.fit(train_df["text"], train_df["label"])
    # The fitted steps are the same objects owned by the pipeline.
    predictions = pipeline.predict(test_df["text"])

    accuracy = float(accuracy_score(test_df["label"], predictions))
    macro_f1 = float(f1_score(test_df["label"], predictions, average="macro"))
    report_str = classification_report(test_df["label"], predictions)
    report_dict = classification_report(test_df["label"], predictions, output_dict=True)
    labels = sorted(test_df["label"].unique().tolist())
    matrix = confusion_matrix(test_df["label"], predictions, labels=labels)

    print(f"Train size: {len(train_df)}, test size: {len(test_df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Classification report:")
    print(report_str)
    print(f"Confusion matrix (labels {labels}):")
    print(matrix)

    trained_at = datetime.now(timezone.utc).isoformat()
    bundle = {
        "vectorizer": vectorizer,
        "model": model,
        "classes": classes,
        "threshold_pct": args.threshold_pct,
        "horizon": HORIZON,
        "trained_at": trained_at,
        "train_size": len(train_df),
        "test_metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "test_size": len(test_df),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "classes": classes,
        "test_labels": labels,
        "threshold_pct": args.threshold_pct,
        "horizon": HORIZON,
        "trained_at": trained_at,
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
    }
    if out_path.stem == "model":
        metrics_path = out_path.parent / "metrics.json"
    elif "model" in out_path.stem:
        metrics_path = out_path.parent / (out_path.stem.replace("model", "metrics") + ".json")
    else:
        metrics_path = out_path.parent / (out_path.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model to {out_path} and metrics to {metrics_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

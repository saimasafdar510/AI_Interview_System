"""Experimental TF-IDF answer-text classifier, separate from rubric scoring."""

import csv
import hashlib
import logging
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)

LABELS = ("poor", "average", "good")
DATASET_PATH = Path(__file__).resolve().parent / "data" / "answer_classifier_training.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "experimental_answer_classifier.joblib"
MODEL_FORMAT_VERSION = 1


def load_training_rows(dataset_path=DATASET_PATH):
    """Load and validate the documented, labeled CSV dataset."""
    rows = []
    with Path(dataset_path).open(encoding="utf-8", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        if reader.fieldnames != ["question_group", "answer", "label"]:
            raise ValueError("The classifier dataset has an unexpected column layout.")
        for row in reader:
            group = (row.get("question_group") or "").strip()
            answer = (row.get("answer") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not group or not answer or label not in LABELS:
                raise ValueError("The classifier dataset contains an invalid row.")
            rows.append({"group": group, "answer": answer, "label": label})

    if not rows or set(row["label"] for row in rows) != set(LABELS):
        raise ValueError("The classifier dataset must contain all supported labels.")
    if len({row["group"] for row in rows}) < 4:
        raise ValueError("The classifier dataset needs at least four question groups.")
    return rows


def split_training_rows(rows, *, test_size=0.25, random_state=42):
    """Split by prompt group to keep answers from one prompt in one partition."""
    answers = [row["answer"] for row in rows]
    labels = [row["label"] for row in rows]
    groups = [row["group"] for row in rows]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(splitter.split(answers, labels, groups))
    train_rows = [rows[index] for index in train_indices]
    test_rows = [rows[index] for index in test_indices]
    train_groups = {row["group"] for row in train_rows}
    test_groups = {row["group"] for row in test_rows}
    if train_groups & test_groups:
        raise ValueError("Question groups leaked across the classifier data split.")
    if set(row["label"] for row in train_rows) != set(LABELS):
        raise ValueError("The training partition must contain every classifier label.")
    if set(row["label"] for row in test_rows) != set(LABELS):
        raise ValueError("The test partition must contain every classifier label.")
    return train_rows, test_rows


def make_pipeline():
    """Create the text vectorization and linear classification pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def train_and_save_model(dataset_path=DATASET_PATH, model_path=MODEL_PATH):
    """Train on grouped examples, evaluate the held-out prompts, and save metadata."""
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    rows = load_training_rows(dataset_path)
    train_rows, test_rows = split_training_rows(rows)

    pipeline = make_pipeline()
    pipeline.fit(
        [row["answer"] for row in train_rows],
        [row["label"] for row in train_rows],
    )
    test_labels = [row["label"] for row in test_rows]
    predictions = pipeline.predict([row["answer"] for row in test_rows])
    metrics = {
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "train_question_groups": len({row["group"] for row in train_rows}),
        "test_question_groups": len({row["group"] for row in test_rows}),
        "accuracy": round(float(accuracy_score(test_labels, predictions)), 4),
        "macro_precision": round(float(precision_score(
            test_labels, predictions, labels=LABELS, average="macro", zero_division=0
        )), 4),
        "macro_recall": round(float(recall_score(
            test_labels, predictions, labels=LABELS, average="macro", zero_division=0
        )), 4),
        "macro_f1": round(float(f1_score(
            test_labels, predictions, labels=LABELS, average="macro", zero_division=0
        )), 4),
    }
    metadata = {
        "format_version": MODEL_FORMAT_VERSION,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "labels": list(LABELS),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "evaluation": metrics,
        "data_notice": "Small synthetic illustrative dataset; metrics are a pipeline sanity check only.",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"metadata": metadata, "pipeline": pipeline}, model_path)
    return metrics, metadata


def predict_answer_category(answer_text, model_path=MODEL_PATH):
    """Return an experimental category or an explicit non-error availability status."""
    if not isinstance(answer_text, str):
        return {"status": "invalid_input", "predicted_label": None}
    if not answer_text.strip():
        return {"status": "skipped_empty", "predicted_label": None}

    model_path = Path(model_path)
    if not model_path.is_file():
        return {"status": "model_missing", "predicted_label": None}

    try:
        artifact = joblib.load(model_path)
        metadata = artifact.get("metadata", {})
        pipeline = artifact.get("pipeline")
        if metadata.get("format_version") != MODEL_FORMAT_VERSION or pipeline is None:
            return {"status": "model_invalid", "predicted_label": None}
        if metadata.get("sklearn_version") != sklearn.__version__:
            return {"status": "version_mismatch", "predicted_label": None}
        if metadata.get("joblib_version") != joblib.__version__:
            return {"status": "version_mismatch", "predicted_label": None}

        label = str(pipeline.predict([answer_text.strip()])[0])
        if label not in LABELS:
            return {"status": "prediction_invalid", "predicted_label": None}
        return {
            "status": "available",
            "predicted_label": label,
            "model_version": metadata["sklearn_version"],
        }
    except Exception as error:
        logger.warning(
            "Experimental answer classifier prediction failed (%s).",
            type(error).__name__,
        )
        return {"status": "prediction_error", "predicted_label": None}

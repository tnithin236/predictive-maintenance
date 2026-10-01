"""Model evaluation: metrics, comparison table, plots. Nothing is hardcoded."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_EXPLANATIONS = {
    "accuracy": "Share of correct predictions. Misleading on imbalanced data: predicting 'no failure' always can score high.",
    "precision": "Of machines flagged as failing, the share that really failed. Low precision = many false alarms / wasted maintenance.",
    "recall": "Of machines that really failed, the share we caught. Low recall = missed failures (usually the costly error).",
    "f1": "Harmonic mean of precision and recall; a single balance score at the chosen threshold.",
    "roc_auc": "Threshold-independent ranking quality. Can look optimistic when failures are rare.",
    "pr_auc": "Average precision; threshold-independent and more informative than ROC-AUC under class imbalance.",
    "confusion_matrix": "Counts of TN / FP / FN / TP, showing exactly which error types occur.",
}
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def compute_metrics(y_true: Any, y_proba: Any, threshold: float = 0.5) -> dict[str, Any]:
    """Compute all metrics from true labels and predicted probabilities."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int = 300) -> dict[str, list[float]]:
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x, y = x[idx], y[idx]
    return {"x": [round(float(v), 5) for v in x], "y": [round(float(v), 5) for v in y]}


def curve_data(y_true: Any, y_proba: Any) -> dict[str, dict[str, list[float]]]:
    """ROC and precision-recall curve points (downsampled for storage)."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return {"roc": _downsample(fpr, tpr), "pr": _downsample(recall, precision)}


def evaluate_models(
    models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
) -> tuple[dict[str, dict], pd.DataFrame, dict[str, dict]]:
    """Evaluate fitted pipelines on the held-out test set."""
    results, curves = {}, {}
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = compute_metrics(y_test, proba, threshold)
        curves[name] = curve_data(y_test, proba)
    table = pd.DataFrame(
        [{"model": n, **{m: r[m] for m in METRIC_COLUMNS}} for n, r in results.items()]
    )
    return results, table, curves


def comparison_markdown(table: pd.DataFrame) -> str:
    """Render the comparison table as Markdown (no extra dependencies)."""
    header = "| " + " | ".join(table.columns) + " |"
    sep = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = [
        "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |"
        for r in table.itertuples(index=False)
    ]
    return "\n".join([header, sep, *rows])


def plot_confusion_matrix(cm: dict[str, int], path: Path, title: str) -> None:
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(curves: dict[str, dict], roc_path: Path, pr_path: Path) -> None:
    for key, path, xlabel, ylabel, title in [
        ("roc", roc_path, "False positive rate", "True positive rate", "ROC curve"),
        ("pr", pr_path, "Recall", "Precision", "Precision-recall curve"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, data in curves.items():
            ax.plot(data[key]["x"], data[key]["y"], label=name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)


def save_evaluation(
    results: dict, table: pd.DataFrame, curves: dict, reports_dir: Path,
    served_model: str, extra: dict[str, Any] | None = None,
) -> None:
    """Persist metrics.json, comparison table (csv + md) and evaluation plots."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "served_model": served_model,
        "models": results,
        "curves": curves,
        "metric_explanations": METRIC_EXPLANATIONS,
        **(extra or {}),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    table.to_csv(reports_dir / "model_comparison.csv", index=False)
    (reports_dir / "model_comparison.md").write_text(comparison_markdown(table) + "\n")
    plot_confusion_matrix(
        results[served_model]["confusion_matrix"], reports_dir / "confusion_matrix.png", served_model
    )
    plot_curves(curves, reports_dir / "roc_curve.png", reports_dir / "pr_curve.png")

"""SHAP explanations for the XGBoost failure model.

Explanations are model-derived feature contributions on the log-odds scale.
They describe what the MODEL used, not guaranteed causes of failure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXPLANATION_DISCLAIMER = (
    "Model-derived feature contributions (SHAP, log-odds scale); "
    "not guaranteed causal explanations."
)


def _clean(name: str) -> str:
    return name.split("__", 1)[1] if "__" in name else name


class ShapExplainer:
    """Wraps a fitted ``Pipeline(preprocess -> clf)`` with a SHAP TreeExplainer."""

    def __init__(self, model: Any) -> None:
        import shap  # imported lazily so non-explain code paths do not need SHAP

        self.preprocessor = model.named_steps["preprocess"]
        self.explainer = shap.TreeExplainer(model.named_steps["clf"])
        self.feature_names = [_clean(n) for n in self.preprocessor.get_feature_names_out()]

    def _values(self, features: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        transformed = self.preprocessor.transform(features)
        values = self.explainer.shap_values(transformed)
        if isinstance(values, list):  # older SHAP: one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., 1]
        return values, transformed

    def explain_rows(self, features: pd.DataFrame, top_k: int = 3) -> list[dict[str, Any]]:
        """Per-row ``top_factors`` (risk-increasing) and ``factor_details`` (by |SHAP|)."""
        values, _ = self._values(features)
        out = []
        for i in range(len(features)):
            row = features.iloc[i]
            contribs = sorted(
                zip(self.feature_names, values[i]), key=lambda kv: abs(kv[1]), reverse=True
            )
            details = [
                {
                    "feature": name,
                    "contribution": round(float(val), 5),
                    "value": (row[name].item() if name in row.index and hasattr(row[name], "item")
                              else (row[name] if name in row.index else None)),
                    "direction": "increases_failure_risk" if val > 0 else "decreases_failure_risk",
                }
                for name, val in contribs[:top_k]
            ]
            positive = [n for n, v in sorted(zip(self.feature_names, values[i]),
                                             key=lambda kv: kv[1], reverse=True) if v > 0][:top_k]
            out.append({"top_factors": positive, "factor_details": details})
        return out

    def global_importance(self, features: pd.DataFrame) -> pd.DataFrame:
        """Mean absolute SHAP value per feature over ``features``."""
        values, _ = self._values(features)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


def save_global_importance(importance: pd.DataFrame, reports_dir: Path) -> None:
    """Save SHAP global importance as JSON (dashboard) and a bar plot (README)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "shap_global_importance.json").write_text(
        json.dumps({"disclaimer": EXPLANATION_DISCLAIMER,
                    "importance": importance.to_dict(orient="records")}, indent=2)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    top = importance.head(12).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("mean |SHAP value| (log-odds)")
    ax.set_title("Global feature importance (model-derived)")
    fig.tight_layout()
    fig.savefig(reports_dir / "shap_global_importance.png", dpi=130)
    plt.close(fig)

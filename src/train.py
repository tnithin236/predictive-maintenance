"""Training entry point (training and inference are separate).

Usage:
    python -m src.train                    # full run on data/raw/ai4i2020.csv
    python -m src.train --n-iter 5 --no-mlflow
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config
from src.anomaly_detection import build_isolation_forest, score_anomalies, summarize_against_labels
from src.data_loader import load_raw_data
from src.evaluate import evaluate_models, save_evaluation
from src.explainability import ShapExplainer, save_global_importance
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    FEATURE_DOCS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from src.preprocessing import (
    build_preprocessor,
    clean_data,
    get_xy,
    rename_columns,
    split_data,
    summarize_dataset,
    validate_columns,
)

logger = logging.getLogger(__name__)

XGB_PARAM_DISTRIBUTIONS: dict[str, list] = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [3, 4, 5, 6],
    "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.7, 0.85, 1.0],
    "clf__min_child_weight": [1, 3, 5],
}


def _pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("clf", estimator)])


def build_baselines(seed: int) -> dict[str, Pipeline]:
    """Logistic Regression and Random Forest baselines (class-weighted)."""
    return {
        "Logistic Regression": _pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        ),
        "Random Forest": _pipeline(
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed
            )
        ),
    }


def tune_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int, n_iter: int, cv_folds: int
) -> RandomizedSearchCV:
    """RandomizedSearchCV over a deliberately small XGBoost space (scored by PR-AUC)."""
    from xgboost import XGBClassifier

    positives = int(y_train.sum())
    scale_pos_weight = (len(y_train) - positives) / max(positives, 1)
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        _pipeline(xgb),
        param_distributions=XGB_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        random_state=seed,
        refit=True,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search


def _simple_params(model: Pipeline) -> dict[str, Any]:
    params = model.named_steps["clf"].get_params()
    return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}


def _log_mlflow(
    models: dict[str, Pipeline], results: dict, best_params: dict, feature_config: dict,
    reports_dir: Path, trained_at: str, seed: int, served: str,
) -> str | None:
    """Log one MLflow run per model. Returns the run id of the served model."""
    import mlflow
    import mlflow.sklearn

    mlruns_dir = config.PROJECT_ROOT / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.MlflowClient().get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT_NAME, artifact_location=mlruns_dir.as_uri())
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    served_run_id = None
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            params = best_params if name == served else _simple_params(model)
            mlflow.log_params({"model": name, "random_seed": seed, **{k: str(v) for k, v in params.items()}})
            mlflow.log_metrics({k: v for k, v in results[name].items() if isinstance(v, float) and k != "threshold"})
            mlflow.set_tags({"trained_at": trained_at, "model_name": name})
            mlflow.log_dict(feature_config, "feature_config.json")
            # cloudpickle: MLflow 3's default "skops" format rejects numpy.dtype inside the pipeline
            mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
            if name == served:
                mlflow.log_artifacts(str(reports_dir), artifact_path="reports")
                served_run_id = run.info.run_id
    return served_run_id


def run_training(
    raw_df: pd.DataFrame | None = None,
    model_dir: Path | None = None,
    reports_dir: Path | None = None,
    n_iter: int | None = None,
    cv_folds: int | None = None,
    log_to_mlflow: bool = True,
    processed_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full training pipeline and save the inference bundle."""
    seed = config.RANDOM_SEED
    model_dir = Path(model_dir or config.MODEL_DIR)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    n_iter = n_iter or config.TUNING_ITERATIONS
    cv_folds = cv_folds or config.CV_FOLDS
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data() if raw_df is None else raw_df
    df = rename_columns(raw_df)
    validate_columns(df)
    df, cleaning_report = clean_data(df)
    (reports_dir / "dataset_summary.json").write_text(
        json.dumps(summarize_dataset(df, cleaning_report), indent=2)
    )

    train_df, test_df = split_data(df)
    if processed_path:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([train_df.assign(split="train"), test_df.assign(split="test")]).sort_index().to_csv(
            processed_path, index=False
        )
    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)
    logger.info("Train %s / test %s; failure rate train=%.4f test=%.4f",
                X_train.shape, X_test.shape, y_train.mean(), y_test.mean())

    models = build_baselines(seed)
    for name, model in models.items():
        logger.info("Fitting baseline: %s", name)
        model.fit(X_train, y_train)

    logger.info("Tuning XGBoost (n_iter=%d, cv=%d)", n_iter, cv_folds)
    search = tune_xgboost(X_train, y_train, seed, n_iter, cv_folds)
    served = "XGBoost"
    models[served] = search.best_estimator_
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    (reports_dir / "best_params.json").write_text(
        json.dumps({"best_params": best_params, "cv_pr_auc": float(search.best_score_)}, indent=2, default=str)
    )

    results, table, curves = evaluate_models(models, X_test, y_test, config.DECISION_THRESHOLD)
    save_evaluation(
        results, table, curves, reports_dir, served,
        extra={"test_rows": int(len(y_test)), "test_failures": int(y_test.sum()),
               "cv_pr_auc_best_xgboost": float(search.best_score_)},
    )

    anomaly_model = build_isolation_forest().fit(X_train)
    scored = score_anomalies(anomaly_model, X_test, test_df["machine_id"])
    (reports_dir / "anomaly_summary.json").write_text(
        json.dumps(summarize_against_labels(scored, y_test), indent=2)
    )

    explainer = ShapExplainer(models[served])
    save_global_importance(explainer.global_importance(X_test.head(1000)), reports_dir)

    now = datetime.now(timezone.utc)
    trained_at = now.isoformat()
    feature_config = {
        "numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
        "model_features": MODEL_FEATURES, "engineered": {f: FEATURE_DOCS[f] for f in ENGINEERED_FEATURES},
        "excluded_leakage_columns": config.LEAKAGE_COLUMNS,
    }
    mlflow_run_id = None
    if log_to_mlflow:
        try:
            mlflow_run_id = _log_mlflow(models, results, best_params, feature_config,
                                        reports_dir, trained_at, seed, served)
        except Exception:  # noqa: BLE001 - surfaced loudly, training result is still saved
            logger.exception("MLflow logging FAILED; the model bundle is still saved.")

    import xgboost

    bundle = {
        "model": models[served],
        "anomaly_model": anomaly_model,
        "feature_config": feature_config,
        "thresholds": {"decision": config.DECISION_THRESHOLD, "risk_medium": config.RISK_MEDIUM_THRESHOLD,
                       "risk_high": config.RISK_HIGH_THRESHOLD},
        "model_version": "v" + now.strftime("%Y%m%d%H%M%S"),
        "trained_at": trained_at,
        "best_params": best_params,
        "test_metrics": results[served],
        "mlflow_run_id": mlflow_run_id,
        "random_seed": seed,
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                             "xgboost": xgboost.__version__, "pandas": pd.__version__},
    }
    bundle_path = model_dir / config.MODEL_PATH.name
    joblib.dump(bundle, bundle_path)
    logger.info("Saved bundle to %s", bundle_path)
    return {"bundle_path": str(bundle_path), "results": results, "comparison": table,
            "mlflow_run_id": mlflow_run_id, "model_version": bundle["model_version"]}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    out = run_training(n_iter=args.n_iter, cv_folds=args.cv_folds,
                       log_to_mlflow=not args.no_mlflow, processed_path=config.PROCESSED_DATA_FILE)
    print(out["comparison"].to_string(index=False))
    print("Model saved:", out["bundle_path"])


if __name__ == "__main__":
    main()
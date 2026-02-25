# adult_income_end2end_pipeline.py
#
# End-to-end KFP v2 pipeline for UCI Adult Income:
# preprocess -> train+eval (scikit-learn)
#
# Output:
#   pipeline/adult_income_end2end.yaml
#
# Why this version works reliably:
# - Uses python:3.11-slim (no missing pytorch image tags).
# - Uses correct KFP v2 artifact types: Dataset/Model/Metrics (no "str" artifact errors).
# - Writes model with joblib, logs metrics into Metrics artifact metadata.

import os
from kfp import compiler, dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

BASE_IMAGE = "python:3.11-slim"


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "numpy==1.26.4",
        "scikit-learn==1.4.2",
        "joblib==1.4.2",
    ],
)
def preprocess_op(
    dataset_url: str,
    output_dataset: Output[Dataset],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> None:
    """Download Adult dataset, preprocess (OHE + scale), split train/val/test, save .npz to output_dataset.path."""
    import csv
    import os
    import urllib.request
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib

    adult_data_url = dataset_url.rstrip("/") + "/adult.data"
    adult_test_url = dataset_url.rstrip("/") + "/adult.test"

    def _download_text(url: str) -> str:
        with urllib.request.urlopen(url) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def _parse_adult_csv(text: str, is_test: bool) -> list[list[str]]:
        lines: list[str] = []
        for i, raw in enumerate(text.splitlines()):
            raw = raw.strip()
            if not raw or raw.startswith("|"):
                continue
            if is_test and i == 0 and raw.lower().startswith("age"):
                continue
            lines.append(raw)

        rows: list[list[str]] = []
        reader = csv.reader(lines, delimiter=",", skipinitialspace=True)
        for row in reader:
            if len(row) != 15:
                continue
            if is_test:
                row[-1] = row[-1].rstrip(".")
            rows.append(row)
        return rows

    data_text = _download_text(adult_data_url)
    test_text = _download_text(adult_test_url)

    train_rows = _parse_adult_csv(data_text, is_test=False)
    test_rows = _parse_adult_csv(test_text, is_test=True)

    all_rows = train_rows + test_rows
    X_raw = [r[:-1] for r in all_rows]
    y_raw = [r[-1] for r in all_rows]

    y = np.array([1 if v.strip() == ">50K" else 0 for v in y_raw], dtype=np.int64)

    # Numeric / categorical indices (Adult dataset)
    numeric_idx = [0, 2, 4, 10, 11, 12]
    categorical_idx = [1, 3, 5, 6, 7, 8, 9, 13]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_idx),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_idx),
        ],
        remainder="drop",
    )

    X_obj = np.array(X_raw, dtype=object)
    X = preprocessor.fit_transform(X_obj).astype(np.float32)

    # Split train vs (val+test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=y,
    )

    # Split tmp into val/test
    val_frac = val_size / max(1e-9, (val_size + test_size))
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=(1.0 - val_frac),
        random_state=random_state,
        stratify=y_tmp,
    )

    os.makedirs(output_dataset.path, exist_ok=True)

    # Save arrays compactly
    np.savez_compressed(
        os.path.join(output_dataset.path, "splits.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )

    # Save preprocessor too (optional but useful)
    joblib.dump(preprocessor, os.path.join(output_dataset.path, "preprocessor.joblib"))


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "numpy==1.26.4",
        "scikit-learn==1.4.2",
        "joblib==1.4.2",
    ],
)
def train_eval_op(
    dataset: Input[Dataset],
    model: Output[Model],
    metrics_out: Output[Metrics],
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 128,
    dropout: float = 0.2,
    random_state: int = 42,
) -> None:
    """Train an MLPClassifier (sklearn) + log val/test metrics + save model.joblib."""
    import os
    import numpy as np
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    data = np.load(os.path.join(dataset.path, "splits.npz"))
    X_train = data["X_train"]; y_train = data["y_train"]
    X_val = data["X_val"]; y_val = data["y_val"]
    X_test = data["X_test"]; y_test = data["y_test"]

    clf = MLPClassifier(
        hidden_layer_sizes=(hidden, hidden),
        activation="relu",
        solver="adam",
        alpha=dropout,              # not true dropout, but keeps your "dropout" knob meaningful
        batch_size=batch_size,
        learning_rate_init=lr,
        max_iter=epochs,
        early_stopping=True,
        n_iter_no_change=5,
        tol=1e-4,
        random_state=random_state,
        verbose=True,               # prints "Iteration X, loss = ..."
    )

    clf.fit(X_train, y_train)

    # Validation metrics
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    val_acc = float(accuracy_score(y_val, val_pred))
    val_f1 = float(f1_score(y_val, val_pred))
    val_ll = float(log_loss(y_val, val_proba))

    # Test metrics
    test_proba = clf.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    test_acc = float(accuracy_score(y_test, test_pred))
    test_f1 = float(f1_score(y_test, test_pred))

    # Log metrics into the Metrics artifact metadata
    metrics_out.log_metric("val_accuracy", val_acc)
    metrics_out.log_metric("val_f1", val_f1)
    metrics_out.log_metric("val_log_loss", val_ll)
    metrics_out.log_metric("test_accuracy", test_acc)
    metrics_out.log_metric("test_f1", test_f1)

    # Save model artifact
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(clf, os.path.join(model.path, "model.joblib"))


@dsl.pipeline(name="adult-income-end2end")
def adult_income_end2end_pipeline(
    dataset_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 128,
    dropout: float = 0.2,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    prep = preprocess_op(
        dataset_url=dataset_url,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    train_eval_op(
        dataset=prep.outputs["output_dataset"],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden=hidden,
        dropout=dropout,
        random_state=random_state,
    )


def main() -> None:
    os.makedirs("pipeline", exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=adult_income_end2end_pipeline,
        package_path="pipeline/adult_income_end2end.yaml",
    )
    print("✅ Compiled to pipeline/adult_income_end2end.yaml")


if __name__ == "__main__":
    main()

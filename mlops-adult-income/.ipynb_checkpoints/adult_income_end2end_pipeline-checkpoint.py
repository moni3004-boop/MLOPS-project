# adult_income_end2end_pipeline.py
#
# End-to-end KFP v2 pipeline for UCI Adult Income:
# preprocess -> train+eval (PyTorch)
#
# Output:
#   pipeline/adult_income_end2end.yaml
#
# Reliable on your cluster:
# - Uses python:3.11-slim (avoids missing/huge pytorch/pytorch tags)
# - Uses correct KFP v2 artifact types: Dataset/Model/Metrics
# - Installs CPU-only torch inside the component (no image pull)
# - Logs metrics to Metrics artifact metadata -> visible in UI

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
    """Download Adult dataset, preprocess (OHE + scale), split train/val/test, save splits + preprocessor."""
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

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=y,
    )

    val_frac = val_size / max(1e-9, (val_size + test_size))
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=(1.0 - val_frac),
        random_state=random_state,
        stratify=y_tmp,
    )

    os.makedirs(output_dataset.path, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dataset.path, "splits.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    joblib.dump(preprocessor, os.path.join(output_dataset.path, "preprocessor.joblib"))


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "numpy==1.26.4",
        "scikit-learn==1.4.2",
        "joblib==1.4.2",
    ],
)
def train_eval_torch_op(
    dataset: Input[Dataset],
    model: Output[Model],
    metrics_out: Output[Metrics],
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 128,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
) -> None:
    """
    Train a simple PyTorch MLP for binary classification.
    Logs val/test metrics and saves TorchScript model to model.path/model.pt
    """
    import os
    import json
    import subprocess
    import numpy as np

    # IMPORTANT: must be defined inside the function (KFP only copies function body)
    TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

    # Install CPU-only torch
    subprocess.check_call([
        "python3", "-m", "pip", "install",
        "--quiet", "--no-warn-script-location",
        "--index-url", TORCH_CPU_INDEX,
        "torch==2.1.2",
    ])

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    def set_seed(s: int) -> None:
        import random
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    set_seed(seed)
    device = torch.device("cpu")

    data = np.load(os.path.join(dataset.path, "splits.npz"))
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train)
    Xv = torch.from_numpy(X_val)
    yv = torch.from_numpy(y_val)
    Xt = torch.from_numpy(X_test)
    yt = torch.from_numpy(y_test)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size, shuffle=False)

    in_dim = X_train.shape[1]

    class MLP(nn.Module):
        def __init__(self, d_in: int, d_hidden: int, p_drop: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(d_hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    net = MLP(in_dim, hidden, dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    last_val_acc = 0.0
    last_val_f1 = 0.0

    for ep in range(1, epochs + 1):
        net.train()
        total = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu().item()) * xb.shape[0]
            n += xb.shape[0]

        train_loss = total / max(1, n)

        net.eval()
        v_logits_all = []
        v_y_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = net(xb)
                v_logits_all.append(logits.cpu())
                v_y_all.append(yb)

        v_logits = torch.cat(v_logits_all, dim=0).numpy()
        v_y = torch.cat(v_y_all, dim=0).numpy()

        v_proba = 1.0 / (1.0 + np.exp(-v_logits))
        v_pred = (v_proba >= 0.5).astype(int)

        val_acc = float(accuracy_score(v_y.astype(int), v_pred))
        val_f1 = float(f1_score(v_y.astype(int), v_pred))
        val_ll = float(log_loss(v_y.astype(int), v_proba))

        last_val_acc = val_acc
        last_val_f1 = val_f1

        print(
            f"epoch={ep}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_log_loss={val_ll:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        if val_ll + 1e-6 < best_val_loss:
            best_val_loss = val_ll
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: no improvement for {patience} epochs.")
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        t_logits = net(Xt.to(device)).cpu().numpy()

    t_proba = 1.0 / (1.0 + np.exp(-t_logits))
    t_pred = (t_proba >= 0.5).astype(int)

    test_acc = float(accuracy_score(y_test.astype(int), t_pred))
    test_f1 = float(f1_score(y_test.astype(int), t_pred))

    metrics_out.log_metric("val_log_loss_best", float(best_val_loss))
    metrics_out.log_metric("val_accuracy", float(last_val_acc))
    metrics_out.log_metric("val_f1", float(last_val_f1))
    metrics_out.log_metric("test_accuracy", float(test_acc))
    metrics_out.log_metric("test_f1", float(test_f1))

    os.makedirs(model.path, exist_ok=True)
    example = torch.zeros((1, in_dim), dtype=torch.float32).to(device)
    scripted = torch.jit.trace(net, example)
    scripted.save(os.path.join(model.path, "model.pt"))

    meta = {
        "model_type": "pytorch_mlp",
        "in_dim": int(in_dim),
        "hidden": int(hidden),
        "dropout": float(dropout),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "epochs_requested": int(epochs),
        "early_stop_patience": int(patience),
        "weight_decay": float(weight_decay),
    }
    with open(os.path.join(model.path, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


@dsl.pipeline(name="adult-income-end2end")
def adult_income_end2end_pipeline(
    dataset_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 128,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    patience: int = 5,
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

    train_eval_torch_op(
        dataset=prep.outputs["output_dataset"],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden=hidden,
        dropout=dropout,
        weight_decay=weight_decay,
        patience=patience,
        seed=random_state,
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

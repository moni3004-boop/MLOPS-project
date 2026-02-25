# adult_income_end2end_pipeline.py
#
# End-to-end KFP v2 pipeline for UCI Adult Income:
# preprocess -> tune -> train+eval -> generate serving manifest -> monitor/retrain gate -> (optional) retrain
#
# Output:
#   pipeline/adult_income_end2end.yaml

import os
from kfp import compiler, dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, Artifact

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
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    joblib.dump(preprocessor, os.path.join(output_dataset.path, "preprocessor.joblib"))


def _install_torch_cpu() -> None:
    import subprocess
    import sys

    TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-warn-script-location",
            "--index-url",
            TORCH_CPU_INDEX,
            "torch",
        ]
    )


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "numpy==1.26.4",
        "scikit-learn==1.4.2",
        "joblib==1.4.2",
    ],
)
def tune_op(
    dataset: Input[Dataset],
    tuning_metrics: Output[Metrics],
    best_params: Output[Artifact],
    trials: int = 10,
    epochs: int = 10,
    seed: int = 42,
) -> None:
    """
    Lightweight random search (Katib-like).
    Writes best_params.json to best_params.path.
    Prints metrics as "name=value" for easy regex.
    """
    import json
    import os
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    lr_space = [0.0005, 0.001, 0.002, 0.003]
    hidden_space = [64, 128, 256]
    dropout_space = [0.0, 0.1, 0.2, 0.3]
    batch_space = [128, 256, 512]

    _install_torch_cpu()

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    data = np.load(os.path.join(dataset.path, "splits.npz"))
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train)
    Xv = torch.from_numpy(X_val)

    in_dim = X_train.shape[1]
    device = torch.device("cpu")

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

    best = None
    best_score = -1.0  # maximize val_f1

    for t in range(1, trials + 1):
        lr = random.choice(lr_space)
        hidden = random.choice(hidden_space)
        dropout = random.choice(dropout_space)
        batch_size = random.choice(batch_space)

        net = MLP(in_dim, hidden, dropout).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            net.train()
            for xb, yb in train_loader:
                opt.zero_grad(set_to_none=True)
                logits = net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        net.eval()
        with torch.no_grad():
            v_logits = net(Xv).cpu().numpy()

        v_proba = 1.0 / (1.0 + np.exp(-v_logits))
        v_pred = (v_proba >= 0.5).astype(int)

        val_acc = float(accuracy_score(y_val, v_pred))
        val_f1 = float(f1_score(y_val, v_pred))
        val_ll = float(log_loss(y_val, v_proba))

        print(
            f"trial={t}/{trials} lr={lr} hidden={hidden} drop={dropout} bs={batch_size} "
            f"val_accuracy={val_acc:.6f} val_f1={val_f1:.6f} val_log_loss={val_ll:.6f}"
        )

        if val_f1 > best_score:
            best_score = val_f1
            best = {
                "lr": lr,
                "hidden": hidden,
                "dropout": dropout,
                "batch_size": batch_size,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "val_log_loss": val_ll,
                "trials": trials,
                "epochs_per_trial": epochs,
            }

    os.makedirs(best_params.path, exist_ok=True)
    with open(os.path.join(best_params.path, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    tuning_metrics.log_metric("best_val_accuracy", float(best["val_accuracy"]))
    tuning_metrics.log_metric("best_val_f1", float(best["val_f1"]))
    tuning_metrics.log_metric("best_val_log_loss", float(best["val_log_loss"]))

    # Optional stdout metrics
    print(f"best_val_accuracy={float(best['val_accuracy']):.6f}")
    print(f"best_val_f1={float(best['val_f1']):.6f}")
    print(f"best_val_log_loss={float(best['val_log_loss']):.6f}")


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
    best_params: Input[Artifact],
    model: Output[Model],
    metrics_out: Output[Metrics],
    epochs: int = 20,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
) -> None:
    """
    Train/eval using best_params.json from tune_op.
    Saves TorchScript model to model.path/model.pt.
    Logs metrics to Metrics artifact AND also writes metrics.json so Kubeflow UI always shows Visualization.
    """
    import os
    import json
    import numpy as np

    _install_torch_cpu()

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

    with open(os.path.join(best_params.path, "best_params.json"), "r", encoding="utf-8") as f:
        bp = json.load(f)

    lr = float(bp["lr"])
    hidden = int(bp["hidden"])
    dropout = float(bp["dropout"])
    batch_size = int(bp["batch_size"])

    data = np.load(os.path.join(dataset.path, "splits.npz"))
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train)
    Xv = torch.from_numpy(X_val)
    yv = torch.from_numpy(y_val.astype(np.float32))
    Xt = torch.from_numpy(X_test)

    in_dim = X_train.shape[1]
    device = torch.device("cpu")

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

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    last_val_acc = 0.0
    last_val_f1 = 0.0

    for _ep in range(1, epochs + 1):
        net.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        net.eval()
        v_logits_all = []
        v_y_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                v_logits_all.append(net(xb).cpu())
                v_y_all.append(yb.cpu())

        v_logits = torch.cat(v_logits_all, dim=0).numpy()
        v_y_float = torch.cat(v_y_all, dim=0).numpy()
        v_y = (v_y_float >= 0.5).astype(int)

        v_proba = 1.0 / (1.0 + np.exp(-v_logits))
        v_pred = (v_proba >= 0.5).astype(int)

        val_acc = float(accuracy_score(v_y, v_pred))
        val_f1 = float(f1_score(v_y, v_pred))
        val_ll = float(log_loss(v_y, v_proba))

        last_val_acc = val_acc
        last_val_f1 = val_f1

        if val_ll + 1e-6 < best_val_loss:
            best_val_loss = val_ll
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    # TEST
    net.eval()
    with torch.no_grad():
        t_logits = net(Xt).cpu().numpy()

    t_proba = 1.0 / (1.0 + np.exp(-t_logits))
    t_pred = (t_proba >= 0.5).astype(int)

    test_acc = float(accuracy_score(y_test, t_pred))
    test_f1 = float(f1_score(y_test, t_pred))

    # Log to Metrics artifact (KFP / MLMD)
    metrics_out.log_metric("val_log_loss_best", float(best_val_loss))
    metrics_out.log_metric("val_accuracy", float(last_val_acc))
    metrics_out.log_metric("val_f1", float(last_val_f1))
    metrics_out.log_metric("test_accuracy", float(test_acc))
    metrics_out.log_metric("test_f1", float(test_f1))

    # StdOut metrics (optional)
    print(f"val_log_loss_best={float(best_val_loss):.6f}")
    print(f"val_accuracy={float(last_val_acc):.6f}")
    print(f"val_f1={float(last_val_f1):.6f}")
    print(f"test_accuracy={float(test_acc):.6f}")
    print(f"test_f1={float(test_f1):.6f}")

    # ✅ FORCE UI VISUALIZATION: write metrics.json next to the metrics artifact
    os.makedirs(metrics_out.path, exist_ok=True)
    metrics_payload = {
        "metrics": [
            {"name": "val_log_loss_best", "numberValue": float(best_val_loss)},
            {"name": "val_accuracy", "numberValue": float(last_val_acc)},
            {"name": "val_f1", "numberValue": float(last_val_f1)},
            {"name": "test_accuracy", "numberValue": float(test_acc)},
            {"name": "test_f1", "numberValue": float(test_f1)},
        ]
    }
    with open(os.path.join(metrics_out.path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f)

    # Save TorchScript
    os.makedirs(model.path, exist_ok=True)
    example = torch.zeros((1, in_dim), dtype=torch.float32).to(device)
    scripted = torch.jit.trace(net, example)
    scripted.save(os.path.join(model.path, "model.pt"))


@component(base_image=BASE_IMAGE)
def build_serving_manifest_op(
    image: str,
    out_manifest: Output[Artifact],
) -> None:
    import os

    os.makedirs(out_manifest.path, exist_ok=True)
    manifest = f"""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adult-income-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: adult-income-api
  template:
    metadata:
      labels:
        app: adult-income-api
    spec:
      containers:
        - name: api
          image: {image}
          ports:
            - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: adult-income-api
spec:
  selector:
    app: adult-income-api
  ports:
    - port: 80
      targetPort: 8080
"""
    with open(os.path.join(out_manifest.path, "serving.yaml"), "w", encoding="utf-8") as f:
        f.write(manifest)


@component(base_image=BASE_IMAGE)
def monitor_and_maybe_retrain_op(
    metrics: Input[Metrics],
    retrain_triggered: Output[Artifact],
    min_test_accuracy: float = 0.84,
) -> None:
    import os
    import json

    m = metrics.metadata or {}
    test_acc = float(m.get("test_accuracy", 0.0))

    os.makedirs(retrain_triggered.path, exist_ok=True)
    out = {
        "test_accuracy": test_acc,
        "min_test_accuracy": float(min_test_accuracy),
        "retrain": bool(test_acc < min_test_accuracy),
        "reason": "below_threshold" if test_acc < min_test_accuracy else "ok",
    }
    with open(os.path.join(retrain_triggered.path, "monitor.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"monitor_test_accuracy={test_acc:.6f}")
    print(f"monitor_retrain={(1 if test_acc < min_test_accuracy else 0)}")


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "numpy==1.26.4",
        "scikit-learn==1.4.2",
        "joblib==1.4.2",
    ],
)
def retrain_op(
    dataset: Input[Dataset],
    best_params: Input[Artifact],
    monitor: Input[Artifact],
    retrained_model: Output[Model],
    retrain_metrics: Output[Metrics],
    epochs: int = 10,
    seed: int = 42,
) -> None:
    import os
    import json
    import numpy as np

    with open(os.path.join(monitor.path, "monitor.json"), "r", encoding="utf-8") as f:
        mon = json.load(f)

    if not bool(mon.get("retrain", False)):
        retrain_metrics.log_metric("retrain_skipped", 1.0)
        retrain_metrics.log_metric("retrain_reason_ok", 1.0)
        os.makedirs(retrained_model.path, exist_ok=True)
        with open(os.path.join(retrained_model.path, "SKIPPED.txt"), "w", encoding="utf-8") as wf:
            wf.write("retrain=false; skipping retrain_op\n")
        print("retrain_skipped=1")
        return

    _install_torch_cpu()

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import accuracy_score, f1_score

    def set_seed(s: int) -> None:
        import random
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    set_seed(seed)

    with open(os.path.join(best_params.path, "best_params.json"), "r", encoding="utf-8") as f:
        bp = json.load(f)

    lr = float(bp["lr"])
    hidden = int(bp["hidden"])
    dropout = float(bp["dropout"])
    batch_size = int(bp["batch_size"])

    data = np.load(os.path.join(dataset.path, "splits.npz"))
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train)
    Xt = torch.from_numpy(X_test)

    in_dim = X_train.shape[1]
    device = torch.device("cpu")

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
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        net.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        t_logits = net(Xt).cpu().numpy()

    t_proba = 1.0 / (1.0 + np.exp(-t_logits))
    t_pred = (t_proba >= 0.5).astype(int)

    test_acc = float(accuracy_score(y_test, t_pred))
    test_f1 = float(f1_score(y_test, t_pred))

    retrain_metrics.log_metric("retrain_triggered", 1.0)
    retrain_metrics.log_metric("retrain_test_accuracy", test_acc)
    retrain_metrics.log_metric("retrain_test_f1", test_f1)

    print("retrain_triggered=1")
    print(f"retrain_test_accuracy={test_acc:.6f}")
    print(f"retrain_test_f1={test_f1:.6f}")

    os.makedirs(retrained_model.path, exist_ok=True)
    example = torch.zeros((1, in_dim), dtype=torch.float32).to(device)
    scripted = torch.jit.trace(net, example)
    scripted.save(os.path.join(retrained_model.path, "model.pt"))


@dsl.pipeline(name="adult-income-end2end")
def adult_income_end2end_pipeline(
    dataset_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult",
    trials: int = 10,
    tune_epochs: int = 5,
    train_epochs: int = 20,
    retrain_epochs: int = 10,
    serving_image: str = "adult-income-api:local",
    min_test_accuracy: float = 0.84,
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

    tune = tune_op(
        dataset=prep.outputs["output_dataset"],
        trials=trials,
        epochs=tune_epochs,
        seed=random_state,
    )

    train = train_eval_torch_op(
        dataset=prep.outputs["output_dataset"],
        best_params=tune.outputs["best_params"],
        epochs=train_epochs,
        patience=5,
        seed=random_state,
    )

    build_serving_manifest_op(image=serving_image)

    mon = monitor_and_maybe_retrain_op(
        metrics=train.outputs["metrics_out"],
        min_test_accuracy=min_test_accuracy,
    )

    retrain_op(
        dataset=prep.outputs["output_dataset"],
        best_params=tune.outputs["best_params"],
        monitor=mon.outputs["retrain_triggered"],
        epochs=retrain_epochs,
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
from kfp import dsl
from kfp.dsl import Output, Input, Dataset, Artifact, Metrics, component


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"],
)
def preprocess_op(dataset_url: str, output_dataset: Output[Dataset], schema_out: Output[Artifact]):
    import json, os, urllib.request
    import numpy as np
    import pandas as pd
    from dataclasses import asdict, dataclass
    from pathlib import Path
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    @dataclass
    class Schema:
        target: str
        numeric_features: list
        categorical_features: list

    def build_preprocessor(numeric_features, categorical_features):
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_features),
                ("cat", cat_pipe, categorical_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    work = Path("/tmp/adult")
    work.mkdir(parents=True, exist_ok=True)
    train_url = dataset_url.rstrip("/") + "/adult.data"
    test_url  = dataset_url.rstrip("/") + "/adult.test"
    train_path = work / "adult.data"
    test_path  = work / "adult.test"
    urllib.request.urlretrieve(train_url, train_path)
    urllib.request.urlretrieve(test_url, test_path)

    cols = [
      "age","workclass","fnlwgt","education","education_num","marital_status","occupation",
      "relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"
    ]
    df_train = pd.read_csv(train_path, header=None, names=cols, skipinitialspace=True)
    df_test  = pd.read_csv(test_path, header=None, names=cols, skiprows=1, skipinitialspace=True)
    df_test["income"] = df_test["income"].astype(str).str.replace(".", "", regex=False)

    df = pd.concat([df_train, df_test], ignore_index=True).replace("?", pd.NA)

    target_col = "income"
    df = df.dropna(subset=[target_col]).copy()
    y = (df[target_col].astype(str).str.contains(">50K")).astype(int).values
    X_df = df.drop(columns=[target_col]).copy()

    numeric_features = X_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [c for c in X_df.columns if c not in numeric_features]

    schema = Schema(target=target_col, numeric_features=numeric_features, categorical_features=categorical_features)
    pre = build_preprocessor(numeric_features, categorical_features)

    X_train_df, X_tmp_df, y_train, y_tmp = train_test_split(
        X_df, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_tmp_df, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    X_train = pre.fit_transform(X_train_df).astype("float32")
    X_val   = pre.transform(X_val_df).astype("float32")
    X_test  = pre.transform(X_test_df).astype("float32")

    os.makedirs(output_dataset.path, exist_ok=True)
    np.save(os.path.join(output_dataset.path, "X_train.npy"), X_train)
    np.save(os.path.join(output_dataset.path, "y_train.npy"), y_train.astype("int64"))
    np.save(os.path.join(output_dataset.path, "X_val.npy"), X_val)
    np.save(os.path.join(output_dataset.path, "y_val.npy"), y_val.astype("int64"))
    np.save(os.path.join(output_dataset.path, "X_test.npy"), X_test)
    np.save(os.path.join(output_dataset.path, "y_test.npy"), y_test.astype("int64"))

    os.makedirs(schema_out.path, exist_ok=True)
    with open(os.path.join(schema_out.path, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(schema), f, indent=2)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["torch", "numpy", "scikit-learn"],
)
def train_eval_op(dataset: Input[Dataset], model: Output[Artifact], metrics_out: Output[Metrics],
                  epochs: int = 10, batch_size: int = 128, lr: float = 1e-3, hidden: int = 128, dropout: float = 0.2):
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score, f1_score

    class MLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )
        def forward(self, x):
            return self.net(x)

    def batch_acc(logits, y):
        preds = torch.argmax(logits, dim=1)
        return (preds == y).float().mean().item()

    X_train = np.load(os.path.join(dataset.path, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(dataset.path, "y_train.npy")).astype(np.int64)
    X_val   = np.load(os.path.join(dataset.path, "X_val.npy")).astype(np.float32)
    y_val   = np.load(os.path.join(dataset.path, "y_val.npy")).astype(np.int64)
    X_test  = np.load(os.path.join(dataset.path, "X_test.npy")).astype(np.float32)
    y_test  = np.load(os.path.join(dataset.path, "y_test.npy")).astype(np.int64)

    device = "cpu"
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=batch_size, shuffle=False)

    model_nn = MLP(in_dim=X_train.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model_nn.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model_nn.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model_nn(xb), yb)
            loss.backward()
            opt.step()

    # val metrics
    model_nn.eval()
    val_losses, val_accs = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model_nn(xb)
            val_losses.append(loss_fn(logits, yb).item())
            val_accs.append(batch_acc(logits, yb))

    val_loss = float(sum(val_losses) / max(1, len(val_losses)))
    val_acc  = float(sum(val_accs) / max(1, len(val_accs)))

    # test metrics
    with torch.no_grad():
        logits = model_nn(torch.from_numpy(X_test).to(device)).cpu()
        preds = torch.argmax(logits, dim=1).numpy()

    test_acc = float(accuracy_score(y_test, preds))
    test_f1  = float(f1_score(y_test, preds))

    metrics_out.log_metric("val_loss", val_loss)
    metrics_out.log_metric("val_acc", val_acc)
    metrics_out.log_metric("test_accuracy", test_acc)
    metrics_out.log_metric("test_f1", test_f1)

    os.makedirs(model.path, exist_ok=True)
    torch.save(model_nn.state_dict(), os.path.join(model.path, "model.pt"))


@dsl.pipeline(name="adult-income-end2end")
def pipeline(
    dataset_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult",
):
    pre = preprocess_op(dataset_url=dataset_url)
    _ = train_eval_op(dataset=pre.outputs["output_dataset"])


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(pipeline_func=pipeline, package_path="pipeline/adult_income_end2end.yaml")
    print("✅ Compiled to pipeline/adult_income_end2end.yaml")

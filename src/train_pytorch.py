import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COLS = [
    "age","workclass","fnlwgt","education","education_num","marital_status",
    "occupation","relationship","race","sex","capital_gain","capital_loss",
    "hours_per_week","native_country","income"
]


def load_adult(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path, header=None, names=COLS, skipinitialspace=True)
    test_df = pd.read_csv(test_path, header=None, names=COLS, skiprows=1, skipinitialspace=True)

    train_df["income"] = train_df["income"].astype(str).str.strip()
    test_df["income"] = test_df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    return train_df, test_df


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # logits


def _to_dense_float32(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _find_best_f1_threshold(y_true, probs):
    best_thr, best_f1 = 0.5, 0.0
    for thr in [i / 100 for i in range(10, 91, 5)]:
        pred = (probs >= thr).astype(np.int32)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="data/adult.data")
    ap.add_argument("--test_path", type=str, default="data/adult.test")
    ap.add_argument("--out_dir", type=str, default="model_out")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    # IMPORTANT: soften class weighting (default: sqrt(neg/pos))
    ap.add_argument("--pos_weight_mode", choices=["none", "ratio", "sqrt_ratio"], default="sqrt_ratio")
    ap.add_argument("--pos_weight_scale", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df, test_df = load_adult(Path(args.train_path), Path(args.test_path))

    X = train_df.drop(columns=["income"])
    y = (train_df["income"] == ">50K").astype(np.float32)

    X_test = test_df.drop(columns=["income"])
    y_test = (test_df["income"] == ">50K").astype(np.float32)

    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    num_cols = X.select_dtypes(exclude=["object", "string"]).columns

    # Preprocessor: impute + scale numeric, impute + one-hot categorical
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    X_tr_t = _to_dense_float32(preprocessor.fit_transform(X_tr))
    X_val_t = _to_dense_float32(preprocessor.transform(X_val))
    X_test_t = _to_dense_float32(preprocessor.transform(X_test))

    in_features = X_tr_t.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(in_features=in_features, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Class weighting (softened)
    pos = float(np.sum(y_tr))
    neg = float(len(y_tr) - pos)
    ratio = neg / max(pos, 1.0)

    if args.pos_weight_mode == "none":
        pos_weight = 1.0
    elif args.pos_weight_mode == "ratio":
        pos_weight = ratio
    else:  # sqrt_ratio
        pos_weight = float(np.sqrt(ratio))

    pos_weight *= float(args.pos_weight_scale)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    print(f"pos_weight: {pos_weight:.4f} (mode={args.pos_weight_mode}, scale={args.pos_weight_scale}, ratio={ratio:.4f})")

    def batches(Xn, yn, bs):
        n = Xn.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        yn = np.asarray(yn, dtype=np.float32)
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            yield Xn[j], yn[j]

    # Track best model by VAL ACC (keeps sanity), but also record best-F1 threshold
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_thr = 0.5
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in batches(X_tr_t, y_tr, args.batch_size):
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)

            opt.zero_grad()
            logits = model(xb_t)
            loss = loss_fn(logits, yb_t)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val_t).to(device))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        # Metrics at fixed threshold 0.5 (accuracy sanity)
        val_pred_05 = (val_probs >= 0.5).astype(np.int32)
        val_acc_05 = accuracy_score(y_val, val_pred_05)
        val_f1_05 = f1_score(y_val, val_pred_05, zero_division=0)

        # Best threshold for F1 (reported, not used to pick best model)
        epoch_thr, epoch_best_f1 = _find_best_f1_threshold(y_val, val_probs)

        # Keep best model by accuracy@0.5 (stable for assignment)
        if val_acc_05 > best_val_acc:
            best_val_acc = float(val_acc_05)
            best_val_f1 = float(epoch_best_f1)
            best_thr = float(epoch_thr)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | loss={np.mean(train_losses):.4f} | "
            f"val_acc@0.5={val_acc_05:.4f} | val_f1@0.5={val_f1_05:.4f} | "
            f"best_f1={epoch_best_f1:.4f} @thr={epoch_thr:.2f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # TEST
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test_t).to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    # Test metrics at 0.5 (primary)
    test_pred_05 = (test_probs >= 0.5).astype(np.int32)
    test_acc_05 = accuracy_score(y_test, test_pred_05)
    test_f1_05 = f1_score(y_test, test_pred_05, zero_division=0)

    # Test metrics at best threshold (secondary)
    test_pred_best = (test_probs >= best_thr).astype(np.int32)
    test_acc_best = accuracy_score(y_test, test_pred_best)
    test_f1_best = f1_score(y_test, test_pred_best, zero_division=0)

    out_dir = Path(args.out_dir)
    art_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_features": in_features,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "best_threshold": best_thr,
        },
        out_dir / "model.pt",
    )
    joblib.dump(preprocessor, art_dir / "preprocessor.joblib")

    metrics = {
        "model": "pytorch_mlp",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "pos_weight_mode": args.pos_weight_mode,
        "pos_weight_scale": float(args.pos_weight_scale),
        "pos_weight_value": float(pos_weight),

        "val_best_accuracy_at_0_5": float(best_val_acc),
        "val_best_f1_threshold": float(best_thr),
        "val_best_f1": float(best_val_f1),

        "test_accuracy_at_0_5": float(test_acc_05),
        "test_f1_at_0_5": float(test_f1_05),
        "test_accuracy_at_best_thr": float(test_acc_best),
        "test_f1_at_best_thr": float(test_f1_best),

        "device": str(device),
    }
    (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("✅ Saved:", out_dir / "model.pt")
    print("✅ Saved:", art_dir / "preprocessor.joblib")
    print("✅ Saved:", art_dir / "metrics.json")
    print(
        f"🎯 TEST @0.5 acc={test_acc_05:.4f} f1={test_f1_05:.4f} | "
        f"@best_thr({best_thr:.2f}) acc={test_acc_best:.4f} f1={test_f1_best:.4f}"
    )


if __name__ == "__main__":
    main()

# print(f"val_best_accuracy_at_0_5={metrics['val_best_accuracy_at_0_5']}")
# print(f"test_accuracy_at_0_5={metrics['test_accuracy_at_0_5']}")
# print(f"test_f1_at_0_5={metrics['test_f1_at_0_5']}")


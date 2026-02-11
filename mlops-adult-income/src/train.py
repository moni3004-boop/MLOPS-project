import argparse
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
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


@dataclass
class TrainMetrics:
    val_loss: float
    val_acc: float


def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--out_dir", default="model_out")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train = np.load(os.path.join(args.artifacts_dir, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(args.artifacts_dir, "y_train.npy")).astype(np.int64)
    X_val   = np.load(os.path.join(args.artifacts_dir, "X_val.npy")).astype(np.float32)
    y_val   = np.load(os.path.join(args.artifacts_dir, "y_val.npy")).astype(np.int64)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MLP(in_dim=X_train.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        val_accs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_losses.append(loss_fn(logits, yb).item())
                val_accs.append(accuracy(logits, yb))

        print(
            f"Epoch {epoch}: "
            f"val_loss={sum(val_losses)/len(val_losses):.4f} "
            f"val_acc={sum(val_accs)/len(val_accs):.4f}"
        )

    metrics = TrainMetrics(
        val_loss=float(sum(val_losses)/len(val_losses)),
        val_acc=float(sum(val_accs)/len(val_accs)),
    )

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    print("✅ Saved:", os.path.join(args.out_dir, "model.pt"))
    print("✅ Metrics:", metrics)


if __name__ == "__main__":
    main()

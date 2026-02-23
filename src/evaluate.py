import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--out_json", default="eval_metrics.json")
    args = p.parse_args()

    X_test = np.load(os.path.join(args.artifacts_dir, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(args.artifacts_dir, "y_test.npy")).astype(np.int64)

    device = "cpu"
    model = MLP(in_dim=X_test.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    state = torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds))
    cm = confusion_matrix(y_test, preds).tolist()

    metrics = {"test_accuracy": acc, "test_f1": f1, "confusion_matrix": cm}

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Test accuracy:", acc)
    print("✅ Test F1:", f1)
    print("Confusion matrix:", cm)
    print("\nClassification report:\n", classification_report(y_test, preds, digits=4))


if __name__ == "__main__":
    main()

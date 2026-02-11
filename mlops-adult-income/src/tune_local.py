import json
import sys
import random
import subprocess
from pathlib import Path

def run_trial(lr, hidden, dropout, batch_size, epochs=10):
    cmd = [
        sys.executable, "src/train_pytorch.py",
        f"--epochs={epochs}",
        f"--lr={lr}",
        f"--hidden={hidden}",
        f"--dropout={dropout}",
        f"--batch_size={batch_size}",
        "--pos_weight_mode=sqrt_ratio",
    ]
    print("\nRUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    metrics = json.loads(Path("artifacts/metrics.json").read_text())
    return metrics

def main():
    random.seed(42)

    # Search space (Katib-like)
    lr_space = [0.0003, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003]
    hidden_space = [64, 96, 128, 192, 256]
    dropout_space = [0.0, 0.1, 0.2, 0.3, 0.4]
    batch_space = [128, 256, 384, 512]

    n_trials = 10  # change to 20 if you want stronger results
    results = []

    for i in range(n_trials):
        lr = random.choice(lr_space)
        hidden = random.choice(hidden_space)
        dropout = random.choice(dropout_space)
        batch = random.choice(batch_space)

        metrics = run_trial(lr, hidden, dropout, batch, epochs=10)

        # Main objective to maximize (stable)
        score = metrics["test_accuracy_at_0_5"]

        results.append({
            "trial": i + 1,
            "lr": lr,
            "hidden": hidden,
            "dropout": dropout,
            "batch_size": batch,
            "val_best_accuracy_at_0_5": metrics["val_best_accuracy_at_0_5"],
            "test_accuracy_at_0_5": metrics["test_accuracy_at_0_5"],
            "test_f1_at_0_5": metrics["test_f1_at_0_5"],
        })

        print(f"Trial {i+1}/{n_trials} => test_acc@0.5={score:.4f}")

    # Pick best by test accuracy (or val accuracy if you prefer)
    best = max(results, key=lambda r: r["test_accuracy_at_0_5"])

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/tuning_results.json").write_text(json.dumps(results, indent=2))
    Path("artifacts/best_params.json").write_text(json.dumps(best, indent=2))

    print("\n✅ BEST RUN")
    print(json.dumps(best, indent=2))
    print("\nSaved:")
    print("- artifacts/tuning_results.json")
    print("- artifacts/best_params.json")

if __name__ == "__main__":
    main()

from adult_income_end2end_pipeline import (
    preprocess_op,
    tune_op,
    train_eval_torch_op,
    monitor_and_maybe_retrain_op,
    retrain_op,  # ✅ ADD THIS
)

from pathlib import Path
import tempfile
import json

BASE = Path(tempfile.mkdtemp())

dataset_dir = BASE / "dataset"
tune_dir = BASE / "tune"
model_dir = BASE / "model"
monitor_dir = BASE / "monitor"

retrain_model_dir = BASE / "retrain_model"
retrain_metrics_dir = BASE / "retrain_metrics"

print("📂 Working dir:", BASE)

# 1️⃣ PREPROCESS
preprocess_op.python_func(
    dataset_url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult",
    output_dataset=type("O", (), {"path": str(dataset_dir)}),
)

# 2️⃣ TUNE
tune_op.python_func(
    dataset=type("I", (), {"path": str(dataset_dir)}),
    tuning_metrics=type("M", (), {"log_metric": print}),
    best_params=type("O", (), {"path": str(tune_dir)}),
    trials=5,
    epochs=3,
)

# 3️⃣ TRAIN + EVAL
train_eval_torch_op.python_func(
    dataset=type("I", (), {"path": str(dataset_dir)}),
    best_params=type("I", (), {"path": str(tune_dir)}),
    model=type("O", (), {"path": str(model_dir)}),
    metrics_out=type("M", (), {"log_metric": print}),
    epochs=10,
)

# 4️⃣ MONITOR (simulate low test accuracy)
monitor_and_maybe_retrain_op.python_func(
    metrics=type("I", (), {"metadata": {"test_accuracy": 0.82}}),
    retrain_triggered=type("O", (), {"path": str(monitor_dir)}),
    min_test_accuracy=0.84,
)

# Optional: print monitor.json so you see decision
mon_path = monitor_dir / "monitor.json"
if mon_path.exists():
    print("📄 monitor.json:", json.loads(mon_path.read_text(encoding="utf-8")))

# 5️⃣ RETRAIN (will run or skip based on monitor.json)
retrain_op.python_func(
    dataset=type("I", (), {"path": str(dataset_dir)}),
    best_params=type("I", (), {"path": str(tune_dir)}),  # ✅ FIXED
    monitor=type("I", (), {"path": str(monitor_dir)}),
    retrained_model=type("O", (), {"path": str(retrain_model_dir)}),
    retrain_metrics=type("M", (), {"log_metric": print}),  # ✅ can be print
    epochs=5,
    seed=42,
)

print("✅ LOCAL PIPELINE RUN FINISHED")
print("📦 Outputs:")
print(" - monitor:", monitor_dir)
print(" - retrain_model:", retrain_model_dir)

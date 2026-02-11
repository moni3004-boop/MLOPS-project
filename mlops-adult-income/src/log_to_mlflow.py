import argparse
import json
import os

import mlflow


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="adult-income-mlp")
    p.add_argument("--run_name", default="baseline")
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--train_metrics", default=None)  # model_out/metrics.json
    p.add_argument("--eval_metrics", default=None)   # model_out/eval_metrics.json
    p.add_argument("--params_json", default=None)    # optional
    p.add_argument("--tracking_uri", default=None)   # optional: http://mlflow...
    args = p.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
        # Params (best effort)
        if args.params_json and os.path.exists(args.params_json):
            params = load_json(args.params_json)
            for k, v in params.items():
                mlflow.log_param(k, v)

        # Metrics
        if args.train_metrics and os.path.exists(args.train_metrics):
            m = load_json(args.train_metrics)
            for k, v in m.items():
                mlflow.log_metric(k, float(v))

        if args.eval_metrics and os.path.exists(args.eval_metrics):
            e = load_json(args.eval_metrics)
            # log the main ones
            mlflow.log_metric("test_accuracy", float(e["test_accuracy"]))
            mlflow.log_metric("test_f1", float(e["test_f1"]))

        # Artifacts
        # model
        model_path = os.path.join(args.model_dir, "model.pt")
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="model")

        # schema + eval outputs
        schema_path = os.path.join(args.artifacts_dir, "schema.json")
        if os.path.exists(schema_path):
            mlflow.log_artifact(schema_path, artifact_path="preprocess")

        if args.eval_metrics and os.path.exists(args.eval_metrics):
            mlflow.log_artifact(args.eval_metrics, artifact_path="eval")

        # record a simple tag for traceability
        mlflow.set_tag("project", "mlops-adult-income")
        mlflow.set_tag("framework", "pytorch")

        print("✅ MLflow run_id:", run.info.run_id)
        print("✅ Experiment:", args.experiment)


if __name__ == "__main__":
    main()

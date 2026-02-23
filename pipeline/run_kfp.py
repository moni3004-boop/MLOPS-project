# pipeline/run_kfp.py
import argparse
from kfp import Client

def get_or_create_experiment(client: Client, name: str, namespace: str) -> str:
    try:
        exp = client.get_experiment(experiment_name=name, namespace=namespace)
        if exp:
            return exp.experiment_id
    except Exception:
        pass
    exp = client.create_experiment(name=name, namespace=namespace)
    return exp.experiment_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="KFP endpoint (e.g. http://ml-pipeline.kubeflow.svc:8888)")
    ap.add_argument("--pipeline-yaml", required=True, help="Path to compiled pipeline yaml")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--namespace", required=True, help="KFP namespace/profile, often participant-17")
    ap.add_argument("--token", default=None, help="Optional bearer token if your KFP requires auth")
    args = ap.parse_args()

    if args.token:
        client = Client(host=args.host, existing_token=args.token)
    else:
        client = Client(host=args.host)

    exp_id = get_or_create_experiment(client, args.experiment_name, args.namespace)

    # Upload pipeline (idempotent-ish по име)
    pipeline_id = client.upload_pipeline(
        pipeline_package_path=args.pipeline_yaml,
        pipeline_name="adult-income-pipeline",
    )

    run = client.create_run_from_pipeline_package(
        pipeline_file=args.pipeline_yaml,
        arguments={},   # ако имаш pipeline params, стави ги тука
        run_name=args.run_name,
        experiment_id=exp_id,
        namespace=args.namespace,
    )

    print("✅ Pipeline uploaded:", pipeline_id)
    print("✅ Run created:", run.run_id)
    print("✅ Experiment:", args.experiment_name, "namespace:", args.namespace)

if __name__ == "__main__":
    main()
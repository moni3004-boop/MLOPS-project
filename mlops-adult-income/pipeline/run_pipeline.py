import os
from kfp import Client

PIPELINE_YAML = "pipeline/adult_income_end2end.yaml"
PIPELINE_NAME = "adult-income-end2end"
RUN_NAME = "adult-income-run-1"
NAMESPACE = os.environ.get("KFP_NAMESPACE") or os.environ.get("NAMESPACE") or "participant-17"

CANDIDATE_HOSTS = [
    os.environ.get("KFP_ENDPOINT"),
    "http://ml-pipeline-ui.kubeflow.svc.cluster.local:80",
    "http://ml-pipeline.kubeflow.svc.cluster.local:8888",
    "http://ml-pipeline.kubeflow.svc.cluster.local:80",
]
CANDIDATE_HOSTS = [h for h in CANDIDATE_HOSTS if h]

last_err = None
client = None
for host in CANDIDATE_HOSTS:
    try:
        client = Client(host=host, namespace=NAMESPACE)
        print("✅ Connected to KFP:", host, "| namespace:", NAMESPACE)
        break
    except Exception as e:
        last_err = e

if client is None:
    raise RuntimeError(f"Could not connect to KFP using {CANDIDATE_HOSTS}. Last error: {last_err}")

# kfp==2.4 uses pipeline_package_path
pipeline = client.upload_pipeline(pipeline_package_path=PIPELINE_YAML, pipeline_name=PIPELINE_NAME)
print("✅ Uploaded pipeline id:", pipeline.pipeline_id)

exp = client.create_experiment(name="adult-income-experiment", namespace=NAMESPACE)
print("✅ Experiment id:", exp.experiment_id)

run = client.create_run_from_pipeline_package(
    pipeline_file=PIPELINE_YAML,
    experiment_id=exp.experiment_id,
    run_name=RUN_NAME,
    arguments={
        # we will change the pipeline to accept dataset_url in the next step
        "input_csv_path": "data/adult.csv",
        "experiment": "adult-income-mlp"
    },
)
print("✅ Run created:", run.run_id)

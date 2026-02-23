import os
import datetime as dt
from kfp import Client

HERE = os.path.dirname(os.path.abspath(__file__))

PIPELINE_YAML = os.path.join(HERE, "adult_income_end2end.yaml")
NAMESPACE = os.environ.get("KFP_NAMESPACE") or os.environ.get("NAMESPACE") or "participant-17"

TS = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
PIPELINE_NAME = f"adult-income-end2end-{TS}"
RUN_NAME = f"adult-income-run-{TS}"

CANDIDATE_HOSTS = [
    os.environ.get("KFP_ENDPOINT"),
    "http://ml-pipeline.kubeflow.svc.cluster.local:8888",
    "http://ml-pipeline-ui.kubeflow.svc.cluster.local:80",
]
CANDIDATE_HOSTS = [h for h in CANDIDATE_HOSTS if h]

last_err = None
client = None

for host in CANDIDATE_HOSTS:
    try:
        client = Client(host=host, namespace=NAMESPACE)
        print(f"✅ Connected to KFP: {host} | namespace: {NAMESPACE}")
        break
    except Exception as e:
        last_err = e
        print(f"❌ Failed to connect: {host} | {repr(e)}")

if client is None:
    raise RuntimeError(f"Could not connect to KFP using {CANDIDATE_HOSTS}. Last error: {last_err}")

pipeline = client.upload_pipeline(
    pipeline_package_path=PIPELINE_YAML,
    pipeline_name=PIPELINE_NAME,
)
print("✅ Uploaded pipeline id:", pipeline.pipeline_id)

exp = client.create_experiment(
    name="adult-income-experiment",
    namespace=NAMESPACE,
)
print("✅ Experiment id:", exp.experiment_id)

run = client.create_run_from_pipeline_package(
    pipeline_file=PIPELINE_YAML,
    experiment_id=exp.experiment_id,
    run_name=RUN_NAME,
    namespace=NAMESPACE,
    arguments={
        "dataset_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult",
        "trials": 10,
        "tune_epochs": 5,
        "train_epochs": 20,
        "min_test_accuracy": 0.84,
    },
)
print("✅ Run created:", run.run_id)

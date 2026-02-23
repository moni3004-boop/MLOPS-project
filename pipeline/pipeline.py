# pipeline/pipeline.py
from kfp import dsl

@dsl.component(base_image="python:3.11-slim")
def hello_op(msg: str) -> str:
    print(f"Hello from pipeline: {msg}")
    return msg

@dsl.pipeline(
    name="adult-income-pipeline",
    description="Minimal pipeline to verify CD -> KFP run"
)
def adult_income_pipeline(message: str = "CD triggered this run"):
    hello_op(msg=message)
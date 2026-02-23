# pipeline/compile.py
from kfp import compiler
from pipeline.pipeline import adult_income_pipeline  # ќе го дефинираш во pipeline.py

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=adult_income_pipeline,
        package_path="pipeline/pipeline.yaml",
    )
    print("✅ Compiled: pipeline/pipeline.yaml")
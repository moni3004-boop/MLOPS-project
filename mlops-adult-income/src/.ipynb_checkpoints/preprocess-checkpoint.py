# src/preprocess.py
import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Schema:
    target: str
    numeric_features: List[str]
    categorical_features: List[str]


def save_np(out_dir: str, name: str, arr: np.ndarray) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npy")
    np.save(path, arr)
    return path


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to Adult dataset CSV (already combined)")
    parser.add_argument("--out_dir", required=True, help="Output directory for .npy and schema.json")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Expected target column name we'll standardize in Step 1:
    target_col = "income"

    # Drop rows where target missing
    df = df.dropna(subset=[target_col]).copy()

    # Binary label: >50K -> 1 else 0
    y = (df[target_col].astype(str).str.contains(">50K")).astype(int).values

    # Features = everything else
    X_df = df.drop(columns=[target_col]).copy()

    numeric_features = X_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [c for c in X_df.columns if c not in numeric_features]

    schema = Schema(target=target_col, numeric_features=numeric_features, categorical_features=categorical_features)

    pre = build_preprocessor(numeric_features, categorical_features)

    # Split: train/val/test
    X_train_df, X_tmp_df, y_train, y_tmp = train_test_split(
        X_df, y, test_size=args.test_size + args.val_size, random_state=args.random_state, stratify=y
    )
    rel_val = args.val_size / (args.test_size + args.val_size)
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_tmp_df, y_tmp, test_size=1 - rel_val, random_state=args.random_state, stratify=y_tmp
    )

    X_train = pre.fit_transform(X_train_df)
    X_val = pre.transform(X_val_df)
    X_test = pre.transform(X_test_df)

    # Save outputs
    save_np(args.out_dir, "X_train", X_train.astype(np.float32))
    save_np(args.out_dir, "y_train", y_train.astype(np.int64))
    save_np(args.out_dir, "X_val", X_val.astype(np.float32))
    save_np(args.out_dir, "y_val", y_val.astype(np.int64))
    save_np(args.out_dir, "X_test", X_test.astype(np.float32))
    save_np(args.out_dir, "y_test", y_test.astype(np.int64))

    with open(os.path.join(args.out_dir, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(schema), f, indent=2)

    print("✅ Preprocessing complete")
    print("Shapes:",
          "X_train", X_train.shape, "y_train", y_train.shape,
          "| X_val", X_val.shape, "y_val", y_val.shape,
          "| X_test", X_test.shape, "y_test", y_test.shape)
    print("Label mean (train):", float(np.mean(y_train)))


if __name__ == "__main__":
    main()

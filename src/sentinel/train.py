"""Model training utilities for Sentinel (Module 3).

Loads processed train/test CSVs, trains an XGBoost classifier, evaluates metrics,
and saves the trained model and a simple report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier


DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path("data/models")
MODEL_FILE = "model.pkl"
REPORT_FILE = "train_report.txt"


def load_processed_data(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = processed_dir / "train_processed.csv"
    test_path = processed_dir / "test_processed.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Processed train/test not found in {processed_dir}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def _split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "Label" not in df.columns:
        raise KeyError("Expected 'Label' column in processed dataframe")
    X = df.drop(columns=["Label"]).astype(float)
    y = df["Label"].astype(int)
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> XGBClassifier:
    model = XGBClassifier(eval_metric="logloss", random_state=random_state)
    model.fit(X, y)
    return model


def evaluate_model(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> str:
    preds = model.predict(X)
    report = classification_report(y, preds)
    cm = confusion_matrix(y, preds)
    summary = f"\nClassification Report:\n{report}\nConfusion Matrix:\n{cm}\n"
    return summary


def train_and_save(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    model_dir: Path = MODEL_DIR,
    model_file: str = MODEL_FILE,
    report_file: str = REPORT_FILE,
    random_state: int = 42,
) -> Tuple[Path, Path]:
    train_df, test_df = load_processed_data(processed_dir)
    X_train, y_train = _split_X_y(train_df)
    X_test, y_test = _split_X_y(test_df)

    model = train_model(X_train, y_train, random_state=random_state)

    train_report = evaluate_model(model, X_train, y_train)
    test_report = evaluate_model(model, X_test, y_test)

    model_dir = Path(processed_dir.parents[0]) / "models" if model_dir is None else Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / model_file
    report_path = model_dir / report_file

    joblib.dump(model, model_path)
    with report_path.open("w", encoding="utf8") as fh:
        fh.write("TRAIN SET:\n")
        fh.write(train_report)
        fh.write("\nTEST SET:\n")
        fh.write(test_report)

    return model_path, report_path


def main() -> None:
    model_path, report_path = train_and_save()
    print(f"Saved model to {model_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()

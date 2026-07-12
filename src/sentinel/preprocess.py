"""Data cleaning and preprocessing utilities for Sentinel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
TRAIN_FILE = "train_processed.csv"
TEST_FILE = "test_processed.csv"
ARTIFACT_FILE = "preprocessing_artifacts.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / RAW_DATA_DIR
DEFAULT_PROCESSED_DATA_DIR = PROJECT_ROOT / PROCESSED_DATA_DIR


@dataclass(frozen=True)
class PreprocessingArtifacts:
    """Artifacts needed to reproduce the preprocessing pipeline."""

    label_encoder: LabelEncoder
    scaler: StandardScaler
    feature_columns: list[str]


def _load_raw_dataset(raw_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path, encoding="latin1", low_memory=False)
        frame.columns = [column.strip() for column in frame.columns]
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    return pd.concat(frames, ignore_index=True)


def _clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]

    if "Label" not in cleaned.columns:
        raise KeyError("Expected a Label column in the raw dataset")

    cleaned["Label"] = cleaned["Label"].astype(str).str.strip()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    feature_columns = [column for column in cleaned.columns if column != "Label"]
    cleaned[feature_columns] = cleaned[feature_columns].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna(axis=0, how="any").reset_index(drop=True)

    return cleaned


def _encode_and_scale(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_labels: pd.Series,
    test_labels: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, PreprocessingArtifacts]:
    label_encoder = LabelEncoder()
    train_encoded = label_encoder.fit_transform(train_labels)
    test_encoded = label_encoder.transform(test_labels)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_test = scaler.transform(test_features)

    train_processed = pd.DataFrame(scaled_train, columns=train_features.columns, index=train_features.index)
    test_processed = pd.DataFrame(scaled_test, columns=test_features.columns, index=test_features.index)

    artifacts = PreprocessingArtifacts(
        label_encoder=label_encoder,
        scaler=scaler,
        feature_columns=list(train_features.columns),
    )

    return (
        train_processed,
        test_processed,
        pd.Series(train_encoded, name="Label", index=train_features.index),
        pd.Series(test_encoded, name="Label", index=test_features.index),
        artifacts,
    )


def preprocess_and_save(
    raw_dir: Path = DEFAULT_RAW_DATA_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DATA_DIR,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[Path, Path, Path]:
    """Clean raw CSV files, encode labels, scale features, and save train/test splits."""

    dataset = _clean_dataset(_load_raw_dataset(raw_dir))
    feature_columns = [column for column in dataset.columns if column != "Label"]

    train_frame, test_frame = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset["Label"],
    )

    train_features = train_frame[feature_columns]
    test_features = test_frame[feature_columns]
    train_labels = train_frame["Label"]
    test_labels = test_frame["Label"]

    train_processed, test_processed, train_encoded, test_encoded, artifacts = _encode_and_scale(
        train_features,
        test_features,
        train_labels,
        test_labels,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)

    train_output = train_processed.copy()
    train_output["Label"] = train_encoded.values
    test_output = test_processed.copy()
    test_output["Label"] = test_encoded.values

    train_path = processed_dir / TRAIN_FILE
    test_path = processed_dir / TEST_FILE
    artifact_path = processed_dir / ARTIFACT_FILE

    train_output.to_csv(train_path, index=False)
    test_output.to_csv(test_path, index=False)
    joblib.dump(artifacts, artifact_path)

    return train_path, test_path, artifact_path


def main() -> None:
    train_path, test_path, artifact_path = preprocess_and_save()
    print(f"Saved processed train set to {train_path}")
    print(f"Saved processed test set to {test_path}")
    print(f"Saved preprocessing artifacts to {artifact_path}")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd

from sentinel.preprocess import _clean_dataset, preprocess_and_save


def test_clean_dataset_drops_null_rows_and_strips_label() -> None:
    frame = pd.DataFrame(
        {
            "Feature 1": [1, 2, None],
            "Feature 2": [3, 4, 5],
            " Label": ["BENIGN ", "DoS Hulk", "BENIGN"],
        }
    )

    cleaned = _clean_dataset(frame)

    assert list(cleaned.columns) == ["Feature 1", "Feature 2", "Label"]
    assert len(cleaned) == 2
    assert cleaned["Label"].tolist() == ["BENIGN", "DoS Hulk"]


def test_preprocess_and_save_writes_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()

    frame = pd.DataFrame(
        {
            "Flow Duration": [1, 2, 3, 4],
            "Total Fwd Packets": [10, 11, 12, 13],
            " Label": ["BENIGN", "BENIGN", "DoS Hulk", "DoS Hulk"],
        }
    )
    frame.to_csv(raw_dir / "sample.csv", index=False)

    train_path, test_path, artifact_path = preprocess_and_save(raw_dir=raw_dir, processed_dir=processed_dir, test_size=0.5, random_state=7)

    assert train_path.exists()
    assert test_path.exists()
    assert artifact_path.exists()

    train_output = pd.read_csv(train_path)
    test_output = pd.read_csv(test_path)

    assert "Label" in train_output.columns
    assert "Label" in test_output.columns
    assert set(train_output["Label"]).issubset({0, 1})
    assert set(test_output["Label"]).issubset({0, 1})
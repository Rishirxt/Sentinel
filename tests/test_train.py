from pathlib import Path
import pandas as pd

from sentinel.train import train_and_save


def test_train_and_save_creates_artifacts(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    # create tiny train/test datasets with two classes
    train = pd.DataFrame({
        "f1": [0.1, 0.2, 0.3, 0.4],
        "f2": [1.0, 0.9, 0.8, 0.7],
        "Label": [0, 0, 1, 1],
    })
    test = pd.DataFrame({
        "f1": [0.15, 0.35],
        "f2": [0.95, 0.75],
        "Label": [0, 1],
    })
    train.to_csv(processed_dir / "train_processed.csv", index=False)
    test.to_csv(processed_dir / "test_processed.csv", index=False)

    model_path, report_path = train_and_save(processed_dir=processed_dir)

    assert model_path.exists()
    assert report_path.exists()

    # basic content checks
    report_text = report_path.read_text(encoding="utf8")
    assert "Classification Report" in report_text

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset


MODULE_ROOT = Path(__file__).resolve().parents[1] / "modules" / "survival_model"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from ecg_survival.data_utils import SurvivalBreaks
from torch_survival.train_survival_from_json import TrainConfig, _export_holdout_risk_scores


class DummyPredictionDataset(Dataset):
    def __init__(self) -> None:
        self.rows = [
            {"patient_SN": "SN001", "patient_id": "P001", "event": 1, "time": 12.0},
            {"patient_SN": "SN002", "patient_id": "P002", "event": 0, "time": 25.0},
            {"patient_SN": "SN003", "patient_id": "P003", "event": 1, "time": 36.0},
            {"patient_SN": "SN004", "patient_id": "P004", "event": 0, "time": 48.0},
        ]
        self.events = np.array([1, 0, 1, 0], dtype=np.int64)
        self.times = np.array([12.0, 25.0, 36.0, 48.0], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        x = torch.zeros((8, 4), dtype=torch.float32)
        y = torch.zeros((2,), dtype=torch.float32)
        event = torch.tensor(float(self.events[idx]), dtype=torch.float32)
        time_value = torch.tensor(float(self.times[idx]), dtype=torch.float32)
        return x, y, event, time_value


class ConstantPredictionModel(nn.Module):
    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return torch.zeros((xb.shape[0], 2), dtype=xb.dtype, device=xb.device)


class RiskScoreExportTest(unittest.TestCase):
    def test_export_holdout_risk_scores_writes_requested_columns(self) -> None:
        dataset = DummyPredictionDataset()
        train_set = Subset(dataset, [0, 1])
        val_set = Subset(dataset, [2])
        test_set = Subset(dataset, [3])

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainConfig(
                task_mode="prediction",
                n_intervals=2,
                max_time=10.0,
                prediction_horizon=4.0,
                batch=2,
                num_workers=0,
                log_dir=Path(tmpdir),
            )
            exports = _export_holdout_risk_scores(
                model=ConstantPredictionModel(),
                dataset=dataset,
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                cfg=cfg,
                device=torch.device("cpu"),
                breaks=SurvivalBreaks.from_uniform(cfg.max_time, cfg.n_intervals),
            )

            self.assertEqual(set(exports.keys()), {"train", "val", "test"})
            train_path = Path(exports["train"])
            self.assertTrue(train_path.exists())
            with train_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertEqual(
                rows[0].keys(),
                {
                    "split",
                    "sample_id",
                    "patient_SN",
                    "patient_id",
                    "event",
                    "time",
                    "risk_score",
                    "task_mode",
                    "risk_horizon",
                },
            )
            self.assertEqual(rows[0]["split"], "train")
            self.assertEqual(rows[0]["sample_id"], "P001")
            self.assertEqual(rows[0]["patient_SN"], "SN001")
            self.assertEqual(rows[0]["patient_id"], "P001")
            self.assertEqual(rows[0]["event"], "1")
            self.assertEqual(rows[0]["time"], "12.0")
            self.assertAlmostEqual(float(rows[0]["risk_score"]), 0.5, places=6)
            self.assertEqual(rows[0]["task_mode"], "prediction")
            self.assertAlmostEqual(float(rows[0]["risk_horizon"]), 5.0, places=6)

    def test_export_holdout_risk_scores_skips_empty_test_split(self) -> None:
        dataset = DummyPredictionDataset()
        train_set = Subset(dataset, [0, 1])
        val_set = Subset(dataset, [2, 3])
        test_set = Subset(dataset, [])

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainConfig(
                task_mode="prediction",
                n_intervals=2,
                max_time=10.0,
                prediction_horizon=4.0,
                batch=2,
                num_workers=0,
                log_dir=Path(tmpdir),
            )
            exports = _export_holdout_risk_scores(
                model=ConstantPredictionModel(),
                dataset=dataset,
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                cfg=cfg,
                device=torch.device("cpu"),
                breaks=SurvivalBreaks.from_uniform(cfg.max_time, cfg.n_intervals),
            )

            self.assertEqual(set(exports.keys()), {"train", "val"})
            self.assertFalse((Path(tmpdir) / "predictions" / "test_risk_scores.csv").exists())


if __name__ == "__main__":
    unittest.main()

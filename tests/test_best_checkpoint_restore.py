import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


MODULE_ROOT = Path(__file__).resolve().parents[1] / "modules" / "survival_model"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from torch_survival.train_survival_from_json import TrainConfig, _train_model


class SingleBatchDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        x = torch.zeros((8, 4), dtype=torch.float32)
        y = torch.ones((1,), dtype=torch.float32)
        event = torch.tensor(1.0, dtype=torch.float32)
        time_value = torch.tensor(5.0, dtype=torch.float32)
        return x, y, event, time_value


class TrackingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("step_marker", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.step_marker += 1.0
        return self.logit.expand(xb.shape[0], 1)


def make_metrics(pr_auc: float) -> dict:
    return {
        "loss": 0.5,
        "auc": pr_auc,
        "pr_auc": pr_auc,
        "c_index": pr_auc,
        "accuracy": 0.5,
        "balanced_acc": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "specificity": 0.5,
        "f1": 0.5,
        "brier": 0.5,
        "best_threshold": 0.5,
        "best_precision": 0.5,
        "best_recall": 0.5,
        "best_specificity": 0.5,
        "best_f1": 0.5,
        "best_accuracy": 0.5,
        "best_balanced_acc": 0.5,
    }


class BestCheckpointRestoreTest(unittest.TestCase):
    def test_train_model_restores_best_validation_checkpoint_before_export(self) -> None:
        dataset = SingleBatchDataset()
        train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        train_eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        def fake_evaluate(model, loader, criterion, device, task_mode, breaks, prediction_horizon, threshold=0.5):
            step_marker = float((model.module if hasattr(model, "module") else model).step_marker.item())
            if loader is train_eval_loader:
                return make_metrics(0.6)
            if step_marker <= 1.0:
                return make_metrics(0.9)
            return make_metrics(0.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainConfig(
                task_mode="classification",
                lead_mode="8lead",
                n_intervals=1,
                max_time=10.0,
                prediction_horizon=None,
                target_len=4,
                batch=1,
                epochs=2,
                lr=1e-3,
                num_workers=0,
                dropout=0.0,
                weight_decay=0.0,
                sched_tmax=2,
                early_stop_metric="val_pr_auc",
                early_stop_patience=0,
                log_dir=Path(tmpdir),
            )

            with patch("torch_survival.train_survival_from_json.build_survival_resnet", return_value=TrackingModel()):
                with patch("torch_survival.train_survival_from_json.evaluate", side_effect=fake_evaluate):
                    with patch("torch_survival.train_survival_from_json._log_and_plot", return_value=None):
                        result = _train_model(
                            cfg,
                            train_loader,
                            train_eval_loader,
                            val_loader,
                            device=torch.device("cpu"),
                            log_dir=Path(tmpdir),
                            pos_weight=None,
                        )

            checkpoint = torch.load(Path(tmpdir) / "model_final.pt", map_location="cpu")
            best_checkpoint = torch.load(Path(tmpdir) / "model_best.pt", map_location="cpu")
            last_checkpoint = torch.load(Path(tmpdir) / "model_last.pt", map_location="cpu")
            self.assertAlmostEqual(float(checkpoint["step_marker"].item()), 1.0, places=6)
            self.assertAlmostEqual(float(best_checkpoint["step_marker"].item()), 1.0, places=6)
            self.assertAlmostEqual(float(last_checkpoint["step_marker"].item()), 2.0, places=6)
            self.assertAlmostEqual(float(result["val"]["pr_auc"]), 0.9, places=6)

            threshold_info = json.loads((Path(tmpdir) / "best_threshold.json").read_text(encoding="utf-8"))
            self.assertEqual(int(threshold_info["epoch"]), 1)
            self.assertAlmostEqual(float(threshold_info["selection_score"]), 0.9, places=6)

            run_summary = json.loads((Path(tmpdir) / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(run_summary["preferred_checkpoint"]).name, "model_best.pt")
            self.assertEqual(Path(run_summary["legacy_checkpoint"]).name, "model_final.pt")
            self.assertEqual(Path(run_summary["latest_checkpoint"]).name, "model_last.pt")
            self.assertEqual(int(run_summary["best_epoch"]), 1)
            self.assertEqual(int(run_summary["last_epoch"]), 2)
            self.assertAlmostEqual(float(run_summary["selection_score"]), 0.9, places=6)
            self.assertTrue(Path(run_summary["archived_best_checkpoint"]).exists())
            self.assertTrue(Path(run_summary["archived_last_checkpoint"]).exists())


if __name__ == "__main__":
    unittest.main()

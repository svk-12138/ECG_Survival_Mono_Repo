import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1] / "modules" / "survival_model"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from torch_survival.infer_survival_risk import _resolve_checkpoint_path


class InferCheckpointResolutionTest(unittest.TestCase):
    def test_prefers_run_summary_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            best_path = log_dir / "model_best.pt"
            final_path = log_dir / "model_final.pt"
            best_path.write_bytes(b"best")
            final_path.write_bytes(b"final")
            (log_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "preferred_checkpoint": str(best_path),
                        "legacy_checkpoint": str(final_path),
                    }
                ),
                encoding="utf-8",
            )

            resolved = _resolve_checkpoint_path(None, str(log_dir))
            self.assertEqual(resolved, best_path.resolve())

    def test_falls_back_to_model_best_alias_when_summary_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            best_path = log_dir / "model_best.pt"
            best_path.write_bytes(b"best")

            resolved = _resolve_checkpoint_path(None, str(log_dir))
            self.assertEqual(resolved, best_path.resolve())

    def test_accepts_directory_via_checkpoint_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            final_path = log_dir / "model_final.pt"
            final_path.write_bytes(b"final")

            resolved = _resolve_checkpoint_path(str(log_dir), None)
            self.assertEqual(resolved, final_path.resolve())


if __name__ == "__main__":
    unittest.main()

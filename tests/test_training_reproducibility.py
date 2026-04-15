import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


MODULE_ROOT = Path(__file__).resolve().parents[1] / "modules" / "survival_model"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from torch_survival.train_survival_from_json import _make_dataloader, _set_training_seed


class NumberDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.values = list(range(size))

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: int) -> int:
        return self.values[idx]


class TrainingReproducibilityTest(unittest.TestCase):
    def test_set_training_seed_resets_python_numpy_and_torch(self) -> None:
        _set_training_seed(123)
        python_values_a = [random.random() for _ in range(3)]
        numpy_values_a = np.random.rand(3)
        torch_values_a = torch.rand(3)

        _set_training_seed(123)
        python_values_b = [random.random() for _ in range(3)]
        numpy_values_b = np.random.rand(3)
        torch_values_b = torch.rand(3)

        self.assertEqual(python_values_a, python_values_b)
        self.assertTrue(np.allclose(numpy_values_a, numpy_values_b))
        self.assertTrue(torch.allclose(torch_values_a, torch_values_b))

    def test_make_dataloader_shuffle_order_is_seeded(self) -> None:
        dataset = NumberDataset(10)

        loader_a = _make_dataloader(dataset, batch_size=3, shuffle=True, num_workers=0, seed=77)
        loader_b = _make_dataloader(dataset, batch_size=3, shuffle=True, num_workers=0, seed=77)
        loader_c = _make_dataloader(dataset, batch_size=3, shuffle=True, num_workers=0, seed=78)

        order_a = [int(item) for batch in loader_a for item in batch]
        order_b = [int(item) for batch in loader_b for item in batch]
        order_c = [int(item) for batch in loader_c for item in batch]

        self.assertEqual(order_a, order_b)
        self.assertNotEqual(order_a, order_c)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


MODULE_ROOT = Path(__file__).resolve().parents[1] / "modules" / "survival_model"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from torch_survival.train_survival_from_json import _make_cv_splits, _split_dataset


class DummyGroupedDataset(Dataset):
    def __init__(self, use_patient_sn: bool) -> None:
        self.rows = [
            {"patient_SN": "SN_A", "patient_id": "P1"},
            {"patient_SN": "SN_A", "patient_id": "P2"},
            {"patient_SN": "SN_B", "patient_id": "P3"},
            {"patient_SN": "SN_C", "patient_id": "P4"},
            {"patient_SN": "SN_C", "patient_id": "P5"},
            {"patient_SN": "SN_D", "patient_id": "P6"},
        ]
        self.events = np.array([1, 1, 0, 0, 0, 1], dtype=np.int64)
        self.patient_ids = np.array([row["patient_id"] for row in self.rows], dtype=object)
        self.group_field = "patient_SN" if use_patient_sn else "patient_id"
        self.group_ids = np.array(
            [row["patient_SN"] if use_patient_sn else row["patient_id"] for row in self.rows],
            dtype=object,
        )
        self.group_to_indices: dict[str, list[int]] = {}
        for idx, group_id in enumerate(self.group_ids):
            self.group_to_indices.setdefault(str(group_id), []).append(idx)
        self.unique_group_ids = np.array(list(self.group_to_indices.keys()), dtype=object)
        self.unique_patient_ids = self.unique_group_ids
        self.group_events = np.array(
            [int(self.events[self.group_to_indices[group_id]].max()) for group_id in self.unique_group_ids],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]

    def subset_indices_for_groups(self, group_ids) -> np.ndarray:
        indices = []
        for group_id in group_ids:
            indices.extend(self.group_to_indices[str(group_id)])
        return np.array(sorted(indices), dtype=np.int64)

    def subset_indices_for_patients(self, patient_ids) -> np.ndarray:
        return self.subset_indices_for_groups(patient_ids)


def group_set(dataset: DummyGroupedDataset, indices) -> set[str]:
    return {str(dataset.group_ids[int(idx)]) for idx in indices}


class GroupedPatientSplitTest(unittest.TestCase):
    def test_holdout_split_keeps_patient_sn_groups_separate(self) -> None:
        dataset = DummyGroupedDataset(use_patient_sn=True)
        train_set, val_set, test_set = _split_dataset(
            dataset,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )

        train_groups = group_set(dataset, train_set.indices)
        val_groups = group_set(dataset, val_set.indices)
        test_groups = group_set(dataset, test_set.indices)

        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(val_groups.isdisjoint(test_groups))
        self.assertEqual(train_groups | val_groups | test_groups, set(dataset.unique_group_ids.tolist()))

    def test_cv_split_keeps_patient_sn_groups_separate(self) -> None:
        dataset = DummyGroupedDataset(use_patient_sn=True)
        splits, _ = _make_cv_splits(dataset, n_splits=2, seed=42)

        self.assertEqual(len(splits), 2)
        for train_idx, val_idx in splits:
            train_groups = group_set(dataset, train_idx)
            val_groups = group_set(dataset, val_idx)
            self.assertTrue(train_groups.isdisjoint(val_groups))

    def test_falls_back_to_patient_id_groups_when_patient_sn_absent(self) -> None:
        dataset = DummyGroupedDataset(use_patient_sn=False)
        train_set, val_set, test_set = _split_dataset(
            dataset,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )

        train_groups = group_set(dataset, train_set.indices)
        val_groups = group_set(dataset, val_set.indices)
        test_groups = group_set(dataset, test_set.indices)

        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(val_groups.isdisjoint(test_groups))


if __name__ == "__main__":
    unittest.main()

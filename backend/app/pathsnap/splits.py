"""Discover explicit dataset partitions without synthesizing trip IDs."""
from dataclasses import dataclass
from pathlib import Path
from app.pathsnap.preprocessing.dataset import discover_trips


@dataclass(frozen=True)
class DatasetSplits:
    train: list
    student_test: list
    evaluator_test: list | None


def discover_dataset_splits(root) -> DatasetSplits:
    """Read only existing train/student/evaluator directories under ``root``.

    Evaluator names are recognized only when present; no evaluator IDs are
    inferred from numeric gaps or copied from the student test partition.
    """
    root = Path(root)
    train_dir, student_dir = root / "train", root / "student_test"
    if not train_dir.is_dir() or not student_dir.is_dir():
        raise ValueError("dataset root must contain separate train/ and student_test/ folders")
    evaluator_candidates = [root / name for name in ("evaluator_test", "evaluator-testing", "evaluator_testing") if (root / name).is_dir()]
    if len(evaluator_candidates) > 1:
        raise ValueError("multiple evaluator test folders found; select and rename the intended folder")
    return DatasetSplits(
        train=discover_trips(train_dir),
        student_test=discover_trips(student_dir),
        evaluator_test=discover_trips(evaluator_candidates[0]) if evaluator_candidates else None,
    )

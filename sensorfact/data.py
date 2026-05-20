from __future__ import annotations

import random
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sensorfact.io import write_json
from sensorfact.schemas import SensorWindow


UCI_LABELS = {
    1: "walking",
    2: "walking_upstairs",
    3: "walking_downstairs",
    4: "sitting",
    5: "standing",
    6: "laying",
}

UCI_CHANNEL_FILES = [
    ("acc_x", "body_acc_x"),
    ("acc_y", "body_acc_y"),
    ("acc_z", "body_acc_z"),
    ("gyro_x", "body_gyro_x"),
    ("gyro_y", "body_gyro_y"),
    ("gyro_z", "body_gyro_z"),
]

WISDM_URL = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
WISDM_RAW_NAME = "WISDM_ar_v1.1_raw.txt"
WISDM_CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z"]
MHEALTH_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"
MHEALTH_CHANNEL_NAMES = [
    "chest_acc_x",
    "chest_acc_y",
    "chest_acc_z",
    "ankle_acc_x",
    "ankle_acc_y",
    "ankle_acc_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_mag_x",
    "ankle_mag_y",
    "ankle_mag_z",
    "arm_acc_x",
    "arm_acc_y",
    "arm_acc_z",
    "arm_gyro_x",
    "arm_gyro_y",
    "arm_gyro_z",
    "arm_mag_x",
    "arm_mag_y",
    "arm_mag_z",
]
MHEALTH_VALUE_INDICES = [0, 1, 2, *range(5, 23)]
MHEALTH_LABELS = {
    1: "standing_still",
    2: "sitting_relaxing",
    3: "lying_down",
    4: "walking",
    5: "climbing_stairs",
    6: "waist_bends_forward",
    7: "frontal_arm_elevation",
    8: "knees_bending",
    9: "cycling",
    10: "jogging",
    11: "running",
    12: "jump_front_back",
}


@dataclass(frozen=True)
class WISDMRawSample:
    user_id: str
    activity: str
    timestamp: int
    values: tuple[float, float, float]


@dataclass(frozen=True)
class MHEALTHRawSample:
    subject_id: str
    activity: str
    values: tuple[float, ...]


def download_modelscope_dataset(dataset_id: str, local_dir: str | Path) -> Path:
    from modelscope.hub.snapshot_download import dataset_snapshot_download

    path = dataset_snapshot_download(dataset_id=dataset_id, local_dir=str(local_dir))
    return Path(path)


def find_ucihar_zip(root: str | Path) -> Path:
    root_path = Path(root)
    candidates = list(root_path.rglob("UCI HAR Dataset.zip"))
    if not candidates:
        candidates = list(root_path.rglob("*HAR*Dataset*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No UCI HAR zip found under {root_path}")
    return candidates[0]


def download_wisdm_dataset(local_dir: str | Path, url: str = WISDM_URL) -> Path:
    out_dir = Path(local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / Path(url).name
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out_dir)
    raw = find_wisdm_raw_file(out_dir)
    return raw


def download_mhealth_dataset(local_dir: str | Path, url: str = MHEALTH_URL) -> Path:
    out_dir = Path(local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / Path(url).name
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(out_dir)
    files = find_mhealth_log_files(out_dir)
    if not files:
        raise FileNotFoundError(f"No MHEALTH subject logs found under {out_dir}")
    return files[0]


def find_wisdm_raw_file(root: str | Path) -> Path:
    root_path = Path(root)
    if root_path.is_file() and root_path.name == WISDM_RAW_NAME:
        return root_path
    candidates = list(root_path.rglob(WISDM_RAW_NAME))
    if not candidates:
        candidates = list(root_path.rglob("*WISDM*raw*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No WISDM raw file found under {root_path}")
    return candidates[0]


def find_mhealth_log_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if root_path.is_file() and root_path.name.lower().startswith("mhealth_subject"):
        return [root_path]
    candidates = list(root_path.rglob("mHealth_subject*.log"))
    if not candidates:
        candidates = list(root_path.rglob("mhealth_subject*.log"))
    return sorted(candidates, key=lambda path: _mhealth_subject_id(path))


def _mhealth_subject_id(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def parse_wisdm_raw_line(line: str) -> WISDMRawSample | None:
    text = line.strip()
    if not text:
        return None
    if text.endswith(";"):
        text = text[:-1]
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 6:
        return None
    user_id, activity, timestamp, x_value, y_value, z_value = parts
    try:
        return WISDMRawSample(
            user_id=str(int(user_id)),
            activity=activity.strip().lower().replace(" ", "_"),
            timestamp=int(timestamp),
            values=(float(x_value), float(y_value), float(z_value)),
        )
    except ValueError:
        return None


def parse_mhealth_raw_line(line: str, subject_id: str) -> MHEALTHRawSample | None:
    text = line.strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 24:
        return None
    try:
        values = [float(item) for item in parts[:-1]]
        label_id = int(float(parts[-1]))
    except ValueError:
        return None
    if label_id == 0:
        return None
    label = MHEALTH_LABELS.get(label_id)
    if label is None:
        return None
    if max(MHEALTH_VALUE_INDICES) >= len(values):
        return None
    selected = tuple(float(values[idx]) for idx in MHEALTH_VALUE_INDICES)
    return MHEALTHRawSample(subject_id=subject_id, activity=label, values=selected)


def _iter_wisdm_samples(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            sample = parse_wisdm_raw_line(line)
            if sample is not None:
                yield sample


def _iter_mhealth_file(path: Path):
    subject_id = str(_mhealth_subject_id(path))
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            sample = parse_mhealth_raw_line(line, subject_id=subject_id)
            if sample is not None:
                yield sample


def _wisdm_split_subjects(subject_ids: list[str], test_subject_ratio: float, seed: int) -> set[str]:
    subjects = sorted(set(subject_ids), key=lambda item: int(item) if item.isdigit() else item)
    if len(subjects) <= 1:
        return set()
    rng = random.Random(seed)
    rng.shuffle(subjects)
    test_count = max(1, min(len(subjects) - 1, round(len(subjects) * test_subject_ratio)))
    return set(subjects[:test_count])


def _split_subjects(subject_ids: list[str], test_subject_ratio: float, seed: int) -> set[str]:
    subjects = sorted(set(subject_ids), key=lambda item: int(item) if item.isdigit() else item)
    if len(subjects) <= 1:
        return set()
    rng = random.Random(seed)
    rng.shuffle(subjects)
    test_count = max(1, min(len(subjects) - 1, round(len(subjects) * test_subject_ratio)))
    return set(subjects[:test_count])


def _emit_wisdm_sequence_windows(
    sequence: list[WISDMRawSample],
    *,
    split: str,
    window_size: int,
    stride: int,
    window_counts: dict[str, int],
    max_windows: int | None,
) -> list[SensorWindow]:
    if len(sequence) < window_size:
        return []
    rows: list[SensorWindow] = []
    user_id = sequence[0].user_id
    activity = sequence[0].activity
    for start in range(0, len(sequence) - window_size + 1, stride):
        if max_windows is not None and max_windows >= 0 and window_counts[split] >= max_windows:
            break
        chunk = sequence[start : start + window_size]
        sensor = np.asarray([item.values for item in chunk], dtype=np.float32)
        window_id = f"wisdm_{split}_{window_counts[split]:06d}"
        rows.append(
            SensorWindow(
                window_id=window_id,
                dataset_id="wisdm",
                split=split,
                label=activity,
                subject_id=user_id,
                sensor=sensor,
            )
        )
        window_counts[split] += 1
    return rows


def _emit_mhealth_sequence_windows(
    sequence: list[MHEALTHRawSample],
    *,
    split: str,
    window_size: int,
    stride: int,
    window_counts: dict[str, int],
    max_windows: int | None,
) -> list[SensorWindow]:
    if len(sequence) < window_size:
        return []
    rows: list[SensorWindow] = []
    subject_id = sequence[0].subject_id
    activity = sequence[0].activity
    for start in range(0, len(sequence) - window_size + 1, stride):
        if max_windows is not None and max_windows >= 0 and window_counts[split] >= max_windows:
            break
        chunk = sequence[start : start + window_size]
        sensor = np.asarray([item.values for item in chunk], dtype=np.float32)
        window_id = f"mhealth_{split}_{window_counts[split]:06d}"
        rows.append(
            SensorWindow(
                window_id=window_id,
                dataset_id="mhealth",
                split=split,
                label=activity,
                subject_id=subject_id,
                sensor=sensor,
            )
        )
        window_counts[split] += 1
    return rows


def load_wisdm(
    raw_root: str | Path,
    window_size: int = 128,
    stride: int = 64,
    max_timestamp_gap: int | None = 250_000_000,
    test_subject_ratio: float = 0.2,
    seed: int = 42,
    max_train_windows: int | None = None,
    max_test_windows: int | None = None,
) -> tuple[list[SensorWindow], list[SensorWindow]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    raw_path = find_wisdm_raw_file(raw_root)
    samples = list(_iter_wisdm_samples(raw_path))
    test_subjects = _wisdm_split_subjects(
        [sample.user_id for sample in samples],
        test_subject_ratio=test_subject_ratio,
        seed=seed,
    )
    train: list[SensorWindow] = []
    test: list[SensorWindow] = []
    counts = {"train": 0, "test": 0}
    current: list[WISDMRawSample] = []

    def flush_sequence() -> None:
        if not current:
            return
        split = "test" if current[0].user_id in test_subjects else "train"
        max_windows = max_test_windows if split == "test" else max_train_windows
        emitted = _emit_wisdm_sequence_windows(
            current,
            split=split,
            window_size=window_size,
            stride=stride,
            window_counts=counts,
            max_windows=max_windows,
        )
        if split == "test":
            test.extend(emitted)
        else:
            train.extend(emitted)

    for sample in samples:
        timestamp_break = False
        if current:
            delta = sample.timestamp - current[-1].timestamp
            timestamp_break = delta < 0 or (
                max_timestamp_gap is not None and delta > max_timestamp_gap
            )
        if current and (
            sample.user_id != current[-1].user_id
            or sample.activity != current[-1].activity
            or timestamp_break
        ):
            flush_sequence()
            current = []
        current.append(sample)
    flush_sequence()
    return train, test


def load_mhealth(
    raw_root: str | Path,
    window_size: int = 128,
    stride: int = 64,
    test_subject_ratio: float = 0.2,
    seed: int = 42,
    max_train_windows: int | None = None,
    max_test_windows: int | None = None,
) -> tuple[list[SensorWindow], list[SensorWindow]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    files = find_mhealth_log_files(raw_root)
    if not files:
        raise FileNotFoundError(f"No MHEALTH subject logs found under {raw_root}")
    samples_by_file = [(path, list(_iter_mhealth_file(path))) for path in files]
    subject_ids = [sample.subject_id for _, samples in samples_by_file for sample in samples]
    test_subjects = _split_subjects(subject_ids, test_subject_ratio=test_subject_ratio, seed=seed)
    train: list[SensorWindow] = []
    test: list[SensorWindow] = []
    counts = {"train": 0, "test": 0}

    def flush_sequence(current: list[MHEALTHRawSample]) -> None:
        if not current:
            return
        split = "test" if current[0].subject_id in test_subjects else "train"
        max_windows = max_test_windows if split == "test" else max_train_windows
        emitted = _emit_mhealth_sequence_windows(
            current,
            split=split,
            window_size=window_size,
            stride=stride,
            window_counts=counts,
            max_windows=max_windows,
        )
        if split == "test":
            test.extend(emitted)
        else:
            train.extend(emitted)

    for _, samples in samples_by_file:
        current: list[MHEALTHRawSample] = []
        for sample in samples:
            if current and (
                sample.subject_id != current[-1].subject_id or sample.activity != current[-1].activity
            ):
                flush_sequence(current)
                current = []
            current.append(sample)
        flush_sequence(current)
    return train, test


def ensure_ucihar_extracted(root: str | Path) -> Path:
    root_path = Path(root)
    extracted = root_path / "UCI HAR Dataset"
    if extracted.exists():
        return extracted
    archive = find_ucihar_zip(root_path)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(root_path)
    if not extracted.exists():
        matches = list(root_path.rglob("UCI HAR Dataset"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not find extracted UCI HAR Dataset under {root_path}")
    return extracted


def _load_split(base: Path, split: str, max_windows: int | None = None) -> list[SensorWindow]:
    split_dir = base / split
    signal_dir = split_dir / "Inertial Signals"
    arrays = []
    for _, stem in UCI_CHANNEL_FILES:
        path = signal_dir / f"{stem}_{split}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        arrays.append(np.loadtxt(path, dtype=np.float32))
    stacked = np.stack(arrays, axis=-1)
    labels = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int32)
    subjects = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=np.int32)
    n = stacked.shape[0] if max_windows is None or max_windows < 0 else min(max_windows, stacked.shape[0])
    windows: list[SensorWindow] = []
    for idx in range(n):
        windows.append(
            SensorWindow(
                window_id=f"ucihar_{split}_{idx:06d}",
                dataset_id="ucihar",
                split=split,
                label=UCI_LABELS.get(int(labels[idx]), str(int(labels[idx]))),
                subject_id=str(int(subjects[idx])),
                sensor=stacked[idx],
            )
        )
    return windows


def load_ucihar(
    raw_root: str | Path,
    max_train_windows: int | None = None,
    max_test_windows: int | None = None,
) -> tuple[list[SensorWindow], list[SensorWindow]]:
    base = ensure_ucihar_extracted(raw_root)
    return (
        _load_split(base, "train", max_windows=max_train_windows),
        _load_split(base, "test", max_windows=max_test_windows),
    )


def make_synthetic_windows(
    train_count: int = 120,
    test_count: int = 60,
    sample_rate: float = 50.0,
    channels: int = 6,
    length: int = 128,
    seed: int = 42,
) -> tuple[list[SensorWindow], list[SensorWindow]]:
    rng = np.random.default_rng(seed)
    labels = ["walking", "sitting", "standing", "walking_upstairs"]

    def build(split: str, count: int) -> list[SensorWindow]:
        rows = []
        for idx in range(count):
            label_idx = idx % len(labels)
            label = labels[label_idx]
            t = np.arange(length, dtype=np.float32) / sample_rate
            sensor = rng.normal(0.0, 0.03, size=(length, channels)).astype(np.float32)
            freq = 0.6 + 0.35 * label_idx
            amp = 0.25 + 0.35 * label_idx
            axis = label_idx % channels
            sensor[:, axis] += amp * np.sin(2 * np.pi * freq * t)
            if label_idx in {0, 3}:
                sensor[:, (axis + 1) % channels] += 0.4 * amp * np.sin(2 * np.pi * freq * t)
            rows.append(
                SensorWindow(
                    window_id=f"synthetic_{split}_{idx:06d}",
                    dataset_id="synthetic_ucihar_like",
                    split=split,
                    label=label,
                    subject_id=str(idx % 12),
                    sensor=sensor,
                )
            )
        return rows

    return build("train", train_count), build("test", test_count)


def write_window_files(
    train: list[SensorWindow],
    test: list[SensorWindow],
    processed_dir: str | Path,
) -> dict[str, str]:
    from sensorfact.io import write_jsonl

    out = Path(processed_dir)
    train_path = out / "ucihar_train.jsonl"
    test_path = out / "ucihar_test.jsonl"
    write_jsonl(train_path, [row.to_json_dict() for row in train])
    write_jsonl(test_path, [row.to_json_dict() for row in test])
    metadata = {
        "train_windows": len(train),
        "test_windows": len(test),
        "channels": [name for name, _ in UCI_CHANNEL_FILES],
        "dataset_id": train[0].dataset_id if train else "unknown",
    }
    metadata_path = out / "ucihar_metadata.json"
    write_json(metadata_path, metadata)
    return {
        "train": str(train_path),
        "test": str(test_path),
        "metadata": str(metadata_path),
    }

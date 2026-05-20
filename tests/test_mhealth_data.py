from pathlib import Path

from sensorfact.data import load_mhealth, parse_mhealth_raw_line


def mhealth_line(label: int, offset: float = 0.0) -> str:
    values = [f"{offset + idx * 0.01:.4f}" for idx in range(23)]
    return "\t".join([*values, str(label)])


def test_parse_mhealth_raw_line_filters_null_class_and_selects_channels():
    parsed = parse_mhealth_raw_line(mhealth_line(4), subject_id="2")

    assert parsed is not None
    assert parsed.subject_id == "2"
    assert parsed.activity == "walking"
    assert len(parsed.values) == 21
    assert parse_mhealth_raw_line(mhealth_line(0), subject_id="2") is None
    assert parse_mhealth_raw_line("bad row", subject_id="2") is None


def test_load_mhealth_windows_are_subject_disjoint(tmp_path: Path):
    raw_dir = tmp_path / "mhealth" / "MHEALTHDATASET"
    raw_dir.mkdir(parents=True)
    for subject_id in range(1, 5):
        rows = []
        for label in [1, 4]:
            for idx in range(6):
                rows.append(mhealth_line(label, offset=subject_id + label + idx))
        (raw_dir / f"mHealth_subject{subject_id}.log").write_text("\n".join(rows), encoding="utf-8")

    train, test = load_mhealth(raw_dir, window_size=4, stride=2, test_subject_ratio=0.25, seed=3)

    assert train
    assert test
    assert {row.subject_id for row in train}.isdisjoint({row.subject_id for row in test})
    assert {row.dataset_id for row in train + test} == {"mhealth"}
    assert {row.sensor.shape for row in train + test} == {(4, 21)}
    assert {row.label for row in train + test} <= {"standing_still", "walking"}

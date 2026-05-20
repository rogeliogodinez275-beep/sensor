from pathlib import Path

from sensorfact.data import load_wisdm, parse_wisdm_raw_line


def test_parse_wisdm_raw_line_handles_semicolon_and_bad_rows():
    parsed = parse_wisdm_raw_line("33,Jogging,49105962326000,-0.6946377,12.680544,0.5039537;")

    assert parsed is not None
    assert parsed.user_id == "33"
    assert parsed.activity == "jogging"
    assert parsed.timestamp == 49105962326000
    assert parsed.values == (-0.6946377, 12.680544, 0.5039537)
    assert parse_wisdm_raw_line("bad,line") is None
    assert parse_wisdm_raw_line("") is None


def test_load_wisdm_windows_are_subject_disjoint(tmp_path: Path):
    raw_dir = tmp_path / "wisdm"
    raw_dir.mkdir()
    raw_path = raw_dir / "WISDM_ar_v1.1_raw.txt"
    lines = []
    for user_id in range(1, 7):
        activity = "Walking" if user_id % 2 else "Jogging"
        for idx in range(6):
            lines.append(
                f"{user_id},{activity},{1000 * user_id + idx},"
                f"{0.1 * idx:.3f},{0.2 * idx:.3f},{0.3 * idx:.3f};"
            )
    raw_path.write_text("\n".join(lines), encoding="utf-8")

    train, test = load_wisdm(
        raw_dir,
        window_size=4,
        stride=2,
        test_subject_ratio=0.34,
        seed=7,
    )

    assert train
    assert test
    assert {row.subject_id for row in train}.isdisjoint({row.subject_id for row in test})
    assert {row.dataset_id for row in train + test} == {"wisdm"}
    assert {row.sensor.shape for row in train + test} == {(4, 3)}
    assert {row.label for row in train + test} <= {"walking", "jogging"}


def test_load_wisdm_splits_windows_at_timestamp_gaps(tmp_path: Path):
    raw_dir = tmp_path / "wisdm"
    raw_dir.mkdir()
    raw_path = raw_dir / "WISDM_ar_v1.1_raw.txt"
    lines = []
    for user_id in [1, 2]:
        for timestamp in [0, 1, 2, 3, 100, 101, 102, 103]:
            lines.append(f"{user_id},Walking,{timestamp},{timestamp},0,0;")
    raw_path.write_text("\n".join(lines), encoding="utf-8")

    train, test = load_wisdm(
        raw_dir,
        window_size=4,
        stride=1,
        max_timestamp_gap=10,
        seed=1,
    )

    windows = train + test
    assert windows
    assert all(float(row.sensor[:, 0].max() - row.sensor[:, 0].min()) <= 3.0 for row in windows)

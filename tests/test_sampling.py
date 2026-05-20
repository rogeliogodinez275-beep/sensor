import json
from pathlib import Path

from sensorfact.sampling import sample_jsonl


def write_rows(path: Path, count: int) -> None:
    path.write_text(
        "".join(json.dumps({"id": idx}) + "\n" for idx in range(count)),
        encoding="utf-8",
    )


def read_ids(path: Path) -> list[int]:
    return [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_sample_jsonl_is_deterministic_and_not_prefix(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_rows(source, 20)

    count_first = sample_jsonl(source, first, sample_size=6, seed=11)
    count_second = sample_jsonl(source, second, sample_size=6, seed=11)

    assert count_first == 6
    assert count_second == 6
    assert read_ids(first) == read_ids(second)
    assert read_ids(first) != list(range(6))


def test_sample_jsonl_keeps_all_rows_when_sample_size_exceeds_count(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "sample.jsonl"
    write_rows(source, 3)

    count = sample_jsonl(source, output, sample_size=10, seed=3)

    assert count == 3
    assert read_ids(output) == [0, 1, 2]

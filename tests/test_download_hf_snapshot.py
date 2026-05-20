import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import download_hf_snapshot


def test_download_snapshot_falls_back_to_modelscope(tmp_path: Path, monkeypatch):
    calls: list[tuple[str, str, str]] = []

    def fake_hf_snapshot_download(*, repo_id: str, local_dir: str, resume_download: bool):
        calls.append(("hf", repo_id, local_dir))
        raise RuntimeError("hf unavailable")

    def fake_ms_snapshot_download(*, model_id: str, local_dir: str):
        calls.append(("ms", model_id, local_dir))
        model_dir = Path(local_dir)
        (model_dir / "model.safetensors.index.json").write_text(
            '{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}',
            encoding="utf-8",
        )
        (model_dir / "model-00001-of-00002.safetensors").write_text("a", encoding="utf-8")
        (model_dir / "model-00002-of-00002.safetensors").write_text("b", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(download_hf_snapshot, "hf_snapshot_download", fake_hf_snapshot_download)
    monkeypatch.setattr(download_hf_snapshot, "ms_snapshot_download", fake_ms_snapshot_download)

    local_dir = tmp_path / "model"
    path = download_hf_snapshot.download_snapshot(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        local_dir=local_dir,
        resume_download=True,
    )

    assert path == local_dir
    assert calls == [
        ("hf", "Qwen/Qwen2.5-Coder-7B-Instruct", str(local_dir)),
        ("ms", "Qwen/Qwen2.5-Coder-7B-Instruct", str(local_dir)),
    ]


def test_download_snapshot_retries_until_all_weight_files_exist(tmp_path: Path, monkeypatch):
    calls: list[int] = []

    def fake_ms_snapshot_download(*, model_id: str, local_dir: str):
        calls.append(len(calls) + 1)
        model_dir = Path(local_dir)
        (model_dir / "model.safetensors.index.json").write_text(
            '{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}',
            encoding="utf-8",
        )
        (model_dir / "model-00001-of-00002.safetensors").write_text("a", encoding="utf-8")
        if len(calls) >= 2:
            (model_dir / "model-00002-of-00002.safetensors").write_text("b", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(download_hf_snapshot, "hf_snapshot_download", None)
    monkeypatch.setattr(download_hf_snapshot, "ms_snapshot_download", fake_ms_snapshot_download)

    path = download_hf_snapshot.download_snapshot(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        local_dir=tmp_path / "model",
        resume_download=True,
        max_attempts=3,
        retry_delay_s=0.0,
    )

    assert path == tmp_path / "model"
    assert calls == [1, 2]


def test_download_snapshot_uses_modelscope_when_hf_returns_incomplete_local_dir(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_hf_snapshot_download(*, repo_id: str, local_dir: str, resume_download: bool):
        calls.append("hf")
        model_dir = Path(local_dir)
        (model_dir / "model.safetensors.index.json").write_text(
            '{"weight_map":{"a":"model-00001-of-00001.safetensors"}}',
            encoding="utf-8",
        )
        return str(local_dir)

    def fake_ms_snapshot_download(*, model_id: str, local_dir: str):
        calls.append("ms")
        model_dir = Path(local_dir)
        (model_dir / "model-00001-of-00001.safetensors").write_text("a", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(download_hf_snapshot, "hf_snapshot_download", fake_hf_snapshot_download)
    monkeypatch.setattr(download_hf_snapshot, "ms_snapshot_download", fake_ms_snapshot_download)

    path = download_hf_snapshot.download_snapshot(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        local_dir=tmp_path / "model",
        resume_download=True,
        max_attempts=1,
        retry_delay_s=0.0,
    )

    assert path == tmp_path / "model"
    assert calls == ["hf", "ms"]


def test_download_snapshot_prefers_modelscope_when_local_snapshot_is_incomplete(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    local_dir = tmp_path / "model"
    local_dir.mkdir()
    (local_dir / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )

    def fake_hf_snapshot_download(*, repo_id: str, local_dir: str, resume_download: bool):
        calls.append("hf")
        return str(local_dir)

    def fake_ms_snapshot_download(*, model_id: str, local_dir: str):
        calls.append("ms")
        (Path(local_dir) / "model-00001-of-00001.safetensors").write_text("a", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(download_hf_snapshot, "hf_snapshot_download", fake_hf_snapshot_download)
    monkeypatch.setattr(download_hf_snapshot, "ms_snapshot_download", fake_ms_snapshot_download)

    path = download_hf_snapshot.download_snapshot(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        local_dir=local_dir,
        resume_download=True,
        max_attempts=1,
        retry_delay_s=0.0,
    )

    assert path == local_dir
    assert calls == ["ms"]


def test_download_snapshot_defaults_are_long_enough_for_large_shards():
    defaults = download_hf_snapshot.download_snapshot.__defaults__

    assert defaults is not None
    assert defaults[0] >= 120
    assert defaults[1] >= 5.0

import json
from pathlib import Path

from scripts.write_repro_manifest import build_manifest, write_manifest


def test_repro_manifest_records_hashes_and_config(tmp_path: Path):
    data_file = tmp_path / "data.jsonl"
    config_file = tmp_path / "config.yaml"
    data_file.write_text('{"x": 1}\n', encoding="utf-8")
    config_file.write_text("seed: 7\n", encoding="utf-8")

    manifest = build_manifest(
        files=[data_file, config_file],
        experiment_name="toy",
        model_id="local-test",
        seed=7,
    )

    assert manifest["experiment_name"] == "toy"
    assert manifest["model_id"] == "local-test"
    assert manifest["seed"] == 7
    assert manifest["files"][0]["sha256"]
    assert manifest["files"][0]["size_bytes"] == data_file.stat().st_size

    out = tmp_path / "manifest.json"
    write_manifest(out, manifest)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["files"][1]["path"].endswith("config.yaml")

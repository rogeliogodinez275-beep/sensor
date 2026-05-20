from pathlib import Path


SCRIPT = Path("scripts/run_experiment_queue.sh")


def test_queue_script_tracks_failures_instead_of_swallowing_them():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "FAILURES=0" in text
    assert "FAILURES=$((FAILURES + 1))" in text
    assert 'exit "$FAILURES"' in text
    assert 'return "$code"' in text


def test_wisdm_build_skip_requires_both_hard_variants():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "data/benchmark/wisdm_sensorfact_hard_v2_test.jsonl,data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl" in text


def test_outputs_ready_does_not_clobber_benchmark_path_loop_variable():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "local candidate_path" in text
    assert 'for candidate_path in "${paths[@]}"' in text
    assert '[[ ! -s "$candidate_path" ]]' in text

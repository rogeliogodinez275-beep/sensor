import copy

from scripts.build_stress_benchmark import build_stress_rows


def make_record(idx: int, evidence_text: str) -> dict:
    return {
        "window_id": f"w{idx}",
        "evidence_text": evidence_text,
        "positive": {"text": f"supported {idx}", "supported": True},
        "counterfactuals": [
            {"text": f"unsupported {idx}", "supported": False, "changed_fact": "intensity"}
        ],
        "caption_selection": {
            "answer_index": 0,
            "candidates": [{"text": f"supported {idx}", "supported": True}],
        },
    }


def test_numeric_mask_redacts_numbers_without_changing_labels():
    rows = [
        make_record(
            1,
            "RMS energy is 0.123, dominant spectral peak is 1.562 Hz, peak count is 7.",
        )
    ]

    stressed = build_stress_rows(rows, variant="numeric_mask", seed=7)

    assert len(stressed) == 1
    assert "0.123" not in stressed[0]["evidence_text"]
    assert "1.562" not in stressed[0]["evidence_text"]
    assert "7" not in stressed[0]["evidence_text"]
    assert stressed[0]["caption_selection"] == rows[0]["caption_selection"]
    assert stressed[0]["stress_variant"] == "numeric_mask"


def test_shuffled_evidence_reassigns_evidence_text_and_keeps_examples():
    rows = [
        make_record(1, "evidence one 0.1"),
        make_record(2, "evidence two 0.2"),
        make_record(3, "evidence three 0.3"),
    ]
    original = copy.deepcopy(rows)

    stressed = build_stress_rows(rows, variant="shuffled_evidence", seed=13)

    assert [row["window_id"] for row in stressed] == ["w1", "w2", "w3"]
    assert all(row["evidence_text"] != source["evidence_text"] for row, source in zip(stressed, rows))
    assert [row["positive"] for row in stressed] == [row["positive"] for row in original]
    assert rows == original
    assert all(row["stress_variant"] == "shuffled_evidence" for row in stressed)


def test_hidden_evidence_removes_instance_specific_evidence():
    rows = [make_record(1, "RMS energy is 0.123 and axis is acc_x.")]

    stressed = build_stress_rows(rows, variant="hidden_evidence", seed=5)

    assert "RMS" not in stressed[0]["evidence_text"]
    assert "acc_x" not in stressed[0]["evidence_text"]
    assert "hidden" in stressed[0]["evidence_text"].lower()
    assert stressed[0]["stress_variant"] == "hidden_evidence"

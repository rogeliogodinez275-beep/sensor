import numpy as np

from sensorfact.qwen_eval import evaluate_embedding_grounding, score_candidates


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def test_score_candidates_prefers_highest_cosine_match():
    embedder = FakeEmbedder(
        {
            "evidence": [1.0, 0.0],
            "supported": [0.9, 0.1],
            "unsupported": [0.0, 1.0],
        }
    )

    scores = score_candidates(embedder, "evidence", ["unsupported", "supported"])

    assert scores[1] > scores[0]


def test_evaluate_embedding_grounding_uses_benchmark_contract():
    records = [
        {
            "window_id": "w1",
            "evidence": {
                "window_id": "w1",
                "dataset_id": "toy",
                "intensity": "high",
                "periodicity": "strong",
                "dominant_axis": "acc_x",
                "dominant_frequency": "mid",
                "cross_channel_relation": "no clear relation",
                "lead_lag": "no clear lag",
                "burstiness": "smooth",
                "trend_segments": ["rise"],
            },
            "caption_selection": {
                "answer_index": 1,
                "candidates": [{"text": "bad caption"}, {"text": "good caption"}],
            },
            "positive": {"text": "good caption"},
            "counterfactuals": [{"text": "bad caption"}],
        }
    ]
    embedder = FakeEmbedder(
        {
            "The motion intensity is high. The signal shows strong periodicity. The dominant movement axis is acc_x. The dominant frequency is mid. The movement is smooth. The main trend pattern is rise.": [
                1.0,
                0.0,
            ],
            "good caption": [1.0, 0.0],
            "bad caption": [0.0, 1.0],
        }
    )

    metrics, rows = evaluate_embedding_grounding(records, embedder)

    assert metrics["caption_selection_accuracy"] == 1.0
    assert metrics["cf_reject_accuracy"] == 1.0
    assert metrics["cf_reject_f1"] == 1.0
    assert metrics["cf_pairwise_reject_accuracy"] == 1.0
    assert metrics["n_eval_records"] == 1
    assert rows[0]["caption_scores"][1] > rows[0]["caption_scores"][0]


def test_evaluate_embedding_grounding_batches_text_encoding():
    records = [
        {
            "window_id": "w1",
            "evidence": {
                "window_id": "w1",
                "dataset_id": "toy",
                "intensity": "high",
                "periodicity": "strong",
                "dominant_axis": "acc_x",
                "dominant_frequency": "mid",
                "cross_channel_relation": "no clear relation",
                "lead_lag": "no clear lag",
                "burstiness": "smooth",
                "trend_segments": ["rise"],
            },
            "caption_selection": {
                "answer_index": 0,
                "candidates": [{"text": "good caption"}, {"text": "bad caption"}],
            },
            "positive": {"text": "good caption"},
            "counterfactuals": [{"text": "bad caption"}],
        }
    ]
    embedder = FakeEmbedder(
        {
            "The motion intensity is high. The signal shows strong periodicity. The dominant movement axis is acc_x. The dominant frequency is mid. The movement is smooth. The main trend pattern is rise.": [
                1.0,
                0.0,
            ],
            "good caption": [1.0, 0.0],
            "bad caption": [0.0, 1.0],
        }
    )

    evaluate_embedding_grounding(records, embedder)

    assert len(embedder.calls) == 1

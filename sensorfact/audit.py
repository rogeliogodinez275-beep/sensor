from __future__ import annotations

from collections import Counter
from typing import Iterable


def _candidate_texts(record: dict) -> list[str]:
    texts = []
    if record.get("positive"):
        texts.append(str(record["positive"].get("text", "")))
    texts.extend(str(item.get("text", "")) for item in record.get("counterfactuals", []))
    texts.extend(
        str(item.get("text", ""))
        for item in record.get("caption_selection", {}).get("candidates", [])
    )
    if record.get("evidence_text"):
        texts.append(str(record["evidence_text"]))
    return texts


def answer_position_counts(records: Iterable[dict]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for record in records:
        candidates = record.get("caption_selection", {}).get("candidates", [])
        for idx, item in enumerate(candidates):
            if item.get("supported") is True:
                counts[idx] += 1
                break
    return dict(sorted(counts.items()))


def forbidden_shortcut_hits(records: Iterable[dict], phrases: list[str]) -> dict[str, int]:
    counts = {phrase: 0 for phrase in phrases}
    lowered_phrases = [(phrase, phrase.lower()) for phrase in phrases]
    for record in records:
        text = "\n".join(_candidate_texts(record)).lower()
        for phrase, lowered in lowered_phrases:
            if lowered in text:
                counts[phrase] += 1
    return counts


def changed_fact_counts(records: Iterable[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for item in record.get("counterfactuals", []):
            changed_facts = item.get("changed_facts")
            if changed_facts:
                counts.update(str(fact) for fact in changed_facts)
            elif item.get("changed_fact"):
                counts[str(item["changed_fact"])] += 1
    return dict(sorted(counts.items()))


def subject_overlap(train_rows: Iterable[dict], test_rows: Iterable[dict]) -> set[str]:
    train_subjects = {
        str(row.get("subject_id"))
        for row in train_rows
        if row.get("subject_id") is not None
    }
    test_subjects = {
        str(row.get("subject_id"))
        for row in test_rows
        if row.get("subject_id") is not None
    }
    return train_subjects & test_subjects

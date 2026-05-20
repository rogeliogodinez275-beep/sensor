from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.io import read_jsonl, write_jsonl
from sensorfact.structured_verifier_eval import extract_claim_fields


def candidate_soft_score(candidate_text: str, evidence_fields: dict[str, str]) -> tuple[int, int]:
    claims = extract_claim_fields(candidate_text)
    if not claims:
        return (0, 0)
    matches = 0
    mismatches = 0
    for field, value in claims.items():
        observed = evidence_fields.get(field)
        if observed is None:
            continue
        if observed == value:
            matches += 1
        else:
            mismatches += 1
    return (matches, -mismatches)


def constrained_candidate_indices(
    structured_row: dict,
    candidate_texts: list[str],
    *,
    top_k: int = 3,
) -> list[int]:
    caption_scores = [float(x) for x in structured_row.get("caption_scores", [])]
    positive = [idx for idx, score in enumerate(caption_scores) if score >= 0.5]
    if len(positive) > 1:
        return positive
    evidence_fields = structured_row.get("evidence_fields", {})
    ranked = sorted(
        range(len(candidate_texts)),
        key=lambda idx: (candidate_soft_score(candidate_texts[idx], evidence_fields), -idx),
        reverse=True,
    )
    return ranked[: min(max(1, top_k), len(candidate_texts))]


def build_subset_records(
    benchmark_records: list[dict],
    structured_rows: list[dict],
    *,
    top_k: int = 3,
    ambiguous_only: bool = True,
) -> list[dict]:
    structured_by_window = {str(row["window_id"]): row for row in structured_rows}
    subset_records: list[dict] = []
    for record in benchmark_records:
        window_id = str(record["window_id"])
        structured = structured_by_window[window_id]
        caption_scores = [float(x) for x in structured.get("caption_scores", [])]
        positive = [idx for idx, score in enumerate(caption_scores) if score >= 0.5]
        ambiguous = len(positive) != 1
        if ambiguous_only and not ambiguous:
            continue
        candidates = list(record["caption_selection"]["candidates"])
        selected = constrained_candidate_indices(structured, [c["text"] for c in candidates], top_k=top_k)
        remapped_candidates = [candidates[idx] for idx in selected]
        answer_index = int(record["caption_selection"]["answer_index"])
        if answer_index not in selected:
            raise ValueError(
                "Constrained subset dropped the gold caption for "
                f"window_id={window_id}, answer_index={answer_index}, selected={selected}"
            )
        remapped_answer = selected.index(answer_index)
        subset_record = json.loads(json.dumps(record))
        subset_record["caption_selection"]["candidates"] = remapped_candidates
        subset_record["caption_selection"]["answer_index"] = remapped_answer
        subset_record["candidate_index_map"] = selected
        subset_record["constrained_caption_subset"] = {
            "ambiguous_window": ambiguous,
            "top_k": top_k,
            "selected_count": len(selected),
            "original_candidate_count": len(candidates),
        }
        subset_records.append(subset_record)
    return subset_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build constrained caption subset benchmark for ambiguous hybrid windows.")
    parser.add_argument("--benchmark-path", required=True)
    parser.add_argument("--structured-rows", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--all-windows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_records = list(read_jsonl(args.benchmark_path))
    structured_rows = list(read_jsonl(args.structured_rows))
    subset_records = build_subset_records(
        benchmark_records,
        structured_rows,
        top_k=args.top_k,
        ambiguous_only=not args.all_windows,
    )
    write_jsonl(args.output_path, subset_records)
    print(f"Wrote {len(subset_records)} constrained records to {args.output_path}")


if __name__ == "__main__":
    main()

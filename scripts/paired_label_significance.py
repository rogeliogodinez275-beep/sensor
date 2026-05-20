from __future__ import annotations

import argparse
from math import comb
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.io import read_jsonl, write_json


def _is_correct(row: dict) -> bool:
    prediction = row.get("caption_prediction")
    if prediction is None:
        return False
    return int(prediction) == int(row["caption_answer_index"])


def _binom_pmf(k: int, n: int) -> float:
    return comb(n, k) * (0.5**n)


def mcnemar_midp(*, primary_only_correct: int, challenger_only_correct: int) -> dict:
    n_discordant = primary_only_correct + challenger_only_correct
    if n_discordant == 0:
        return {
            "primary_only_correct": primary_only_correct,
            "challenger_only_correct": challenger_only_correct,
            "discordant_count": 0,
            "two_sided_p_value": 1.0,
            "midp_value": 1.0,
        }
    smaller = min(primary_only_correct, challenger_only_correct)
    exact_tail = sum(_binom_pmf(k, n_discordant) for k in range(0, smaller + 1))
    exact_two_sided = min(1.0, 2.0 * exact_tail)
    mid_tail = exact_tail - 0.5 * _binom_pmf(smaller, n_discordant)
    midp = min(1.0, 2.0 * mid_tail)
    return {
        "primary_only_correct": primary_only_correct,
        "challenger_only_correct": challenger_only_correct,
        "discordant_count": n_discordant,
        "two_sided_p_value": exact_two_sided,
        "midp_value": midp,
    }


def compare_paired_rows(primary_rows: list[dict], challenger_rows: list[dict]) -> dict:
    challenger_by_window = {str(row["window_id"]): row for row in challenger_rows}
    n_pairs = 0
    primary_correct_count = 0
    challenger_correct_count = 0
    primary_only = 0
    challenger_only = 0
    both_correct = 0
    both_wrong = 0
    missing_challenger = 0
    for primary in primary_rows:
        challenger = challenger_by_window.get(str(primary["window_id"]))
        if challenger is None:
            missing_challenger += 1
            continue
        n_pairs += 1
        primary_correct = _is_correct(primary)
        challenger_correct = _is_correct(challenger)
        primary_correct_count += int(primary_correct)
        challenger_correct_count += int(challenger_correct)
        if primary_correct and challenger_correct:
            both_correct += 1
        elif primary_correct and not challenger_correct:
            primary_only += 1
        elif challenger_correct and not primary_correct:
            challenger_only += 1
        else:
            both_wrong += 1
    primary_accuracy = primary_correct_count / max(1, n_pairs)
    challenger_accuracy = challenger_correct_count / max(1, n_pairs)
    result = {
        "n_pairs": n_pairs,
        "missing_challenger": missing_challenger,
        "primary_accuracy": primary_accuracy,
        "challenger_accuracy": challenger_accuracy,
        "delta": challenger_accuracy - primary_accuracy,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "primary_only_correct": primary_only,
        "challenger_only_correct": challenger_only,
    }
    result["mcnemar"] = mcnemar_midp(
        primary_only_correct=primary_only,
        challenger_only_correct=challenger_only,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired label-level significance for caption rows.")
    parser.add_argument("--primary-rows", required=True)
    parser.add_argument("--challenger-rows", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--system-name", default="paired_label_significance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_rows = list(read_jsonl(args.primary_rows))
    challenger_rows = list(read_jsonl(args.challenger_rows))
    result = compare_paired_rows(primary_rows, challenger_rows)
    result["system"] = args.system_name
    result["primary_rows"] = args.primary_rows
    result["challenger_rows"] = args.challenger_rows
    write_json(args.output_json, result)
    print("Paired label significance finished.")
    for key in ["n_pairs", "primary_accuracy", "challenger_accuracy", "delta"]:
        print(f"{key}: {result[key]}")
    print(f"mcnemar_midp: {result['mcnemar']['midp_value']}")


if __name__ == "__main__":
    main()


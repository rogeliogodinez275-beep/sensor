from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASETS = {
    "ucihar": "UCI HAR",
    "wisdm": "WISDM",
    "mhealth": "MHEALTH",
}

VOTE5_CAPTION = {
    "ucihar": 0.8565,
    "wisdm": 0.8859,
    "mhealth": 0.9351,
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_json(root: Path, patterns: list[str]) -> dict | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return _read_json(matches[0])
    return None


def _metric(metrics: dict | None, *keys: str) -> float | None:
    if not metrics:
        return None
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return None


def _fmt(value: float | None) -> str:
    return "信息不足" if value is None else f"{value:.4f}"


def _bootstrap_table(root: Path) -> tuple[str, str]:
    payload = _first_json(root, ["confidence_gated_forced_choice_paired_bootstrap.json"])
    if not payload:
        return "信息不足", "未找到 paired bootstrap 输出。"
    rows = ["| Dataset | Delta | 95% CI | 结论 |", "|---|---:|---:|---|"]
    significant = 0
    positive_nonsig = 0
    for dataset, label in DATASETS.items():
        item = payload.get(dataset) or payload.get(label) or payload.get(f"{dataset}_hard_v3")
        if not item:
            rows.append(f"| {label} | 信息不足 | 信息不足 | 信息不足 |")
            continue
        delta = _metric(item, "delta", "mean_delta")
        low = _metric(item, "ci_low", "lower", "p025")
        high = _metric(item, "ci_high", "upper", "p975")
        is_sig = low is not None and high is not None and low > 0.0
        if is_sig:
            significant += 1
            conclusion = "显著正向"
        elif delta is not None and delta > 0:
            positive_nonsig += 1
            conclusion = "非显著正向"
        else:
            conclusion = "未支持正向"
        rows.append(f"| {label} | {_fmt(delta)} | [{_fmt(low)}, {_fmt(high)}] | {conclusion} |")
    headline = "两显著一非显著" if significant == 2 and positive_nonsig >= 1 else f"{significant} 个显著正向"
    return headline, "\n".join(rows)


def build_lock_report(outputs_dir: str | Path) -> str:
    root = Path(outputs_dir)
    headline, bootstrap_md = _bootstrap_table(root)
    main_rows = ["| Dataset | Vote5 Caption Acc | Gated Caption Acc | Delta | CF F1 |", "|---|---:|---:|---:|---:|"]
    risk_rows = ["| Dataset | Full-candidate FC | Hidden-evidence FC | 风险解释 |", "|---|---:|---:|---|"]

    for dataset, label in DATASETS.items():
        gated = _first_json(
            root,
            [f"*gated*choice*logprob*{dataset}*metrics.json", f"*{dataset}*gated*metrics.json"],
        )
        full = _first_json(root, [f"*choice_logprob*{dataset}*full_baseline_metrics.json"])
        hidden = _first_json(root, [f"*choice_logprob*{dataset}*hidden_evidence*metrics.json"])
        gated_acc = _metric(gated, "caption_selection_accuracy")
        vote5_acc = VOTE5_CAPTION.get(dataset)
        delta = None if gated_acc is None or vote5_acc is None else gated_acc - vote5_acc
        cf_f1 = _metric(gated, "cf_reject_f1", "counterfactual_f1", "cf_f1")
        main_rows.append(f"| {label} | {_fmt(vote5_acc)} | {_fmt(gated_acc)} | {_fmt(delta)} | {_fmt(cf_f1)} |")
        hidden_acc = _metric(hidden, "caption_selection_accuracy")
        full_acc = _metric(full, "caption_selection_accuracy")
        risk = "需正面讨论语言先验/候选偏差" if hidden_acc is not None and hidden_acc >= 0.7 else "信息不足或风险较低"
        risk_rows.append(f"| {label} | {_fmt(full_acc)} | {_fmt(hidden_acc)} | {risk} |")

    return f"""# Evidence-First Result Lock

## Locked Claim

当前主张锁定为：confidence-gated forced-choice reranking 相对 order-vote5 的 caption 结果为 **{headline}**。CF F1 不归功于 reranker；support / counterfactual rejection 仍由 structured-only backbone 负责。

## Main Caption Results

{chr(10).join(main_rows)}

## Paired Bootstrap

{bootstrap_md}

## Risk Controls

{chr(10).join(risk_rows)}

## Writing Guardrails

- 可以写：structured verifier gives verifiable support/rejection.
- 可以写：gated forced-choice improves residual caption ambiguity.
- 必须写：hidden-evidence / shuffled / numeric-mask controls reveal and bound language-prior risk.
- 不要写：reranker 提升了 CF F1 或证明 LLM 已经完整理解 sensor evidence.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock evidence-first main results into a Markdown report.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_lock_report(args.outputs_dir)
    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8", newline="\n")
    print("Evidence-first lock report written.")
    print(f"output_md: {output}")


if __name__ == "__main__":
    main()


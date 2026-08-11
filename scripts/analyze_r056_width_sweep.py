#!/usr/bin/env python3
"""Rank R056 forest widths by SGLang acceptance, parity, and paired speed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_r056_sglang_e2e import (
    bootstrap_ratio,
    indexed,
    load,
    median_rows,
    parity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--domino", type=Path, required=True)
    parser.add_argument("--forest", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = load(args.target)
    domino = load(args.domino)
    domino_rows = median_rows([domino])
    domino_index = indexed(domino)
    ids = sorted(domino_rows)
    domino_verifies = sum(int(row["spec_verify_ct"]) for row in domino_index.values())
    domino_accepted = sum(
        int(row["spec_accept_token_num"]) for row in domino_index.values()
    )
    domino_eal = domino_accepted / domino_verifies

    widths = {}
    for forest_path in args.forest:
        forest = load(forest_path)
        label = str(forest["mode"])
        forest_rows = median_rows([forest])
        ratio = bootstrap_ratio(
            ids,
            domino_rows,
            forest_rows,
            key="server_e2e_seconds",
            draws=args.bootstrap_draws,
            seed=20260810 + sum(ord(char) for char in label),
        )
        forest_index = indexed(forest)
        verifies = sum(int(row["spec_verify_ct"]) for row in forest_index.values())
        accepted = sum(
            int(row["spec_accept_token_num"]) for row in forest_index.values()
        )
        forest_eal = accepted / verifies
        widths[label] = {
            "target_parity": parity(target, forest),
            "domino_parity": parity(domino, forest),
            "accepted_draft_eal": forest_eal,
            "accepted_draft_eal_delta_vs_domino": forest_eal - domino_eal,
            "output_advance": forest["total_completion_tokens"] / verifies,
            "aggregate_server_tokens_per_second": forest[
                "aggregate_server_tokens_per_second"
            ],
            "throughput_ratio_vs_domino": ratio,
        }

    speed_order = sorted(
        widths,
        key=lambda label: widths[label]["throughput_ratio_vs_domino"]["point"],
        reverse=True,
    )
    report = {
        "format": "r056_width_sweep_v1",
        "workload": {
            "num_prompts": target["num_prompts"],
            "tokens_per_prompt": target["max_new_tokens"],
        },
        "domino": {
            "target_parity": parity(target, domino),
            "accepted_draft_eal": domino_eal,
            "output_advance": domino["total_completion_tokens"] / domino_verifies,
            "aggregate_server_tokens_per_second": domino[
                "aggregate_server_tokens_per_second"
            ],
        },
        "widths": widths,
        "speed_order": speed_order,
        "best_screen_width": speed_order[0],
        "inputs": {
            "target": str(args.target),
            "domino": str(args.domino),
            "forest": [str(path) for path in args.forest],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

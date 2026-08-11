#!/usr/bin/env python3
"""Freeze the independently recomputable PCLD capacity-support receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sph.japd_data import load_rollout_records
from sph.pcld_data import (
    attach_pcld_sidecar,
    build_capacity_support_receipt,
    calibrate_epsilon_from_records,
    load_manifest,
    load_pcld_sidecar,
    select_manifest_group,
    validate_manifest_source,
    validate_sidecar_receipt,
    validate_sidecar_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--group", choices=("capacity",), default="capacity")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest = load_manifest(args.manifest)
    validate_manifest_source(manifest, rollout=args.rollout, split=args.split)
    _, rollout = load_rollout_records(args.rollout, split=args.split)
    records = select_manifest_group(rollout, manifest, args.group)
    metadata, sidecar = load_pcld_sidecar(args.sidecar)
    validate_sidecar_source(
        metadata,
        rollout=args.rollout,
        target=args.target,
        split=args.split,
        group=args.group,
    )
    replay = validate_sidecar_receipt(args.sidecar, metadata)
    if replay.get("verified") is not True:
        raise RuntimeError("PCLD capacity receipt requires verified replay")
    records = attach_pcld_sidecar(records, sidecar, require_exact_keys=True)
    epsilon_num = calibrate_epsilon_from_records(records)
    receipt = build_capacity_support_receipt(
        records,
        epsilon_num,
        rollout=args.rollout,
        manifest=args.manifest,
        target=args.target,
        sidecar=args.sidecar,
        split=args.split,
        group=args.group,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "denominators": {
                    name: branch["j2_denominator"]
                    for name, branch in receipt["support"]["branches"].items()
                },
                "epsilon_num": receipt["support"]["epsilon_num"],
                "stable_effective_blocks": receipt["support"][
                    "stable_effective_blocks"
                ],
                "stable_support_rows": receipt["support"]["stable_support_rows"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return receipt


if __name__ == "__main__":
    run(parse_args())

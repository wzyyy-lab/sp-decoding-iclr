from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


PROJECT = Path(__file__).resolve().parents[1]
SLURM = PROJECT / "scripts/slurm"
WRAPPERS = [
    SLURM / "pros_gate_split.sbatch",
    SLURM / "pros_gate_outcomes.sbatch",
    SLURM / "pros_gate_artifact_audit.sbatch",
    SLURM / "pros_gate_capacity_materialize.sbatch",
    SLURM / "pros_gate_capacity.sbatch",
]
NUMERIC_CUDA_SMOKE = SLURM / "pros_gate_numeric_cuda_smoke.sbatch"
FAILED_FIT_WRAPPER = SLURM / "pros_gate_fit.sbatch"
FIT_WRAPPER = SLURM / "pros_gate_fit_publication_rescue.sbatch"
FALSIFIER_WRAPPER = SLURM / "pros_gate_falsifier.sbatch"
NUMERIC_PORTABILITY_DIAGNOSTIC = (
    SLURM / "pros_gate_numeric_portability_diagnostic.sbatch"
)
STATIC_PIN = re.compile(r'^echo "([0-9a-f]{64})  ([^"$]+)" \| sha256sum -c -$')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_static_source_pin_matches_the_current_file() -> None:
    observed = 0
    historical_manifest = json.loads(
        (
            PROJECT
            / "refine-logs/direct-safety-gate/R079_SOURCE_CLOSURE_NUMERIC_V2.json"
        ).read_text(encoding="utf-8")
    )
    historical_hashes = {
        row["path"]: row["sha256"] for row in historical_manifest["files"]
    }
    for wrapper in [
        *WRAPPERS,
        NUMERIC_CUDA_SMOKE,
        FIT_WRAPPER,
        FALSIFIER_WRAPPER,
    ]:
        for line in wrapper.read_text(encoding="utf-8").splitlines():
            match = STATIC_PIN.match(line)
            if match is None:
                continue
            expected, raw_path = match.groups()
            # Real-data manifests are hash-pinned in the wrappers but remain
            # unopened during CPU/static review.
            if raw_path.startswith("artifacts/"):
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = PROJECT / path
            assert path.is_file(), (wrapper.name, raw_path)
            if wrapper in {FIT_WRAPPER, FALSIFIER_WRAPPER} or raw_path.startswith(
                "tests/"
            ) or raw_path.endswith(".json"):
                assert _sha256(path) == expected, (wrapper.name, raw_path)
            else:
                # Completed R079/R080 wrappers intentionally remain bound to
                # their immutable historical source snapshot.  Later protocol
                # edits must make those wrappers non-rerunnable rather than
                # silently retargeting them to current files.
                assert historical_hashes.get(raw_path) == expected, (
                    wrapper.name,
                    raw_path,
                )
            observed += 1
    assert observed >= 20


def test_numeric_cuda_smoke_is_single_v2_roundtrip_fail_on_skip_test() -> None:
    source = NUMERIC_CUDA_SMOKE.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --time=00:05:00" in source
    assert "PROS_REQUIRE_CUDA=1" in source
    assert "assert torch.cuda.is_available()" in source
    assert "test_cuda_same_device_and_portable_independent_roundtrip" in source
    assert "artifacts/pros_gate" not in source
    assert "--expected-manifest-sha256 \"$SOURCE_MANIFEST_SHA256\"" in source


def test_numeric_portability_diagnostic_is_single_label_blind_read_only_job() -> None:
    source = NUMERIC_PORTABILITY_DIAGNOSTIC.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --array" not in source
    assert "#SBATCH --time=00:30:00" in source
    assert "diagnose_direct_safety_numeric_portability.py" in source
    assert "--expected-split-manifest-sha256 \"$SPLIT_MANIFEST_SHA256\"" in source
    assert "R079_CONTINUOUS_NUMERIC_DIAGNOSTIC_SOURCE_CLOSURE.json" in source
    assert "--manifest \"$SOURCE_MANIFEST\"" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "pytest" not in source
    assert "\\\n  --output" not in source
    assert "falsifier" not in source
    assert "validation" not in source
    assert "reserved" not in source
    assert "capacity" not in source
    assert "outcomes" not in source
    assert ">/dev/null" in source
    assert _sha256(NUMERIC_PORTABILITY_DIAGNOSTIC) == (
        "1bedcf8b3418ebff72378d0c02473b4fae9a2ba027e8fd42ad7939996b9fefcb"
    )


def test_stage_wrappers_require_prior_independent_go_receipts() -> None:
    outcomes = (SLURM / "pros_gate_outcomes.sbatch").read_text(encoding="utf-8")
    capacity_data = (
        SLURM / "pros_gate_capacity_materialize.sbatch"
    ).read_text(encoding="utf-8")
    capacity = (SLURM / "pros_gate_capacity.sbatch").read_text(encoding="utf-8")
    assert "SPLIT_AUDIT_SHA256" in outcomes
    assert "OUTCOMES_AUDIT_SHA256" in capacity_data
    assert "CAPACITY_AUDIT_SHA256" in capacity
    for source in (outcomes, capacity_data, capacity):
        assert 'scripts/verify_pros_gate_receipt.py' in source
        assert "--expected-receipt-sha256" in source
        assert '--source-manifest-sha256 "$SOURCE_MANIFEST_SHA256"' in source

    assert "verify_pros_gate_receipt.py split" in outcomes
    assert '--split-manifest-sha256 "$SPLIT_MANIFEST_SHA256"' in outcomes

    assert "CHECKPOINT_METADATA_SHA256" in capacity_data
    assert "verify_pros_gate_receipt.py outcomes" in capacity_data
    assert '--fit-metadata-sha256 "$FIT_METADATA_SHA256"' in capacity_data
    assert (
        '--checkpoint-metadata-sha256 "$CHECKPOINT_METADATA_SHA256"'
        in capacity_data
    )

    assert "FIT_METADATA_SHA256" in capacity
    assert "SPLIT_MANIFEST_SHA256" in capacity
    assert "verify_pros_gate_receipt.py capacity" in capacity
    assert (
        '--capacity-metadata-sha256 "$CAPACITY_METADATA_SHA256"' in capacity
    )
    assert '--fit-metadata-sha256 "$FIT_METADATA_SHA256"' in capacity
    assert '--split-manifest-sha256 "$SPLIT_MANIFEST_SHA256"' in capacity

    for source in (outcomes, capacity_data, capacity):
        assert "jq " not in source


def test_artifact_audit_exclusion_manifests_are_split_stage_only() -> None:
    source = (
        SLURM / "pros_gate_artifact_audit.sbatch"
    ).read_text(encoding="utf-8")
    prefix, branches = source.split('case "$AUDIT_STAGE" in', maxsplit=1)
    split_branch, after_split = branches.split("  outcomes)", maxsplit=1)
    outcomes_branch, _ = after_split.split("  capacity)", maxsplit=1)
    exclusion_manifests = (
        "open_perfectblend_100k_v2.jsonl",
        "phase3_development_v3.jsonl",
        "phase3_reserved_test_v3.jsonl",
    )
    for manifest in exclusion_manifests:
        assert manifest not in prefix
        assert manifest in split_branch
        assert manifest not in outcomes_branch


def test_every_wrapper_pins_the_complete_reviewed_source_manifest() -> None:
    expected = (
        "2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4"
    )
    for wrapper in WRAPPERS:
        source = wrapper.read_text(encoding="utf-8")
        assert f"SOURCE_MANIFEST_SHA256={expected}" in source
        assert (
            f'echo "{expected}  '
            "refine-logs/direct-safety-gate/R079_SOURCE_CLOSURE_NUMERIC_V2.json\""
        ) in source
        preflight = '"$PYTHON" src/sph/source_closure.py'
        assert preflight in source
        assert '--expected-manifest-sha256 "$SOURCE_MANIFEST_SHA256"' in source
        assert '--expected-source-manifest-sha256 "$SOURCE_MANIFEST_SHA256"' in source


def test_no_r079_execution_wrapper_exposes_falsifier_outcomes() -> None:
    outcomes = (SLURM / "pros_gate_outcomes.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1" in outcomes
    assert "OUTCOME_SPLIT=fit" in outcomes
    assert "OUTCOME_SPLIT=checkpoint" in outcomes
    assert "falsifier" not in outcomes
    capacity = (SLURM / "pros_gate_capacity.sbatch").read_text(encoding="utf-8")
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in capacity


def test_r082_fit_wrapper_is_one_exact_review_gated_job() -> None:
    source = FIT_WRAPPER.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --array" not in source
    assert "#SBATCH --time=00:30:00" in source
    assert 'pros_gate_fit_${SLURM_JOB_ID}"' in source
    assert 'OUTPUT="$RUN_ROOT/seed0"' in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "R082_SOURCE_CLOSURE_PUBLICATION_RESCUE_V1.json" in source
    assert (
        "SOURCE_MANIFEST_SHA256="
        "f36291a961ea793dbaa888950bc4312d8b53954fcc5ecdb01a5caad4af97e184"
        in source
    )
    assert '--manifest "$SOURCE_MANIFEST"' in source
    assert "verify_pros_gate_receipt.py outcomes" in source
    assert "verify_pros_gate_receipt.py capacity-adjudication" in source
    assert "train_direct_safety_fit.py" in source
    assert '--fit-bundle "$FIT"' in source
    assert '--checkpoint-bundle "$CHECKPOINT"' in source
    assert '--source-manifest "$SOURCE_MANIFEST"' in source
    assert '--expected-wrapper-sha256 "$WRAPPER_SHA256"' in source
    smoke = source.index("publication_filesystem_smoke")
    first_records = source.index("outcomes/fit/records.pt")
    training = source.rindex('"$PYTHON" scripts/train_direct_safety_fit.py')
    assert smoke < first_records < training
    assert _sha256(FIT_WRAPPER) == (
        "1bffe45017da30393f2dbda5bd33b1e14ff1e0ca1a60c0be4fc86f4ad1885c74"
    )
    assert _sha256(FAILED_FIT_WRAPPER) == (
        "059e06cfe90d17a929d6d59999b72eb810ac446c5f6115d936c66e1b2c684e69"
    )
    assert "afterok" not in source
    for forbidden in ("falsifier", "validation", "reserved", "formal"):
        assert forbidden not in source


def test_r083_falsifier_wrapper_is_one_exact_no_retry_opening() -> None:
    source = FALSIFIER_WRAPPER.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --array" not in source
    assert "#SBATCH --time=00:30:00" in source
    assert 'pros_gate_falsifier_${SLURM_JOB_ID}"' in source
    assert 'OUTPUT="$RUN_ROOT/seed0"' in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "R083_SOURCE_CLOSURE_RESCUE_V2.json" in source
    assert (
        "SOURCE_MANIFEST_SHA256="
        "204c025305a9665803e714708dc0eab29394644d5905ad76f1715c7309020878"
        in source
    )
    assert '--manifest "$SOURCE_MANIFEST"' in source
    assert "verify_pros_gate_receipt.py split" in source
    assert "verify_published_directory(Path(sys.argv[1]))" in source
    assert '--split-audit-receipt "$SPLIT_AUDIT"' in source
    assert '--r082-output "$R082_OUTPUT"' in source
    assert '--source-manifest "$SOURCE_MANIFEST"' in source
    assert '--expected-wrapper-sha256 "$WRAPPER_SHA256"' in source
    assert source.count('"$PYTHON" scripts/evaluate_direct_safety_falsifier.py') == 1
    assert "set +e" in source
    assert "EVALUATOR_RC=$?" in source
    assert '"$EVALUATOR_RC" -ne 0 && "$EVALUATOR_RC" -ne 2' in source
    assert 'exit "$EVALUATOR_RC"' in source

    smoke = source.index("publication_filesystem_smoke")
    canonical_pin = source.index(
        'echo "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320'
    )
    split_pin = source.index(
        'echo "7a572670867bbf6f811aad58c7ed1365f179083cda8187f5402221d554d1f1c0'
    )
    selected_pin = source.index(
        'echo "f3e7c68dafd93528c03deda9710e3d23cf5b0e9e51a7b2ef66200f08201066dc'
    )
    evaluator = source.index(
        '"$PYTHON" scripts/evaluate_direct_safety_falsifier.py'
    )
    consumer = source.rindex(
        "from sph.direct_safety_publication import verify_published_directory"
    )
    assert smoke < canonical_pin < split_pin < selected_pin < evaluator < consumer

    for role in ("producer_train", "validation", "reserved"):
        assert f'--exclusion "{role}=' in source
    for forbidden in (
        "--threshold",
        "--seed",
        "--checkpoint-pass",
        "--validation",
        "--reserved",
        "--formal",
        "--fit-bundle",
        "afterok",
    ):
        assert forbidden not in source
    assert _sha256(FALSIFIER_WRAPPER) == (
        "b20fc9461daac0385b09fbd0840aa5d476dcf71fadd3b6ba3288938b9d124560"
    )

#!/bin/bash
#SBATCH --job-name=pros-falsifier
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-falsifier-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/pros-falsifier-%j.err

set -euo pipefail
umask 077

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
ASSETS="/hpc2hdd/home/zwang668/sp decoding/hf_assets"
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
DATA="$PROJECT/artifacts/canonical/qwen3_4b_phase3_tier1_10035436"
DIRECT_RUN="$PROJECT/artifacts/training/gcls_v4_feature_100k_10133585/compact_axial_additive_d64_full_seed0"
SPLIT_MANIFEST="$PROJECT/artifacts/pros_gate/r079_numeric_v2/split_manifest.json"
SPLIT_AUDIT="$PROJECT/artifacts/pros_gate/r079_numeric_v2/audits/split.json"
R082_OUTPUT="$PROJECT/artifacts/training/pros_gate_fit_10141115/seed0"
SOURCE_MANIFEST="$PROJECT/refine-logs/direct-safety-gate/R083_SOURCE_CLOSURE_RESCUE_V2.json"
SOURCE_MANIFEST_SHA256=204c025305a9665803e714708dc0eab29394644d5905ad76f1715c7309020878
RUN_ROOT="$PROJECT/artifacts/evaluation/pros_gate_falsifier_${SLURM_JOB_ID}"
OUTPUT="$RUN_ROOT/seed0"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$PROJECT"
read -r WRAPPER_SHA256 _ < <(sha256sum "$0")
echo "wrapper_sha256=$WRAPPER_SHA256"

# Freeze the reviewed execution surface before the one-shot data opening.
echo "54d430a6d9d92118e2005e6c985c6c04f0424cfcc86afe60c5eecce5f39aa571  scripts/evaluate_direct_safety_falsifier.py" | sha256sum -c -
echo "90001f5b2f0224e79d8d205bdd781876f7dcbed89273fb55d9ca4c3f53d95b2b  scripts/materialize_direct_safety_artifacts.py" | sha256sum -c -
echo "141abdef88320173fb03c438c5c54907118a64bcf0a29932268be389fb4f5f1c  src/sph/direct_safety_publication.py" | sha256sum -c -
echo "9bc069a5d70ff6f074e323a076c065eb7ae8ff903f4ed83cd1f6a317b790bff4  scripts/train_direct_safety_fit.py" | sha256sum -c -
echo "844e2aac1149d88af5710a3186ba3e6f8384f6dc346a2ad151a534e9ca74e4dd  scripts/verify_pros_gate_receipt.py" | sha256sum -c -
echo "852915273e330344c1afe472f9d0c2b4789b564e6636af897051486fbaa3a6d9  src/sph/direct_safety_artifacts.py" | sha256sum -c -
echo "a8aa727c0dc6e82c7d0c1a2ac65f2cef78941ac4619f02bf0740d3bb03d7b30f  src/sph/direct_safety_gate.py" | sha256sum -c -
echo "3b2f0fbab3a0a683a7f5e142b88f79c96960b6427e48253861c090e85768437c  src/sph/direct_safety_numeric_policy.py" | sha256sum -c -
echo "47f91341c7d222fa8a62d55b9aad16d75e54582d30e632aa8b9d63e6a678a0ea  src/sph/direct_safety_protocol.py" | sha256sum -c -
echo "9fac5193a09c3def52b8e1360c0790d2497597f4daa8f5b67a16df5bbb1a31d4  src/sph/source_closure.py" | sha256sum -c -
echo "204c025305a9665803e714708dc0eab29394644d5905ad76f1715c7309020878  refine-logs/direct-safety-gate/R083_SOURCE_CLOSURE_RESCUE_V2.json" | sha256sum -c -
"$PYTHON" src/sph/source_closure.py \
  --project "$PROJECT" \
  --manifest "$SOURCE_MANIFEST" \
  --expected-manifest-sha256 "$SOURCE_MANIFEST_SHA256"

# Exercise the exact no-clobber directory-commit protocol before reading any
# split content, canonical data, frozen checkpoint, or exclusion manifest.
"$PYTHON" -c 'import json,sys; from pathlib import Path; from sph.direct_safety_publication import publication_filesystem_smoke; print(json.dumps(publication_filesystem_smoke(Path(sys.argv[1]), project=Path(sys.argv[2]), job_id=sys.argv[3], wrapper_sha256=sys.argv[4], source_manifest=Path(sys.argv[5]), source_manifest_sha256=sys.argv[6], entrypoint_path=sys.argv[7]), sort_keys=True))' "$RUN_ROOT" "$PROJECT" "$SLURM_JOB_ID" "$WRAPPER_SHA256" "$SOURCE_MANIFEST" "$SOURCE_MANIFEST_SHA256" scripts/evaluate_direct_safety_falsifier.py

# Only after the publication smoke succeeds may this unique opening inspect
# any frozen scientific input.  Every identity is pinned before CUDA work.
echo "0dbca3e9667f6578b02559e2934ee79a677e45d058c68271c94db6e8f4338320  artifacts/canonical/qwen3_4b_phase3_tier1_10035436/metadata.json" | sha256sum -c -
echo "7a572670867bbf6f811aad58c7ed1365f179083cda8187f5402221d554d1f1c0  artifacts/pros_gate/r079_numeric_v2/split_manifest.json" | sha256sum -c -
echo "3df67764ef6dc7e8a827277c34730233bc7a5155451fe42c6b98b99bf7a7ef76  artifacts/pros_gate/r079_numeric_v2/audits/split.json" | sha256sum -c -
echo "9486d976c115fc4313864bb5e2b9b84ddf0145311e58f838b0925c184d6d9a0e  artifacts/training/gcls_v4_feature_100k_10133585/compact_axial_additive_d64_full_seed0/best.pt" | sha256sum -c -
echo "9ec91a1faceec05c1f798482d9ebf45a8ed194f2bddfaa18228317476aad4aef  artifacts/training/gcls_v4_feature_100k_10133585/compact_axial_additive_d64_full_seed0/metrics.json" | sha256sum -c -
echo "91c51864339321436ffa560470667bb99d1a22e955e7273fcd7a3d5711f3f508  artifacts/training/pros_gate_fit_10141115/seed0/READY.json" | sha256sum -c -
echo "9c17020a95d42725a097e6847b5023d6c6e7971a0647fb4475727e93d2297a5e  artifacts/training/pros_gate_fit_10141115/seed0/PUBLICATION_MANIFEST.json" | sha256sum -c -
echo "ba7fd8264813b7baa4927d94d1acd8d697bad5175d1acdfcc62dea5a103491b0  artifacts/training/pros_gate_fit_10141115/seed0/metrics.json" | sha256sum -c -
echo "a0abcfd4e56229647afd1dda5ca2fe861f7dbf21d00c09fe275ba8a66826c142  artifacts/training/pros_gate_fit_10141115/seed0/selected_bundle.json" | sha256sum -c -
echo "f3e7c68dafd93528c03deda9710e3d23cf5b0e9e51a7b2ef66200f08201066dc  artifacts/training/pros_gate_fit_10141115/seed0/selected.pt" | sha256sum -c -
echo "1fd6beb846c8a46e874875628b37ac1094e3ac61eaddbfef3bb9d7b7dbd88749  artifacts/training/pros_gate_fit_10141115/seed0/selected_checkpoint_records.pt" | sha256sum -c -
echo "2c5a76ca96f9f6afb08d47a116cdb17fdce6de11386b6350c5e3e485732f4f16  artifacts/training/pros_gate_fit_10141115/seed0/ridge_model.pt" | sha256sum -c -
echo "2a4f6457fc1d85a3ad42ec7430ecb4d28d532d4bf57f06371ac019526c3dc809  artifacts/training/pros_gate_fit_10141115/seed0/ridge_freeze_receipt.json" | sha256sum -c -
test "$(stat -c %s artifacts/manifests/open_perfectblend_100k_v2.jsonl)" -eq 92590866
test "$(stat -c %s artifacts/manifests/phase3_development_v3.jsonl)" -eq 1418202
test "$(stat -c %s artifacts/manifests/phase3_reserved_test_v3.jsonl)" -eq 429359
echo "b05087a56e8e717605415026421f7bae23092eb7cb9509361a36932f80260e3a  artifacts/manifests/open_perfectblend_100k_v2.jsonl" | sha256sum -c -
echo "e16374068e9c8904214fbf282b4adb6187a0b099db5c37e79660fc46a2801d01  artifacts/manifests/phase3_development_v3.jsonl" | sha256sum -c -
echo "ae25467fbb52b7091c8d9a5f98776b11ccf76e87e781850b5638734548a53bb4  artifacts/manifests/phase3_reserved_test_v3.jsonl" | sha256sum -c -

"$PYTHON" scripts/verify_pros_gate_receipt.py split \
  --project "$PROJECT" \
  --receipt "$SPLIT_AUDIT" \
  --expected-receipt-sha256 3df67764ef6dc7e8a827277c34730233bc7a5155451fe42c6b98b99bf7a7ef76 \
  --split-manifest-sha256 7a572670867bbf6f811aad58c7ed1365f179083cda8187f5402221d554d1f1c0 \
  --source-manifest-sha256 2bd264d770b9aa89e1b25598add7ecf3755a457e9f2f542f0533cfe04f3d48a4
"$PYTHON" -c 'import json,sys; from pathlib import Path; from scripts.train_direct_safety_fit import verify_published_directory; print(json.dumps(verify_published_directory(Path(sys.argv[1])), sort_keys=True))' "$R082_OUTPUT"
nvidia-smi

# There is intentionally no retry path.  Exit 2 is a fully published
# scientific FAIL; operational failures use their original non-{0,2} status.
set +e
"$PYTHON" scripts/evaluate_direct_safety_falsifier.py \
  --data "$DATA" \
  --split-manifest "$SPLIT_MANIFEST" \
  --split-audit-receipt "$SPLIT_AUDIT" \
  --direct-run "$DIRECT_RUN" \
  --target "$ASSETS/models/Qwen3-4B" \
  --r082-output "$R082_OUTPUT" \
  --source-manifest "$SOURCE_MANIFEST" \
  --expected-source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --expected-wrapper-sha256 "$WRAPPER_SHA256" \
  --output "$OUTPUT" \
  --exclusion "producer_train=$PROJECT/artifacts/manifests/open_perfectblend_100k_v2.jsonl" \
  --exclusion "validation=$PROJECT/artifacts/manifests/phase3_development_v3.jsonl" \
  --exclusion "reserved=$PROJECT/artifacts/manifests/phase3_reserved_test_v3.jsonl"
EVALUATOR_RC=$?
set -e

if [[ "$EVALUATOR_RC" -ne 0 && "$EVALUATOR_RC" -ne 2 ]]; then
  exit "$EVALUATOR_RC"
fi
"$PYTHON" -c 'import json,sys; from pathlib import Path; from sph.direct_safety_publication import verify_published_directory; print(json.dumps(verify_published_directory(Path(sys.argv[1])), sort_keys=True))' "$OUTPUT"
exit "$EVALUATOR_RC"

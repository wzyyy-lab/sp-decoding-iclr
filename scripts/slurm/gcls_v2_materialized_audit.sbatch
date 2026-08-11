#!/bin/bash
#SBATCH --job-name=gcls-v2-matcheck
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v2-matcheck-%j.out
#SBATCH --error=/hpc2hdd/home/zwang668/dflash-iclr/artifacts/logs/gcls-v2-matcheck-%j.err

set -euo pipefail

PROJECT=/hpc2hdd/home/zwang668/dflash-iclr
PYTHON="$PROJECT/third_party/Domino/.venv/bin/python"
COLLECTION="$PROJECT/artifacts/canonical/qwen3_4b_open_perfectblend_25k_seed20260730"

export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=2

cd "$PROJECT"
"$PYTHON" -c 'import hashlib,json,sys; from pathlib import Path; from sph.data import CanonicalBlockDataset; root=Path(sys.argv[1]); dataset=CanonicalBlockDataset(root); metadata=dataset.metadata; ids={str(record["sample_id"]) for record in dataset.records}; digest=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(); expected="a3d25eba926ea8dc474d59b8a4bf3eabef6953d198bf2630d525344c3236fa73"; assert len(dataset.records)==199818; assert len(ids)==25000; assert digest==expected; assert metadata["num_blocks"]==199818; assert metadata["num_collected_samples"]==25000; assert dataset.base_greedy_witness_status=="none_legacy"; print("MATERIALIZED_AUDIT_PASS", len(dataset.records), len(ids), digest, len(metadata["shards"]), dataset.base_greedy_witness_status)' "$COLLECTION"

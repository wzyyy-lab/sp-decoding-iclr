import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.train_survival_head import (
    assert_prompt_disjoint_splits,
    validate_target_embedding_identity,
)


class TrainingProtocolTest(unittest.TestCase):
    def test_prompt_overlap_is_rejected(self) -> None:
        train = SimpleNamespace(records=[{"sample_id": "shared"}])
        validation = SimpleNamespace(records=[{"sample_id": "shared"}])
        with self.assertRaisesRegex(RuntimeError, "prompt leakage"):
            assert_prompt_disjoint_splits(
                {"train": train, "validation": validation}
            )

    def test_target_embedding_files_must_match_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            files = {
                "config.json": b"{}",
                "model.safetensors.index.json": json.dumps(
                    {
                        "weight_map": {
                            "model.embed_tokens.weight": "embedding.safetensors"
                        }
                    }
                ).encode(),
                "embedding.safetensors": b"frozen embedding bytes",
            }
            expected = []
            for name, contents in files.items():
                path = target / name
                path.write_bytes(contents)
                expected.append(
                    {
                        "path": name,
                        "bytes": len(contents),
                        "sha256": hashlib.sha256(contents).hexdigest(),
                    }
                )
            metadata = {
                "format_version": 2,
                "provenance": {"target_files": expected},
            }
            verified = validate_target_embedding_identity(metadata, target)
            self.assertEqual(len(verified), 3)

            (target / "embedding.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "size differs"):
                validate_target_embedding_identity(metadata, target)


if __name__ == "__main__":
    unittest.main()

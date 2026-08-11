from __future__ import annotations

import unittest

from scripts.build_open_perfectblend_manifest import (
    source_domain,
    stable_rank,
)


class OpenPerfectBlendManifestTest(unittest.TestCase):
    def test_source_domains_match_the_preregistered_policy(self) -> None:
        self.assertEqual(source_domain("meta-math/MetaMathQA"), "math")
        self.assertEqual(
            source_domain("microsoft/orca-math-word-problems-200k"),
            "math",
        )
        self.assertEqual(
            source_domain("HuggingFaceH4/orca-math-word-problems-200k"),
            "math",
        )
        self.assertEqual(
            source_domain("theblackcat102/evol-codealpaca-v1"),
            "code",
        )
        self.assertEqual(
            source_domain("HuggingFaceH4/ultrachat_200k"),
            "chat",
        )
        self.assertEqual(source_domain("new-source"), "chat")

    def test_stable_rank_is_repeatable_and_namespace_separated(self) -> None:
        self.assertEqual(
            stable_rank(7, "a", "prompt"),
            stable_rank(7, "a", "prompt"),
        )
        self.assertNotEqual(
            stable_rank(7, "a", "prompt"),
            stable_rank(7, "b", "prompt"),
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.build_phase3_manifests import (
    NgramOverlapIndex,
    first_human_turn,
    make_record,
    normalize_prompt,
    stable_select,
)


class Phase3ManifestTest(unittest.TestCase):
    def test_normalization_collapses_unicode_case_and_whitespace(self) -> None:
        self.assertEqual(normalize_prompt("  Ａ  B\n"), "a b")

    def test_ngram_overlap_detects_copy(self) -> None:
        reference = "alpha beta gamma delta epsilon zeta eta theta iota"
        index = NgramOverlapIndex([reference], size=4)
        self.assertEqual(index.maximum_jaccard(reference), 1.0)
        self.assertEqual(index.maximum_jaccard("entirely unrelated words"), 0.0)

    def test_stable_selection_is_order_invariant_and_nested(self) -> None:
        records = [
            make_record(
                source="source",
                raw_id=index,
                domain="chat",
                prompt=f"A sufficiently long unique prompt number {index}",
            )
            for index in range(10)
        ]
        first = stable_select(records, 4, seed=7, namespace="test")
        reversed_selection = stable_select(
            reversed(records), 4, seed=7, namespace="test"
        )
        larger = stable_select(records, 7, seed=7, namespace="test")
        self.assertEqual(first, reversed_selection)
        self.assertEqual(first, larger[:4])

    def test_arrow_json_turns_are_supported(self) -> None:
        row = {
            "conversations": [
                '{"from":"human","value":"A real user prompt"}',
                '{"from":"gpt","value":"An answer"}',
            ]
        }
        self.assertEqual(first_human_turn(row), "A real user prompt")


if __name__ == "__main__":
    unittest.main()

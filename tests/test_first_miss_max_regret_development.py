from __future__ import annotations

import ast
from copy import deepcopy

import pytest

from scripts import train_first_miss_max_regret_development as development


def _summary(eal: float, first_token_accuracy: float = 0.9) -> dict[str, float]:
    return {
        "mean_accepted_draft_tokens": eal,
        "mean_verification_advance": eal + 1.0,
        "mean_accepted_draft_tokens_prompt_balanced": eal,
        "mean_verification_advance_prompt_balanced": eal + 1.0,
        "first_token_accuracy": first_token_accuracy,
    }


def _gate_inputs() -> tuple[dict, dict]:
    examples = []
    controls = []
    for index in range(development.EXPECTED_VALIDATION_BLOCKS):
        sample_id = f"prompt-{index % development.EXPECTED_VALIDATION_PROMPTS}"
        examples.append(
            {
                "sample_id": sample_id,
                "domain": "chat",
                "accepted_draft_tokens": {
                    "base": 5,
                    "camrs": 6,
                    "single_edit_oracle": 6,
                },
                "first_token_correct": {
                    "base": True,
                    "camrs": True,
                    "single_edit_oracle": True,
                },
                "candidate_path_indices": {
                    "base": [0, 0],
                    "single_edit_oracle": [0, 1],
                },
                "oracle_action": 2,
            }
        )
        controls.append(
            {
                "sample_id": sample_id,
                "domain": "chat",
                "accepted_draft_tokens": {
                    "base": 5,
                    "direct_native": 5,
                    "fmas": 5,
                    "single_edit_oracle": 6,
                },
                "first_token_correct": {
                    "base": True,
                    "direct_native": True,
                    "fmas": True,
                    "single_edit_oracle": True,
                },
                "candidate_path_indices": {
                    "base": [0, 0],
                    "single_edit_oracle": [0, 1],
                },
                "target_action": 2,
            }
        )
    evaluation = {
        "base": development.direct._method_summary(examples, "base"),
        "camrs": development.direct._method_summary(examples, "camrs"),
        "decision": {"harmed_fraction": 0.0},
        "loss": {"mean_block_hinge": 0.2},
        "blocks": development.EXPECTED_VALIDATION_BLOCKS,
        "prompts": development.EXPECTED_VALIDATION_PROMPTS,
        "examples": examples,
    }
    control = {
        "base": development.direct._method_summary(controls, "base"),
        "direct_native": development.direct._method_summary(
            controls, "direct_native"
        ),
        "direct_one_edit": development.direct._method_summary(controls, "fmas"),
        "examples": controls,
    }
    return evaluation, control


def test_runtime_source_paths_cover_local_python_import_closure() -> None:
    source_paths = development.development_source_paths()
    captured = {
        path.resolve() for path in source_paths.values() if path.suffix == ".py"
    }
    assert (
        development.PROJECT / "src/sph/first_miss_value_selector.py"
    ).resolve() in captured
    missing: list[tuple[str, str]] = []
    for source_path in sorted(captured):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            candidates = []
            if isinstance(node, ast.ImportFrom):
                if node.level == 1 and source_path.parent.name == "sph":
                    candidates.append(source_path.parent / f"{node.module}.py")
                elif node.module and node.module.startswith("sph."):
                    module = node.module.removeprefix("sph.")
                    candidates.append(
                        development.PROJECT
                        / "src/sph"
                        / f"{module.replace('.', '/')}.py"
                    )
                elif node.module == "scripts":
                    candidates.extend(
                        development.PROJECT / "scripts" / f"{alias.name}.py"
                        for alias in node.names
                    )
            elif isinstance(node, ast.Import):
                candidates.extend(
                    development.PROJECT / "scripts" / f"{alias.name}.py"
                    for alias in node.names
                )
            for candidate in candidates:
                if candidate.is_file() and candidate.resolve() not in captured:
                    missing.append((str(source_path), str(candidate.resolve())))
    assert not missing


def test_checkpoint_selection_key_uses_frozen_lexicographic_order() -> None:
    evaluation, _ = _gate_inputs()
    evaluation["decision"]["harmed_fraction"] = 0.04
    assert development.checkpoint_selection_key(evaluation) == (6.0, -0.04, -0.2)
    lower_harm = deepcopy(evaluation)
    lower_harm["decision"]["harmed_fraction"] = 0.03
    assert development.checkpoint_selection_key(
        lower_harm
    ) > development.checkpoint_selection_key(evaluation)
    lower_eal = deepcopy(evaluation)
    lower_eal["camrs"] = _summary(5.99)
    lower_eal["decision"]["harmed_fraction"] = 0.0
    assert development.checkpoint_selection_key(
        lower_eal
    ) < development.checkpoint_selection_key(evaluation)


def test_development_gate_passes_exact_contract() -> None:
    evaluation, control = _gate_inputs()
    gate = development.development_gate_report(evaluation, control)
    assert gate["passed"]
    assert all(gate["checks"].values())
    assert gate["values"]["first_token_shortfall_blocks"] == 0


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("dflash", "delta_vs_dflash"),
        ("native", "delta_vs_direct_native"),
        ("one_edit", "delta_vs_direct_one_edit"),
        ("harm", "harmed_fraction"),
        ("first_token", "first_token_shortfall"),
        ("nonfinite", "finite_values"),
        ("blocks", "validation_blocks"),
        ("prompts", "validation_prompts"),
    ],
)
def test_development_gate_fails_closed(mutation: str, failed_check: str) -> None:
    evaluation, control = _gate_inputs()
    if mutation == "dflash":
        for example in evaluation["examples"]:
            example["accepted_draft_tokens"]["camrs"] = 5
        evaluation["camrs"] = development.direct._method_summary(
            evaluation["examples"], "camrs"
        )
    elif mutation == "native":
        for example in control["examples"]:
            example["accepted_draft_tokens"]["direct_native"] = 6
        control["direct_native"] = development.direct._method_summary(
            control["examples"], "direct_native"
        )
    elif mutation == "one_edit":
        for example in control["examples"]:
            example["accepted_draft_tokens"]["fmas"] = 6
        control["direct_one_edit"] = development.direct._method_summary(
            control["examples"], "fmas"
        )
    elif mutation == "harm":
        harmed_blocks = 59
        for example in evaluation["examples"][:harmed_blocks]:
            example["accepted_draft_tokens"]["camrs"] = 4
        evaluation["camrs"] = development.direct._method_summary(
            evaluation["examples"], "camrs"
        )
        evaluation["decision"]["harmed_fraction"] = (
            harmed_blocks / development.EXPECTED_VALIDATION_BLOCKS
        )
    elif mutation == "first_token":
        evaluation["examples"][0]["first_token_correct"]["camrs"] = False
        evaluation["examples"][1]["first_token_correct"]["camrs"] = False
        evaluation["camrs"] = development.direct._method_summary(
            evaluation["examples"], "camrs"
        )
    elif mutation == "nonfinite":
        evaluation["decision"]["harmed_fraction"] = float("nan")
    elif mutation == "blocks":
        evaluation["blocks"] -= 1
    elif mutation == "prompts":
        evaluation["prompts"] -= 1
    gate = development.development_gate_report(evaluation, control)
    assert not gate["passed"]
    assert not gate["checks"][failed_check]


def test_development_gate_rejects_example_misalignment() -> None:
    evaluation, control = _gate_inputs()
    control["examples"][0], control["examples"][1] = (
        control["examples"][1],
        control["examples"][0],
    )
    with pytest.raises(RuntimeError, match="example order"):
        development.development_gate_report(evaluation, control)


def test_development_gate_rejects_prompt_set_mismatch() -> None:
    evaluation, control = _gate_inputs()
    control["examples"][5]["sample_id"] = "different"
    with pytest.raises(RuntimeError, match="prompt sets"):
        development.development_gate_report(evaluation, control)


def test_development_gate_rejects_summary_example_mismatch() -> None:
    evaluation, control = _gate_inputs()
    evaluation["camrs"]["mean_accepted_draft_tokens_prompt_balanced"] -= 0.1
    with pytest.raises(RuntimeError, match="camrs.camrs"):
        development.development_gate_report(evaluation, control)


def test_development_gate_rejects_base_realization_mismatch() -> None:
    evaluation, control = _gate_inputs()
    control["examples"][5]["accepted_draft_tokens"]["base"] = 4
    with pytest.raises(RuntimeError, match="base realization"):
        development.development_gate_report(evaluation, control)


def test_development_gate_rejects_oracle_action_mismatch() -> None:
    evaluation, control = _gate_inputs()
    control["examples"][5]["target_action"] = 3
    with pytest.raises(RuntimeError, match="oracle action"):
        development.development_gate_report(evaluation, control)


def test_development_gate_rejects_unreconstructed_harm_fraction() -> None:
    evaluation, control = _gate_inputs()
    evaluation["decision"]["harmed_fraction"] = 0.01
    with pytest.raises(RuntimeError, match="harmed fraction"):
        development.development_gate_report(evaluation, control)

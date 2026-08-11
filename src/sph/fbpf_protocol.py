"""Deterministic prospective-v2 data and inference protocol primitives.

All helpers in this module operate on caller-provided synthetic/in-memory data.
They do not open the prospective source pool or any sealed outcome artifact.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
SPLIT_ORDER = ("fit", "checkpoint", "falsifier")
DOMAIN_ORDER = ("math", "code", "chat")
MINIMUM_CONTINUATION_TOKENS = 19
BLOCK_SIZE = 16
PREDICTED_POSITIONS = 15
RESERVE_EXHAUSTED = "TERMINAL_SEQUENCE_RESERVE_EXHAUSTED"
EXACT_QUOTA_UNAVAILABLE = "TERMINAL_EXACT_COMPONENT_QUOTA_UNAVAILABLE"
LATENCY_T_CRITICAL_DF19_90_CI = 1.729132811521367


def deterministic_complete_offsets(continuation_length: int) -> tuple[int, ...]:
    """Return the four frozen Python-rounded complete b16 offsets."""

    if continuation_length < MINIMUM_CONTINUATION_TOKENS:
        raise ValueError(
            f"continuation requires at least {MINIMUM_CONTINUATION_TOKENS} tokens"
        )
    maximum = continuation_length - BLOCK_SIZE
    offsets = tuple(int(round(j * maximum / 3)) for j in range(4))
    if len(set(offsets)) != 4:
        raise AssertionError("the frozen offsets must be distinct")
    if any(offset < 0 or offset + BLOCK_SIZE > continuation_length for offset in offsets):
        raise AssertionError("every frozen offset must define a complete b16 block")
    return offsets


@dataclass(frozen=True)
class ContinuationBlocks:
    offsets: tuple[int, ...]
    anchors: Tensor
    gold: Tensor


def extract_continuation_blocks(continuation: Tensor) -> ContinuationBlocks:
    """Extract four anchors and their 15 verifier-defined future labels."""

    if continuation.ndim != 1:
        raise ValueError("continuation must be a one-dimensional token tensor")
    offsets = deterministic_complete_offsets(int(continuation.numel()))
    anchors = torch.stack([continuation[offset] for offset in offsets])
    gold = torch.stack(
        [continuation[offset + 1 : offset + BLOCK_SIZE] for offset in offsets]
    )
    if gold.shape != (4, PREDICTED_POSITIONS):
        raise AssertionError("each retained continuation must yield exactly four blocks")
    return ContinuationBlocks(offsets=offsets, anchors=anchors, gold=gold)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return " ".join(normalized.split())


def normalized_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def tokenize_for_overlap(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(normalize_text(text)))


def overlap_ngrams(text: str, size: int = 8) -> frozenset[tuple[str, ...]]:
    if size <= 0:
        raise ValueError("ngram size must be positive")
    tokens = tokenize_for_overlap(text)
    if len(tokens) < size:
        return frozenset((tokens,))
    return frozenset(tuple(tokens[start : start + size]) for start in range(len(tokens) - size + 1))


def exact_jaccard(
    left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]
) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


@dataclass(frozen=True)
class CandidateRow:
    source_ordinal: int
    domain: str
    text: str

    def __post_init__(self) -> None:
        if self.source_ordinal < 0:
            raise ValueError("source_ordinal must be non-negative")
        if self.domain not in DOMAIN_ORDER:
            raise ValueError(f"unknown domain {self.domain!r}")

    @property
    def normalized(self) -> str:
        return normalize_text(self.text)

    @property
    def normalized_hash(self) -> str:
        return hashlib.sha256(self.normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NearDuplicateComponent:
    component_id: str
    domain: str
    rows: tuple[CandidateRow, ...]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        low, high = sorted((root_left, root_right))
        self.parent[high] = low


def component_identifier(rows: Sequence[CandidateRow]) -> str:
    hashes = sorted(row.normalized_hash for row in rows)
    return hashlib.sha256(b"\0".join(value.encode("utf-8") for value in hashes)).hexdigest()


@dataclass(frozen=True)
class ComponentBuildResult:
    components: tuple[NearDuplicateComponent, ...]
    excluded_cross_domain_ordinals: tuple[int, ...]
    candidate_pair_count: int
    edge_count: int


def build_near_duplicate_components(
    rows: Sequence[CandidateRow], *, jaccard_threshold: float = 0.5
) -> ComponentBuildResult:
    """Build exact deterministic Jaccard components from inverted postings."""

    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must lie in [0,1]")
    ordered_rows = tuple(sorted(rows, key=lambda row: row.source_ordinal))
    if len({row.source_ordinal for row in ordered_rows}) != len(ordered_rows):
        raise ValueError("source ordinals must be unique")

    grams = tuple(overlap_ngrams(row.text) for row in ordered_rows)
    postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for row_index, row_grams in enumerate(grams):
        for ngram in sorted(row_grams):
            postings[ngram].append(row_index)

    candidate_pairs: set[tuple[int, int]] = set()
    for ngram in sorted(postings):
        indices = sorted(postings[ngram])
        for left_index, left in enumerate(indices):
            for right in indices[left_index + 1 :]:
                candidate_pairs.add((left, right))

    union_find = _UnionFind(len(ordered_rows))
    edges: list[tuple[int, int]] = []
    for left, right in sorted(candidate_pairs):
        if exact_jaccard(grams[left], grams[right]) >= jaccard_threshold:
            edges.append((left, right))
            union_find.union(left, right)

    groups: dict[int, list[CandidateRow]] = defaultdict(list)
    for index, row in enumerate(ordered_rows):
        groups[union_find.find(index)].append(row)

    components: list[NearDuplicateComponent] = []
    excluded: list[int] = []
    for root in sorted(groups):
        component_rows = tuple(sorted(groups[root], key=lambda row: row.source_ordinal))
        domains = {row.domain for row in component_rows}
        if len(domains) != 1:
            excluded.extend(row.source_ordinal for row in component_rows)
            continue
        components.append(
            NearDuplicateComponent(
                component_id=component_identifier(component_rows),
                domain=next(iter(domains)),
                rows=component_rows,
            )
        )
    components.sort(key=lambda component: component.component_id)
    return ComponentBuildResult(
        components=tuple(components),
        excluded_cross_domain_ordinals=tuple(sorted(excluded)),
        candidate_pair_count=len(candidate_pairs),
        edge_count=len(edges),
    )


def _decimal_bytes(value: int) -> bytes:
    if value < 0:
        raise ValueError("rank integers must be non-negative")
    return str(value).encode("ascii")


def _lower_hex_bytes(value: str) -> bytes:
    if value != value.lower() or not re.fullmatch(r"[0-9a-f]+", value):
        raise ValueError("rank hashes must be lowercase hexadecimal")
    return value.encode("utf-8")


def component_rank(
    split_seed: int, split: str, domain: str, component_id: str
) -> str:
    material = b"\0".join(
        (
            _decimal_bytes(split_seed),
            split.encode("utf-8"),
            domain.encode("utf-8"),
            _lower_hex_bytes(component_id),
        )
    )
    return hashlib.sha256(material).hexdigest()


def row_rank(component_rank_hex: str, row: CandidateRow) -> str:
    material = b"\0".join(
        (
            _lower_hex_bytes(component_rank_hex),
            _lower_hex_bytes(row.normalized_hash),
            _decimal_bytes(row.source_ordinal),
        )
    )
    return hashlib.sha256(material).hexdigest()


def reserve_quota(active_quota: int) -> int:
    if active_quota < 0:
        raise ValueError("active_quota must be non-negative")
    return max(64, math.ceil(0.10 * active_quota))


@dataclass(frozen=True)
class AllocationRecord:
    source_ordinal: int
    component_id: str
    domain: str
    split: str
    status: str
    order: int
    component_rank: str
    row_rank: str


@dataclass(frozen=True)
class AllocationResult:
    records: tuple[AllocationRecord, ...]
    assigned_component_ids: tuple[str, ...]

    def rows(self, split: str, domain: str, status: str) -> tuple[AllocationRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.split == split
            and record.domain == domain
            and record.status == status
        )


class AllocationFailure(RuntimeError):
    failure_class = EXACT_QUOTA_UNAVAILABLE


def allocate_components(
    components: Sequence[NearDuplicateComponent],
    active_quotas: Mapping[tuple[str, str], int],
    *,
    split_seed: int = 20_260_806,
) -> AllocationResult:
    """Assign whole components, then exact active/reserve rows within each bucket."""

    available = {component.component_id: component for component in components}
    if len(available) != len(components):
        raise ValueError("component IDs must be unique")
    used: set[str] = set()
    records: list[AllocationRecord] = []

    for split in SPLIT_ORDER:
        for domain in DOMAIN_ORDER:
            key = (split, domain)
            if key not in active_quotas:
                continue
            active = int(active_quotas[key])
            reserve = reserve_quota(active)
            required = active + reserve
            candidates = [
                component
                for component in components
                if component.domain == domain and component.component_id not in used
            ]
            candidates.sort(
                key=lambda component: component_rank(
                    split_seed, split, domain, component.component_id
                )
            )
            selected: list[NearDuplicateComponent] = []
            row_count = 0
            for component in candidates:
                selected.append(component)
                row_count += len(component.rows)
                if row_count >= required:
                    break
            if row_count < required:
                raise AllocationFailure(
                    f"{EXACT_QUOTA_UNAVAILABLE}: {split}/{domain} needs {required}, "
                    f"has {row_count}"
                )

            ranked_rows: list[tuple[str, str, CandidateRow]] = []
            for component in selected:
                c_rank = component_rank(
                    split_seed, split, domain, component.component_id
                )
                for row in component.rows:
                    ranked_rows.append((c_rank, row_rank(c_rank, row), row))
                used.add(component.component_id)
            ranked_rows.sort(key=lambda item: (item[0], item[1]))

            status_counts = {"active": 0, "reserve": 0, "discarded": 0}
            selected_ids = {
                row.source_ordinal: component.component_id
                for component in selected
                for row in component.rows
            }
            for index, (c_rank, r_rank, row) in enumerate(ranked_rows):
                if index < active:
                    status = "active"
                elif index < active + reserve:
                    status = "reserve"
                else:
                    status = "discarded"
                order = status_counts[status]
                status_counts[status] += 1
                records.append(
                    AllocationRecord(
                        source_ordinal=row.source_ordinal,
                        component_id=selected_ids[row.source_ordinal],
                        domain=domain,
                        split=split,
                        status=status,
                        order=order,
                        component_rank=c_rank,
                        row_rank=r_rank,
                    )
                )

    records.sort(
        key=lambda record: (
            SPLIT_ORDER.index(record.split),
            DOMAIN_ORDER.index(record.domain),
            {"active": 0, "reserve": 1, "discarded": 2}[record.status],
            record.order,
        )
    )
    return AllocationResult(records=tuple(records), assigned_component_ids=tuple(sorted(used)))


class ReserveExhausted(RuntimeError):
    failure_class = RESERVE_EXHAUSTED


@dataclass(frozen=True)
class SequenceSelection:
    selected_ordinals: tuple[int, ...]
    consumed_attempt_ordinals: tuple[int, ...]


def select_complete_sequence_rows(
    allocation: AllocationResult,
    *,
    split: str,
    domain: str,
    continuation_lengths: Mapping[int, int],
) -> SequenceSelection:
    """Consume active slots and split-local reserves without reassignment."""

    active = allocation.rows(split, domain, "active")
    reserve = allocation.rows(split, domain, "reserve")
    reserve_index = 0
    selected: list[int] = []
    consumed: list[int] = []
    for active_record in active:
        candidate = active_record
        while True:
            consumed.append(candidate.source_ordinal)
            if continuation_lengths[candidate.source_ordinal] >= MINIMUM_CONTINUATION_TOKENS:
                selected.append(candidate.source_ordinal)
                break
            if reserve_index >= len(reserve):
                raise ReserveExhausted(f"{RESERVE_EXHAUSTED}: {split}/{domain}")
            candidate = reserve[reserve_index]
            reserve_index += 1
    return SequenceSelection(
        selected_ordinals=tuple(selected),
        consumed_attempt_ordinals=tuple(consumed),
    )


@dataclass(frozen=True)
class PromptMetricBundle:
    eal: Tensor
    harm_rate: Tensor
    mean_harm: Tensor
    first_token_contrast: Tensor


def prompt_metrics(released: Tensor, model: Tensor) -> PromptMetricBundle:
    """Aggregate four accepted-count blocks equally within each prompt."""

    if released.shape != model.shape or released.ndim != 2:
        raise ValueError("accepted counts must both have shape [prompts, blocks]")
    if released.shape[1] != 4:
        raise ValueError("the frozen prompt metric uses exactly four blocks")
    released64 = released.to(dtype=torch.float64)
    model64 = model.to(dtype=torch.float64)
    return PromptMetricBundle(
        eal=(1.0 + model64).mean(dim=1),
        harm_rate=(model64 < released64).to(torch.float64).mean(dim=1),
        mean_harm=torch.relu(released64 - model64).mean(dim=1),
        first_token_contrast=(
            (model64 >= 1).to(torch.float64)
            - (released64 >= 1).to(torch.float64)
        ).mean(dim=1),
    )


def mean_matched_seed_prompt_metric(seed_prompt_metrics: Tensor) -> Tensor:
    if seed_prompt_metrics.ndim != 2 or seed_prompt_metrics.shape[0] != 3:
        raise ValueError("expected [three matched seeds, prompts]")
    return seed_prompt_metrics.to(dtype=torch.float64).mean(dim=0)


def cluster_bootstrap(
    prompt_values: Tensor,
    component_ids: Sequence[str],
    domains: Sequence[str],
    *,
    replicates: int = 10_000,
    seed: int = 2_026_080_601,
) -> Tensor:
    """Domain-stratified connected-component cluster bootstrap."""

    values = prompt_values.detach().to(dtype=torch.float64, device="cpu")
    if values.ndim != 1:
        raise ValueError("prompt_values must be one-dimensional")
    if len(component_ids) != values.numel() or len(domains) != values.numel():
        raise ValueError("component/domain labels must align with prompt values")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    strata: dict[str, dict[str, list[int]]] = {
        domain: defaultdict(list) for domain in DOMAIN_ORDER
    }
    for index, (component_id, domain) in enumerate(zip(component_ids, domains, strict=True)):
        if domain not in strata:
            raise ValueError(f"unknown domain {domain!r}")
        strata[domain][component_id].append(index)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    statistics = torch.empty(replicates, dtype=torch.float64)
    for replicate in range(replicates):
        sampled_values: list[Tensor] = []
        for domain in DOMAIN_ORDER:
            component_groups = [
                strata[domain][component_id]
                for component_id in sorted(strata[domain])
            ]
            if not component_groups:
                continue
            draws = torch.randint(
                len(component_groups),
                (len(component_groups),),
                generator=generator,
            )
            for draw in draws.tolist():
                sampled_values.append(values[component_groups[draw]])
        if not sampled_values:
            raise ValueError("at least one component is required")
        statistics[replicate] = torch.cat(sampled_values).mean()
    return statistics


def percentile_interval(
    bootstrap_statistics: Tensor, *, one_sided: str | None = None
) -> tuple[float, ...]:
    values = bootstrap_statistics.to(dtype=torch.float64)
    if one_sided is None:
        quantiles = torch.tensor([0.025, 0.975], dtype=torch.float64)
    elif one_sided == "lower":
        quantiles = torch.tensor([0.05], dtype=torch.float64)
    elif one_sided == "upper":
        quantiles = torch.tensor([0.95], dtype=torch.float64)
    else:
        raise ValueError("one_sided must be None, 'lower', or 'upper'")
    result = torch.quantile(values, quantiles, interpolation="linear")
    return tuple(float(value) for value in result)


@dataclass(frozen=True)
class PowerResult:
    design_effect: float
    requirements: Mapping[str, int]
    n_power: int


def power_requirements(
    *,
    sd_upper_by_contrast: Mapping[str, float],
    icc_upper: float,
    cv_cluster_size_upper: float,
    mean_cluster_size_upper: float,
    sd_mean_harm_upper: float,
) -> PowerResult:
    """Apply the frozen normal-approximation and unequal-cluster design effect."""

    if not 0.0 <= icc_upper <= 1.0:
        raise ValueError("icc_upper must lie in [0,1]")
    if cv_cluster_size_upper < 0.0 or mean_cluster_size_upper < 1.0:
        raise ValueError("invalid cluster-size upper bounds")
    if sd_mean_harm_upper < 0.0:
        raise ValueError("sd_mean_harm_upper must be non-negative")
    design_effect = 1.0 + (
        ((1.0 + cv_cluster_size_upper**2) * mean_cluster_size_upper) - 1.0
    ) * icc_upper
    z975 = NormalDist().inv_cdf(0.975)
    z95 = NormalDist().inv_cdf(0.95)
    z80 = NormalDist().inv_cdf(0.80)
    superiority = {
        "d_vs_released_eal": 0.30,
        "d_vs_a_eal": 0.10,
        "d_vs_b_eal": 0.10,
        "d_vs_c_released_referenced_harm": 0.02,
    }
    noninferiority = {
        "d_vs_c_eal": 0.05,
        "first_token_vs_released": 0.005,
    }
    expected = set(superiority) | set(noninferiority)
    if set(sd_upper_by_contrast) != expected:
        missing = sorted(expected - set(sd_upper_by_contrast))
        extra = sorted(set(sd_upper_by_contrast) - expected)
        raise ValueError(f"contrast SD fields mismatch; missing={missing}, extra={extra}")

    requirements: dict[str, int] = {}
    for name, effect in superiority.items():
        sd_upper = sd_upper_by_contrast[name]
        requirements[name] = math.ceil(
            design_effect * ((z975 + z80) * sd_upper / effect) ** 2
        )
    for name, distance in noninferiority.items():
        sd_upper = sd_upper_by_contrast[name]
        requirements[name] = math.ceil(
            design_effect * ((z95 + z80) * sd_upper / distance) ** 2
        )
    requirements["harm_rate_upper_bound_precision"] = math.ceil(
        design_effect * 0.25 * (z95 / 0.015) ** 2
    )
    requirements["mean_harm_upper_bound_precision"] = math.ceil(
        design_effect * (z95 * sd_mean_harm_upper / 0.03) ** 2
    )
    return PowerResult(
        design_effect=design_effect,
        requirements=requirements,
        n_power=max(requirements.values()),
    )


@dataclass(frozen=True)
class LatencyTostResult:
    estimate: float
    sample_standard_deviation: float
    ci_lower: float
    ci_upper: float
    log_equivalence_lower: float
    log_equivalence_upper: float
    passed: bool


def latency_tost(restart_median_log_ratios: Sequence[float]) -> LatencyTostResult:
    """Frozen 20-restart Student-t 90% CI and strict equivalence decision."""

    if len(restart_median_log_ratios) != 20:
        raise ValueError("latency TOST requires exactly 20 restart values")
    values = torch.tensor(restart_median_log_ratios, dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("restart values must be finite")
    estimate = float(values.mean())
    sample_sd = float(values.std(correction=1))
    half_width = LATENCY_T_CRITICAL_DF19_90_CI * sample_sd / math.sqrt(20)
    ci_lower = estimate - half_width
    ci_upper = estimate + half_width
    equivalence_lower = math.log(0.98)
    equivalence_upper = math.log(1.02)
    return LatencyTostResult(
        estimate=estimate,
        sample_standard_deviation=sample_sd,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        log_equivalence_lower=equivalence_lower,
        log_equivalence_upper=equivalence_upper,
        passed=equivalence_lower < ci_lower and ci_upper < equivalence_upper,
    )

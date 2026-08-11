"""Draft-only Fast-K beam and budgeted prefix-tree primitives for R053.

R053 keeps the ordinary speculative-decoding contract of one target-model
invocation.  A frozen Fast-K64 Domino beam proposes several complete paths;
those paths are compressed to a prefix-closed tree using only draft scores.
The target can then verify every retained branch in one masked forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class FastBeamOutput:
    """Final frozen Fast-K beam and its protected greedy trunk."""

    token_ids: Tensor
    edge_log_probs: Tensor
    map_scores: Tensor
    trunk_token_ids: Tensor
    candidate_ids: Tensor


@dataclass
class _TrieNode:
    token_id: int | None
    depth: int
    parent: int | None
    children: dict[int, int]
    descendant_logmass: float = -math.inf


@dataclass(frozen=True)
class BudgetedTrie:
    """A deterministic, prefix-closed subset of a complete beam trie."""

    budget: int
    used_nodes_including_anchor: int
    full_nodes_including_anchor: int
    packed_token_ids: tuple[int, ...]
    packed_depths: tuple[int, ...]
    packed_parent_rows: tuple[int, ...]
    selected_prefixes: frozenset[tuple[int, ...]]
    horizon: int


@dataclass(frozen=True)
class PackedTreeTraversal:
    """Device-local static metadata for fixed-shape tree traversal."""

    token_ids: Tensor
    parent_rows: Tensor
    row_ids: Tensor
    sentinel: Tensor


def _head_state(domino: Any, state: Tensor) -> Tensor:
    value = state.transpose(0, 1)
    if bool(getattr(domino, "use_bias_norm", False)):
        value = domino.bias_norm(value)
    return value


@torch.no_grad()
def fast_candidate_domino_beam_from_base(
    *,
    domino: Any,
    target_weight: Tensor,
    anchors: Tensor,
    hidden: Tensor,
    base_logits: Tensor,
    candidate_pool_topk: int = 64,
    tree_support_size: int = 16,
    beam_width: int = 16,
) -> FastBeamOutput:
    """Generate a batch-1 Fast-K beam with the greedy Fast-K path protected.

    Candidate IDs come from the FP32-ranked DFlash vocabulary projection, but
    base/correction addition remains in the checkpoint dtype.  The protected
    trunk is the current Fast-K64 greedy path.  Each tree branch scores only a
    fixed K16 support: DFlash Top-15 plus the trunk token (or unchanged Top-16
    when the trunk is already present).  No per-beam vocabulary GEMM is used.
    """

    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("R053 beam requires hidden [1, positions, width]")
    _, positions, width = hidden.shape
    if anchors.shape != (1,):
        raise ValueError("R053 beam requires one anchor")
    if target_weight.ndim != 2 or target_weight.shape[1] != width:
        raise ValueError("target embedding is incompatible with hidden")
    if base_logits.shape != (1, positions, target_weight.shape[0]):
        raise ValueError("base logits have the wrong shape")
    if not 2 <= candidate_pool_topk <= target_weight.shape[0]:
        raise ValueError("candidate_pool_topk lies outside the vocabulary")
    if not 2 <= tree_support_size <= candidate_pool_topk:
        raise ValueError("tree support lies outside the candidate pool")
    if not 1 <= beam_width <= tree_support_size:
        raise ValueError("beam_width must lie in [1, tree_support_size]")
    if positions < 1:
        raise ValueError("proposal horizon must be positive")

    candidate_pool_ids = base_logits.float().topk(
        candidate_pool_topk, dim=-1
    ).indices
    canonical_tree_ids = base_logits.float().topk(
        tree_support_size, dim=-1
    ).indices

    trunk_first = base_logits[0, 0].float().argmax()
    first_support = canonical_tree_ids[0, 0].clone()
    first_present = first_support.eq(trunk_first).any()
    first_support[-1] = torch.where(
        first_present, first_support[-1], trunk_first
    )
    first_scores = base_logits[0, 0].gather(0, first_support).float()
    first_log_probs = F.log_softmax(first_scores, dim=-1)
    _, first_indices = first_log_probs.topk(beam_width)
    trunk_matches = first_support.eq(trunk_first)
    trunk_child = trunk_matches.to(torch.long).argmax()
    # Keep the protected trunk without a device-to-host branch.  Top-k and the
    # replacement both have fixed shapes, so this path is CUDA-graph safe.
    first_indices[-1] = torch.where(
        first_indices.eq(trunk_child).any(), first_indices[-1], trunk_child
    )

    first_tokens = first_support[first_indices]
    paths = first_tokens[:, None]
    edge_log_probs = first_log_probs[first_indices, None]
    map_scores = edge_log_probs[:, 0]
    trunk_index = first_indices.eq(trunk_child).to(torch.long).argmax()
    trunk_tokens = [trunk_first]
    support_rows = [first_support]

    anchor_rows = anchors.expand(beam_width)[:, None]
    prefix_ids = torch.cat([anchor_rows, first_tokens[:, None]], dim=1)
    _, state = domino.prefix_gru(F.embedding(prefix_ids, target_weight))
    # Preserve the exact batch-1 Fast-K64 numerical path.  Reusing the W16
    # beam parent's GRU/code changes BF16 GEMM geometry and can flip near ties.
    trunk_prefix = torch.cat([anchors[:, None], trunk_first.view(1, 1)], dim=1)
    _, trunk_state = domino.prefix_gru(F.embedding(trunk_prefix, target_weight))
    output_basis = domino.embed_proj[2].weight

    for position in range(1, positions):
        trunk_joined = torch.cat(
            [hidden[:, position : position + 1], _head_state(domino, trunk_state)],
            dim=-1,
        )
        trunk_code = domino.embed_proj[1](
            domino.embed_proj[0](trunk_joined)
        )[:, 0]
        active = beam_width
        joined = torch.cat(
            [
                hidden[:, position : position + 1].expand(active, -1, -1),
                _head_state(domino, state),
            ],
            dim=-1,
        )
        code = domino.embed_proj[1](domino.embed_proj[0](joined))[:, 0]
        # The trunk independently keeps the current gathered K64 Fast policy.
        pool_ids = candidate_pool_ids[0, position]
        pool_basis = F.embedding(pool_ids[None], output_basis)
        trunk_pool_correction = torch.einsum(
            "bd,bkd->bk", trunk_code, pool_basis
        )
        trunk_pool_base = base_logits[:, position].gather(1, pool_ids[None])
        trunk_pool_scores = trunk_pool_base + trunk_pool_correction
        trunk_pool_child = trunk_pool_scores.float().argmax(dim=-1)[0]
        trunk_token = pool_ids.gather(0, trunk_pool_child.view(1))[0]

        local_candidate_ids = canonical_tree_ids[0, position].clone()
        trunk_present = local_candidate_ids.eq(trunk_token).any()
        local_candidate_ids[-1] = torch.where(
            trunk_present, local_candidate_ids[-1], trunk_token
        )
        local_base = base_logits[0, position].gather(0, local_candidate_ids)
        basis = F.embedding(local_candidate_ids, output_basis)
        correction = torch.einsum("bd,kd->bk", code, basis)
        corrected = local_base[None] + correction
        local_log_probs = F.log_softmax(corrected.float(), dim=-1)
        expanded_scores = map_scores[:, None] + local_log_probs
        _, flat_indices = expanded_scores.flatten().topk(beam_width)

        trunk_matches = local_candidate_ids.eq(trunk_token)
        trunk_local_child = trunk_matches.to(torch.long).argmax()
        protected_flat = trunk_index * tree_support_size + trunk_local_child
        flat_indices[-1] = torch.where(
            flat_indices.eq(protected_flat).any(),
            flat_indices[-1],
            protected_flat,
        )

        parents = torch.div(
            flat_indices, tree_support_size, rounding_mode="floor"
        )
        children = flat_indices.remainder(tree_support_size)
        chosen_tokens = local_candidate_ids[children]
        chosen_edges = local_log_probs[parents, children]
        paths = torch.cat([paths[parents], chosen_tokens[:, None]], dim=1)
        edge_log_probs = torch.cat(
            [edge_log_probs[parents], chosen_edges[:, None]], dim=1
        )
        map_scores = expanded_scores[parents, children]
        trunk_index = flat_indices.eq(protected_flat).to(torch.long).argmax()
        trunk_tokens.append(trunk_token)
        support_rows.append(local_candidate_ids)

        if position + 1 < positions:
            _, trunk_state = domino.prefix_gru(
                F.embedding(trunk_token.view(1, 1), target_weight), trunk_state
            )

        state = state[:, parents, :]
        if position + 1 < positions:
            _, state = domino.prefix_gru(
                F.embedding(chosen_tokens[:, None], target_weight), state
            )

    trunk = torch.stack(trunk_tokens)[None]
    return FastBeamOutput(
        token_ids=paths[None],
        edge_log_probs=edge_log_probs[None],
        map_scores=map_scores[None],
        trunk_token_ids=trunk,
        candidate_ids=torch.stack(support_rows, dim=0)[None],
    )


def build_budgeted_trie(
    paths: Tensor,
    path_scores: Tensor,
    trunk: Tensor,
    *,
    budget: int,
) -> BudgetedTrie:
    """Build a prefix-closed tree with the full Fast trunk always retained.

    After retaining the 16-token trunk, remaining nodes are added best-first.
    A node's fixed draft-only priority is the maximum gamma-discounted draft
    score among all final beam paths below it.  A child becomes eligible only
    after its parent is selected, which guarantees prefix closure without
    target information.
    """

    if paths.ndim != 2 or path_scores.shape != (paths.shape[0],):
        raise ValueError("paths/scores must have shape [beam,horizon]/[beam]")
    if trunk.shape != (paths.shape[1],):
        raise ValueError("trunk must contain exactly one complete path")
    beam_width, horizon = paths.shape
    if beam_width < 1 or horizon < 1:
        raise ValueError("beam and horizon must be non-empty")
    if budget < horizon + 1:
        raise ValueError("budget must retain anchor plus the full trunk")

    cpu_paths = paths.detach().to(device="cpu", dtype=torch.long).tolist()
    cpu_scores = path_scores.detach().float().cpu().tolist()
    trunk_tuple = tuple(trunk.detach().to(device="cpu", dtype=torch.long).tolist())
    if trunk_tuple not in {tuple(path) for path in cpu_paths}:
        raise ValueError("protected trunk is absent from the beam")

    nodes = [_TrieNode(None, -1, None, {})]
    prefix_to_node: dict[tuple[int, ...], int] = {(): 0}
    descendants: list[list[float]] = [[]]
    for path, score in zip(cpu_paths, cpu_scores, strict=True):
        parent = 0
        prefix: tuple[int, ...] = ()
        descendants[parent].append(float(score))
        for depth, token in enumerate(path):
            prefix = (*prefix, int(token))
            node_index = prefix_to_node.get(prefix)
            if node_index is None:
                node_index = len(nodes)
                prefix_to_node[prefix] = node_index
                nodes.append(_TrieNode(int(token), depth, parent, {}))
                descendants.append([])
                nodes[parent].children[int(token)] = node_index
            parent = node_index
            descendants[parent].append(float(score))
    for node, scores in zip(nodes, descendants, strict=True):
        node.descendant_logmass = max(scores)

    selected = {0}
    prefix: tuple[int, ...] = ()
    for token in trunk_tuple:
        prefix = (*prefix, token)
        selected.add(prefix_to_node[prefix])

    target_size = min(int(budget), len(nodes))
    heap: list[tuple[float, int, int, int]] = []
    queued: set[int] = set()

    def queue_children(parent: int) -> None:
        for token, child in nodes[parent].children.items():
            if child in selected or child in queued:
                continue
            queued.add(child)
            heapq.heappush(
                heap,
                (
                    -nodes[child].descendant_logmass,
                    nodes[child].depth,
                    int(token),
                    child,
                ),
            )

    for node_index in tuple(selected):
        queue_children(node_index)
    while len(selected) < target_size:
        if not heap:
            raise RuntimeError("prefix-closed allocator exhausted before budget")
        _, _, _, node_index = heapq.heappop(heap)
        parent = nodes[node_index].parent
        if node_index in selected or parent not in selected:
            continue
        selected.add(node_index)
        queue_children(node_index)

    ordered_nodes = [0] + sorted(
        (index for index in selected if index != 0),
        key=lambda index: (nodes[index].depth, index),
    )
    old_to_row = {node_index: row for row, node_index in enumerate(ordered_nodes)}
    packed_tokens: list[int] = []
    packed_depths = [0]
    packed_parents = [-1]
    selected_prefixes: set[tuple[int, ...]] = {()}
    for node_index in ordered_nodes[1:]:
        node = nodes[node_index]
        if node.parent not in old_to_row:
            raise RuntimeError("selected tree is not prefix closed")
        packed_tokens.append(int(node.token_id))
        packed_depths.append(node.depth + 1)
        packed_parents.append(old_to_row[int(node.parent)])

    for path in cpu_paths:
        prefix = ()
        for token in path:
            prefix = (*prefix, int(token))
            node_index = prefix_to_node[prefix]
            if node_index in selected:
                selected_prefixes.add(prefix)
            else:
                break

    return BudgetedTrie(
        budget=int(budget),
        used_nodes_including_anchor=len(ordered_nodes),
        full_nodes_including_anchor=len(nodes),
        packed_token_ids=tuple(packed_tokens),
        packed_depths=tuple(packed_depths),
        packed_parent_rows=tuple(packed_parents),
        selected_prefixes=frozenset(selected_prefixes),
        horizon=horizon,
    )


def simulated_tree_acceptance(tree: BudgetedTrie, target_tokens: Tensor) -> int:
    """Return the exact greedy target-prefix length retained by ``tree``."""

    if target_tokens.ndim != 1 or target_tokens.numel() != tree.horizon:
        raise ValueError("target continuation has the wrong horizon")
    values = target_tokens.detach().to(device="cpu", dtype=torch.long).tolist()
    accepted = 0
    prefix: tuple[int, ...] = ()
    for token in values:
        prefix = (*prefix, int(token))
        if prefix not in tree.selected_prefixes:
            break
        accepted += 1
    return accepted


def full_pool_oracle_acceptance(paths: Tensor, target_tokens: Tensor) -> int:
    """Longest clean-target prefix present in any complete beam path."""

    if paths.ndim != 2 or target_tokens.shape != (paths.shape[1],):
        raise ValueError("paths and target continuation have incompatible shapes")
    matches = paths.eq(target_tokens[None]).to(torch.long).cumprod(dim=1).sum(dim=1)
    return int(matches.max())


def hindsight_budget_acceptance(
    paths: Tensor,
    trunk: Tensor,
    target_tokens: Tensor,
    *,
    budget: int,
) -> int:
    """Gold-aware per-request N-node structural upper bound.

    The bound retains the full trunk, then spends the remaining budget only on
    clean-target prefixes that exist in the full beam trie.  It is intentionally
    non-deployable and must never be used to allocate the actual tree.
    """

    if paths.ndim != 2 or trunk.shape != target_tokens.shape != (paths.shape[1],):
        raise ValueError("hindsight inputs have incompatible horizons")
    horizon = int(paths.shape[1])
    if budget < horizon + 1:
        raise ValueError("budget must retain anchor plus the full trunk")
    full_oracle = full_pool_oracle_acceptance(paths, target_tokens)
    trunk_values = trunk.detach().cpu().long().tolist()
    target_values = target_tokens.detach().cpu().long().tolist()
    trunk_prefixes = {tuple(trunk_values[:depth]) for depth in range(1, horizon + 1)}
    used = 1 + len(trunk_prefixes)
    accepted = 0
    for depth in range(1, full_oracle + 1):
        prefix = tuple(target_values[:depth])
        extra = int(prefix not in trunk_prefixes)
        if used + extra > budget:
            break
        used += extra
        accepted = depth
    return accepted


def pack_tree_tensors(
    tree: BudgetedTrie,
    *,
    anchor_token_id: int,
    prefix_length: int,
    device: torch.device,
    mask_dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    """Materialize tree inputs, RoPE positions and a 4-D additive mask."""

    if prefix_length < 1:
        raise ValueError("cached prefix must be non-empty")
    input_ids = torch.tensor(
        [anchor_token_id, *tree.packed_token_ids],
        dtype=torch.long,
    )[None]
    rows = int(input_ids.shape[1])
    if rows != tree.used_nodes_including_anchor:
        raise RuntimeError("packed tree row count differs from selected budget")
    position_ids = (
        torch.tensor(tree.packed_depths, dtype=torch.long)[None]
        + prefix_length
    )
    mask = torch.full(
        (1, 1, rows, prefix_length + rows),
        float("-inf"),
        dtype=mask_dtype,
    )
    mask[:, :, :, :prefix_length] = 0.0
    for row in range(rows):
        cursor = row
        while cursor >= 0:
            mask[0, 0, row, prefix_length + cursor] = 0.0
            cursor = tree.packed_parent_rows[cursor]
    return (
        input_ids.to(device=device),
        position_ids.to(device=device),
        mask.to(device=device),
    )


def traverse_tree_logits_path(
    tree: BudgetedTrie, logits: Tensor
) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
    """Return selected draft tokens, next token and packed-row path."""

    rows = tree.used_nodes_including_anchor
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != rows:
        raise ValueError("tree logits have incompatible rows")
    tokens_by_row = [None, *tree.packed_token_ids]
    children: list[dict[int, int]] = [dict() for _ in range(rows)]
    for row in range(1, rows):
        parent = tree.packed_parent_rows[row]
        token = int(tokens_by_row[row])
        if token in children[parent]:
            raise RuntimeError("tree contains duplicate sibling tokens")
        children[parent][token] = row

    current_row = 0
    selected_tokens: list[int] = []
    selected_rows: list[int] = []
    posterior = logits[0].float().argmax(dim=-1)
    while len(selected_tokens) < tree.horizon:
        predicted = int(posterior[current_row])
        child = children[current_row].get(predicted)
        if child is None:
            return tuple(selected_tokens), predicted, tuple(selected_rows)
        selected_tokens.append(predicted)
        selected_rows.append(child)
        current_row = child
    return (
        tuple(selected_tokens),
        int(posterior[current_row]),
        tuple(selected_rows),
    )


def traverse_tree_logits(tree: BudgetedTrie, logits: Tensor) -> tuple[int, int]:
    """Traverse verifier argmaxes and return accepted drafts and next token."""

    path, next_token, _ = traverse_tree_logits_path(tree, logits)
    return len(path), next_token


def pack_tree_traversal(
    tree: BudgetedTrie, *, device: torch.device
) -> PackedTreeTraversal:
    rows = tree.used_nodes_including_anchor
    return PackedTreeTraversal(
        token_ids=torch.tensor(
            [-1, *tree.packed_token_ids], dtype=torch.long, device=device
        ),
        parent_rows=torch.tensor(
            tree.packed_parent_rows, dtype=torch.long, device=device
        ),
        row_ids=torch.arange(rows, dtype=torch.long, device=device),
        sentinel=torch.full((), rows, dtype=torch.long, device=device),
    )


def traverse_tree_logits_tensor(
    tree: BudgetedTrie,
    logits: Tensor,
    *,
    packed: PackedTreeTraversal | None = None,
) -> tuple[Tensor, Tensor]:
    """Fixed-shape device traversal suitable for timing the decision path."""

    rows = tree.used_nodes_including_anchor
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != rows:
        raise ValueError("tree logits have incompatible rows")
    device = logits.device
    if packed is None:
        packed = pack_tree_traversal(tree, device=device)
    if any(
        tensor.device != device
        for tensor in (
            packed.token_ids,
            packed.parent_rows,
            packed.row_ids,
            packed.sentinel,
        )
    ):
        raise ValueError("packed traversal metadata is on the wrong device")
    posterior = logits.float().argmax(dim=-1)
    current = torch.zeros((), dtype=torch.long, device=device)
    alive = torch.ones((), dtype=torch.bool, device=device)
    accepted = torch.zeros((), dtype=torch.long, device=device)
    for _ in range(tree.horizon):
        predicted = posterior[0].gather(0, current.view(1))[0]
        matches = packed.parent_rows.eq(current) & packed.token_ids.eq(predicted)
        child = torch.where(matches, packed.row_ids, packed.sentinel).min()
        found = child.lt(rows)
        advance = alive & found
        accepted = accepted + advance.to(torch.long)
        current = torch.where(advance, child, current)
        alive = advance
    next_token = posterior[0].gather(0, current.view(1))[0]
    return accepted.view(1), next_token.view(1)

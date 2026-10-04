"""Global path decoding for span candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor

from .model_types import Edge
from .normalization import is_kanji


@dataclass(frozen=True)
class BatchedDecodedPath:
    """A Viterbi result expressed in padded-edge indices."""

    edge_indices: list[int]
    score: float


@dataclass(frozen=True)
class DecodedPath:
    edge_indices: list[int]
    edges: list[Edge]
    score: float

    @property
    def reading(self) -> str:
        return "".join(edge.reading for edge in self.edges)


def _forced_edges(edges: list[Edge], length: int) -> set[int]:
    """Select locked spans and reject overlapping alternatives."""
    locked = [(index, edge) for index, edge in enumerate(edges) if edge.locked]
    locked.sort(key=lambda item: (item[1].start, -(item[1].end - item[1].start), -item[1].confidence))
    selected: set[int] = set()
    occupied: list[tuple[int, int]] = []
    for index, edge in locked:
        if any(edge.start < end and start < edge.end for start, end in occupied):
            continue
        selected.add(index)
        occupied.append((edge.start, edge.end))
    if any(start < 0 or end > length for start, end in occupied):
        raise ValueError("locked edge is outside the input")
    return selected


def _allowed_edges(edges: list[Edge], length: int) -> set[int]:
    forced = _forced_edges(edges, length)
    if not forced:
        return set(range(len(edges)))
    occupied = [(edges[index].start, edges[index].end) for index in forced]
    allowed = set(forced)
    for index, edge in enumerate(edges):
        if index in forced:
            continue
        if any(edge.start < end and start < edge.end for start, end in occupied):
            continue
        allowed.add(index)
    return allowed


def allowed_edge_mask(edges: list[Edge], length: int,
                      *, device: torch.device | None = None) -> Tensor:
    """Return the pre-neural locked-edge filter as a tensor."""
    mask = torch.zeros(len(edges), dtype=torch.bool, device=device)
    if edges:
        mask[list(_allowed_edges(edges, length))] = True
    return mask


_COUNTER_SURFACES = frozenset((
    "つ", "時", "時間", "分", "秒", "年", "月", "日", "人", "回", "曲",
    "目", "匹", "個", "本", "枚", "台", "番", "度", "階", "杯", "冊", "倍", "段", "才", "歳", "歩"
))


_SINO_DIGIT_CHARS = frozenset("ぜろいちにさんよんしごろくななしちはちきゅうくじゅうひゃくせんまんおくちょう")


def _morpho_penalty(prev: Edge, curr: Edge) -> float:
    """Soft morphosyntactic constraints on adjacent candidate transitions."""
    # 1. Numeral + Japanese Counter or Kanji:
    # Digits followed immediately by counters/kanji take Sino-Japanese or native readings,
    # never unpronounced COPY, English numerals, or irregular fragments.
    if prev.surface.isdigit():
        is_counter = curr.surface in _COUNTER_SURFACES or (curr.surface and is_kanji(curr.surface[0]))
        if is_counter:
            if prev.reading == prev.surface:
                return -10.0
            if prev.reading == "みっ":
                # Native reading みっ is only valid before つ/日/か (みっつ,
                # みっか); the charset check below would otherwise reject it
                # (み/っ are not Sino-Japanese digits) and make this branch
                # unreachable.  つ itself is covered by the つ whitelist.
                if curr.surface not in ("つ", "日", "か"):
                    return -10.0
            else:
                if any(ch not in _SINO_DIGIT_CHARS for ch in prev.reading):
                    return -10.0
                if curr.surface == "つ":
                    # 'つ' takes Yamato compound counters (ひとつ, ふたつ),
                    # separate digit + つ is invalid
                    return -10.0

    # 2. On-yomi / ungrammatical verb stem inflection transitions
    if prev.surface == "微笑" and prev.reading == "びしょう":
        if curr.surface in ("ん", "んで", "む", "まない", "み", "めば", "もう", "んだ", "た", "て"):
            return -10.0
    elif prev.surface == "気付" and prev.reading in ("きつけ", "ぎづけ"):
        if curr.surface in ("い", "いて", "いた", "く", "かず", "けば", "き"):
            return -10.0
    elif prev.surface == "失" and prev.reading in ("しつ", "うしな"):
        if curr.surface in ("く", "くし", "くして", "くした", "くさ"):
            return -10.0
    elif prev.surface == "逃" and prev.reading in ("とう", "に"):
        if curr.surface in ("し", "した", "して", "す", "せば"):
            return -10.0
    elif prev.surface == "止" and prev.reading == "し":
        if curr.surface in ("ま", "まる", "まり", "まった", "まない", "め", "める", "めて"):
            return -10.0
    return 0.0


def decode_viterbi(scores: Tensor, edges: list[Edge], length: int,
                   *, force_locked: bool = True,
                   transition_scores: Tensor | None = None) -> DecodedPath:
    """Return the highest-scoring non-overlapping path covering ``[0, length)``.

    Supports both 0th-order semi-Markov DP (when ``transition_scores`` is None)
    and exact 1st-order semi-Markov DP (when pairwise transition scores are provided).
    """
    if scores.ndim != 1 or scores.numel() != len(edges):
        raise ValueError("scores must have one value for each edge")
    allowed = (_allowed_edges(edges, length) if force_locked
               else set(range(len(edges))))
    values = scores.detach().float().cpu().tolist()
    if transition_scores is not None:
        trans_vals = transition_scores.detach().float().cpu().tolist()
    else:
        trans_vals = None

    # Exact semi-Markov DP with morphosyntactic transition gating
    num_edges = len(edges)
    v_edge = [-math.inf] * num_edges
    back_edge = [-1] * num_edges
    by_end: list[list[int]] = [[] for _ in range(length + 1)]
    by_start: list[list[int]] = [[] for _ in range(length + 1)]
    for index in allowed:
        edge = edges[index]
        if 0 <= edge.start < edge.end <= length:
            by_start[edge.start].append(index)
            by_end[edge.end].append(index)

    for start in range(length):
        incoming = by_end[start]
        for curr in by_start[start]:
            val = values[curr]
            if start == 0:
                if val > v_edge[curr]:
                    v_edge[curr] = val
                    back_edge[curr] = -1
            elif incoming:
                best_score = -math.inf
                best_prev = -1
                for prev in incoming:
                    if not math.isfinite(v_edge[prev]):
                        continue
                    trans = trans_vals[prev][curr] if trans_vals is not None else 0.0
                    penalty = _morpho_penalty(edges[prev], edges[curr])
                    cand = v_edge[prev] + trans + val + penalty
                    if cand > best_score:
                        best_score = cand
                        best_prev = prev
                if best_score > v_edge[curr]:
                    v_edge[curr] = best_score
                    back_edge[curr] = best_prev

    terminal = by_end[length]
    best_terminal = -1
    best_total = -math.inf
    for edge_idx in terminal:
        if v_edge[edge_idx] > best_total:
            best_total = v_edge[edge_idx]
            best_terminal = edge_idx

    if best_terminal < 0 or not math.isfinite(best_total):
        raise ValueError("candidate graph has no path covering the input")

    indices_out = []
    cursor_edge = best_terminal
    while cursor_edge >= 0:
        indices_out.append(cursor_edge)
        cursor_edge = back_edge[cursor_edge]
    indices_out.reverse()
    return DecodedPath(indices_out, [edges[index] for index in indices_out], best_total)


def _check_batched_inputs(scores: Tensor, starts: Tensor, ends: Tensor,
                          mask: Tensor, lengths: Tensor | Sequence[int], *,
                          check_lengths: bool = True) -> Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, edges]")
    if starts.shape != scores.shape or ends.shape != scores.shape or mask.shape != scores.shape:
        raise ValueError("starts, ends, and mask must have the same shape as scores")
    if starts.device != scores.device or ends.device != scores.device or mask.device != scores.device:
        raise ValueError("scores, starts, ends, and mask must be on the same device")
    if starts.dtype != torch.long or ends.dtype != torch.long:
        raise ValueError("starts and ends must have dtype torch.long")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=scores.device)
    if lengths.ndim != 1 or lengths.numel() != scores.size(0):
        raise ValueError("lengths must have one value for each batch row")
    if check_lengths and bool((lengths < 0).any()):
        raise ValueError("lengths must be non-negative")
    return lengths


def _effective_edge_mask(scores: Tensor, mask: Tensor,
                         allowed_mask: Tensor | None) -> Tensor:
    if allowed_mask is None:
        return mask
    if allowed_mask.shape != scores.shape or allowed_mask.dtype != torch.bool:
        raise ValueError("allowed_mask must be boolean with the same shape as scores")
    if allowed_mask.device != scores.device:
        raise ValueError("allowed_mask must be on the same device as scores")
    return mask & allowed_mask


def _valid_edge_mask(starts: Tensor, ends: Tensor, mask: Tensor,
                     lengths: Tensor) -> Tensor:
    return (mask & (starts >= 0) & (ends > starts)
            & (ends <= lengths.unsqueeze(1)))


def _check_finite_scores(scores: Tensor, mask: Tensor) -> None:
    if not bool(torch.isfinite(scores[mask]).all()):
        raise FloatingPointError("candidate edge scores contain NaN or infinity")


def _scatter_logsumexp(values: Tensor, indices: Tensor, size: int) -> Tensor:
    """Differentiable 1-D logsumexp grouped by ``indices``."""
    if not values.numel():
        return values.new_full((size,), -torch.inf)
    # The detached maxima are only numerical offsets.  Keeping them outside the
    # value graph avoids max tie semantics while scatter_add carries gradients.
    maxima = values.detach().new_full((size,), -torch.inf)
    maxima.scatter_reduce_(
        0, indices, values.detach(), reduce="amax", include_self=True)
    safe_maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    totals = values.new_zeros(size)
    totals.scatter_add_(0, indices, torch.exp(values - safe_maxima[indices]))
    result = safe_maxima + torch.log(totals.clamp_min(torch.finfo(values.dtype).tiny))
    return result.masked_fill(totals == 0, -torch.inf)


def _batched_forward(scores: Tensor, starts: Tensor, ends: Tensor,
                     valid: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
    batch_size, edge_count = scores.shape
    max_length = int(lengths.max().item()) if batch_size else 0
    span = max_length + 1
    alpha = scores.new_full((batch_size, span), -torch.inf)
    alpha[:, 0] = 0.0
    if max_length == 0:
        return alpha, alpha.gather(1, lengths.unsqueeze(1)).squeeze(1)
    # Dense sync-free relaxation: mask over ALL edges per position instead of
    # extracting outgoing groups with nonzero()/any(), each of which drains
    # the CUDA pipeline. -inf masked entries weigh zero and receive zero
    # gradient, matching grouped semantics exactly.
    flat_rows = torch.arange(batch_size, device=scores.device
                             ).unsqueeze(1).expand(batch_size, edge_count
                                                   ).reshape(-1)
    destinations = flat_rows * span + ends.reshape(-1)
    positions = torch.arange(max_length, device=scores.device)
    selectors = (starts.reshape(-1).unsqueeze(0) == positions.unsqueeze(1)
                 ) & valid.reshape(-1)
    scores_flat = scores.reshape(-1)
    zero = scores.new_zeros(())
    for start in range(max_length):
        selected = selectors[start]
        row_alpha = alpha[:, start]
        finite_row = torch.isfinite(row_alpha)
        values = scores_flat + torch.where(
            finite_row, row_alpha, zero
        ).unsqueeze(1).expand(batch_size, edge_count).reshape(-1)
        mask = selected & finite_row.reshape(-1, 1).expand(
            batch_size, edge_count).reshape(-1)
        values = values.masked_fill(~mask, -torch.inf)
        updates = _scatter_logsumexp(values, destinations, batch_size * span)
        touched = torch.isfinite(updates).view(batch_size, span)
        finite_alpha = torch.isfinite(alpha)
        finite_target = finite_alpha | touched
        combined = torch.logaddexp(
            alpha.masked_fill(~finite_target, 0.0),
            updates.view(batch_size, span).masked_fill(~touched, -torch.inf))
        alpha = combined.masked_fill(~finite_target, -torch.inf)
    partitions = alpha.gather(1, lengths.unsqueeze(1)).squeeze(1)
    return alpha, partitions


def _batched_backward(scores: Tensor, starts: Tensor, ends: Tensor,
                      valid: Tensor, lengths: Tensor, max_length: int) -> Tensor:
    batch_size, edge_count = scores.shape
    span = max_length + 1
    beta = scores.new_full((batch_size, span), -torch.inf)
    beta.scatter_(1, lengths.unsqueeze(1), 0.0)
    if max_length == 0:
        return beta
    flat_rows = torch.arange(batch_size, device=scores.device
                             ).unsqueeze(1).expand(batch_size, edge_count
                                                   ).reshape(-1)
    destinations = flat_rows * span + ends.reshape(-1)
    positions = torch.arange(max_length, device=scores.device)
    selectors = (starts.reshape(-1).unsqueeze(0) == positions.unsqueeze(1)
                 ) & valid.reshape(-1)
    scores_flat = scores.reshape(-1)
    zero = scores.new_zeros(())
    for start in range(max_length - 1, -1, -1):
        selected = selectors[start]
        edge_beta = beta.reshape(-1)[destinations]
        finite_edge = torch.isfinite(edge_beta)
        values = scores_flat + torch.where(finite_edge, edge_beta, zero)
        mask = selected & finite_edge
        values = values.masked_fill(~mask, -torch.inf)
        updates = _scatter_logsumexp(values, flat_rows, batch_size)
        touched = torch.isfinite(updates)
        finite_beta = torch.isfinite(beta[:, start])
        finite_target = finite_beta | touched
        combined = torch.logaddexp(
            beta[:, start].masked_fill(~finite_target, 0.0),
            updates.masked_fill(~touched, -torch.inf))
        beta[:, start] = combined.masked_fill(~finite_target, -torch.inf)
    return beta


def _check_sparse_topology(scores: Tensor, edge_order: Tensor,
                           start_offsets: Sequence[int],
                           max_length: int) -> tuple[int, ...]:
    if not isinstance(edge_order, Tensor) or edge_order.ndim != 1:
        raise ValueError("edge_order must be a one-dimensional tensor")
    if edge_order.dtype != torch.long:
        raise ValueError("edge_order must have dtype torch.long")
    if edge_order.device != scores.device:
        raise ValueError("edge_order must be on the same device as scores")
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 0:
        raise ValueError("max_length must be a non-negative integer")
    offsets = tuple(start_offsets)
    if len(offsets) != max_length + 1:
        raise ValueError("start_offsets must have length max_length + 1")
    if any(not isinstance(offset, int) or isinstance(offset, bool)
           for offset in offsets):
        raise ValueError("start_offsets must contain integers")
    if (not offsets or offsets[0] != 0
            or offsets[-1] != edge_order.numel()
            or any(left > right for left, right in zip(offsets, offsets[1:]))):
        raise ValueError("start_offsets must delimit all of edge_order in order")
    if edge_order.numel() > scores.numel():
        raise ValueError("edge_order cannot contain more entries than scores")
    return offsets


def _sparse_fused_forward(
        scores: Tensor, ends: Tensor, valid: Tensor, gold_edge_mask: Tensor,
        lengths: Tensor, edge_order: Tensor, start_offsets: tuple[int, ...],
        max_length: int) -> tuple[Tensor, Tensor]:
    batch_size, edge_count = scores.shape
    span = max_length + 1
    alpha = scores.new_full((2, batch_size, span), -torch.inf)
    alpha[:, :, 0] = 0.0
    flat_scores = scores.reshape(-1)
    flat_ends = ends.reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_gold = gold_edge_mask.reshape(-1)
    channel_dest = torch.arange(2, device=scores.device) * batch_size * span
    for start in range(max_length):
        indices = edge_order[start_offsets[start]:start_offsets[start + 1]]
        if indices.numel() == 0:
            continue
        rows = torch.div(indices, edge_count, rounding_mode="floor")
        destinations = rows * span + flat_ends[indices].clamp(0, max_length)
        source = alpha[:, rows, start]
        edge_valid = flat_valid[indices]
        channel_valid = torch.stack((edge_valid, edge_valid & flat_gold[indices]))
        values = source + flat_scores[indices].unsqueeze(0)
        values = values.masked_fill(~(channel_valid & torch.isfinite(source)), -torch.inf)
        grouped = destinations.unsqueeze(0) + channel_dest.unsqueeze(1)
        updates = _scatter_logsumexp(values.reshape(-1), grouped.reshape(-1),
                                     2 * batch_size * span)
        alpha = torch.logaddexp(alpha, updates.view_as(alpha))
    partitions = alpha.gather(
        2, lengths.view(1, batch_size, 1).expand(2, -1, -1)).squeeze(2)
    return alpha, partitions


class _LogPartition(torch.autograd.Function):
    """Semi-Markov log partition with the closed-form CRF backward.

    d logZ / d score_e is exactly the edge marginal P(e), obtained from the
    forward-backward quantities already computed in forward.  Wrapping the DP
    in a custom Function keeps the enormous per-position autograd graph out of
    memory and turns backward into a single elementwise multiply.
    """

    @staticmethod
    def forward(ctx, scores: Tensor, starts: Tensor, ends: Tensor,
                valid: Tensor, lengths: Tensor) -> Tensor:
        with torch.no_grad():
            if scores.is_cuda:
                from .triton_dp import (fused_backward_beta,
                                        fused_forward_partitions)
                alpha, partitions = fused_forward_partitions(
                    scores, starts, ends, valid, lengths)
                max_length = (int(lengths.max().item())
                              if scores.size(0) and partitions.numel() else 0)
                beta = fused_backward_beta(
                    scores, starts, ends, valid, lengths, max_length)
            else:
                alpha, partitions = _batched_forward(
                    scores, starts, ends, valid, lengths)
                max_length = (int(lengths.max().item())
                              if scores.size(0) and partitions.numel() else 0)
                beta = _batched_backward(
                    scores, starts, ends, valid, lengths, max_length)
            span = alpha.size(1)
            starts_safe = starts.clamp(0, span - 1)
            ends_safe = ends.clamp(0, span - 1)
            alpha_edge = alpha.gather(1, starts_safe)
            beta_edge = beta.gather(1, ends_safe)
            finite = (valid & torch.isfinite(alpha_edge)
                      & torch.isfinite(beta_edge))
            scores_safe = torch.where(
                torch.isfinite(scores), scores, torch.zeros_like(scores))
            logits = (alpha_edge + scores_safe + beta_edge
                      - partitions.unsqueeze(1))
            posterior = torch.exp(
                logits.masked_fill(~finite, 0.0)
            ).masked_fill(~finite, 0.0)
        ctx.save_for_backward(posterior)
        return partitions

    @staticmethod
    def backward(ctx, grad_partitions: Tensor):
        (posterior,) = ctx.saved_tensors
        return posterior * grad_partitions.unsqueeze(1), None, None, None, None


def _sparse_fused_backward(
        scores: Tensor, ends: Tensor, valid: Tensor, gold_edge_mask: Tensor,
        lengths: Tensor, edge_order: Tensor, start_offsets: tuple[int, ...],
        max_length: int) -> Tensor:
    batch_size, edge_count = scores.shape
    span = max_length + 1
    beta = scores.new_full((2, batch_size, span), -torch.inf)
    beta.scatter_(2, lengths.view(1, batch_size, 1).expand(2, -1, -1), 0.0)
    flat_scores = scores.reshape(-1)
    flat_ends = ends.reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_gold = gold_edge_mask.reshape(-1)
    channel_rows = torch.arange(2, device=scores.device) * batch_size
    for start in range(max_length - 1, -1, -1):
        indices = edge_order[start_offsets[start]:start_offsets[start + 1]]
        if indices.numel() == 0:
            continue
        rows = torch.div(indices, edge_count, rounding_mode="floor")
        edge_beta = beta[:, rows, flat_ends[indices].clamp(0, max_length)]
        edge_valid = flat_valid[indices]
        channel_valid = torch.stack((edge_valid, edge_valid & flat_gold[indices]))
        values = edge_beta + flat_scores[indices].unsqueeze(0)
        values = values.masked_fill(
            ~(channel_valid & torch.isfinite(edge_beta)), -torch.inf)
        grouped = rows.unsqueeze(0) + channel_rows.unsqueeze(1)
        updates = _scatter_logsumexp(
            values.reshape(-1), grouped.reshape(-1), 2 * batch_size,
        ).view(2, batch_size)
        # Rows have different terminal positions.  Preserve beta[length] = 0
        # when the loop reaches a shorter row's terminal start.
        beta[:, :, start] = torch.logaddexp(beta[:, :, start], updates)
    return beta


def _check_sparse_partitions(partitions: Tensor) -> None:
    if not bool(torch.isfinite(partitions[0]).all()):
        rows = (~torch.isfinite(partitions[0])).nonzero(
            as_tuple=False).flatten().tolist()
        raise ValueError(
            f"candidate graph does not admit a complete path for rows {rows}")
    if not bool(torch.isfinite(partitions[1]).all()):
        rows = (~torch.isfinite(partitions[1])).nonzero(
            as_tuple=False).flatten().tolist()
        raise ValueError(
            "gold constraints do not admit a complete candidate path "
            f"for rows {rows}")


class _SparsePartialPathNll(torch.autograd.Function):
    """Fused full/gold CSR DP whose backward is the posterior difference."""

    @staticmethod
    def forward(ctx, scores: Tensor, ends: Tensor, valid: Tensor,
                gold_edge_mask: Tensor, lengths: Tensor, edge_order: Tensor,
                start_offsets: tuple[int, ...], max_length: int,
                check_finite: bool) -> Tensor:
        with torch.no_grad():
            alpha, partitions = _sparse_fused_forward(
                scores, ends, valid, gold_edge_mask, lengths, edge_order,
                start_offsets, max_length)
            if check_finite:
                _check_sparse_partitions(partitions)
            beta = _sparse_fused_backward(
                scores, ends, valid, gold_edge_mask, lengths, edge_order,
                start_offsets, max_length)
            batch_size, edge_count = scores.shape
            flat_indices = torch.arange(
                scores.numel(), device=scores.device, dtype=torch.long)
            rows = torch.div(flat_indices, edge_count, rounding_mode="floor")
            flat_ends = ends.reshape(-1).clamp(0, max_length)
            flat_valid = valid.reshape(-1)
            flat_gold = gold_edge_mask.reshape(-1)
            posterior = scores.new_zeros((2, scores.numel()))
            for start in range(max_length):
                indices = edge_order[start_offsets[start]:start_offsets[start + 1]]
                if indices.numel() == 0:
                    continue
                edge_rows = rows[indices]
                alpha_edge = alpha[:, edge_rows, start]
                beta_edge = beta[:, edge_rows, flat_ends[indices]]
                channel_valid = torch.stack(
                    (flat_valid[indices], flat_valid[indices] & flat_gold[indices]))
                finite = (channel_valid & torch.isfinite(alpha_edge)
                          & torch.isfinite(beta_edge))
                logits = (alpha_edge + scores.reshape(-1)[indices].unsqueeze(0)
                          + beta_edge - partitions[:, edge_rows])
                posterior[:, indices] = torch.exp(
                    logits.masked_fill(~finite, 0.0)
                ).masked_fill(~finite, 0.0)
            gradient = (posterior[0] - posterior[1]).view(batch_size, edge_count)
        ctx.save_for_backward(gradient)
        return partitions[0] - partitions[1]

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (gradient,) = ctx.saved_tensors
        return (gradient * grad_output.unsqueeze(1), None, None, None, None,
                None, None, None, None)


def _sparse_partial_path_nll_no_grad(
        scores: Tensor, ends: Tensor, valid: Tensor, gold_edge_mask: Tensor,
        lengths: Tensor, edge_order: Tensor, start_offsets: tuple[int, ...],
        max_length: int, check_finite: bool) -> Tensor:
    with torch.no_grad():
        _, partitions = _sparse_fused_forward(
            scores, ends, valid, gold_edge_mask, lengths, edge_order,
            start_offsets, max_length)
        if check_finite:
            _check_sparse_partitions(partitions)
    return partitions[0] - partitions[1]


def batched_log_partition(scores: Tensor, starts: Tensor, ends: Tensor,
                          mask: Tensor, lengths: Tensor | Sequence[int],
                          *, allowed_mask: Tensor | None = None) -> Tensor:
    """FP32 semi-Markov log partitions for padded edge batches.

    Edge work is tensorized across batch rows and padded edge slots; only the
    sequence-position recurrence is a Python loop. ``allowed_mask`` can apply
    an additional per-row constraint without changing the padding mask.
    Gradients use the closed-form edge marginals (see :class:`_LogPartition`).
    """
    lengths = _check_batched_inputs(scores, starts, ends, mask, lengths)
    mask = _effective_edge_mask(scores, mask, allowed_mask)
    scores = scores.float()
    valid = _valid_edge_mask(starts, ends, mask, lengths)
    _check_finite_scores(scores, valid)
    partitions = _LogPartition.apply(scores, starts, ends, valid, lengths)
    if not bool(torch.isfinite(partitions).all()):
        rows = (~torch.isfinite(partitions)).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"candidate graph does not admit a complete path for rows {rows}")
    return partitions


# Concise alias for callers that name batched operations with a ``batch_`` prefix.
batch_log_partition = batched_log_partition


def _first_order_forward_backward(
        scores: Tensor, transition: Tensor, starts: Tensor, ends: Tensor,
        valid: Tensor, lengths: Tensor,
        max_length: int) -> tuple[Tensor, Tensor, Tensor]:
    """First-order semi-Markov forward-backward (edge-indexed states).

    A path is a sequence of non-overlapping edges covering ``[0, length)`` with
    consecutive edges sharing a boundary (``end[p] == start[c]``). Its score is
    ``sum_e score[e] + sum_{p->c} transition[p, c]``. Returns ``(Z, edge_marg,
    trans_marg)`` where ``edge_marg[b, e] = P(edge e on path)`` and
    ``trans_marg[b, p, c] = P(p, c consecutive on path)`` -- exactly the
    closed-form gradients of ``logZ`` w.r.t. ``score`` and ``transition``. Run
    under ``no_grad`` by the autograd Function below.

    ``transition[b, p, c]`` is only read where ``end[p] == start[c]`` (both
    valid), so its value on non-adjacent pairs (0 as emitted by the model) is
    irrelevant.
    """
    batch, edges = scores.shape
    neg = float("-inf")
    # Preserve the caller's float dtype (callers pass float32; tests may pass
    # float64 for tight gradient checks).
    s = scores if scores.is_floating_point() else scores.float()
    t = transition if transition.is_floating_point() else transition.float()
    isfin = torch.isfinite
    # alpha[b, e] = logsum over prefixes ENDING with edge e (incl score[e] and
    # the transition into e). Base: edges starting at 0 have no incoming edge.
    start0 = (starts == 0) & valid
    alpha = torch.where(start0, s, s.new_full((batch, edges), neg))
    for pos in range(1, max_length + 1):
        out_mask = (starts == pos) & valid
        in_mask = (ends == pos) & valid                       # predecessors p
        # contrib[b, p, c] = alpha[p] + transition[p, c]
        contrib = alpha.unsqueeze(2) + t
        contrib = contrib.masked_fill(
            ~(in_mask & isfin(alpha)).unsqueeze(2), neg)
        agg = torch.logsumexp(contrib, dim=1)                 # over p -> [b, c]
        alpha = torch.where(out_mask, s + agg, alpha)
    # beta[b, e] = logsum over suffixes AFTER edge e (excl score[e]); the
    # transition out of e and the successor's score are included.
    term = (ends == lengths.unsqueeze(1)) & valid
    beta = torch.where(term, s.new_zeros(()), s.new_full((batch, edges), neg))
    for pos in range(max_length - 1, -1, -1):
        in_mask = (ends == pos) & valid                       # edges p to fill
        out_mask = (starts == pos) & valid                    # successors c
        succ = s + beta                                       # score[c] + beta[c]
        # contrib[b, p, c] = transition[p, c] + succ[c]
        contrib = t + succ.unsqueeze(1)
        contrib = contrib.masked_fill(
            ~(out_mask & isfin(succ)).unsqueeze(1), neg)
        agg = torch.logsumexp(contrib, dim=2)                 # over c -> [b, p]
        # Only non-terminal edges take the successor sum; terminal edges (their
        # end is the row length) have no successor and must keep beta = 0. With
        # mixed lengths in a batch a shorter row's terminal sits at pos < max
        # length, so without the ``~term`` guard this loop would overwrite its 0
        # with logsumexp(no successors) = -inf and wipe out every path.
        beta = torch.where(in_mask & ~term, agg, beta)
    partition = torch.logsumexp(
        torch.where(start0, s + beta, s.new_full((batch, edges), neg)), dim=1)
    z = partition.unsqueeze(1)
    edge_logits = alpha + beta - z
    edge_marg = torch.where(
        valid & isfin(alpha) & isfin(beta) & isfin(z),
        torch.exp(edge_logits.masked_fill(~isfin(edge_logits), neg)),
        s.new_zeros(()))
    adjacent = ((ends.unsqueeze(2) == starts.unsqueeze(1))
                & valid.unsqueeze(2) & valid.unsqueeze(1))
    trans_logits = (alpha.unsqueeze(2) + t + (s + beta).unsqueeze(1)
                    - z.unsqueeze(2))
    trans_marg = torch.where(
        adjacent & isfin(trans_logits),
        torch.exp(trans_logits.masked_fill(~isfin(trans_logits), neg)),
        s.new_zeros(()))
    return partition, edge_marg, trans_marg


class _FirstOrderPartialPathNll(torch.autograd.Function):
    """First-order full/gold NLL whose backward is the posterior difference.

    ``d(logZ_full - logZ_gold)/d score`` is the edge-marginal difference and
    the analogous transition gradient is the pairwise-marginal difference, both
    produced by the closed-form forward-backward above. The DP graph stays out
    of autograd; backward is an elementwise multiply.
    """

    @staticmethod
    def forward(ctx, scores: Tensor, transition: Tensor, starts: Tensor,
                ends: Tensor, valid: Tensor, gold_valid: Tensor,
                lengths: Tensor, max_length: int,
                check_finite: bool) -> Tensor:
        with torch.no_grad():
            full, em_f, tm_f = _first_order_forward_backward(
                scores, transition, starts, ends, valid, lengths, max_length)
            if check_finite and not bool(torch.isfinite(full).all()):
                rows = (~torch.isfinite(full)).nonzero(
                    as_tuple=False).flatten().tolist()
                raise ValueError(
                    f"candidate graph does not admit a complete path for rows {rows}")
            gold, em_g, tm_g = _first_order_forward_backward(
                scores, transition, starts, ends, gold_valid, lengths, max_length)
            if check_finite and not bool(torch.isfinite(gold).all()):
                rows = (~torch.isfinite(gold)).nonzero(
                    as_tuple=False).flatten().tolist()
                raise ValueError(
                    "gold constraints do not admit a complete candidate path "
                    f"for rows {rows}")
            ctx.save_for_backward(em_f - em_g, tm_f - tm_g)
        return full - gold

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        edge_grad, trans_grad = ctx.saved_tensors
        g = grad_output.unsqueeze(1)
        return (edge_grad * g, trans_grad * g.unsqueeze(2),
                None, None, None, None, None, None, None)


def batched_partial_path_nll(
        scores: Tensor, starts: Tensor, ends: Tensor, mask: Tensor,
        gold_edge_mask: Tensor, lengths: Tensor | Sequence[int], *,
        transition: Tensor | None = None,
        allowed_mask: Tensor | None = None, edge_order: Tensor | None = None,
        start_offsets: Sequence[int] | None = None,
        max_length: int | None = None, check_finite: bool = True) -> Tensor:
    """Per-row sparse-gold NLL for a padded semi-Markov edge batch.

    Always uses the closed-form-gradient :class:`_LogPartition` operator: the
    DP runs graph-free in forward and its backward is an elementwise multiply
    with the edge marginals. ``edge_order``/``start_offsets``/``max_length``
    are accepted for call-site compatibility with the collator's precomputed
    metadata but are no longer required by the dense sync-free recurrence.
    """
    lengths = _check_batched_inputs(
        scores, starts, ends, mask, lengths, check_lengths=check_finite)
    mask = _effective_edge_mask(scores, mask, allowed_mask)
    if gold_edge_mask.shape != scores.shape or gold_edge_mask.dtype != torch.bool:
        raise ValueError("gold_edge_mask must be boolean with the same shape as scores")
    if gold_edge_mask.device != scores.device:
        raise ValueError("gold_edge_mask must be on the same device as scores")
    scores = scores.float()
    valid = _valid_edge_mask(starts, ends, mask, lengths)
    if check_finite:
        _check_finite_scores(scores, valid)
    if transition is not None:
        if transition.shape != (scores.size(0), scores.size(1), scores.size(1)):
            raise ValueError("transition must have shape [batch, edges, edges]")
        transition = transition.to(scores.dtype)
        ml = int(lengths.max().item()) if scores.size(0) else 0
        return _FirstOrderPartialPathNll.apply(
            scores, transition, starts, ends, valid, valid & gold_edge_mask,
            lengths, ml, check_finite)
    full = _LogPartition.apply(scores, starts, ends, valid, lengths)
    if check_finite and not bool(torch.isfinite(full).all()):
        rows = (~torch.isfinite(full)).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            f"candidate graph does not admit a complete path for rows {rows}")
    gold = _LogPartition.apply(
        scores, starts, ends, valid & gold_edge_mask, lengths)
    if check_finite and not bool(torch.isfinite(gold).all()):
        rows = (~torch.isfinite(gold)).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            "gold constraints do not admit a complete candidate path "
            f"for rows {rows}")
    return full - gold

batch_partial_path_nll = batched_partial_path_nll


def batched_edge_log_posterior(scores: Tensor, starts: Tensor, ends: Tensor,
                               mask: Tensor, lengths: Tensor | Sequence[int],
                               *, allowed_mask: Tensor | None = None) -> Tensor:
    """Return edge log posteriors, with invalid/padded entries set to ``-inf``."""
    lengths = _check_batched_inputs(scores, starts, ends, mask, lengths)
    mask = _effective_edge_mask(scores, mask, allowed_mask)
    scores = scores.float()
    valid = _valid_edge_mask(starts, ends, mask, lengths)
    _check_finite_scores(scores, valid)
    alpha, partitions = _batched_forward(scores, starts, ends, valid, lengths)
    if not bool(torch.isfinite(partitions).all()):
        rows = (~torch.isfinite(partitions)).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"candidate graph does not admit a complete path for rows {rows}")
    beta = _batched_backward(scores, starts, ends, valid, lengths, alpha.size(1) - 1)
    safe_starts = starts.clamp(min=0, max=alpha.size(1) - 1)
    safe_ends = ends.clamp(min=0, max=beta.size(1) - 1)
    edge_values = (alpha.gather(1, safe_starts) + scores
                   + beta.gather(1, safe_ends) - partitions.unsqueeze(1))
    reachable = valid & torch.isfinite(alpha.gather(1, safe_starts)) & torch.isfinite(beta.gather(1, safe_ends))
    return edge_values.masked_fill(~reachable, -torch.inf)


def batched_edge_posterior(scores: Tensor, starts: Tensor, ends: Tensor,
                           mask: Tensor, lengths: Tensor | Sequence[int],
                           *, allowed_mask: Tensor | None = None) -> Tensor:
    """Return edge posterior probabilities, with invalid entries set to zero."""
    log_posterior = batched_edge_log_posterior(
        scores, starts, ends, mask, lengths, allowed_mask=allowed_mask)
    # Do not backpropagate through exp(-inf); its local derivative is benign,
    # but replacing padding first makes the zero-gradient intent explicit.
    finite = torch.isfinite(log_posterior)
    return torch.exp(log_posterior.masked_fill(~finite, 0.0)).masked_fill(~finite, 0.0)


batch_edge_log_posterior = batched_edge_log_posterior
batch_edge_posterior = batched_edge_posterior


def edge_log_posterior(scores: Tensor, starts: Tensor, ends: Tensor,
                       mask: Tensor, length: int, *,
                       allowed_mask: Tensor | None = None) -> Tensor:
    """Single-example forward-backward edge log posterior."""
    batched_allowed = (allowed_mask.unsqueeze(0)
                       if allowed_mask is not None else None)
    return batched_edge_log_posterior(
        scores.unsqueeze(0), starts.unsqueeze(0), ends.unsqueeze(0),
        mask.unsqueeze(0), [length], allowed_mask=batched_allowed,
    ).squeeze(0)


def edge_posterior(scores: Tensor, starts: Tensor, ends: Tensor,
                   mask: Tensor, length: int, *,
                   allowed_mask: Tensor | None = None) -> Tensor:
    """Single-example edge posterior probabilities."""
    batched_allowed = (allowed_mask.unsqueeze(0)
                       if allowed_mask is not None else None)
    return batched_edge_posterior(
        scores.unsqueeze(0), starts.unsqueeze(0), ends.unsqueeze(0),
        mask.unsqueeze(0), [length], allowed_mask=batched_allowed,
    ).squeeze(0)


def batched_decode_viterbi(scores: Tensor, starts: Tensor, ends: Tensor,
                           mask: Tensor, lengths: Tensor | Sequence[int], *,
                           allowed_mask: Tensor | None = None
                           ) -> list[BatchedDecodedPath]:
    """Viterbi decode padded edge batches, tensorizing each position's edges."""
    lengths = _check_batched_inputs(scores, starts, ends, mask, lengths)
    mask = _effective_edge_mask(scores, mask, allowed_mask)
    scores = scores.float()
    valid = _valid_edge_mask(starts, ends, mask, lengths)
    _check_finite_scores(scores, valid)
    batch_size = scores.size(0)
    max_length = int(lengths.max().item()) if batch_size else 0
    best = scores.new_full((batch_size, max_length + 1), -torch.inf)
    best[:, 0] = 0.0
    back = torch.full((batch_size, max_length + 1), -1,
                      dtype=torch.long, device=scores.device)
    for start in range(max_length):
        outgoing = valid & (starts == start) & torch.isfinite(best[:, start]).unsqueeze(1)
        if not bool(outgoing.any()):
            continue
        rows, edge_indices = outgoing.nonzero(as_tuple=True)
        values = best[rows, start] + scores[rows, edge_indices]
        destinations = rows * (max_length + 1) + ends[rows, edge_indices]
        flat_size = batch_size * (max_length + 1)
        updates = values.new_full((flat_size,), -torch.inf)
        updates.scatter_reduce_(
            0, destinations, values, reduce="amax", include_self=True)
        winners = values == updates[destinations]
        winner_destinations = destinations[winners]
        winner_edges = edge_indices[winners]
        # amin gives deterministic first-edge tie breaking.
        update_back = torch.full((flat_size,), scores.size(1),
                                 dtype=torch.long, device=scores.device)
        update_back.scatter_reduce_(
            0, winner_destinations, winner_edges, reduce="amin", include_self=True)
        flat_best = best.flatten()
        improve = updates > flat_best
        updated_best = torch.where(improve, updates, flat_best)
        best = updated_best.view(batch_size, max_length + 1)
        flat_back = back.flatten()
        updated_back = torch.where(improve, update_back, flat_back)
        back = updated_back.view(batch_size, max_length + 1)

    results: list[BatchedDecodedPath] = []
    for row, row_length in enumerate(lengths.tolist()):
        score = best[row, row_length]
        if not bool(torch.isfinite(score)):
            raise ValueError(
                f"candidate graph does not admit a complete path for row {row}")
        indices: list[int] = []
        cursor = row_length
        while cursor:
            edge_index = int(back[row, cursor].item())
            if edge_index < 0:
                raise RuntimeError(f"broken Viterbi backpointer for row {row}")
            indices.append(edge_index)
            cursor = int(starts[row, edge_index].item())
        indices.reverse()
        results.append(BatchedDecodedPath(indices, float(score.detach().cpu())))
    return results


batch_decode_viterbi = batched_decode_viterbi


def log_partition(scores: Tensor, starts: Tensor, ends: Tensor,
                  mask: Tensor, length: int) -> Tensor:
    """Differentiable semi-Markov partition function for one example."""
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    alpha = [scores.new_full((), -torch.inf) for _ in range(length + 1)]
    alpha[0] = scores.new_zeros(())
    for start in range(length):
        # Do not create logaddexp(-inf, -inf) nodes for unreachable starts.
        # Their forward value is harmless, but their backward derivative is
        # undefined and can silently put NaNs into otherwise valid gradients.
        if not bool(torch.isfinite(alpha[start]).detach()):
            continue
        outgoing = ((starts == start) & mask & (ends > starts) & (ends <= length)).nonzero(as_tuple=False).flatten()
        if not outgoing.numel():
            continue
        values = scores[outgoing] + alpha[start]
        for edge_index, value in zip(outgoing.tolist(), values):
            end = int(ends[edge_index])
            alpha[end] = torch.logaddexp(alpha[end], value)
    return alpha[length]


def partial_path_nll(scores: Tensor, starts: Tensor, ends: Tensor,
                     mask: Tensor, gold_edge_mask: Tensor, length: int) -> Tensor:
    """NLL where ``gold_edge_mask`` describes edges compatible with gold.

    This is the training primitive for sparse ruby: unlabelled positions can
    leave many edges enabled, while an explicit ruby span can restrict the
    allowed path without inventing a unique character-level alignment.
    """
    # AMP is useful for the scorer, but the small dynamic program benefits from
    # FP32 accumulation, especially when many paths are nearly tied.
    scores = scores.float()
    if not torch.isfinite(scores[mask]).all():
        raise FloatingPointError("candidate edge scores contain NaN or infinity")
    full = log_partition(scores, starts, ends, mask, length)
    if not torch.isfinite(full):
        raise ValueError("candidate graph does not admit a complete path")
    gold = log_partition(scores, starts, ends, mask & gold_edge_mask, length)
    if not torch.isfinite(gold):
        raise ValueError("gold constraints do not admit a complete candidate path")
    return full - gold

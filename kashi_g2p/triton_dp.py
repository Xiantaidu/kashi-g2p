"""Triton fused semi-Markov DP: one kernel launch per forward/backward pass.

The reference implementation (decoder._batched_forward/_batched_backward)
runs a Python loop over text positions, launching ~8 small CUDA kernels per
position -- ~2000 launches per training step, which saturates the CPU launch
path at ~2 s/step while the GPU idles at 30 W.  These kernels keep the whole
DP in ONE launch: one Triton program per batch row, the row's alpha/beta
recurrence serialized inside the program (the position dependency is
inherent), and the per-position edge scatter-logsumexp vectorized over the
row's edges in BLOCK-sized tiles.

Semantics match decoder._batched_forward/_batched_backward exactly:
  forward  alpha[0]=0; alpha[t] = logsumexp over valid edges (s=t-1, e<=S-1)
           of score + alpha[t-1]; -inf stays -inf (no phantom mass)
  backward beta[L]=0;   beta[t] = logsumexp over valid edges (s=t) of
           score + beta[e]

`fused_forward_partitions` returns (alpha, partitions); partitions are
gathered by the caller.  `fused_backward_beta` returns beta.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_forward_kernel(
    scores_ptr,   # [B, E] fp32
    starts_ptr,   # [B, E] int32
    ends_ptr,     # [B, E] int32
    valid_ptr,    # [B, E] int8
    alpha_ptr,    # [B, S] fp32 out
    lengths_ptr,  # [B] int32
    E: tl.constexpr,
    S: tl.constexpr,
    S_M,
    EDGE_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    edge_base = scores_ptr + row.to(tl.int64) * E

    # alpha row kept in REGISTERS (S <= 65 << BLOCK_S); global-memory
    # store-then-load across the t loop is not reliably ordered by Triton
    soffs = tl.arange(0, BLOCK_S)
    alpha = tl.where(soffs == 0, tl.zeros((), dtype=tl.float32) + 0.0,
                     tl.full((), float("-inf"), dtype=tl.float32))

    for t in range(1, S_M + 1):
        # alpha[t] = logsumexp over ALL valid edges ending at t of
        # score + alpha[start]
        a_prev_vec = tl.where(soffs < t, alpha, float("-inf"))
        acc_max = float("-inf")
        acc_sum = 0.0
        for tile in range(EDGE_TILES):
            offs = tile * BLOCK + tl.arange(0, BLOCK)
            em = offs < E
            es = tl.load(starts_ptr + row.to(tl.int64) * E + offs, mask=em, other=-1)
            en = tl.load(ends_ptr + row.to(tl.int64) * E + offs, mask=em, other=-1)
            v = tl.load(valid_ptr + row.to(tl.int64) * E + offs, mask=em, other=0)
            sc = tl.load(edge_base + offs, mask=em, other=0.0)
            safe_idx = tl.minimum(tl.maximum(es, 0), BLOCK_S - 1)
            a_start = tl.sum(tl.where(soffs[None, :] == safe_idx[:, None],
                                      a_prev_vec[None, :], 0.0), axis=1)
            a_start = tl.where((es >= 0) & (es < t), a_start, float("-inf"))
            take = em & (v != 0) & (es >= 0) & (es < t) & (en == t)
            contrib = tl.where(take, sc + a_start, float("-inf"))
            tile_max = tl.max(contrib, axis=0)
            new_max = tl.maximum(acc_max, tile_max)
            safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
            old_scale = tl.exp(acc_max - safe_max)
            acc_sum = acc_sum * old_scale                 + tl.sum(tl.where(take, tl.exp(contrib - safe_max), 0.0), axis=0)
            acc_max = new_max
        upd = tl.where(acc_max > float("-inf"),
                       acc_max + tl.log(tl.maximum(acc_sum, 1e-45)),
                       float("-inf"))
        prev = tl.sum(tl.where(soffs == t, alpha, 0.0), axis=0)
        diff = prev - upd
        new_val = tl.where(
            upd > float("-inf"),
            tl.where(prev > float("-inf"),
                     tl.maximum(prev, upd) + tl.log(1.0 + tl.exp(-tl.abs(diff))),
                     upd),
            prev)
        alpha = tl.where(soffs == t, new_val.to(tl.float32), alpha)

    tl.store(alpha_ptr + row.to(tl.int64) * S + soffs, alpha, mask=soffs < S)


@triton.jit
def _fused_backward_kernel(
    scores_ptr,   # [B, E] fp32
    starts_ptr,   # [B, E] int32
    ends_ptr,     # [B, E] int32
    valid_ptr,    # [B, E] int8
    beta_ptr,     # [B, S] fp32 out
    lengths_ptr,  # [B] int32
    E: tl.constexpr,
    S: tl.constexpr,
    S_M,
    EDGE_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    edge_base = scores_ptr + row.to(tl.int64) * E
    length = tl.load(lengths_ptr + row)

    soffs = tl.arange(0, BLOCK_S)
    beta = tl.where(soffs == length, tl.zeros((), dtype=tl.float32) + 0.0,
                    tl.full((), float("-inf"), dtype=tl.float32))

    for t in range(S_M - 1, -1, -1):
        # beta[t] = logsumexp over valid edges with start == t of
        # score + beta[end]
        acc_max = float("-inf")
        acc_sum = 0.0
        for tile in range(EDGE_TILES):
            offs = tile * BLOCK + tl.arange(0, BLOCK)
            em = offs < E
            es = tl.load(starts_ptr + row.to(tl.int64) * E + offs, mask=em, other=-1)
            en = tl.load(ends_ptr + row.to(tl.int64) * E + offs, mask=em, other=-1)
            v = tl.load(valid_ptr + row.to(tl.int64) * E + offs, mask=em, other=0)
            sc = tl.load(edge_base + offs, mask=em, other=0.0)
            safe_idx = tl.minimum(tl.maximum(en, 0), BLOCK_S - 1)
            b_end = tl.sum(tl.where(soffs[None, :] == safe_idx[:, None],
                                    beta[None, :], 0.0), axis=1)
            take = em & (v != 0) & (es == t) & (en >= 0) & (en < S)
            contrib = tl.where(take, sc + b_end, float("-inf"))
            tile_max = tl.max(contrib, axis=0)
            new_max = tl.maximum(acc_max, tile_max)
            safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
            old_scale = tl.exp(acc_max - safe_max)
            acc_sum = acc_sum * old_scale                 + tl.sum(tl.where(take, tl.exp(contrib - safe_max), 0.0), axis=0)
            acc_max = new_max
        upd = tl.where(acc_max > float("-inf"),
                       acc_max + tl.log(tl.maximum(acc_sum, 1e-45)),
                       float("-inf"))
        prev = tl.sum(tl.where(soffs == t, beta, 0.0), axis=0)
        diff = prev - upd
        new_val = tl.where(
            upd > float("-inf"),
            tl.where(prev > float("-inf"),
                     tl.maximum(prev, upd) + tl.log(1.0 + tl.exp(-tl.abs(diff))),
                     upd),
            prev)
        beta = tl.where(soffs == t, new_val.to(tl.float32), beta)

    tl.store(beta_ptr + row.to(tl.int64) * S + soffs, beta, mask=soffs < S)


def _edge_tiles(edge_count: int, block: int = 128) -> int:
    return (edge_count + block - 1) // block


def fused_forward_partitions(
    scores: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor,
    valid: torch.Tensor, lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference-equivalent forward in ONE kernel launch.

    Returns (alpha [B, S], partitions [B]); partitions gathered by the caller.
    """
    batch_size, edge_count = scores.shape
    max_length = int(lengths.max().item()) if batch_size else 0
    span = max_length + 1
    S_P = 80
    alpha = scores.new_full((batch_size, S_P), float("-inf"))
    if max_length == 0:
        return alpha, alpha.gather(1, lengths.unsqueeze(1)).squeeze(1)
    starts32 = starts.to(torch.int32)
    ends32 = ends.to(torch.int32)
    valid8 = valid.to(torch.int8)
    lengths32 = lengths.to(torch.int32)
    # fixed-size specialization: Triton recompiles per distinct runtime int
    # (S/E values and their divisibility), which spikes ~200-350ms per new
    # shape. Pad to fixed sizes so ONE compiled kernel serves every batch.
    # fixed E_P: one compiled kernel; max_edges=256 + fusion headroom < 384
    E_P = 384
    S_P = 80
    tiles = E_P // 128
    if E_P > edge_count:
        pad = torch.zeros(batch_size, E_P - edge_count, dtype=torch.int32,
                          device=scores.device)
        starts32 = torch.cat((starts32, pad), dim=1)
        ends32 = torch.cat((ends32, pad), dim=1)
        valid8 = torch.cat((valid8, torch.zeros_like(pad)), dim=1)
        scores = torch.cat((scores, torch.zeros(batch_size, E_P - edge_count,
                                                dtype=scores.dtype,
                                                device=scores.device)), dim=1)
    _fused_forward_kernel[(batch_size,)](
        scores, starts32, ends32, valid8, alpha, lengths32,
        E_P, S_P, span - 1, tiles, BLOCK=128, BLOCK_S=128, num_warps=4)
    partitions = alpha.gather(1, lengths.unsqueeze(1)).squeeze(1)
    return alpha, partitions


def fused_backward_beta(
    scores: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor,
    valid: torch.Tensor, lengths: torch.Tensor, max_length: int,
) -> torch.Tensor:
    """Reference-equivalent backward in ONE kernel launch."""
    batch_size, edge_count = scores.shape
    span = max_length + 1
    S_P = 80
    beta = scores.new_full((batch_size, S_P), float("-inf"))
    if max_length == 0:
        return beta
    starts32 = starts.to(torch.int32)
    ends32 = ends.to(torch.int32)
    valid8 = valid.to(torch.int8)
    lengths32 = lengths.to(torch.int32)
    # fixed E_P: one compiled kernel; max_edges=256 + fusion headroom < 384
    E_P = 384
    S_P = 80
    tiles = E_P // 128
    if E_P > edge_count:
        pad = torch.zeros(batch_size, E_P - edge_count, dtype=torch.int32,
                          device=scores.device)
        starts32 = torch.cat((starts32, pad), dim=1)
        ends32 = torch.cat((ends32, pad), dim=1)
        valid8 = torch.cat((valid8, torch.zeros_like(pad)), dim=1)
        scores = torch.cat((scores, torch.zeros(batch_size, E_P - edge_count,
                                                dtype=scores.dtype,
                                                device=scores.device)), dim=1)
    _fused_backward_kernel[(batch_size,)](
        scores, starts32, ends32, valid8, beta, lengths32,
        E_P, S_P, span - 1, tiles, BLOCK=128, BLOCK_S=128, num_warps=4)
    return beta

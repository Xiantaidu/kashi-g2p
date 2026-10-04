"""Uncertainty Gated Inference Engine for kashi-g2p v4."""

from __future__ import annotations

import torch
from torch import Tensor


def gated_rerank_candidates(
    candidate_readings: list[str],
    candidate_model_scores: list[float],
    cross_encoder_scores: list[float] | Tensor,
    threshold: float = 0.8,
    beta: float = 0.5,
) -> tuple[str, bool, int]:
    """Select final candidate under uncertainty gating and z-score calibration.

    Args:
        candidate_readings: list of folded candidate reading strings.
        candidate_model_scores: log-probability scores from exp32 main model.
        cross_encoder_scores: raw logits from Exp32WarmStartCrossEncoder.
        threshold: margin threshold (S0 - S1).
                   If gap >= threshold: Main model is confident -> Bypass Reranker (0 False Positives).
                   If gap < threshold: Main model is uncertain -> Trigger Reranker arbitration.
        beta: scale factor for z-score normalized cross-encoder logits.

    Returns:
        tuple of (selected_reading, is_gated_triggered, selected_index).
    """
    if len(candidate_readings) <= 1:
        return candidate_readings[0], False, 0

    s0 = candidate_model_scores[0]
    s1 = candidate_model_scores[1]
    score_gap = s0 - s1

    # High-confidence zone: base model is confident -> Bypass Reranker
    if score_gap >= threshold:
        return candidate_readings[0], False, 0

    # Ambiguity / Competitive zone: trigger Reranker
    if isinstance(cross_encoder_scores, list):
        c_scores = torch.tensor(cross_encoder_scores, dtype=torch.float)
    else:
        c_scores = cross_encoder_scores.detach().float().cpu()

    # z-score normalization within candidate list
    std = c_scores.std()
    if std > 1e-6:
        z_scores = (c_scores - c_scores.mean()) / std
    else:
        z_scores = torch.zeros_like(c_scores)

    m_scores = torch.tensor(candidate_model_scores, dtype=torch.float)
    final_scores = m_scores + beta * z_scores
    best_idx = int(torch.argmax(final_scores).item())

    return candidate_readings[best_idx], True, best_idx

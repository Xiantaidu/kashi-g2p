"""Unified span generator and scorer for kashi-g2p v4."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        value_fp32 = value.float()
        norm = value_fp32 * torch.rsqrt(
            value_fp32.square().mean(-1, keepdim=True) + self.eps
        )
        return (norm * self.weight.float()).to(dtype=value.dtype)


def _rotate_half(value: Tensor) -> Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        inverse = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        frequencies = torch.einsum("i,j->ij", positions, inverse)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        length = query.size(-2)
        if length > self.cos.size(-2):
            raise ValueError(
                f"sequence length {length} exceeds RoPE limit {self.cos.size(-2)}"
            )
        cos = self.cos[..., :length, :].to(query.dtype)
        sin = self.sin[..., :length, :].to(query.dtype)
        return (query * cos + _rotate_half(query) * sin,
                key * cos + _rotate_half(key) * sin)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(dim, d_ff, bias=False)
        self.up = nn.Linear(dim, d_ff, bias=False)
        self.down = nn.Linear(d_ff, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        return self.down(self.dropout(F.silu(self.gate(value)) * self.up(value)))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, d_ff: int, dropout: float,
                 max_seq_len: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm_attn = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.attn_dropout = dropout
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim, d_ff, dropout)

    def forward(self, value: Tensor, attention_mask: Tensor) -> Tensor:
        batch, length, dim = value.shape
        query = self.norm_attn(value)
        q, k, v = self.qkv(query).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask[:, None, None, :].bool(),
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        value = value + self.attn_out(attended)
        return value + self.ffn(self.norm_ffn(value))


class EdgeFusion(nn.Module):
    """Refresh characters from all candidate edges covering each position."""

    def __init__(self, dim: int, edge_dim: int, heads: int, dropout: float):
        super().__init__()
        if edge_dim % heads:
            raise ValueError("edge_dim must be divisible by fusion heads")
        self.heads = heads
        self.head_dim = edge_dim // heads
        self.norm = RMSNorm(dim)
        self.query = nn.Linear(dim, edge_dim, bias=False)
        self.key = nn.Linear(edge_dim, edge_dim, bias=False)
        self.value = nn.Linear(edge_dim, edge_dim, bias=False)
        self.output = nn.Linear(edge_dim, dim, bias=False)
        self.gate = nn.Linear(dim, 1)
        self.dropout = dropout
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, hidden: Tensor, edge_features: Tensor,
                coverage: Tensor, attention_mask: Tensor) -> Tensor:
        batch, length, _ = hidden.shape
        edges = edge_features.size(1)
        if not edges:
            return hidden
        query = self.query(self.norm(hidden)).view(
            batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(edge_features).view(
            batch, edges, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(edge_features).view(
            batch, edges, self.heads, self.head_dim).transpose(1, 2)
        allowed = coverage[:, None].bool()
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = weights * allowed.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        weights = F.dropout(weights, self.dropout, self.training)
        update = torch.matmul(weights, value).transpose(1, 2).reshape(
            batch, length, -1)
        update = self.output(update) * torch.sigmoid(self.gate(hidden))
        return hidden + update * attention_mask.unsqueeze(-1).to(update.dtype)


def _attention_bias(attention_mask: Tensor, *, dtype: torch.dtype,
                    device: torch.device | None = None) -> Tensor:
    """Return the additive key mask retained for checkpoint compatibility."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be [batch, length]")
    valid = attention_mask.to(
        device=device or attention_mask.device, dtype=torch.bool)
    bias = torch.zeros(
        (valid.size(0), 1, 1, valid.size(1)), dtype=dtype,
        device=valid.device)
    return bias.masked_fill(~valid[:, None, None, :], float("-inf"))


def _sinusoidal_rows(length: int, dimension: int, device: torch.device,
                     dtype: torch.dtype, *, offset: int = 0) -> Tensor:
    if length <= 0:
        return torch.empty((0, dimension), device=device, dtype=dtype)
    positions = torch.arange(
        offset, offset + length, device=device, dtype=torch.float32
    )[:, None]
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, dimension))
    )
    angles = positions * frequencies[None]
    result = torch.zeros((length, dimension), device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(angles)
    result[:, 1::2] = torch.cos(angles[:, :result[:, 1::2].shape[1]])
    return result.to(dtype=dtype)


def _position_rows(table: nn.Embedding, length: int, device: torch.device) -> Tensor:
    learned_length = min(length, table.num_embeddings)
    learned = table(torch.arange(learned_length, device=device))
    if learned_length == length:
        return learned
    return torch.cat((learned, _sinusoidal_rows(
        length - learned_length, table.embedding_dim, device,
        table.weight.dtype, offset=learned_length)), dim=0)


class TransitionHead(nn.Module):
    """Pairwise bilinear transition head between adjacent candidate edges."""

    def __init__(self, edge_dim: int, transition_dim: int = 64, *,
                 use_reading: bool = False):
        super().__init__()
        self.left_proj = nn.Linear(edge_dim, transition_dim, bias=False)
        self.right_proj = nn.Linear(edge_dim, transition_dim, bias=False)
        self.reading_left_proj = (nn.Linear(edge_dim, transition_dim, bias=False)
                                  if use_reading else None)
        self.reading_right_proj = (nn.Linear(edge_dim, transition_dim, bias=False)
                                   if use_reading else None)
        self.gate = nn.Linear(edge_dim * 2, 1)
        nn.init.zeros_(self.left_proj.weight)
        nn.init.zeros_(self.right_proj.weight)
        if self.reading_left_proj is not None:
            nn.init.zeros_(self.reading_left_proj.weight)
            nn.init.zeros_(self.reading_right_proj.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        bilinear = (self.left_proj(left) * self.right_proj(right)).sum(-1)
        gate = torch.sigmoid(self.gate(torch.cat((left, right), -1))).squeeze(-1)
        return bilinear * gate

    def project_candidates(self, context: Tensor,
                           reading: Tensor) -> tuple[Tensor, Tensor]:
        """Project each candidate, retaining its reading at shared boundaries.

        Separate reading projections preserve legacy context weight shapes.
        This is equivalent to a linear projection of [context; reading].
        """
        left, right = self.left_proj(context), self.right_proj(context)
        if self.reading_left_proj is not None:
            left = left + self.reading_left_proj(reading)
            right = right + self.reading_right_proj(reading)
        return left, right


@dataclass
class SpanModelOutput:
    edge_scores: Tensor
    hidden: Tensor
    edge_features: Tensor
    boundary_logits: Tensor | None = None
    generator_logits: Tensor | None = None
    transition_scores: Tensor | None = None
    transition_pairs: Tensor | None = None
    first_order_loss: bool = False


@dataclass
class GeneratedReadings:
    span_start: Tensor
    span_end: Tensor
    token_ids: Tensor
    lengths: Tensor
    log_probs: Tensor


class ReadingDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm_self = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True, bias=False)
        self.norm_cross = RMSNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True, bias=False)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim, d_ff, dropout)

    def forward(self, value: Tensor, memory: Tensor, causal_mask: Tensor) -> Tensor:
        normalized = self.norm_self(value)
        attended, _ = self.self_attn(
            normalized, normalized, normalized, attn_mask=causal_mask,
            need_weights=False)
        value = value + attended
        normalized = self.norm_cross(value)
        attended, _ = self.cross_attn(
            normalized, memory, memory, need_weights=False)
        value = value + attended
        return value + self.ffn(self.norm_ffn(value))


class OpenReadingGenerator(nn.Module):
    """Small conditional decoder that proposes dictionary-OOV readings."""

    def __init__(self, vocab_size: int, context_dim: int, *, dim: int = 256,
                 layers: int = 3, heads: int = 8, d_ff: int = 640,
                 max_length: int = 64, dropout: float = 0.1,
                 pad_id: int = 0, bos_id: int = 2, eos_id: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_length = max_length
        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.position_embed = nn.Embedding(max_length + 1, dim)
        self.context_proj = nn.Linear(context_dim, dim, bias=False)
        self.blocks = nn.ModuleList([
            ReadingDecoderBlock(dim, heads, d_ff, dropout) for _ in range(layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.token_embed.weight
        self.dropout = nn.Dropout(dropout)

    def forward(self, context: Tensor, decoder_input_ids: Tensor) -> Tensor:
        if decoder_input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must be [spans, target_length]")
        length = decoder_input_ids.size(1)
        if length > self.max_length:
            raise ValueError("reading target exceeds generator max_length")
        positions = self.position_embed(
            torch.arange(length, device=decoder_input_ids.device))[None]
        value = self.dropout(self.token_embed(decoder_input_ids) + positions)
        memory = self.context_proj(context).unsqueeze(1)
        causal = torch.ones(
            (length, length), dtype=torch.bool, device=value.device).triu(1)
        for block in self.blocks:
            value = block(value, memory, causal)
        return self.output(self.norm(value))

    @torch.inference_mode()
    def beam_search(self, context: Tensor, *, beam_size: int = 4,
                    max_length: int | None = None,
                    allowed_token_ids: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.size(0) != 1:
            raise ValueError("beam_search accepts one span context")
        limit = min(max_length or self.max_length, self.max_length)
        beams = [(torch.tensor([self.bos_id], device=context.device), 0.0)]
        finished: list[tuple[Tensor, float]] = []
        for _ in range(limit):
            candidates: list[tuple[Tensor, float]] = []
            for tokens, score in beams:
                if int(tokens[-1]) == self.eos_id:
                    finished.append((tokens, score))
                    continue
                logits = self(context, tokens.unsqueeze(0))[0, -1].float()
                valid = torch.zeros_like(logits, dtype=torch.bool)
                if allowed_token_ids is None:
                    valid[:] = True
                else:
                    valid[allowed_token_ids.to(logits.device)] = True
                valid[self.eos_id] = True
                valid[self.pad_id] = False
                valid[self.bos_id] = False
                logits = logits.masked_fill(~valid, -torch.inf)
                values, indices = torch.topk(
                    F.log_softmax(logits, -1), min(beam_size, int(valid.sum())))
                for value, index in zip(values.tolist(), indices.tolist()):
                    candidates.append((torch.cat((tokens, tokens.new_tensor([index]))),
                                       score + float(value)))
            if not candidates:
                break
            candidates.sort(key=lambda item: item[1] / max(1, item[0].numel() - 1),
                            reverse=True)
            beams = candidates[:beam_size]
        finished.extend(beams)
        finished.sort(key=lambda item: item[1] / max(1, item[0].numel() - 1),
                      reverse=True)
        selected = finished[:beam_size]
        width = max(tokens.numel() for tokens, _ in selected)
        result = torch.full((len(selected), width), self.pad_id,
                            dtype=torch.long, device=context.device)
        scores = torch.empty(len(selected), device=context.device)
        for index, (tokens, score) in enumerate(selected):
            result[index, :tokens.numel()] = tokens
            scores[index] = score
        return result, scores


class SpanG2P(nn.Module):
    """Unified word/character path model with an optional open reading proposer."""

    def __init__(self, vocab_size: int = 7027, *, layers: int = 8,
                 dim: int = 512, heads: int = 8, d_ff: int = 1280,
                 max_seq_len: int = 256, max_reading_len: int = 64,
                 component_vocab_size: int = 1024, component_dim: int = 256,
                 char_type_count: int = 7, source_count: int = 16,
                 source_dim: int = 32, edge_dim: int = 256,
                 dropout: float = 0.1, reading_kernel_size: int = 5,
                 fusion_layers: tuple[int, ...] | list[int] = (2, 4, 6, 8),
                 fusion_heads: int = 8, max_span_length: int = 16,
                 use_open_generator: bool = True, generator_dim: int = 256,
                 use_prior_table: bool = False, prior_table_size: int = 1,
                 prior_dim: int = 4,
                 generator_layers: int = 3, generator_heads: int = 8,
                 generator_d_ff: int = 640, generator_max_length: int = 64,
                 activation_checkpointing: bool = False,
                 reading_chunk_size: int = 0,
                 use_per_layer_edge_seed: bool = False,
                 use_transition_head: bool = False,
                 transition_dim: int = 64,
                 first_order_loss: bool = False,
                 transition_features: str = "context"):
        super().__init__()
        if max_seq_len < 1 or max_reading_len < 1:
            raise ValueError("sequence and reading limits must be positive")
        if reading_kernel_size < 1 or reading_kernel_size % 2 == 0:
            raise ValueError("reading_kernel_size must be a positive odd number")
        if transition_features not in {"context", "context_reading"}:
            raise ValueError("transition_features must be context or context_reading")
        if transition_features != "context" and not use_transition_head:
            raise ValueError("reading-aware transitions require use_transition_head")
        self.pad_id = 0
        self.dim = dim
        self.edge_dim = edge_dim
        self.max_span_length = max_span_length
        self.max_reading_len = max_reading_len
        self.activation_checkpointing = bool(activation_checkpointing)
        self.reading_chunk_size = int(reading_chunk_size)
        self.use_per_layer_edge_seed = bool(use_per_layer_edge_seed)
        self.use_transition_head = bool(use_transition_head)
        self.transition_dim = int(transition_dim)
        self.transition_features = transition_features
        # When set, the transition head is trained: its scores flow into the
        # first-order path loss with gradients (see forward below). Off by
        # default so 0th-order experiments are byte-for-byte unchanged.
        self.first_order_loss = bool(first_order_loss)
        # Opt-in only and deliberately not a parameter/buffer, so checkpoints and
        # state_dict keys are unchanged.
        self.profile_ranges = False
        self.char_embed = nn.Embedding(vocab_size, dim, padding_idx=self.pad_id)
        self.component_embed = nn.Embedding(
            component_vocab_size, component_dim, padding_idx=0)
        self.component_proj = nn.Linear(component_dim, dim, bias=False)
        self.char_type_embed = nn.Embedding(char_type_count, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, heads, d_ff, dropout, max_seq_len)
            for _ in range(layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.reading_pos = nn.Embedding(max_reading_len, dim)
        self.reading_conv = nn.Conv1d(
            dim, dim, kernel_size=reading_kernel_size,
            padding=reading_kernel_size // 2, groups=dim, bias=False)
        self.reading_norm = RMSNorm(dim)
        self.reading_proj = nn.Linear(dim, edge_dim, bias=False)
        self.context_proj = nn.Linear(dim * 3, edge_dim, bias=False)
        self.source_embed = nn.Embedding(source_count, source_dim)
        requested_fusions = tuple(sorted(set(int(value) for value in fusion_layers)))
        self.fusion_layers = tuple(
            value for value in requested_fusions if 1 <= value <= layers)
        if not self.fusion_layers and layers > 0 and requested_fusions:
            self.fusion_layers = (layers,)
        self.edge_seed_in_dim = edge_dim * 2 + source_dim + 2
        if self.use_per_layer_edge_seed:
            self.edge_seeds = nn.ModuleDict({
                str(value): nn.Linear(self.edge_seed_in_dim, edge_dim, bias=False)
                for value in self.fusion_layers
            })
            self.edge_seed = None
        else:
            self.edge_seeds = None
            self.edge_seed = nn.Linear(self.edge_seed_in_dim, edge_dim, bias=False)
        self.fusions = nn.ModuleDict({
            str(value): EdgeFusion(
                dim, edge_dim,
                min(fusion_heads, edge_dim), dropout)
            for value in self.fusion_layers
        })
        self.use_prior_table = bool(use_prior_table)
        self.prior_dim = prior_dim if self.use_prior_table else 0
        self.prior_table = (nn.Embedding(prior_table_size, self.prior_dim)
                            if self.use_prior_table else None)
        if self.prior_table is not None:
            nn.init.normal_(self.prior_table.weight, mean=0.0, std=0.02)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim * 3 + source_dim + 2 + self.prior_dim, edge_dim),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim // 2), nn.SiLU(),
            nn.Linear(edge_dim // 2, 1, bias=False),
        )
        self.transition_head = (TransitionHead(
            edge_dim, self.transition_dim,
            use_reading=transition_features == "context_reading")
                                if self.use_transition_head else None)
        self.boundary_start = nn.Linear(dim, max_span_length, bias=False)
        self.boundary_end = nn.Linear(dim, max_span_length, bias=False)
        self.use_open_generator = bool(use_open_generator)
        self.generator = (OpenReadingGenerator(
            vocab_size, edge_dim, dim=generator_dim, layers=generator_layers,
            heads=generator_heads, d_ff=generator_d_ff,
            max_length=generator_max_length, dropout=dropout)
            if self.use_open_generator else None)
        self._initialize(reading_kernel_size)
        if self.first_order_loss and self.transition_head is not None:
            self.reinitialize_transition_for_training()

    def reinitialize_transition_for_training(self, std: float = 0.02) -> None:
        """Break the transition head off its zero init so it can be trained.

        ``TransitionHead`` zero-inits both bilinear projections so the head is a
        decode-time no-op when bolted onto a 0th-order model. That is a dead
        gradient for first-order training: the bilinear ``left*right`` and its
        gradient w.r.t. both factors are identically zero, so it never leaves
        zero. Re-seed both projections with small noise. Call this on a fresh
        model and again after a warm start that loaded the zero weights.
        """
        if self.transition_head is None:
            return
        with torch.no_grad():
            nn.init.normal_(self.transition_head.left_proj.weight, mean=0.0, std=std)
            nn.init.normal_(self.transition_head.right_proj.weight, mean=0.0, std=std)
            if self.transition_head.reading_left_proj is not None:
                # Add the new condition without overwhelming pretrained unary
                # scores. Context projections are nonzero, so both reading
                # branches can learn through the bilinear cross terms.
                nn.init.zeros_(self.transition_head.reading_left_proj.weight)
                nn.init.zeros_(self.transition_head.reading_right_proj.weight)

    def _initialize(self, reading_kernel_size: int) -> None:
        for embedding in (self.char_embed, self.component_embed,
                          self.char_type_embed, self.reading_pos, self.source_embed):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.char_embed.weight[self.pad_id].zero_()
            self.component_embed.weight[self.pad_id].zero_()
            self.reading_conv.weight.zero_()
            self.reading_conv.weight[:, 0, reading_kernel_size // 2] = 1.0
        scale = (2.0 * max(1, len(self.blocks))) ** -0.5
        for block in self.blocks:
            block.attn_out.weight.data.mul_(scale)
            block.ffn.down.weight.data.mul_(scale)

    @staticmethod
    def _gather(value: Tensor, index: Tensor) -> Tensor:
        return torch.gather(
            value, 1, index.unsqueeze(-1).expand(*index.shape, value.size(-1)))

    def _span_features(self, hidden: Tensor, starts: Tensor, ends: Tensor) -> Tensor:
        batch, length, dim = hidden.shape
        # Span-boundary aggregation accumulates over variable-length spans; keep
        # it in fp32 so BF16 rounding cannot push it into an overflow that the
        # semi-Markov DP then propagates into an inf loss.
        value = hidden.float()
        prefix = torch.cat((value.new_zeros((batch, 1, dim)), value.cumsum(1)), 1)
        start_hidden = self._gather(value, starts.clamp(0, length - 1))
        end_hidden = self._gather(value, (ends - 1).clamp(0, length - 1))
        span_sum = torch.gather(
            prefix, 1, ends.clamp(0, length).unsqueeze(-1).expand(*ends.shape, dim))
        span_sum = span_sum - torch.gather(
            prefix, 1, starts.clamp(0, length).unsqueeze(-1).expand(*starts.shape, dim))
        span_length = (ends - starts).clamp_min(1).unsqueeze(-1).to(value.dtype)
        return self.context_proj(torch.cat(
            (start_hidden, end_hidden, span_sum / span_length), -1))

    def _encode_reading_chunk(self, ids: Tensor, mask: Tensor,
                              positions: Tensor) -> Tensor:
        # Pooling over reading positions sums up to 64 tokens; accumulate in
        # fp32 so BF16 autocast cannot round it into an inf that eventually
        # destabilizes the edge scores and the path NLL.
        _batch, _edges, length = ids.shape
        encoded = F.embedding(ids, self.char_embed.weight) + positions
        encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)
        shape = encoded.shape
        encoded = self.reading_conv(
            encoded.reshape(-1, length, self.dim).transpose(1, 2))
        encoded = encoded.transpose(1, 2).reshape(*shape)
        encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)
        encoded = encoded.float()
        denom = mask.sum(-1, keepdim=True).to(encoded.dtype).clamp_min(1)
        return self.reading_norm(encoded.sum(-2) / denom)

    def _encode_readings(self, ids: Tensor, mask: Tensor) -> Tensor:
        length = ids.size(-1)
        positions = _position_rows(self.reading_pos, length, ids.device)[None, None]
        chunk_size = self.reading_chunk_size or ids.size(1)
        outputs = []
        for start in range(0, ids.size(1), chunk_size):
            args = (ids[:, start:start + chunk_size],
                    mask[:, start:start + chunk_size], positions)
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(
                    self._encode_reading_chunk, *args, use_reentrant=False))
            else:
                outputs.append(self._encode_reading_chunk(*args))
        return torch.cat(outputs, 1)

    def _encode_unique_readings(self, ids: Tensor, mask: Tensor,
                                batch_size: int) -> Tensor:
        length = ids.size(-1)
        if ids.size(0) == 0:
            return self.char_embed.weight.new_empty((0, self.dim))
        positions = _position_rows(self.reading_pos, length, ids.device)[None, None]
        chunk_size = (batch_size * self.reading_chunk_size
                      if self.reading_chunk_size else ids.size(0))
        outputs = []
        for start in range(0, ids.size(0), chunk_size):
            args = (ids[start:start + chunk_size].unsqueeze(0),
                    mask[start:start + chunk_size].unsqueeze(0), positions)
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                encoded = checkpoint(
                    self._encode_reading_chunk, *args, use_reentrant=False)
            else:
                encoded = self._encode_reading_chunk(*args)
            outputs.append(encoded.squeeze(0))
        return torch.cat(outputs, 0)

    def _reading_features(
            self, edge_reading_ids: Tensor | None,
            edge_reading_mask: Tensor | None, edge_mask: Tensor,
            unique_reading_ids: Tensor | None,
            unique_reading_mask: Tensor | None,
            edge_reading_inverse: Tensor | None) -> Tensor:
        use_unique = (unique_reading_ids is not None
                      or unique_reading_mask is not None
                      or edge_reading_inverse is not None)
        if use_unique:
            if (unique_reading_ids is None or unique_reading_mask is None
                    or edge_reading_inverse is None):
                raise ValueError("unique reading inputs must be provided together")
            if edge_reading_inverse.shape != edge_mask.shape:
                raise ValueError("edge_reading_inverse must match edge_mask")
            encoded = self.reading_proj(self._encode_unique_readings(
                unique_reading_ids, unique_reading_mask, edge_mask.size(0)))
            if encoded.size(0) == 0:
                reading = encoded.new_zeros((*edge_mask.shape, self.edge_dim))
            else:
                inverse = edge_reading_inverse.clamp(0, encoded.size(0) - 1)
                reading = encoded[inverse]
            return reading * edge_mask.unsqueeze(-1).to(reading.dtype)
        if edge_reading_ids is None or edge_reading_mask is None:
            raise ValueError("dense reading ids and mask are required")
        return self.reading_proj(
            self._encode_readings(edge_reading_ids, edge_reading_mask))

    @staticmethod
    def _coverage(starts: Tensor, ends: Tensor, edge_mask: Tensor,
                  length: int) -> Tensor:
        positions = torch.arange(length, device=starts.device)[None, :, None]
        return ((positions >= starts[:, None]) & (positions < ends[:, None])
                & edge_mask[:, None].bool())

    def _input_embedding(self, input_ids: Tensor,
                         component_ids: Tensor | None,
                         char_type_ids: Tensor | None) -> Tensor:
        hidden = self.char_embed(input_ids)
        if component_ids is not None:
            components = self.component_embed(component_ids).sum(-2)
            hidden = hidden + self.component_proj(components)
        if char_type_ids is not None:
            hidden = hidden + self.char_type_embed(char_type_ids)
        return self.dropout(hidden)

    def _profile(self, name: str):
        return (torch.autograd.profiler.record_function(name)
                if self.profile_ranges else nullcontext())

    def encode(self, input_ids: Tensor, attention_mask: Tensor,
               component_ids: Tensor | None = None,
               char_type_ids: Tensor | None = None) -> Tensor:
        length = input_ids.size(1)
        if length > self.blocks[0].rope.cos.size(-2):
            raise ValueError(
                f"input sequence length {length} exceeds max_seq_len "
                f"{self.blocks[0].rope.cos.size(-2)}"
            )
        with self._profile("model/input_embedding"):
            hidden = self._input_embedding(input_ids, component_ids, char_type_ids)
        for layer_index, block in enumerate(self.blocks, start=1):
            with self._profile(f"model/encode/block_{layer_index:02d}"):
                if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                    hidden = checkpoint(
                        block, hidden, attention_mask, use_reentrant=False)
                else:
                    hidden = block(hidden, attention_mask)
        return self.final_norm(hidden)

    def boundary_logits(self, hidden: Tensor) -> Tensor:
        start = self.boundary_start(hidden)
        end = self.boundary_end(hidden)
        return start + end

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                edge_start: Tensor, edge_end: Tensor,
                edge_reading_ids: Tensor | None = None,
                edge_reading_mask: Tensor | None = None,
                edge_mask: Tensor | None = None,
                edge_source_ids: Tensor | None = None,
                edge_prior: Tensor | None = None,
                edge_locked: Tensor | None = None,
                component_ids: Tensor | None = None,
                char_type_ids: Tensor | None = None,
                generator_context: Tensor | None = None,
                decoder_input_ids: Tensor | None = None,
                unique_reading_ids: Tensor | None = None,
                unique_reading_mask: Tensor | None = None,
                edge_reading_inverse: Tensor | None = None,
                edge_pack_ids: Tensor | None = None,
                target_offset: Tensor | int | None = None,
                target_length: Tensor | int | None = None) -> SpanModelOutput:
        if edge_mask is None or edge_source_ids is None:
            raise ValueError("edge_mask and edge_source_ids are required")
        if target_offset is not None:
            offset = (target_offset if isinstance(target_offset, Tensor)
                      else torch.tensor(target_offset, device=edge_start.device))
            if offset.ndim == 0:
                offset = offset.view(1, 1).expand(edge_start.size(0), 1)
            elif offset.ndim == 1:
                offset = offset.unsqueeze(1)
            span_start = edge_start + offset
            span_end = edge_end + offset
        else:
            offset = None
            span_start = edge_start
            span_end = edge_end

        with self._profile("model/input_embedding"):
            hidden = self._input_embedding(input_ids, component_ids, char_type_ids)
        with self._profile("model/reading"):
            reading = self._reading_features(
                edge_reading_ids, edge_reading_mask, edge_mask,
                unique_reading_ids, unique_reading_mask, edge_reading_inverse)
        with self._profile("model/edge_static"):
            source = self.source_embed(
                edge_source_ids.clamp(0, self.source_embed.num_embeddings - 1))
            length_feature = (edge_end - edge_start).clamp_min(1).float().log1p().unsqueeze(-1)
            prior = (edge_prior.float().unsqueeze(-1) if edge_prior is not None
                     else torch.zeros_like(length_feature))
            coverage = self._coverage(span_start, span_end, edge_mask, input_ids.size(1))
        for layer_index, block in enumerate(self.blocks, start=1):
            with self._profile(f"model/encode/block_{layer_index:02d}"):
                if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                    hidden = checkpoint(block, hidden, attention_mask, use_reentrant=False)
                else:
                    hidden = block(hidden, attention_mask)
            fusion = self.fusions[str(layer_index)] if str(layer_index) in self.fusions else None
            if fusion is not None:
                with self._profile(f"model/edge_fusion/layer_{layer_index:02d}"):
                    context = self._span_features(hidden, span_start, span_end)
                    seed_module = (self.edge_seeds[str(layer_index)]
                                   if self.edge_seeds is not None and str(layer_index) in self.edge_seeds
                                   else self.edge_seed)
                    seed = seed_module(torch.cat(
                        (context, reading, source, length_feature, prior), -1))
                    hidden = fusion(hidden, seed, coverage, attention_mask)
        with self._profile("model/scoring"):
            hidden = self.final_norm(hidden)
            context = self._span_features(hidden, span_start, span_end)
            features = torch.cat(
                (context, reading, context * reading, source, length_feature, prior), -1)
            if self.prior_table is not None and edge_pack_ids is not None:
                features = torch.cat((features, self.prior_table(
                    edge_pack_ids.clamp(0, self.prior_table.num_embeddings - 1))), -1)
            scores = self.edge_mlp(features).squeeze(-1)
            if edge_locked is not None and edge_prior is not None:
                scores = torch.where(edge_locked.bool(), edge_prior.float(), scores)
            scores = scores.masked_fill(~edge_mask.bool(), torch.finfo(scores.dtype).min)
            boundary = self.boundary_logits(hidden)

        transition_scores = None
        if self.transition_head is not None:
            with self._profile("model/transition"):
                # Detach only when the head is inference-only (0th-order
                # training). With first_order_loss the scores must carry
                # gradients into the path loss so the head actually learns.
                grad_ctx = (torch.no_grad()
                            if (self.training and not self.first_order_loss)
                            else nullcontext())
                with grad_ctx:
                    left_proj, right_proj = self.transition_head.project_candidates(
                        context, reading)
                    trans_matrix = torch.bmm(left_proj, right_proj.transpose(1, 2))  # [batch, edges, edges]
                    is_adjacent = (
                        (edge_end.unsqueeze(2) == edge_start.unsqueeze(1)) &
                        edge_mask.unsqueeze(2).bool() &
                        edge_mask.unsqueeze(1).bool()
                    )
                    transition_scores = trans_matrix.masked_fill(~is_adjacent, 0.0)

        generator_logits = None
        if decoder_input_ids is not None:
            if self.generator is None:
                raise ValueError("decoder_input_ids require use_open_generator=true")
            if generator_context is None:
                raise ValueError("generator_context is required with decoder_input_ids")
            generator_logits = self.generator(generator_context, decoder_input_ids)
        return SpanModelOutput(scores, hidden, features, boundary, generator_logits,
                               transition_scores=transition_scores,
                               first_order_loss=self.first_order_loss)

    @torch.inference_mode()
    def propose_readings(self, hidden: Tensor, *, top_spans: int = 8,
                         beam_size: int = 4,
                         max_reading_length: int | None = None,
                         allowed_token_ids: Tensor | None = None) -> GeneratedReadings:
        if self.generator is None:
            raise RuntimeError("open reading generator is disabled")
        if hidden.ndim != 3 or hidden.size(0) != 1:
            raise ValueError("propose_readings currently accepts one encoded sentence")
        length = hidden.size(1)
        candidates: list[tuple[float, int, int, Tensor]] = []
        logits = self.boundary_logits(hidden)[0]
        for start in range(length):
            for span_length in range(1, min(self.max_span_length, length - start) + 1):
                end = start + span_length
                starts = torch.tensor([[start]], device=hidden.device)
                ends = torch.tensor([[end]], device=hidden.device)
                context = self._span_features(hidden, starts, ends)[0, 0]
                candidates.append((float(logits[start, span_length - 1]),
                                   start, end, context))
        candidates.sort(key=lambda item: item[0], reverse=True)
        rows: list[tuple[int, int, Tensor, float]] = []
        for _score, start, end, context in candidates[:top_spans]:
            tokens, scores = self.generator.beam_search(
                context, beam_size=beam_size, max_length=max_reading_length,
                allowed_token_ids=allowed_token_ids)
            for token_row, score in zip(tokens, scores.tolist()):
                rows.append((start, end, token_row, float(score)))
        width = max((row[2].numel() for row in rows), default=1)
        token_ids = torch.full((len(rows), width), self.pad_id,
                               dtype=torch.long, device=hidden.device)
        lengths = torch.zeros(len(rows), dtype=torch.long, device=hidden.device)
        starts = torch.zeros(len(rows), dtype=torch.long, device=hidden.device)
        ends = torch.zeros(len(rows), dtype=torch.long, device=hidden.device)
        scores = torch.zeros(len(rows), device=hidden.device)
        for index, (start, end, tokens, score) in enumerate(rows):
            token_ids[index, :tokens.numel()] = tokens
            lengths[index] = tokens.numel()
            starts[index] = start
            ends[index] = end
            scores[index] = score
        return GeneratedReadings(starts, ends, token_ids, lengths, scores)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

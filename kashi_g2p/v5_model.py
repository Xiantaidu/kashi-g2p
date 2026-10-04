"""V5 architecture: reading-sequence decoder, calibrated unary scores, V5-G relations.

Design: docs/V5_ARCHITECTURE.md. The text backbone (char embedding + IDS +
char-type + Transformer blocks + optional EdgeFusion) is reused verbatim from
``model.py`` so exp35/exp39 checkpoints warm-start with matching state names.
On top of it:

- a 4-layer 384-dim span reading Transformer decoder produces the conditional
  reading-sequence score ``q(e) = sum log P(y_j | y_<j, H, s, t)`` including
  EOS, and the EOS-position hidden state ``r_e`` (the position that has seen
  the full candidate reading);
- a candidate representation ``z_e`` fuses the span condition, ``r_e``, source,
  lengths, and prior; the unary head scores ``u(e) = MLP(z_e) + alpha q(e)
  + beta len(reading) + gamma len(surface)`` with alpha/beta/gamma learned and
  zero-initialized so the sequence branch starts as a no-op;
- the first-order transition head operates on ``z_e``;
- V5-G (``use_relation_attention``) adds candidate relation-attention blocks
  (self / same-span / predecessor / successor / overlap, plus a boundary
  distance bias) before the unary head.

Everything here is a single jointly trained network. Training losses
(L_path via the existing first-order DP, L_read teacher-forcing CE, L_contrast
margin) are wired in ``train.py``; the model only exposes their inputs.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import RMSNorm, SpanG2P, SwiGLU, TransitionHead
from .model_types import SOURCE_TO_ID


@dataclass
class V5ModelOutput:
    """Outputs consumed by train/pipeline: same contract as SpanModelOutput."""

    edge_scores: Tensor
    hidden: Tensor
    edge_features: Tensor
    sequence_log_probs: Tensor | None = None
    transition_scores: Tensor | None = None
    first_order_loss: bool = False
    read_loss: Tensor | None = None
    contrast_loss: Tensor | None = None
    contrast_negatives: Tensor | None = None


class V5DecoderBlock(nn.Module):
    """Causal self-attention + full-text cross-attention block (384-dim).

    Cross-attention uses the row's text key-padding mask so padded positions
    never contribute, unlike the single-span memory in model.ReadingDecoderBlock.
    """

    def __init__(self, dim: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm_self = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.self_out = nn.Linear(dim, dim, bias=False)
        self.norm_cross = RMSNorm(dim)
        self.cross_q = nn.Linear(dim, dim, bias=False)
        self.cross_kv = nn.Linear(dim, dim * 2, bias=False)
        self.cross_out = nn.Linear(dim, dim, bias=False)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim, d_ff, dropout)
        self.dropout = dropout

    def _split(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, value: Tensor, memory: Tensor,
                memory_mask: Tensor) -> Tensor:
        length = value.size(1)
        normalized = self.norm_self(value)
        q, k, v = (self._split(part) for part in
                   self.qkv(normalized).chunk(3, dim=-1))
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(value.shape)
        value = value + self.self_out(attended)

        normalized = self.norm_cross(value)
        query = self._split(self.cross_q(normalized))
        keys, values = (self._split(part) for part in
                        self.cross_kv(memory).chunk(2, dim=-1))
        # memory_mask [rows, text_len] broadcasts over heads and query positions.
        attended = F.scaled_dot_product_attention(
            query, keys, values,
            attn_mask=memory_mask[:, None, None, :].bool(),
            dropout_p=self.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(value.shape)
        value = value + self.cross_out(attended)
        return value + self.ffn(self.norm_ffn(value))


class RelationAttentionBlock(nn.Module):
    """V5-G candidate interaction: attention over edges with relation bias.

    Relations are defined on real source boundaries (not list order):
    0 self, 1 same-span competitor, 2 predecessor, 3 successor, 4 overlap.
    A gated residual keeps the block a no-op at init.
    """

    SELF = 0
    SAME_SPAN = 1
    PREDECESSOR = 2
    SUCCESSOR = 3
    OVERLAP = 4
    RELATION_COUNT = 5

    def __init__(self, dim: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.relation_bias = nn.Parameter(torch.zeros(self.RELATION_COUNT, heads))
        self.distance_scale = nn.Parameter(torch.zeros(heads))
        self.gate = nn.Linear(dim, 1)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim, d_ff, dropout)
        self.dropout = dropout
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    @staticmethod
    def relation_types(starts: Tensor, ends: Tensor,
                       edge_mask: Tensor) -> Tensor:
        """Classify every candidate pair; padded pairs fall back to self."""
        same_span = ((starts.unsqueeze(2) == starts.unsqueeze(1))
                     & (ends.unsqueeze(2) == ends.unsqueeze(1)))
        predecessor = (ends.unsqueeze(2) == starts.unsqueeze(1))
        successor = (starts.unsqueeze(2) == ends.unsqueeze(1))
        overlap = ((starts.unsqueeze(2) < ends.unsqueeze(1))
                   & (starts.unsqueeze(1) < ends.unsqueeze(2)))
        pair_valid = edge_mask.unsqueeze(2) & edge_mask.unsqueeze(1)
        relation = torch.full(same_span.shape, 0, device=starts.device,
                              dtype=torch.long)
        relation = torch.where(overlap & ~predecessor & ~successor, 4, relation)
        relation = torch.where(successor & ~same_span, 3, relation)
        relation = torch.where(predecessor & ~same_span, 2, relation)
        relation = torch.where(same_span, 1, relation)
        return torch.where(pair_valid, relation, 0)

    @staticmethod
    def boundary_distance(starts: Tensor, ends: Tensor) -> Tensor:
        gap_start = (starts.unsqueeze(2) - ends.unsqueeze(1)).abs()
        gap_end = (starts.unsqueeze(1) - ends.unsqueeze(2)).abs()
        return torch.minimum(gap_start, gap_end).float().clamp_max(1e6)

    def forward(self, value: Tensor, relation: Tensor,
                distance: Tensor) -> Tensor:
        batch, edges, dim = value.shape
        normalized = self.norm(value)
        heads, head_dim = self.heads, self.head_dim
        q, k, v = (part.view(batch, edges, heads, head_dim).transpose(1, 2)
                   for part in self.qkv(normalized).chunk(3, dim=-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        # relation [B, E, E] indexes to [B, E, E, heads]; move heads to axis 1
        # and add the per-head distance bias for the same [B, heads, E, E].
        relation_part = self.relation_bias[relation].permute(0, 3, 1, 2)
        distance_part = (-torch.log1p(distance))[:, None]  # [B, heads, E, E]
        bias = relation_part + self.distance_scale[None, :, None, None] * distance_part
        scores = scores + bias
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = F.dropout(weights, self.dropout, self.training)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(
            batch, edges, heads * head_dim)
        update = self.out(attended) * torch.sigmoid(self.gate(value))
        value = value + update
        return value + self.ffn(self.norm_ffn(value))


class SpanG2Pv5(SpanG2P):
    """V5: text backbone + span reading decoder + calibrated path scoring.

    Backbone state names match ``SpanG2P`` so exp35/exp39 weights warm-start;
    the conv-pooled reading encoder, edge MLP, and open generator are absent.
    """

    def __init__(self, vocab_size: int = 7027, *, layers: int = 8,
                 dim: int = 512, heads: int = 8, d_ff: int = 1280,
                 max_seq_len: int = 256, max_reading_len: int = 64,
                 component_vocab_size: int = 1024, component_dim: int = 256,
                 char_type_count: int = 7, source_count: int = 16,
                 source_dim: int = 32, edge_dim: int = 384,
                 dropout: float = 0.1,
                 fusion_layers: tuple[int, ...] | list[int] = (),
                 fusion_heads: int = 8, max_span_length: int = 16,
                 use_edge_fusion: bool = False,
                 reading_layers: int = 4, reading_heads: int = 8,
                 reading_d_ff: int = 1024,
                 use_relation_attention: bool = False,
                 relation_layers: int = 2, relation_heads: int = 8,
                 relation_d_ff: int = 1024,
                 use_prior_table: bool = False, prior_table_size: int = 1,
                 prior_dim: int = 4,
                 bos_id: int = 2, eos_id: int = 3,
                 activation_checkpointing: bool = False,
                 reading_chunk_size: int = 0,
                 use_transition_head: bool = True, transition_dim: int = 64,
                 first_order_loss: bool = True,
                 transition_features: str = "context",
                 reading_token_ids: Sequence[int] | Tensor | None = None):
        # The parent builds the shared backbone (embeddings, blocks, fusions,
        # prior table, boundary heads); its reading conv / edge MLP / open
        # generator are removed below because V5 replaces them.
        super().__init__(
            vocab_size, layers=layers, dim=dim, heads=heads, d_ff=d_ff,
            max_seq_len=max_seq_len, max_reading_len=max_reading_len,
            component_vocab_size=component_vocab_size,
            component_dim=component_dim, char_type_count=char_type_count,
            source_count=source_count, source_dim=source_dim,
            edge_dim=edge_dim, dropout=dropout,
            fusion_layers=fusion_layers, fusion_heads=fusion_heads,
            max_span_length=max_span_length, use_open_generator=False,
            use_prior_table=use_prior_table,
            prior_table_size=prior_table_size, prior_dim=prior_dim,
            activation_checkpointing=activation_checkpointing,
            reading_chunk_size=reading_chunk_size,
            use_transition_head=use_transition_head,
            transition_dim=transition_dim,
            first_order_loss=first_order_loss,
            transition_features=transition_features)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.use_edge_fusion = bool(use_edge_fusion)
        self.reading_layers = int(reading_layers)
        self.use_relation_attention = bool(use_relation_attention)
        self.read_loss_weight = 0.5
        self.contrast_loss_weight = 0.1
        self.contrast_margin = 1.0
        self.max_contrast_pairs = 2048
        # L_contrast negative selection. "all" (default) keeps the existing
        # behavior: every same-span rival of a fully-determined group is a
        # negative. "hard" selects only the highest-scoring rival per group,
        # and requires a trusted_gold_span_mask so negatives are drawn only
        # from spans whose supervision is provably complete (no partial
        # labels, no special/creative spans, natural gold candidates only).
        self.contrast_mode = "all"
        # Removed v4 modules: keep them out of state_dict entirely so warm
        # starts report them as unexpected instead of carrying dead weights.
        self.reading_conv = None
        self.reading_norm = None
        self.reading_proj = None
        self.reading_pos = None
        self.edge_mlp = None
        # The v4 conv-pooled reading feature projection is replaced by the
        # span decoder's conditioning; removing it avoids a name clash with
        # warm-started checkpoints of a different edge_dim.
        self.context_proj = None
        if not self.use_edge_fusion:
            self.fusions = nn.ModuleDict()
            self.edge_seed = None
            self.edge_seeds = None
        # Span conditioning and full-text memory, both edge_dim wide.
        self.memory_proj = nn.Linear(dim, edge_dim, bias=False)
        self.span_proj = nn.Linear(edge_dim * 3, edge_dim, bias=False)
        self.boundary_start_marker = nn.Parameter(torch.zeros(edge_dim))
        self.boundary_end_marker = nn.Parameter(torch.zeros(edge_dim))
        # Reading sequence decoder over [BOS, y1..yn]. The output head is an
        # independent projection: the decoder lives in edge_dim while char
        # embeddings live in dim, so a weight tie is shape-invalid whenever
        # dim != edge_dim (the production config).
        self.reading_token_proj = nn.Linear(dim, edge_dim, bias=False)
        self.reading_position = nn.Embedding(max_reading_len + 1, edge_dim)
        self.reading_decoder = nn.ModuleList([
            V5DecoderBlock(edge_dim, reading_heads, reading_d_ff, dropout)
            for _ in range(self.reading_layers)
        ])
        self.reading_final_norm = RMSNorm(edge_dim)
        # The reading output predicts kana tokens + EOS, not the full char
        # vocab: readings are hiragana by construction, so a full 7027-way
        # head wastes ~77x memory on logits that backward must retain
        # (B*E*chunk*width*vocab). When reading_token_ids is given the head
        # is restricted to those tokens plus a dedicated EOS index; labels
        # are remapped via reading_label_map. None falls back to the full
        # vocab for tests / backward compatibility.
        if reading_token_ids is not None:
            ids = sorted({int(i) for i in reading_token_ids})
            # Old-checkpoint compatible layout: output indices 0..n-1 are the
            # emittable tokens in sorted order and index n is the dedicated
            # EOS. Width = len(ids) + 1. Non-kana vocab ids (Latin/digit COPY
            # characters) have no output index of their own; their labels are
            # excluded from q/L_read through the vocab-level emittable mask
            # below, so they never contribute loss or inflate q(e).
            self.reading_output = nn.Linear(edge_dim, len(ids) + 1, bias=False)
            mapping = torch.zeros(vocab_size, dtype=torch.long)
            for out_idx, vocab_idx in enumerate(ids):
                mapping[vocab_idx] = out_idx
            self.register_buffer("reading_label_map", mapping,
                                 persistent=False)
            self.reading_eos_output_index = len(ids)
            # Vocab-level mask: True only for emittable tokens. EOS positions
            # bypass the mask via the labels == eos_id check in
            # _decode_readings.
            emittable = torch.zeros(vocab_size, dtype=torch.bool)
            emittable[torch.tensor(ids, dtype=torch.long)] = True
            self.register_buffer("reading_emittable", emittable,
                                 persistent=False)
        else:
            self.reading_output = nn.Linear(edge_dim, vocab_size, bias=False)
            self.register_buffer(
                "reading_label_map",
                torch.arange(vocab_size, dtype=torch.long),
                persistent=False)
            self.register_buffer(
                "reading_emittable",
                torch.ones(vocab_size, dtype=torch.bool), persistent=False)
            self.reading_eos_output_index = self.eos_id
        # Candidate representation and calibrated unary head.
        self.candidate_mlp = nn.Sequential(
            nn.Linear(edge_dim * 3 + source_dim + 3 + self.prior_dim, edge_dim),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim // 2), nn.SiLU(),
            nn.Linear(edge_dim // 2, edge_dim, bias=False),
        )
        self.unary_head = nn.Linear(edge_dim, 1, bias=False)
        self.relation_blocks = nn.ModuleList([
            RelationAttentionBlock(edge_dim, relation_heads, relation_d_ff,
                                   dropout)
            for _ in range(int(relation_layers) if use_relation_attention else 0)
        ])
        if self.transition_head is not None:
            # z_e lives in the V5 edge_dim now; rebuild the head at that width
            # (the parent built it before its own transition init policy).
            self.transition_head = TransitionHead(
                edge_dim, self.transition_dim,
                use_reading=transition_features == "context_reading")
        nn.init.normal_(self.reading_position.weight, mean=0.0, std=0.02)
        # Sequence-branch calibration: alpha starts at zero so the initial
        # scores equal the pure MLP head and the warm-started model is
        # unchanged. beta/gamma start slightly negative: q(e) grows more
        # negative with reading length, so an unpenalized alpha would bias
        # the path score toward short readings and different segmentations
        # (docs/V5_ARCHITECTURE.md requires the length calibration).
        self.alpha_sequence = nn.Parameter(torch.zeros(()))
        self.beta_reading_length = nn.Parameter(torch.tensor(-0.01))
        self.gamma_surface_length = nn.Parameter(torch.tensor(-0.01))
        if self.first_order_loss and self.transition_head is not None:
            self.reinitialize_transition_for_training()

    @property
    def first_order_enabled(self) -> bool:
        return self.transition_head is not None and self.first_order_loss

    def _span_condition(self, memory: Tensor, starts: Tensor,
                        ends: Tensor) -> Tensor:
        """c_e[384]: start/end hidden + interval mean, with boundary markers."""
        length = memory.size(1)
        # Interval sums come from a cumulative prefix over the text axis, so
        # the only large tensor is [B, L+1, D] (a few MiB) rather than a
        # [B, E, L, D] broadcast product (1.5 GiB at B=64, E=256, L=64).
        # Accumulate in fp32 so BF16 rounding cannot overflow the sum, then
        # cast back to the memory dtype so the reading decoder stays in the
        # autocast dtype (returning fp32 would pull the whole decoder out of
        # bf16, and a mixed-dtype matmul fails outright under plain bf16).
        value = memory.float()
        prefix = torch.cat(
            (value.new_zeros((value.size(0), 1, value.size(2))),
             value.cumsum(1)), dim=1)
        start_hidden = self._gather(value, starts.clamp(0, length - 1))
        end_hidden = self._gather(value, (ends - 1).clamp(0, length - 1))
        span_sum = (self._gather(prefix, ends.clamp(0, length))
                    - self._gather(prefix, starts.clamp(0, length)))
        span_length = (ends - starts).clamp_min(1).unsqueeze(-1).to(value.dtype)
        span_mean = span_sum / span_length
        inputs = torch.cat(
            (start_hidden + self.boundary_start_marker.float(),
             end_hidden + self.boundary_end_marker.float(), span_mean), -1)
        condition = self.span_proj(inputs.to(memory.dtype))
        return condition.to(memory.dtype)

    @staticmethod
    def _reading_labels(reading_ids: Tensor, reading_mask: Tensor,
                        eos_id: int) -> tuple[Tensor, Tensor]:
        """Labels for q(e)/L_read: y_j at steps 0..n-1 and EOS at step n.

        The EOS is INSERTED at position n (a plain cat after the padded width
        would read the next token or a PAD; slicing decoder_input[..., 1:]
        would read a PAD whenever n < read_len). Returns (labels, valid).
        """
        batch, edges, read_len = reading_ids.shape
        width = read_len + 1
        device = reading_ids.device
        lengths = reading_mask.sum(-1)  # [B, E]
        eos = reading_ids.new_full((batch, edges, width), eos_id)
        labels = eos.clone()
        body = (torch.arange(read_len, device=device)[None, None]
                < lengths.unsqueeze(-1))
        labels[:, :, :read_len] = torch.where(
            body, reading_ids, labels[:, :, :read_len])
        valid = (torch.arange(width, device=device)[None, None]
                 <= lengths.unsqueeze(-1))
        return labels, valid

    def _decode_readings(self, reading_ids: Tensor, reading_mask: Tensor,
                         memory: Tensor, memory_mask: Tensor,
                         span_start: Tensor, span_end: Tensor,
                         edge_mask: Tensor,
                         gold_edge_mask: Tensor | None = None,
                         condition: Tensor | None = None
                         ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Teacher-force all candidates through the reading decoder.

        Returns (q per edge including EOS, EOS-position hidden r_e, L_read).
        Each row's text memory and padding mask are shared by its edges.
        q and L_read are computed chunk-locally so the full-vocab logits are
        never materialized for the whole batch at once.

        ``condition`` may be supplied by the caller (forward computes it for
        the candidate features anyway) to avoid a second span aggregation.
        """
        batch, edges, read_len = reading_ids.shape
        width = read_len + 1
        device = reading_ids.device
        bos = reading_ids.new_full((batch, edges, 1), self.bos_id)
        decoder_input = torch.cat((bos, reading_ids), dim=2)

        token_proj = self.reading_token_proj(self.char_embed.weight)
        value = F.embedding(
            decoder_input.reshape(-1, width), token_proj
        ).view(batch, edges, width, -1)
        positions = self.reading_position(
            torch.arange(width, device=device))[None, None]
        value = value + positions
        if condition is None:
            condition = self._span_condition(memory, span_start, span_end)
        value = value + condition[:, :, None, :]
        # Embedding lookups are not autocast-eligible and return fp32; cast
        # the decoder input to the memory dtype so the reading decoder stays
        # in bf16 under AMP instead of running the whole stack in fp32.
        value = value.to(memory.dtype)
        edge_rows = torch.arange(batch, device=device)[:, None].expand(
            batch, edges).reshape(-1)
        flat_valid = edge_mask.reshape(-1).bool()
        active_rows = flat_valid.nonzero(as_tuple=False).flatten()
        total_rows = flat_valid.numel()
        value = value.reshape(-1, width, value.size(-1)).index_select(
            0, active_rows)
        chunk_memory = memory.index_select(0, edge_rows.index_select(0, active_rows))
        chunk_mask = memory_mask.index_select(0, edge_rows.index_select(0, active_rows))

        lengths = reading_mask.sum(-1).reshape(-1)
        lengths_active = lengths.index_select(0, active_rows)
        labels, label_valid = self._reading_labels(
            reading_ids, reading_mask, self.eos_id)
        labels = labels.reshape(-1, width).index_select(0, active_rows)
        label_valid = label_valid.reshape(-1, width).index_select(0, active_rows)
        # Remap full-vocab labels to the (emittable + EOS) output-head indices
        # so the gather targets match the restricted logit width. EOS
        # positions (label == eos_id) map to the dedicated EOS output index;
        # the fallback arange map makes this a no-op when the head is
        # unrestricted. Labels whose vocab id is not emittable (non-kana COPY
        # characters such as Latin letters and digits) are masked from q and
        # L_read via label_valid before remapping, so they neither receive
        # gradient mass nor inflate q(e). The mask lives in vocab-id space:
        # non-emittable ids have no output index at all and must never be
        # gathered as a real token.
        if not bool(self.reading_emittable.all()):
            # EOS positions (label == eos_id) bypass the vocab-level mask:
            # eos_id itself is not a kana token but the EOS step is always
            # supervised, so a non-kana COPY edge keeps exactly its EOS
            # target.
            is_eos = labels == self.eos_id
            label_valid = label_valid & (
                is_eos | self.reading_emittable[
                    labels.clamp(0, self.reading_emittable.numel() - 1)])
        labels = torch.where(
            labels == self.eos_id,
            labels.new_full(labels.shape, self.reading_eos_output_index),
            self.reading_label_map[labels])
        gold_flat = (gold_edge_mask.reshape(-1).bool().index_select(
            0, active_rows) if gold_edge_mask is not None else None)

        dense_chunk_size = self.reading_chunk_size or total_rows
        chunk_size = dense_chunk_size
        q_out = []
        eos_out = []
        use_ckpt = (self.activation_checkpointing and self.training
                    and torch.is_grad_enabled())
        for start in range(0, value.size(0), chunk_size):
            chunk = value[start:start + chunk_size]
            rows_memory = chunk_memory[start:start + chunk_size]
            rows_mask = chunk_mask[start:start + chunk_size]
            for block in self.reading_decoder:
                # Checkpoint the decoder blocks: backward recomputes the
                # q/k/v/attention/ffn intermediates instead of retaining
                # them for every B*E candidate row.  Without this the 4
                # layers' activations across all chunks reach ~11 GiB at
                # B=64, E=256; with it only the block inputs are kept.
                if use_ckpt:
                    chunk = checkpoint(block, chunk, rows_memory, rows_mask,
                                       use_reentrant=False)
                else:
                    chunk = block(chunk, rows_memory, rows_mask)
            chunk = self.reading_final_norm(chunk)
            # Per-chunk logits exist only here: q and L_read consume them
            # inside the loop, so full-vocab logits are never held for the
            # whole batch at once (they alone reach 7+ GiB at B=64, E=256).
            logits = self.reading_output(chunk).float()
            log_probs = F.log_softmax(logits, dim=-1)
            chunk_labels = labels[start:start + chunk_size]
            picked = torch.gather(
                log_probs, 2,
                chunk_labels.clamp(0, log_probs.size(-1) - 1)[..., None]
            ).squeeze(-1)
            picked = picked * label_valid[start:start + chunk_size].to(
                picked.dtype)
            q_out.append(picked.sum(1))
            eos_out.append(chunk[
                torch.arange(chunk.size(0), device=device),
                lengths_active[start:start + chunk_size].clamp_max(width - 1)])
        q_active = torch.cat(q_out, 0)
        sequence_log_probs = q_active.new_zeros((total_rows,)).index_copy(
            0, active_rows, q_active).view(batch, edges)
        read_loss = None
        if gold_flat is not None:
            # Preserve the old chunk-weighted objective while decoding only
            # active rows: each active edge maps back to its original dense
            # chunk, whose gold token mean is averaged across non-empty chunks.
            original_chunk = active_rows // chunk_size
            chunk_total = q_active.new_zeros((
                (total_rows + chunk_size - 1) // chunk_size,))
            chunk_count = q_active.new_zeros(chunk_total.shape)
            # q(e) is already the sum over valid target positions, so use the
            # same per-edge token count as the original decoder and aggregate
            # by original dense chunk without any device-to-host decision.
            edge_token_count = label_valid.sum(-1).to(q_active.dtype)
            edge_loss_sum = q_active * gold_flat.to(q_active.dtype)
            chunk_total.scatter_add_(0, original_chunk, edge_loss_sum)
            chunk_count.scatter_add_(
                0, original_chunk,
                gold_flat.to(q_active.dtype) * edge_token_count)
            nonempty = (chunk_count > 0).to(q_active.dtype)
            read_loss = -(
                (chunk_total / chunk_count.clamp_min(1)) * nonempty
            ).sum() / nonempty.sum().clamp_min(1)
        eos_hidden = torch.cat(eos_out, 0).new_zeros(
            (total_rows, self.reading_final_norm.weight.numel())).index_copy(
                0, active_rows, torch.cat(eos_out, 0)).view(batch, edges, -1)
        return sequence_log_probs, eos_hidden, read_loss

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
                gold_edge_mask: Tensor | None = None,
                edge_pack_ids: Tensor | None = None,
                trusted_gold_span_mask: Tensor | None = None,
                target_offset: Tensor | int | None = None,
                target_length: Tensor | int | None = None) -> V5ModelOutput:
        if edge_mask is None or edge_source_ids is None:
            raise ValueError("edge_mask and edge_source_ids are required")
        if edge_reading_ids is None or edge_reading_mask is None:
            raise ValueError("V5 requires dense reading ids and mask")
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
            span_start = edge_start
            span_end = edge_end

        hidden = self._input_embedding(input_ids, component_ids, char_type_ids)
        source = self.source_embed(
            edge_source_ids.clamp(0, self.source_embed.num_embeddings - 1))
        reading_length_feature = (
            edge_reading_mask.sum(-1).clamp_min(1).float().log1p().unsqueeze(-1))
        surface_length_feature = (
            (edge_end - edge_start).clamp_min(1).float().log1p().unsqueeze(-1))
        prior = (edge_prior.float().unsqueeze(-1) if edge_prior is not None
                 else torch.zeros_like(surface_length_feature))

        if self.use_edge_fusion:
            coverage = self._coverage(span_start, span_end, edge_mask,
                                      input_ids.size(1))
            static = torch.cat(
                (source, surface_length_feature, prior), -1).expand(
                    -1, edge_mask.size(1), -1)
        for layer_index, block in enumerate(self.blocks, start=1):
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, attention_mask,
                                    use_reentrant=False)
            else:
                hidden = block(hidden, attention_mask)
            if self.use_edge_fusion:
                fusion = self.fusions.get(str(layer_index))
                if fusion is not None:
                    with torch.no_grad():
                        context_seed = self._span_features(
                            hidden, span_start, span_end)
                    seed_module = (self.edge_seeds[str(layer_index)]
                                   if self.edge_seeds is not None
                                   and str(layer_index) in self.edge_seeds
                                   else self.edge_seed)
                    if seed_module is not None:
                        seed = seed_module(torch.cat((context_seed, static), -1))
                        hidden = fusion(hidden, seed, coverage, attention_mask)
        hidden = self.final_norm(hidden)

        memory = self.memory_proj(hidden)
        # Computed once: feeds both the reading decoder and the candidate
        # features (the interval aggregation is the expensive part).
        condition = self._span_condition(memory, span_start, span_end)
        want_read_loss = (gold_edge_mask is not None and self.read_loss_weight > 0
                          and (self.training or torch.is_grad_enabled()))
        sequence_log_probs, eos_hidden, read_loss = self._decode_readings(
            edge_reading_ids, edge_reading_mask, memory, attention_mask,
            span_start, span_end, edge_mask,
            gold_edge_mask=(gold_edge_mask if want_read_loss else None),
            condition=condition)
        # Length features are computed in fp32 (log1p of counts); cast the
        # assembled feature vector back so the candidate MLP runs in the
        # model dtype instead of hitting a mixed-dtype matmul.
        candidate_features = torch.cat(
            (condition, eos_hidden, condition * eos_hidden, source,
             reading_length_feature, surface_length_feature, prior), -1
        ).to(condition.dtype)
        if self.prior_table is not None and edge_pack_ids is not None:
            candidate_features = torch.cat(
                (candidate_features, self.prior_table(
                    edge_pack_ids.clamp(0, self.prior_table.num_embeddings - 1))),
                -1)
        z = self.candidate_mlp(candidate_features)
        for block in self.relation_blocks:
            relation = block.relation_types(span_start, span_end, edge_mask)
            distance = block.boundary_distance(span_start, span_end)
            z = block(z, relation, distance)
        u = self.unary_head(z).squeeze(-1)
        u = (u + self.alpha_sequence * sequence_log_probs
             + self.beta_reading_length * reading_length_feature.squeeze(-1)
             + self.gamma_surface_length * surface_length_feature.squeeze(-1))
        if edge_locked is not None and edge_prior is not None:
            u = torch.where(edge_locked.bool(), edge_prior.float(), u)
        u = u.masked_fill(~edge_mask.bool(), torch.finfo(u.dtype).min)
        u = u.float()

        transition_scores = None
        if self.transition_head is not None:
            # Detach only when the head is inference-only; with first_order_loss
            # the transition scores must carry gradients into the path loss.
            grad_ctx = (torch.no_grad()
                        if (self.training and not self.first_order_loss)
                        else nullcontext())
            with grad_ctx:
                left_proj, right_proj = self.transition_head.project_candidates(
                    z, z)
                trans_matrix = torch.bmm(left_proj, right_proj.transpose(1, 2))
                is_adjacent = (
                    (edge_end.unsqueeze(2) == edge_start.unsqueeze(1))
                    & edge_mask.unsqueeze(2).bool()
                    & edge_mask.unsqueeze(1).bool())
                transition_scores = trans_matrix.masked_fill(~is_adjacent, 0.0)

        contrast_loss = None
        contrast_negatives = None
        if gold_edge_mask is not None and (self.training or torch.is_grad_enabled()):
            if self.contrast_loss_weight > 0:
                trusted = (
                    trusted_gold_span_mask.bool() & edge_mask.bool()
                    if trusted_gold_span_mask is not None else None)
                contrast_loss, contrast_negatives = self._contrast_loss(
                    u, gold_edge_mask, edge_mask, span_start, span_end,
                    source_ids=edge_source_ids, trusted_mask=trusted,
                    margin=self.contrast_margin,
                    max_pairs=self.max_contrast_pairs,
                    mode=self.contrast_mode)

        return V5ModelOutput(
            edge_scores=u,
            hidden=hidden,
            edge_features=z,
            sequence_log_probs=sequence_log_probs,
            transition_scores=transition_scores,
            first_order_loss=self.first_order_loss,
            read_loss=read_loss,
            contrast_loss=contrast_loss,
            contrast_negatives=contrast_negatives)

    @staticmethod
    def _contrast_loss(scores: Tensor, gold_edge_mask: Tensor,
                       edge_mask: Tensor, starts: Tensor,
                       ends: Tensor, *,
                       source_ids: Tensor | None = None,
                       trusted_mask: Tensor | None = None,
                       margin: float = 1.0,
                       max_pairs: int = 2048,
                       mode: str = "all") -> tuple[Tensor, Tensor | None]:
        """L_contrast: margin loss between gold and same-span rivals.

        Only fully-determined groups (exactly one gold-compatible candidate
        per span) contribute; capped at ``max_pairs`` per batch.

        ``mode`` selects the negative per group:
        - "all" (default): every rival is a negative (existing behavior).
        - "hard": only the highest-scoring rival is a negative, and groups
          must lie inside ``trusted_mask`` (spans with provably complete
          supervision) and contain no gold-injected edge, so an imperfect
          gold reading can never be turned into a wrong negative signal.
          With no trusted mask this mode contributes no pairs (fail-closed).

        Returns (loss, flat negative index tensor [N, 2] of (row, edge)).
        """
        mode = str(mode).lower()
        if mode not in {"all", "hard"}:
            raise ValueError(f"contrast_mode must be 'all' or 'hard'; got {mode!r}")
        gold_id = SOURCE_TO_ID["gold"]
        gold = (gold_edge_mask.bool() & edge_mask.bool()).detach().cpu().tolist()
        valid = edge_mask.bool().detach().cpu().tolist()
        starts_cpu = starts.detach().cpu().tolist()
        ends_cpu = ends.detach().cpu().tolist()
        source_cpu = (source_ids.detach().cpu().tolist()
                      if source_ids is not None else None)
        trusted_cpu = (trusted_mask.bool().detach().cpu().tolist()
                       if trusted_mask is not None else None)
        rows: list[int] = []
        gold_indices: list[int] = []
        rival_indices: list[int] = []
        for row in range(scores.size(0)):
            groups: dict[tuple[int, int], list[int]] = {}
            for index in range(edge_mask.shape[1]):
                if not valid[row][index]:
                    continue
                groups.setdefault(
                    (starts_cpu[row][index], ends_cpu[row][index]), []
                ).append(index)
            for (start, end), indices in groups.items():
                gold_flags = [gold[row][index] for index in indices]
                if gold_flags.count(True) != 1:
                    continue
                gold_index = indices[gold_flags.index(True)]
                if mode == "hard":
                    # Trusted supervision only: without a trusted mask, or if
                    # the group contains a gold-injected (dataset-supplied)
                    # edge, the gold reading is not provably complete and the
                    # group is skipped rather than turned into negatives.
                    if trusted_cpu is None or not trusted_cpu[row][gold_index]:
                        continue
                    if source_cpu is not None and any(
                            int(source_cpu[row][index]) == gold_id
                            for index in indices):
                        continue
                    rival_scores = scores[row, indices].detach()
                    rival_flags = [
                        index for index in indices
                        if index != gold_index
                        and (source_cpu is None
                             or int(source_cpu[row][index]) != gold_id)
                    ]
                    if not rival_flags:
                        continue
                    # Pick the highest-scoring rival directly by index.
                    best_index = max(
                        rival_flags, key=lambda i: float(rival_scores[
                            indices.index(i)]))
                    rows.append(row)
                    gold_indices.append(gold_index)
                    rival_indices.append(best_index)
                    if len(rows) >= max_pairs:
                        break
                    continue
                for index in indices:
                    if index != gold_index:
                        rows.append(row)
                        gold_indices.append(gold_index)
                        rival_indices.append(index)
                if len(rows) >= max_pairs:
                    break
            if len(rows) >= max_pairs:
                break
        if not rows:
            return scores.new_zeros(()), None
        gold_scores = scores[rows, gold_indices]
        rival_scores = scores[rows, rival_indices]
        loss = F.softplus(margin - (gold_scores - rival_scores)).mean()
        negatives = torch.stack(
            (torch.as_tensor(rows, dtype=torch.long, device=scores.device),
             torch.as_tensor(rival_indices, dtype=torch.long,
                             device=scores.device)), dim=1)
        return loss, negatives


def parameter_count_v5(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def kana_token_ids(vocab: dict[str, int]) -> list[int]:
    """Vocab ids of tokens the reading decoder may emit: kana + ー.

    Readings are normalized to hiragana, but include katakana and the long
    vowel mark in practice. Restricting the output head to these (plus a
    dedicated EOS index) cuts the full-vocab logits 7027 -> ~180, which is
    where the B*E*width*vocab activation memory actually lives. Vocab ids
    outside this set (Latin letters, digits, kanji) are not emittable: edges
    whose reading contains them (non-kana COPY, unmatched alphanumeric
    fallback) are scored by the candidate MLP on z_e, not by q(e), whose
    non-emittable label positions are masked out.
    """
    ids: list[int] = []
    for token, index in vocab.items():
        if len(token) != 1:
            continue
        code = ord(token)
        if (0x3041 <= code <= 0x3096          # hiragana
                or 0x30A1 <= code <= 0x30FC   # katakana + ー
                or token == "ー"):
            ids.append(int(index))
    return sorted(set(ids))

"""Tiny Cross-Encoder Reranker for kashi-g2p v4."""

from __future__ import annotations

import math
from typing import Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TinyCrossEncoder(nn.Module):
    """A lightweight (~1.3M parameter) 2-layer Transformer Cross-Encoder.
    
    Inputs:
        input_ids: [B, L] token sequence containing:
                   [CLS] surface_text [SEP] candidate_reading [SEP]
        token_type_ids: [B, L] 0 for surface text, 1 for reading
        attention_mask: [B, L] 1 for real tokens, 0 for pad
    Output:
        scores: [B] scalar matching score
    """

    def __init__(
        self,
        vocab_size: int = 7027,
        d_model: int = 128,
        nhead: int = 4,
        d_ff: int = 512,
        num_layers: int = 2,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.char_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.token_type_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.score_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.score_head.bias)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        b, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        embeds = (
            self.char_embed(input_ids)
            + self.token_type_embed(token_type_ids)
            + self.pos_embed(positions)
        )
        embeds = self.dropout(self.norm_in(embeds))

        # PyTorch TransformerEncoder expects src_key_padding_mask where True = masked out (pad)
        padding_mask = ~attention_mask.bool()

        hidden = self.encoder(embeds, src_key_padding_mask=padding_mask)
        cls_hidden = self.norm_out(hidden[:, 0, :])  # [B, d_model]
        scores = self.score_head(cls_hidden).squeeze(-1)  # [B]
        return scores


class CrossEncoderTokenizer:
    """Helper to convert (text, candidate_reading) pairs into tensors."""

    def __init__(self, vocab_path: str = "models/bert-base-japanese-char-v3/vocab.txt"):
        self.vocab = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vocab[line.strip()] = i
        self.pad_id = self.vocab.get("[PAD]", 0)
        self.unk_id = self.vocab.get("[UNK]", 1)
        self.cls_id = self.vocab.get("[CLS]", 2)
        self.sep_id = self.vocab.get("[SEP]", 3)

    def encode_pair(self, text: str, reading: str, max_len: int = 128) -> tuple[list[int], list[int], list[int]]:
        text_tokens = [self.vocab.get(c, self.unk_id) for c in text]
        reading_tokens = [self.vocab.get(c, self.unk_id) for c in reading]

        # [CLS] text [SEP] reading [SEP]
        input_ids = [self.cls_id] + text_tokens + [self.sep_id] + reading_tokens + [self.sep_id]
        token_type_ids = [0] * (len(text_tokens) + 2) + [1] * (len(reading_tokens) + 1)
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            token_type_ids = token_type_ids[:max_len]
            attention_mask = attention_mask[:max_len]

        return input_ids, token_type_ids, attention_mask

    def collate_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        max_len: int = 128,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded = [self.encode_pair(t, r, max_len) for t, r in pairs]
        batch_len = min(max_len, max(len(e[0]) for e in encoded))

        b_ids = torch.full((len(pairs), batch_len), self.pad_id, dtype=torch.long)
        b_types = torch.zeros((len(pairs), batch_len), dtype=torch.long)
        b_mask = torch.zeros((len(pairs), batch_len), dtype=torch.long)

        for i, (ids, types, mask) in enumerate(encoded):
            cur_len = min(len(ids), batch_len)
            b_ids[i, :cur_len] = torch.tensor(ids[:cur_len], dtype=torch.long)
            b_types[i, :cur_len] = torch.tensor(types[:cur_len], dtype=torch.long)
            b_mask[i, :cur_len] = torch.tensor(mask[:cur_len], dtype=torch.long)

        return b_ids.to(device), b_types.to(device), b_mask.to(device)

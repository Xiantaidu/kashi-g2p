"""Exp32 Warm-Start 512d Cross-Encoder for kashi-g2p v4."""

from __future__ import annotations

from typing import Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import EncoderBlock, RMSNorm
from .reranker import CrossEncoderTokenizer


class Exp32WarmStartCrossEncoder(nn.Module):
    """512-dimensional 2-layer Cross-Encoder matching exp32's native Transformer architecture.
    
    100% inherits:
      - char_embed (7027, 512)
      - blocks[0] (Layer 0 Transformer: RMSNorm, RoPE, SwiGLU)
      - blocks[1] (Layer 1 Transformer: RMSNorm, RoPE, SwiGLU)
    """

    def __init__(
        self,
        vocab_size: int = 7027,
        dim: int = 512,
        heads: int = 8,
        d_ff: int = 1280,
        num_layers: int = 6,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.char_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.token_type_embed = nn.Embedding(2, dim)
        # CRITICAL: Zero-init token_type_embed so it introduces 0 noise to pretrained char_embed (std 0.024)
        nn.init.zeros_(self.token_type_embed.weight)

        self.norm_in = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            EncoderBlock(dim, heads, d_ff, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm_out = RMSNorm(dim)
        self.norm_max = RMSNorm(dim)

        # 3 numerical features: [rel_score, score_gap, candidate_len_diff]
        self.feature_proj = nn.Linear(3, 64)
        # Cross interaction: [surf_mean (512), read_mean (512), |diff| (512), prod (512), read_max (512), feats (64)] = 2624
        self.score_head = nn.Sequential(
            nn.Linear(dim * 5 + 64, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self._init_head()

    def _init_head(self):
        for m in (self.feature_proj, self.score_head):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_exp32_weights(self, checkpoint_path: str, freeze_bottom_layers: int = 2):
        """Zero-loss transfer of exp32 pretrained char_embed and all layers up to len(self.blocks)."""
        print(f"Loading exp32 weights from {checkpoint_path}...")
        payload = torch.load(checkpoint_path, map_location="cpu")
        model_state = payload["model"]

        # 1. char_embed
        self.char_embed.weight.data.copy_(model_state["char_embed.weight"])

        # 2. blocks[0] through blocks[len(self.blocks) - 1]
        loaded_count = 0
        for l in range(len(self.blocks)):
            block_prefix = f"blocks.{l}."
            block_dict = {}
            for k, v in model_state.items():
                if k.startswith(block_prefix):
                    sub_k = k[len(block_prefix):]
                    block_dict[sub_k] = v
            if block_dict:
                self.blocks[l].load_state_dict(block_dict, strict=True)
                loaded_count += 1
                print(f"  Successfully loaded exp32 Layer {l} ({len(block_dict)} parameters).")

        if freeze_bottom_layers > 0:
            self.char_embed.weight.requires_grad = False
            for l in range(min(freeze_bottom_layers, len(self.blocks))):
                for p in self.blocks[l].parameters():
                    p.requires_grad = False
            print(f"  Frozen char_embed and bottom {freeze_bottom_layers} layers. Top {len(self.blocks) - freeze_bottom_layers} layers trainable!")

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        numerical_features: Tensor,
    ) -> Tensor:
        batch, length = input_ids.shape
        x = self.char_embed(input_ids) + self.token_type_embed(token_type_ids)
        x = self.dropout(self.norm_in(x))

        for block in self.blocks:
            x = block(x, attention_mask)

        # Segment-aware cross-interaction pooling
        surf_valid = (token_type_ids == 0) & (input_ids > 3) & (attention_mask == 1)
        surf_valid = torch.where(surf_valid.any(dim=-1, keepdim=True), surf_valid, (token_type_ids == 0) & (attention_mask == 1))
        surf_mask = surf_valid.unsqueeze(-1).float()

        read_valid = (token_type_ids == 1) & (input_ids > 3) & (attention_mask == 1)
        read_valid = torch.where(read_valid.any(dim=-1, keepdim=True), read_valid, (token_type_ids == 1) & (attention_mask == 1))
        read_mask = read_valid.unsqueeze(-1).float()

        surf_rep = self.norm_out((x * surf_mask).sum(dim=1) / surf_mask.sum(dim=1).clamp_min(1.0))
        read_rep = self.norm_out((x * read_mask).sum(dim=1) / read_mask.sum(dim=1).clamp_min(1.0))

        diff_rep = torch.abs(surf_rep - read_rep)
        prod_rep = surf_rep * read_rep

        # Salient local token anomaly pooling: max-pooling over reading tokens
        # Prevents a 1-character discrepancy from being diluted by 95% identical characters!
        read_x = x.masked_fill(read_mask == 0, -1e4)
        read_max = self.norm_max(read_x.max(dim=1).values)

        feat_rep = F.silu(self.feature_proj(numerical_features))
        combined = torch.cat([surf_rep, read_rep, diff_rep, prod_rep, read_max, feat_rep], dim=-1)
        scores = self.score_head(combined).squeeze(-1)
        return scores

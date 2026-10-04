"""Native Tohoku Japanese BERT Cross-Encoder for kashi-g2p v4."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import BertModel


class BertCrossEncoder(nn.Module):
    """Native Tohoku Japanese Character BERT Cross-Encoder (110M parameters).

    Takes [CLS] Surface Text [SEP] Candidate Reading [SEP]
    Pretrained on Japanese Wikipedia character-level with true bidirectional cross-attention.
    Combines BERT pooled [CLS] representation with beam search numerical features.
    """

    def __init__(
        self,
        bert_model_path: str = "models/bert-base-japanese-char-v3",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        hidden_size = self.bert.config.hidden_size  # 768

        self.dropout = nn.Dropout(dropout)
        self.feature_proj = nn.Linear(3, 64)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size + 64, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        for m in (self.feature_proj, self.score_head):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        numerical_features: Tensor,
    ) -> Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls_rep = outputs.pooler_output  # [B, 768] (pretrained pooled [CLS])
        cls_rep = self.dropout(cls_rep)

        feat_rep = F.silu(self.feature_proj(numerical_features))
        combined = torch.cat([cls_rep, feat_rep], dim=-1)
        scores = self.score_head(combined).squeeze(-1)
        return scores

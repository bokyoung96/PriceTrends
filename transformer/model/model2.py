from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model1 import GLU, GRN, VSN, PosEncoding


class MultiHeadTransformer(nn.Module):
    def __init__(
        self,
        n_feat: int,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        drop: float = 0.1,
        n_class: int = 2,
        max_len: int = 5000,
    ):
        super().__init__()
        self.vsn = VSN(n_feat, 1, d_model, drop)
        self.pe = PosEncoding(d_model, max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.shared_grn = GRN(d_model, d_ff, d_model, drop)
        self.norm = nn.LayerNorm(d_model)

        self.head_cls = nn.Linear(d_model, n_class)
        self.head_dd = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_ff, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE: x: (batch, seq, n_feat)
        x = self.vsn(x)
        x = self.pe(x)
        x = self.enc(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        x = self.shared_grn(x)
        x = self.norm(x)

        logits = self.head_cls(x)
        dd_pred = self.head_dd(x).squeeze(-1)
        return logits, dd_pred

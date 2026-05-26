import torch
import torch.nn as nn
import torch.nn.functional as F


class StoryPooling(nn.Module):

    def __init__(

        self,

        hidden_dim=768,

        output_dim=512,

        dropout=0.3     # ALIGNED with rest of model
    ):

        super().__init__()

        # =================================================
        # CLS TOKEN
        # =================================================

        self.cls_token = nn.Parameter(

            torch.randn(
                1,
                1,
                hidden_dim
            )
        )

        # =================================================
        # ATTENTION POOLING
        # =================================================

        self.attention = nn.MultiheadAttention(

            embed_dim=hidden_dim,

            num_heads=8,

            dropout=dropout,

            batch_first=True
        )

        # =================================================
        # NORMALIZATION
        # =================================================

        self.norm = nn.LayerNorm(hidden_dim)

        # =================================================
        # DROPOUT
        # =================================================

        self.dropout = nn.Dropout(dropout)

        # =================================================
        # PROJECTION
        # =================================================

        self.projection = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim)
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, fused_tokens):

        B, T, Q, D = fused_tokens.shape

        # =================================================
        # FLATTEN TOKENS
        # =================================================

        x = fused_tokens.view(
            B,
            T * Q,
            D
        )

        # =================================================
        # CLS TOKEN
        # =================================================

        cls = self.cls_token.expand(
            B,
            -1,
            -1
        )

        # =================================================
        # ATTENTION POOLING
        # =================================================

        pooled, _ = self.attention(

            cls,

            x,

            x
        )

        pooled = pooled.squeeze(1)

        # =================================================
        # NORMALIZATION
        # =================================================

        pooled = self.norm(pooled)

        # =================================================
        # DROPOUT
        # =================================================

        pooled = self.dropout(pooled)

        # =================================================
        # PROJECTION
        # =================================================

        pooled = self.projection(pooled)

        # =================================================
        # EMBEDDING NOISE
        # =================================================

        if self.training:

            pooled = pooled + (
                0.02
                * torch.randn_like(pooled)
            )

        # =================================================
        # NORMALIZATION
        # =================================================

        pooled = F.normalize(
            pooled,
            dim=-1
        )

        return pooled
import torch
import torch.nn as nn
import torch.nn.functional as F


class StoryHead(nn.Module):

    def __init__(
        self,
        hidden_dim=768,
        output_dim=512,
        dropout=0.2
    ):

        super().__init__()

        self.projection = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, fused_tokens):

        """
        fused_tokens:
        (B, 3, 32, 768)
        """

        B, T, Q, D = fused_tokens.shape

        # =================================================
        # FLATTEN TOKENS
        # =================================================

        x = fused_tokens.view(B, T * Q, D)

        # =================================================
        # GLOBAL AVERAGE POOLING
        # =================================================

        x = x.mean(dim=1)

        # =================================================
        # PROJECTION
        # =================================================

        x = self.projection(x)

        # =================================================
        # NORMALIZATION
        # =================================================

        x = F.normalize(x, dim=-1)

        return x
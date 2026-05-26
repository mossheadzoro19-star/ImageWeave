import torch
import torch.nn as nn


class CrossImageAttention(nn.Module):

    def __init__(
        self,
        hidden_dim=768,
        num_heads=8,
        num_layers=2,          # REDUCED
        dropout=0.3            # ADDED
    ):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=hidden_dim,

            nhead=num_heads,

            dropout=dropout,   # IMPORTANT

            batch_first=True,

            norm_first=True,

            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, q_tokens):

        B, T, Q, D = q_tokens.shape

        # =================================================
        # FLATTEN TEMPORAL TOKENS
        # =================================================

        x = q_tokens.view(B, T * Q, D)

        # =================================================
        # TRANSFORMER
        # =================================================

        x = self.transformer(x)

        # =================================================
        # DROPOUT
        # =================================================

        x = self.dropout(x)

        # =================================================
        # RESHAPE BACK
        # =================================================

        x = x.view(B, T, Q, D)

        return x
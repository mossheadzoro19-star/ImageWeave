import torch
import torch.nn as nn


class QFormer(nn.Module):

    def __init__(

        self,

        num_queries=8,         # REDUCED

        hidden_dim=768,

        num_heads=8,

        num_layers=2,          # REDUCED

        dropout=0.35           # INCREASED to 0.35
    ):

        super().__init__()

        # =================================================
        # QUERY TOKENS
        # =================================================

        self.query_tokens = nn.Parameter(

            torch.randn(
                num_queries,
                hidden_dim
            )
        )

        # =================================================
        # TRANSFORMER
        # =================================================

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=hidden_dim,

            nhead=num_heads,

            dropout=dropout,

            batch_first=True,

            norm_first=True,

            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers
        )

        # =================================================
        # DROPOUT
        # =================================================

        self.dropout = nn.Dropout(dropout)

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, image_tokens):

        B, T, N, D = image_tokens.shape

        outputs = []

        context = None

        for t in range(T):

            tokens = image_tokens[:, t]

            queries = self.query_tokens.unsqueeze(0).expand(

                B,

                -1,

                -1
            )

            # =================================================
            # TEMPORAL CONTEXT
            # =================================================

            if context is not None:

                queries = queries + context

            # =================================================
            # CONCAT QUERY + IMAGE TOKENS
            # =================================================

            x = torch.cat(
                [queries, tokens],
                dim=1
            )

            # =================================================
            # TRANSFORMER
            # =================================================

            x = self.transformer(x)

            # =================================================
            # QUERY OUTPUTS
            # =================================================

            q = x[:, :queries.shape[1]]

            # =================================================
            # DROPOUT
            # =================================================

            q = self.dropout(q)

            outputs.append(q)

            # =================================================
            # TEMPORAL MEMORY
            # =================================================

            context = q.mean(
                dim=1,
                keepdim=True
            )

        outputs = torch.stack(
            outputs,
            dim=1
        )

        return outputs